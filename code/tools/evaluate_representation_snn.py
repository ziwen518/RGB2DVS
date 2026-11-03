from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset

from isgd.data import make_event_dataset
from isgd.model import PureSpikeResNet


class OperationCounter:
    """Estimate Conv2d synaptic operations from measured presynaptic activity."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.rows: dict[str, dict[str, float]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name: str, module: nn.Conv2d):
        def hook(_module, inputs, output):
            x = inputs[0].detach()
            batch = output.shape[0]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            kernel_ops *= module.in_channels // module.groups
            dense_per_sample = float(output[0].numel() * kernel_ops)
            activity = float(x.ne(0).float().mean())
            row = self.rows.setdefault(name, {"calls": 0.0, "dense": 0.0, "synops": 0.0, "activity_sum": 0.0})
            row["calls"] += 1.0
            row["dense"] += dense_per_sample
            row["activity_sum"] += activity
            # The stem consumes continuous event counts; later convolutions
            # consume binary presynaptic spikes and therefore use ACs.
            row["synops"] += dense_per_sample if name == "stem" else dense_per_sample * activity
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self):
        layers = {}
        total_dense = total_mac = total_ac = 0.0
        for name, row in self.rows.items():
            dense = row["dense"]
            activity = row["activity_sum"] / max(row["calls"], 1.0)
            if name == "stem":
                mac, ac = row["synops"], 0.0
            else:
                mac, ac = 0.0, row["synops"]
            total_dense += dense
            total_mac += mac
            total_ac += ac
            layers[name] = {
                "calls_per_sample": int(row["calls"]),
                "input_activity": activity,
                "dense_ops": dense,
                "macs": mac,
                "acs": ac,
            }
        mac_pj, ac_pj = 4.6, 0.9
        energy_pj = total_mac * mac_pj + total_ac * ac_pj
        dense_energy_pj = total_dense * mac_pj
        return {
            "assumption": {"fp32_mac_pj": mac_pj, "fp32_ac_pj": ac_pj, "technology": "45nm estimate"},
            "macs_per_sample": total_mac,
            "acs_per_sample": total_ac,
            "synops_per_sample": total_mac + total_ac,
            "dense_ops_per_sample": total_dense,
            "estimated_energy_mj": energy_pj / 1.0e9,
            "dense_mac_energy_mj": dense_energy_pj / 1.0e9,
            "estimated_energy_ratio": energy_pj / max(dense_energy_pj, 1.0),
            "layers": layers,
        }


class SpikeActivityCounter:
    """Measure binary activation rates emitted by every spiking neuron."""

    def __init__(self, model: nn.Module):
        self.rows: dict[str, dict[str, float]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "PLIF":
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            spikes = output[0] if isinstance(output, tuple) else output
            row = self.rows.setdefault(name, {"calls": 0.0, "rate_sum": 0.0})
            row["calls"] += 1.0
            row["rate_sum"] += float(spikes.detach().float().mean())
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self):
        return {
            name: row["rate_sum"] / max(row["calls"], 1.0)
            for name, row in self.rows.items()
        }


@torch.inference_mode()
def collect(model, loader, device, counters=()):
    model.eval()
    features, labels = [], []
    for batch_index, (events, target, *_) in enumerate(loader):
        # Count a single representative batch; hooks report per-sample ops.
        if batch_index == 1:
            for counter in counters:
                counter.close()
        features.append(model(events.to(device, non_blocking=True))["embedding"].float().cpu())
        labels.append(torch.as_tensor(target))
    return torch.cat(features), torch.cat(labels)


def knn(bank, bank_y, query, query_y, k):
    indices = (query @ bank.t()).topk(k, dim=-1).indices
    prediction = torch.mode(bank_y[indices], dim=-1).values
    return float((prediction == query_y).float().mean())


def few_shot_knn(bank, bank_y, query, query_y, shots, repeats, seed):
    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(repeats):
        selected = []
        for class_index in bank_y.unique(sorted=True):
            candidates = torch.where(bank_y == class_index)[0]
            order = torch.randperm(len(candidates), generator=generator)
            selected.append(candidates[order[: min(shots, len(candidates))]])
        selected = torch.cat(selected)
        scores.append(knn(bank[selected], bank_y[selected], query, query_y, min(5, shots)))
    values = torch.tensor(scores)
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)), "repeats": repeats}


def retrieval_metrics(bank, bank_y, query, query_y, ks=(1, 5, 10)):
    recalls = {k: [] for k in ks}
    average_precision = []
    chunk_size = 128
    for start in range(0, len(query), chunk_size):
        similarity = query[start : start + chunk_size] @ bank.t()
        order = similarity.argsort(dim=-1, descending=True)
        relevant = bank_y[order].eq(query_y[start : start + chunk_size, None])
        for k in ks:
            recalls[k].append(relevant[:, :k].any(dim=-1).float())
        precision = relevant.cumsum(dim=-1) / torch.arange(1, len(bank) + 1).float()
        average_precision.append(
            (precision * relevant).sum(dim=-1) / relevant.sum(dim=-1).clamp_min(1)
        )
    return {
        "recall": {str(k): float(torch.cat(values).mean()) for k, values in recalls.items()},
        "map": float(torch.cat(average_precision).mean()),
    }


def linear_probe(train_x, train_y, test_x, test_y, classes, epochs, device, seed):
    # Select the epoch on an internal stratified validation split, never test.
    generator = torch.Generator().manual_seed(seed)
    train_indices, validation_indices = [], []
    for class_index in range(classes):
        candidates = torch.where(train_y == class_index)[0]
        candidates = candidates[torch.randperm(len(candidates), generator=generator)]
        validation_count = max(1, len(candidates) // 10)
        validation_indices.append(candidates[:validation_count])
        train_indices.append(candidates[validation_count:])
    train_indices = torch.cat(train_indices)
    validation_indices = torch.cat(validation_indices)
    fit_x, fit_y = train_x[train_indices], train_y[train_indices]
    validation_x, validation_y = train_x[validation_indices], train_y[validation_indices]
    probe = nn.Linear(train_x.shape[-1], classes).to(device)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1.0e-2, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader = DataLoader(TensorDataset(fit_x, fit_y), 512, shuffle=True, generator=generator)
    train_x_device = fit_x.to(device)
    mean = train_x_device.mean(0, keepdim=True)
    std = train_x_device.std(0, keepdim=True).clamp_min(1.0e-4)
    best_validation = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        probe.train()
        for features, labels in loader:
            features = ((features.to(device) - mean) / std).clamp(-10.0, 10.0)
            loss = F.cross_entropy(probe(features), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        probe.eval()
        with torch.no_grad():
            normalized_validation = ((validation_x.to(device) - mean) / std).clamp(-10.0, 10.0)
            score = float((probe(normalized_validation).argmax(-1).cpu() == validation_y).float().mean())
            if score > best_validation:
                best_validation = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        normalized_test = ((test_x.to(device) - mean) / std).clamp(-10.0, 10.0)
        test_score = float((probe(normalized_test).argmax(-1).cpu() == test_y).float().mean())
    return {"accuracy": test_score, "validation_accuracy": best_validation, "selected_epoch": best_epoch}


def save_tsne(features, labels, class_names, path, seed):
    reduced = PCA(n_components=50, random_state=seed).fit_transform(features.numpy())
    points = TSNE(
        n_components=2, perplexity=30, learning_rate="auto", init="pca",
        max_iter=1500, random_state=seed,
    ).fit_transform(reduced)
    plt.figure(figsize=(9, 8), dpi=180)
    palette = plt.get_cmap("tab10")
    labels_np = labels.numpy()
    for class_index, class_name in enumerate(class_names):
        mask = labels_np == class_index
        plt.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.72, color=palette(class_index), label=class_name)
    plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=2, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return float(silhouette_score(points, labels_np))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("motion", "cifar10dvs", "ncaltech101"), default="motion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--readout-mode", choices=("checkpoint", "final", "multiscale", "temporal", "temporal_unit", "temporal_delta", "spatiotemporal", "spatial_temporal"), default="checkpoint")
    parser.add_argument("--few-shot-repeats", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload["args"]
    steps = cfg.get("steps", 16)
    device = torch.device("cuda")
    readout_mode = cfg.get("readout_mode", "temporal") if args.readout_mode == "checkpoint" else args.readout_mode
    model = PureSpikeResNet(
        blocks_per_stage=cfg["blocks_per_stage"], binary_input=cfg["binary_input"],
        readout_mode=readout_mode, temporal_steps=steps,
        preserve_final_resolution=cfg.get("preserve_final_resolution", False),
        mixed_lif=cfg.get("mixed_lif", False),
        mixed_lif_spread=cfg.get("mixed_lif_spread", 1.0),
    ).to(device)
    model.load_state_dict(payload["model"])
    common = dict(
        steps=steps, size=cfg.get("size", 48), two_views=False,
        frame_mode=cfg.get("frame_mode", "count_global"), preload_frames=True,
    )
    train_set = make_event_dataset(args.dataset, args.data_root, True, max_samples=9000, **common)
    test_set = make_event_dataset(args.dataset, args.data_root, False, max_samples=1000, **common)
    counter = OperationCounter(model)
    activity_counter = SpikeActivityCounter(model)
    train_x, train_y = collect(
        model, DataLoader(train_set, 64, pin_memory=True), device, (counter, activity_counter)
    )
    test_x, test_y = collect(
        model, DataLoader(test_set, 64, pin_memory=True), device
    )
    result = {
        "protocol": {"dataset": args.dataset, "train": len(train_set), "test": len(test_set), "steps": steps, "readout_mode": readout_mode},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_dim": train_x.shape[-1],
        "knn": {str(k): knn(train_x, train_y, test_x, test_y, k) for k in (1, 3, 5, 10, 20, 50)},
        "linear_probe": linear_probe(
            train_x, train_y, test_x, test_y, len(train_set.class_names),
            args.probe_epochs, device, args.seed,
        ),
        "few_shot": {
            str(shots): few_shot_knn(train_x, train_y, test_x, test_y, shots, args.few_shot_repeats, args.seed + shots)
            for shots in (1, 5, 10, 45)
        },
        "retrieval": retrieval_metrics(train_x, train_y, test_x, test_y),
        "snn": {**counter.report(), "firing_rate_by_layer": activity_counter.report()},
    }
    result["tsne_silhouette"] = save_tsne(
        test_x, test_y, train_set.class_names, output_dir / "tsne_test.png", args.seed
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
