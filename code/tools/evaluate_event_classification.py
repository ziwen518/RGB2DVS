"""Fixed kNN and linear-probe evaluation for frozen PureSpikeFormer features.

Labels never enter Student training. They are used only by kNN evaluation and
the downstream linear head. The linear probe runs for 100 epochs and validates
every five epochs; the held-out test split is evaluated once after validation
has selected the best head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class OperationEnergyMeter:
    def __init__(self, model: nn.Module) -> None:
        self.rows: dict[str, dict[str, float]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name: str, module: nn.Module):
        def hook(_module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            source = inputs[0].detach()
            if isinstance(module, nn.Linear):
                kernel_ops = module.in_features
            else:
                kh, kw = module.kernel_size
                kernel_ops = kh * kw * (module.in_channels // module.groups)
            dense = float(output[0].numel() * kernel_ops)
            activity = float(source.float().ne(0).float().mean())
            row = self.rows.setdefault(name, {"calls": 0.0, "dense": 0.0, "synops": 0.0})
            row["calls"] += 1.0
            row["dense"] += dense
            row["synops"] += dense * activity
        return hook

    def report(self, batches: int) -> dict[str, float | int]:
        batches = max(1, int(batches))
        dense = sum(row["dense"] for row in self.rows.values()) / batches
        synops = sum(row["synops"] for row in self.rows.values()) / batches
        calls = sum(row["calls"] for row in self.rows.values()) / batches
        ac_pj, mac_pj = 0.9, 4.6
        return {
            "synops_per_sample": synops,
            "dense_ops_per_sample": dense,
            "activity_fraction": synops / max(dense, 1.0),
            "estimated_energy_pj": synops * ac_pj,
            "estimated_energy_mj": synops * ac_pj / 1.0e9,
            "dense_mac_energy_pj": dense * mac_pj,
            "estimated_energy_ratio": (synops * ac_pj) / max(dense * mac_pj, 1.0),
            "hook_calls_per_sample": int(calls),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", choices=("motion", "ncaltech101", "cifar10dvs"), required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(model_class, checkpoint: dict, device: torch.device) -> nn.Module:
    cfg = checkpoint["args"]
    kwargs = dict(
        in_channels=2, dim=int(cfg["dim"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), patch_size=int(cfg["patch_size"]),
        threshold=float(cfg["threshold"]),
        use_cls_token=bool(cfg.get("use_cls_token", False)), norm="bntt",
        image_size=int(cfg["size"]), temporal_readout="learned",
        local_stem=not bool(cfg.get("pyramid_stem", False)),
        pyramid_stem=bool(cfg.get("pyramid_stem", False)),
        use_positional_bias=True, attention_mode="normalized",
        signed_readout=bool(cfg.get("signed_readout", True)),
        multidepth_readout=bool(cfg.get("multidepth_readout", False)),
        multidepth_readout_layers=int(cfg.get("multidepth_readout_layers", 3)),
        membrane_readout=bool(cfg.get("membrane_readout", False)),
        block_spiking=cfg.get("block_spiking", "full_respike"),
        hybrid_attention=bool(cfg.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(cfg.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=cfg.get("qkv_temporal_mode", "standard"),
        continuous=bool(cfg.get("continuous_relaxation", False)),
        ternary_threshold=float(cfg.get("ternary_threshold", 0.5)),
        temporal_steps=int(cfg["steps"]),
        population_bins=int(cfg.get("population_bins", 1)),
        event_prompt=bool(cfg.get("event_prompt", False)),
        prompt_strength=float(cfg.get("prompt_strength", 0.7)),
    )
    if "local_structure_mixer" in cfg:
        kwargs.update(
            local_structure_mixer=bool(cfg["local_structure_mixer"]),
            local_mixer_dilations=tuple(cfg.get("local_mixer_dilations", [1, 2])),
        )
    model = model_class(**kwargs).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_loaders(args, make_event_dataset, cfg):
    frame_mode = cfg.get("frame_mode", "count")
    if args.dataset_name == "ncaltech101" and frame_mode == "count_global":
        frame_mode = "count"
    common = dict(
        steps=int(cfg["steps"]), size=int(cfg["size"]), two_views=False,
        frame_mode=frame_mode,
    )
    if args.dataset_name in {"motion", "ncaltech101"}:
        if args.split_manifest is None:
            raise ValueError(f"{args.dataset_name} formal evaluation requires --split-manifest")
        dataset_kwargs = dict(common)
        if args.dataset_name == "ncaltech101":
            dataset_kwargs["use_all"] = True
        train_source = make_event_dataset(
            args.dataset_name, args.data_root, True, **dataset_kwargs
        )
        test_source = (
            make_event_dataset(args.dataset_name, args.data_root, False, **dataset_kwargs)
            if args.dataset_name == "motion" else train_source
        )
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
        train_by_id = {
            str(sample[0].relative_to(train_source.root)).replace("\\", "/"): index
            for index, sample in enumerate(train_source.samples)
        }
        test_by_id = {
            str(sample[0].relative_to(test_source.root)).replace("\\", "/"): index
            for index, sample in enumerate(test_source.samples)
        }
        train_ids = list(map(str, manifest["train_ids"]))
        validation_ids = list(map(str, manifest["validation_ids"]))
        test_ids = list(map(str, manifest.get("test_ids", [])))
        split_sets = [set(train_ids), set(validation_ids), set(test_ids)]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("train, validation, and test IDs must be disjoint")
        missing_trainval = (split_sets[0] | split_sets[1]) - set(train_by_id)
        missing_test = split_sets[2] - set(test_by_id)
        if missing_trainval or missing_test:
            raise KeyError(
                "classification split is missing "
                f"{len(missing_trainval)} train/validation and {len(missing_test)} test IDs"
            )
        if args.dataset_name == "motion":
            leaked_trainval = (split_sets[0] | split_sets[1]) & set(test_by_id)
            leaked_test = split_sets[2] & set(train_by_id)
            if leaked_trainval or leaked_test:
                raise ValueError("motion manifest assigns IDs to the wrong physical split")
        train = IndexedDataset(train_source, [train_by_id[key] for key in train_ids])
        validation = IndexedDataset(
            train_source, [train_by_id[key] for key in validation_ids]
        )
        test = (
            IndexedDataset(test_source, [test_by_id[key] for key in test_ids])
            if test_ids else None
        )
        if tuple(train_source.class_names) != tuple(test_source.class_names):
            raise ValueError("train and test class order differs")
        class_names = train_source.class_names
    else:
        train = make_event_dataset(args.dataset_name, args.data_root, True, **common)
        validation = make_event_dataset(args.dataset_name, args.data_root, False, **common)
        test = None
        class_names = tuple(str(index) for index in range(10))
    train_loader = DataLoader(
        train, args.extract_batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    validation_loader = DataLoader(
        validation, args.extract_batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = (
        DataLoader(test, args.extract_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        if test is not None else None
    )
    return train_loader, validation_loader, test_loader, class_names


@torch.inference_mode()
def extract(model, loader, device, meter=None):
    model.eval()
    signed, core, labels = [], [], []
    spike_rate = 0.0
    for events, target, *_ in loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(events.to(device, non_blocking=True))
        signed.append(output["embedding"].float().cpu())
        core.append(output["core_embedding"].float().cpu())
        labels.append(torch.as_tensor(target).long())
        spike_rate += float(output["spike_rate"])
    return (
        torch.cat(signed), torch.cat(core), torch.cat(labels),
        spike_rate / max(len(loader), 1),
    )


@torch.inference_mode()
def knn_top1(
    bank: torch.Tensor, bank_y: torch.Tensor,
    query: torch.Tensor, query_y: torch.Tensor, k: int,
) -> float:
    bank = F.normalize(bank.float(), dim=-1)
    query = F.normalize(query.float(), dim=-1)
    neighbors = (query @ bank.t()).topk(min(k, len(bank)), dim=-1).indices
    prediction = torch.mode(bank_y[neighbors], dim=-1).values
    return float(prediction.eq(query_y).float().mean())


@torch.inference_mode()
def score(head, features, labels, mean, std, device) -> float:
    head.eval()
    correct = total = 0
    for start in range(0, len(features), 1024):
        x = ((features[start:start + 1024].to(device) - mean) / std).clamp(-10, 10)
        y = labels[start:start + 1024].to(device)
        correct += int(head(x).argmax(-1).eq(y).sum())
        total += len(y)
    return correct / max(total, 1)


def training_regime(history: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [row for row in history if row.get("validation_top1") is not None]
    if len(evaluated) < 3:
        return {
            "state": "insufficient_evidence", "evaluations": len(evaluated),
            "required_evaluations": 3, "recommendation": "continue_to_next_validation",
        }
    window = evaluated[-3:]
    accuracy = [float(row["validation_top1"]) for row in window]
    loss = [float(row["train_loss"]) for row in window]
    delta = accuracy[-1] - accuracy[0]
    recent = accuracy[-1] - accuracy[-2]
    loss_reduction = (loss[0] - loss[-1]) / max(abs(loss[0]), 1.0e-8)
    if delta >= 0.005 and recent >= -0.002:
        state, recommendation = "undertrained_improving", "continue_training"
    elif recent <= -0.005:
        state, recommendation = "downstream_overfit", "stop_at_best_epoch"
    elif abs(delta) < 0.003 and loss_reduction < 0.01:
        state, recommendation = "converged_plateau", "stop_training"
    elif loss_reduction >= 0.02 and delta < 0.003:
        state = "representation_limited_or_objective_mismatch"
        recommendation = "stop_head_extension_and_improve_student"
    else:
        state, recommendation = "ambiguous_slow_progress", "continue_one_validation_interval_only"
    return {
        "state": state, "recommendation": recommendation,
        "evaluation_epochs": [int(row["epoch"]) for row in window],
        "accuracy_delta": delta, "recent_accuracy_delta": recent,
        "relative_loss_reduction": loss_reduction,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.eval_every <= 0:
        raise ValueError("epochs and eval-every must be positive")
    if args.knn_k <= 0:
        raise ValueError("knn-k must be positive")
    if args.epochs != 100 or args.eval_every != 5:
        raise ValueError("formal classification protocol requires 100 epochs and eval-every=5")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    core = args.project_root / "code" / "core"
    sys.path.insert(0, str(core))
    import model as model_module
    from data import make_event_dataset
    from model import PureSpikeFormer

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(PureSpikeFormer, checkpoint, device)
    train_loader, validation_loader, test_loader, class_names = make_loaders(
        args, make_event_dataset, checkpoint["args"]
    )
    energy_meter = OperationEnergyMeter(model)
    train_x, train_core_x, train_y, spike_rate = extract(
        model, train_loader, device, energy_meter
    )
    encoder_energy = energy_meter.report(len(train_loader))
    energy_meter.close()
    validation_x, validation_core_x, validation_y, _ = extract(
        model, validation_loader, device
    )
    validation_signed_knn = knn_top1(
        train_x, train_y, validation_x, validation_y, args.knn_k
    )
    validation_core_knn = knn_top1(
        train_core_x, train_y, validation_core_x, validation_y, args.knn_k
    )
    mean = train_x.to(device).mean(0, keepdim=True)
    std = train_x.to(device).std(0, keepdim=True).clamp_min(1.0e-4)
    head = nn.Linear(train_x.shape[1], len(class_names)).to(device)
    counts = torch.bincount(train_y, minlength=len(class_names)).float()
    class_weight = counts.sum() / counts.clamp_min(1.0)
    class_weight = (class_weight / class_weight.mean()).sqrt().to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y), args.batch_size, shuffle=True,
        generator=generator, pin_memory=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        samples = 0
        for features, labels in loader:
            features = ((features.to(device) - mean) / std).clamp(-10, 10)
            labels = labels.to(device)
            loss = F.cross_entropy(head(features), labels, weight=class_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        scheduler.step()
        row: dict[str, object] = {
            "epoch": epoch, "train_loss": total_loss / max(samples, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row["train_top1"] = score(head, train_x, train_y, mean, std, device)
            row["validation_top1"] = score(
                head, validation_x, validation_y, mean, std, device
            )
            # kNN and encoder cost are properties of this frozen Student. Keep
            # them in every five-epoch record so each validation point is a
            # self-contained comparison rather than a linear-head-only curve.
            row["validation_signed_knn_top1"] = validation_signed_knn
            row["validation_core_knn_top1"] = validation_core_knn
            row["knn_k"] = args.knn_k
            row["encoder_spike_rate"] = spike_rate
            row["encoder_energy"] = encoder_energy
            row["training_regime"] = training_regime([*history, row])
            if float(row["validation_top1"]) > best:
                best, best_epoch = float(row["validation_top1"]), epoch
                torch.save({"head": head.state_dict(), "epoch": epoch, "metrics": row},
                           args.output_dir / "best_linear_head.pt")
        history.append(row)
        (args.output_dir / "classification_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

    test_top1 = test_signed_knn = test_core_knn = None
    test_x = test_y = None
    if test_loader is not None:
        # The test split remains untouched until validation has selected the head.
        test_x, test_core_x, test_y, _ = extract(model, test_loader, device)
        best_head = torch.load(
            args.output_dir / "best_linear_head.pt", map_location=device, weights_only=False
        )
        head.load_state_dict(best_head["head"])
        test_top1 = score(head, test_x, test_y, mean, std, device)
        test_signed_knn = knn_top1(train_x, train_y, test_x, test_y, args.knn_k)
        test_core_knn = knn_top1(
            train_core_x, train_y, test_core_x, test_y, args.knn_k
        )

    result = {
        "dataset": args.dataset_name,
        "protocol": "frozen_knn_and_100_epoch_linear_probe",
        "protocol_locked": True,
        "linear_probe_epochs": args.epochs,
        "validation_every_epochs": args.eval_every,
        "knn_k": args.knn_k,
        "student_labels_used": False,
        "labels_used_for_knn_and_downstream_head_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("metrics", {}).get("epoch"),
        "model_file": str(Path(model_module.__file__).resolve()),
        "train_samples": len(train_x), "validation_samples": len(validation_x),
        "test_samples": 0 if test_x is None else len(test_x),
        "classes": len(class_names),
        "validation_signed_knn_top1": validation_signed_knn,
        "validation_core_knn_top1": validation_core_knn,
        "test_signed_knn_top1": test_signed_knn,
        "test_core_knn_top1": test_core_knn,
        "best_validation_linear_top1": best,
        "best_epoch": best_epoch,
        "test_linear_top1_at_best_validation": test_top1,
        "test_evaluations": 0 if test_top1 is None else 1,
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "linear_head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "encoder_spike_rate": spike_rate, "encoder_energy": encoder_energy,
        "history": history,
    }
    (args.output_dir / "classification_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
