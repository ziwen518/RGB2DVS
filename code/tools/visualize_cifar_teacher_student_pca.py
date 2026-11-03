"""Visualize CIFAR RGB teacher, observable target, and PureSpikeFormer embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset


def tensor_payload(payload: dict, preferred: list[str]) -> torch.Tensor:
    for key in preferred:
        value = payload.get(key)
        if torch.is_tensor(value):
            tensor = value.float()
            if tensor.ndim == 2:
                return tensor
            if tensor.ndim == 3:
                return tensor.mean(dim=1)
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim in {2, 3}:
            tensor = value.float()
            return tensor if tensor.ndim == 2 else tensor.mean(dim=1)
    raise KeyError(f"no 2D/3D feature tensor in payload keys={list(payload)}")


def load_bank(path: Path, preferred: list[str]) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "ids" not in payload:
        raise KeyError(f"cache {path} has no ids")
    values = tensor_payload(payload, preferred)
    ids = [str(item).replace("\\", "/") for item in payload["ids"]]
    if len(ids) != len(values):
        raise ValueError(f"cache {path}: ids={len(ids)} features={len(values)}")
    return {key: value for key, value in zip(ids, values)}, payload


@torch.inference_mode()
def extract_student(model, dataset, indices, batch_size, device):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels, ids = [], [], []
    model.eval()
    for events, target, _ordinal, sample_ids in loader:
        out = model(events.to(device, non_blocking=True))
        features.append(out["embedding"].float().cpu())
        labels.append(torch.as_tensor(target).view(-1))
        ids.extend(str(item).replace("\\", "/") for item in sample_ids)
    return {key: value for key, value in zip(ids, torch.cat(features))}, torch.cat(labels)


def plot_split(name, ids, labels, teacher, observable, student, output_dir, seed):
    arrays = [teacher, observable, student]
    stacked = torch.cat(arrays, dim=0).numpy()
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(stacked)
    n = len(ids)
    points = [projected[i * n : (i + 1) * n] for i in range(3)]
    labels_np = labels.numpy()
    unique = np.unique(labels_np)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180, sharex=True, sharey=True)
    titles = ["RGB teacher", "Observable bridge target", "PureSpikeFormer student"]
    for axis, point, title in zip(axes, points, titles):
        for label in unique:
            mask = labels_np == label
            axis.scatter(point[mask, 0], point[mask, 1], s=10, alpha=0.70, color=colors(int(label) % 10))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle(f"CIFAR10-RCLS-medium {name}: joint PCA (n={n})", y=1.01)
    fig.text(0.5, 0.01, f"PC1/PC2 explained variance: {pca.explained_variance_ratio_[0]:.3f} / {pca.explained_variance_ratio_[1]:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / f"pca_{name}.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "split": name,
        "samples": n,
        "classes": int(len(unique)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "teacher_student_cosine": torch.nn.functional.cosine_similarity(teacher, student, dim=-1).mean().item(),
        "observable_student_cosine": torch.nn.functional.cosine_similarity(observable, student, dim=-1).mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--observable-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    core = args.project_root / "code" / "core"
    sys.path.insert(0, str(core))
    from data import make_event_dataset
    from model import PureSpikeFormer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = PureSpikeFormer(
        in_channels=2, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        patch_size=cfg["patch_size"], threshold=cfg["threshold"],
        use_cls_token=cfg.get("use_cls_token", False), norm="bntt", image_size=cfg["size"],
        temporal_readout=cfg.get("temporal_readout", "learned"),
        local_stem=not cfg.get("pyramid_stem", False),
        pyramid_stem=cfg.get("pyramid_stem", False), use_positional_bias=True,
        attention_mode="normalized", block_spiking=cfg.get("block_spiking", "full_respike"),
        hybrid_attention=cfg.get("hybrid_attention", False),
        hybrid_attention_suffix=cfg.get("hybrid_attention_suffix", 0),
        qkv_temporal_mode=cfg.get("qkv_temporal_mode", "standard"),
        signed_readout=cfg.get("signed_readout", False),
        multidepth_readout=cfg.get("multidepth_readout", False),
        multidepth_readout_layers=cfg.get("multidepth_readout_layers", 3),
        membrane_readout=cfg.get("membrane_readout", False),
        continuous=cfg.get("continuous_relaxation", False),
        ternary_threshold=cfg.get("ternary_threshold", 0.5),
        population_bins=cfg.get("population_bins", 1), temporal_steps=cfg["steps"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    common = dict(steps=cfg["steps"], size=cfg["size"], two_views=False,
                  frame_mode=cfg.get("frame_mode", "count_global"), preload_frames=True)
    train_set = make_event_dataset("motion", args.data_root, True, **common)
    test_set = make_event_dataset("motion", args.data_root, False, **common)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_map = {str(path.relative_to(train_set.root)).replace("\\", "/"): i for i, (path, _, _) in enumerate(train_set.samples)}
    test_map = {str(path.relative_to(test_set.root)).replace("\\", "/"): i for i, (path, _, _) in enumerate(test_set.samples)}
    rgb, rgb_payload = load_bank(args.rgb_cache, ["prefix", "features", "embeddings", "rgb_features", "dino"])
    observable, _ = load_bank(args.observable_cache, ["transported", "observable", "features", "embeddings"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, ids, dataset, index_map in [
        ("validation", split["validation_ids"], train_set, train_map),
        ("test", split["test_ids"], test_set, test_map),
    ]:
        ids = [str(item).replace("\\", "/") for item in ids]
        missing = [key for key in ids if key not in rgb or key not in observable or key not in index_map]
        if missing:
            raise KeyError(f"{name}: missing {len(missing)} ids, first={missing[:2]}")
        student_map, labels_tensor = extract_student(model, dataset, [index_map[key] for key in ids], args.batch_size, device)
        teacher = torch.stack([rgb[key] for key in ids])
        target = torch.stack([observable[key] for key in ids])
        student = torch.stack([student_map[key] for key in ids])
        reports.append(plot_split(name, ids, labels_tensor, teacher, target, student, args.output_dir, args.seed))
    report = {"checkpoint": str(args.checkpoint), "rgb_cache": str(args.rgb_cache), "observable_cache": str(args.observable_cache), "splits": reports}
    (args.output_dir / "pca_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
