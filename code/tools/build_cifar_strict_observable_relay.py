"""Build a label-free, test-zero-touch relay for paired CIFAR RCLS events."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def event_descriptor(frames: torch.Tensor, grid: int) -> torch.Tensor:
    """Summarize event activity without images, class IDs, or learned labels."""
    if frames.ndim != 4:
        raise ValueError(f"expected [T,C,H,W] frames, got {tuple(frames.shape)}")
    frames = frames.float().div_(255.0)
    pooled = F.adaptive_avg_pool2d(frames, (grid, grid))
    temporal_delta = pooled[1:] - pooled[:-1]
    channel_mean = frames.mean(dim=(-1, -2))
    channel_std = frames.std(dim=(-1, -2))
    return torch.cat([
        pooled.flatten(), temporal_delta.flatten(),
        channel_mean.flatten(), channel_std.flatten(),
    ])


def main() -> int:
    args = parse_args()
    if args.grid <= 0 or args.ridge <= 0:
        raise ValueError("grid and ridge must be positive")
    split_bytes = args.split_manifest.read_bytes()
    split = json.loads(split_bytes.decode("utf-8-sig"))
    train_ids = list(map(str, split["train_ids"]))
    validation_ids = list(map(str, split["validation_ids"]))
    test_ids = list(map(str, split.get("test_ids", [])))
    train_set, validation_set, test_set = map(set, (train_ids, validation_ids, test_ids))
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise ValueError("train, validation, and test IDs must be disjoint")

    # Test IDs are validated against the split but never dereferenced below.
    target_ids = train_ids + validation_ids
    rgb = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    rgb_by_id = {
        str(sample_id): F.normalize(feature.float().mean(dim=0), dim=-1)
        for sample_id, feature in zip(rgb["ids"], rgb["prefix"])
    }
    missing_rgb = set(target_ids) - set(rgb_by_id)
    if missing_rgb:
        raise KeyError(f"RGB cache misses {len(missing_rgb)} train/validation IDs")

    descriptors = []
    for sample_id in target_ids:
        frame_path = args.frame_cache / Path(sample_id).with_suffix(".npy")
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        frames = torch.from_numpy(np.load(frame_path, allow_pickle=False))
        descriptors.append(event_descriptor(frames, args.grid))
    event = torch.stack(descriptors)
    teacher = torch.stack([rgb_by_id[sample_id] for sample_id in target_ids])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit = torch.arange(len(train_ids), device=device)
    x = event.to(device)
    y = teacher.to(device)
    x_mean = x[fit].mean(dim=0, keepdim=True)
    x_std = x[fit].std(dim=0, keepdim=True).clamp_min(1.0e-6)
    y_mean = y[fit].mean(dim=0, keepdim=True)
    centered_x = (x - x_mean) / x_std
    centered_y = y - y_mean
    x_fit, y_fit = centered_x[fit], centered_y[fit]
    identity = torch.eye(x_fit.shape[1], device=device, dtype=x_fit.dtype)
    weight = torch.linalg.solve(
        x_fit.T @ x_fit + args.ridge * identity,
        x_fit.T @ y_fit,
    )
    prediction = F.normalize(centered_x @ weight + y_mean, dim=-1).cpu()

    output = {
        "ids": target_ids,
        "observable": prediction,
        "transported": prediction,
        "fit_ids": train_ids,
        "validation_ids": validation_ids,
        "excluded_test_ids": test_ids,
        "fit_only_estimation": True,
        "test_used": False,
        "labels_used": False,
        "source": "label-free event-statistics ridge relay to paired RGB-DINO",
        "target_type": "strict_event_statistics_observable",
        "descriptor_grid": args.grid,
        "descriptor_dim": event.shape[1],
        "ridge": args.ridge,
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({
        "output": str(args.output), "fit": len(train_ids),
        "validation": len(validation_ids), "excluded_test": len(test_ids),
        "descriptor_dim": event.shape[1], "labels_used": False,
        "test_used": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
