"""Prepare N-Caltech101 event frames, deterministic splits, and paired RGB index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_bin(path: Path) -> np.ndarray:
    from tonic.io import read_mnist_file
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    return read_mnist_file(str(path), dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--validation", type=int, default=900)
    parser.add_argument("--test", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
    from data import events_to_frames
    root = args.root
    event_root = root / "events"
    paths = sorted(event_root.rglob("*.bin"))
    classes = sorted({p.parent.name for p in paths})
    class_to_id = {name: i for i, name in enumerate(classes)}
    samples = [(p, class_to_id[p.parent.name]) for p in paths]
    rng = np.random.default_rng(args.seed)
    train, validation, test = [], [], []
    for label in range(len(classes)):
        group = [item for item in samples if item[1] == label]
        order = rng.permutation(len(group))
        n_test = min(args.test // len(classes) + int(label < args.test % len(classes)), len(group) // 3)
        n_val = min(args.validation // len(classes) + int(label < args.validation % len(classes)), len(group) - n_test)
        test.extend(group[int(i)] for i in order[:n_test])
        validation.extend(group[int(i)] for i in order[n_test:n_test + n_val])
        train.extend(group[int(i)] for i in order[n_test + n_val:])
    cache_root = root / f"frames_ncaltech101_t{args.steps}_s{args.size}"
    records = []
    for path, label in samples:
        rel = path.relative_to(event_root).with_suffix("")
        cache = cache_root / rel.with_suffix(".npy")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            try:
                frames = events_to_frames(read_bin(path), args.steps, args.size, mode="count_global")
                np.save(cache, frames.mul(255).round().to(torch.uint8).numpy(), allow_pickle=False)
            except Exception as exc:
                print(f"ERROR {path}: {type(exc).__name__}: {exc}", flush=True)
                continue
        sample_id = str(path.relative_to(root)).replace("\\", "/")
        records.append({"id": sample_id, "label": label, "class_name": path.parent.name, "event_path": sample_id,
                        "rgb_path": f"rgb/{path.parent.name}/{path.stem}.jpg"})
    ids = {"train": [str(p.relative_to(root)).replace("\\", "/") for p, _ in train],
           "validation": [str(p.relative_to(root)).replace("\\", "/") for p, _ in validation],
           "test": [str(p.relative_to(root)).replace("\\", "/") for p, _ in test]}
    out = root / "manifests"; out.mkdir(exist_ok=True)
    (out / "event_split.json").write_text(json.dumps({"seed": args.seed, "classes": classes, **ids}, indent=2), encoding="utf-8")
    (out / "paired_index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(samples), "classes": len(classes), "train": len(train), "validation": len(validation), "test": len(test), "cache": str(cache_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
