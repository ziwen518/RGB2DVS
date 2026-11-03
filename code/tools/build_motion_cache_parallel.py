from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def build_one(item: tuple[str, str, int, int, str]) -> str:
    source_name, target_name, steps, size, mode = item
    source = Path(source_name); target = Path(target_name)
    if target.exists():
        return "cached"
    import torch
    project_root = Path(__file__).resolve().parent
    core_root = project_root.parent / "core"
    if str(core_root) not in sys.path:
        sys.path.insert(0, str(core_root))
    from data import events_to_frames

    with np.load(source, allow_pickle=False) as payload:
        events = np.zeros(
            len(payload["x"]),
            dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)],
        )
        for field in events.dtype.names:
            events[field] = payload[field]
    frames = events_to_frames(events, steps, size, mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, frames.mul(255).round().to(torch.uint8).numpy(), allow_pickle=False)
    return "built"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=("count_global", "count", "log_count", "occupancy", "count_recency", "count_recency_local", "rcls_recency_fixed", "rcls_recency_fixed_global", "count_motion", "rcls_direction", "poisson_rate", "poisson_current"),
        default="count_global",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    root = Path(args.data_root)
    tasks = []
    for split in ("train", "test"):
        source_root = root / split
        cache_root = root / f"frames_motion_{args.mode}_t{args.steps}_s{args.size}" / split
        limit = args.max_train_samples if split == "train" else args.max_test_samples
        for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            sources = sorted(class_dir.rglob("*.npz"))
            if limit is not None:
                per_class = max(1, limit // len([p for p in source_root.iterdir() if p.is_dir()]))
                sources = sources[:per_class]
            for source in sources:
                target = cache_root / source.relative_to(source_root).with_suffix(".npy")
                if not target.exists():
                    tasks.append((str(source), str(target), args.steps, args.size, args.mode))
    print(f"missing={len(tasks)}", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in executor.map(build_one, tasks, chunksize=8):
            completed += 1
            if completed % 500 == 0 or completed == len(tasks):
                print(f"completed={completed}/{len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
