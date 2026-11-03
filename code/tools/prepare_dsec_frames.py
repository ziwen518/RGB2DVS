"""Materialize DSEC event windows as uint8 frame tensors for fast reruns."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from data import DSECEvents  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--frame-mode", default="count_global")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--max-samples", type=int)
    args = p.parse_args()
    ds = DSECEvents(args.root, args.split == "train", steps=args.steps, size=args.size,
                    frame_mode=args.frame_mode, max_samples=args.max_samples)
    out_root = args.root / f"frames_dsec_{args.frame_mode}_t{args.steps}_s{args.size}" / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for index, sample in enumerate(ds.samples):
        seq, frame, _, _, _ = sample
        out = out_root / seq / f"{frame:06d}.npy"
        if out.exists():
            done += 1
            continue
        frames, _, _, _ = ds[index]
        out.parent.mkdir(parents=True, exist_ok=True)
        # Keep the temporary filename from acquiring an extra .npy suffix.
        tmp = out.with_suffix(".part.npy")
        with tmp.open("wb") as handle:
            np.save(handle, (frames.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()), allow_pickle=False)
        tmp.replace(out)
        done += 1
        if done % 50 == 0 or done == len(ds):
            print(json.dumps({"done": done, "total": len(ds)}), flush=True)
    print(json.dumps({"out": str(out_root), "done": done, "total": len(ds)}), flush=True)

if __name__ == "__main__":
    main()
