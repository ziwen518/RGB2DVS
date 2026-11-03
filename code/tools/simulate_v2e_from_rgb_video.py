from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def simulate(video_dir: Path, output: Path, contrast_threshold: float = 0.10, refractory: int = 1) -> dict:
    paths = sorted(video_dir.glob("*.png"))
    if len(paths) < 2:
        raise ValueError("video directory must contain at least two frames")
    frames = [np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0 for path in paths]
    height, width = frames[0].shape
    previous = np.log(np.maximum(frames[0], 1.0e-3))
    last_event = np.full((height, width), -10**9, dtype=np.int32)
    xs, ys, ts, ps = [], [], [], []
    for frame_index, frame in enumerate(frames[1:], start=1):
        current = np.log(np.maximum(frame, 1.0e-3))
        delta = current - previous
        magnitude = np.abs(delta)
        polarity = (delta >= 0).astype(np.uint8)
        count = np.floor(magnitude / max(contrast_threshold, 1.0e-4)).astype(np.int32)
        count = np.clip(count, 0, 4)
        yy, xx = np.where(count > 0)
        for y, x in zip(yy.tolist(), xx.tolist()):
            if frame_index - int(last_event[y, x]) <= refractory:
                continue
            repeat = int(count[y, x])
            xs.extend([x] * repeat); ys.extend([y] * repeat)
            ts.extend([frame_index * 1000 + j for j in range(repeat)])
            ps.extend([int(polarity[y, x])] * repeat)
            last_event[y, x] = frame_index
        previous = current
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.asarray(xs, dtype=np.int16), y=np.asarray(ys, dtype=np.int16),
                        t=np.asarray(ts, dtype=np.int64), p=np.asarray(ps, dtype=np.uint8),
                        frames=len(frames), height=height, width=width,
                        contrast_threshold=float(contrast_threshold), refractory=int(refractory))
    return {"frames": len(frames), "height": height, "width": width, "events": len(xs),
            "event_rate_per_frame": len(xs) / max(len(frames) - 1, 1),
            "positive_fraction": float(np.mean(ps)) if ps else 0.0,
            "contrast_threshold": contrast_threshold, "refractory": refractory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast-threshold", type=float, default=0.10)
    parser.add_argument("--refractory", type=int, default=1)
    args = parser.parse_args()
    print(simulate(args.video_dir, args.output, args.contrast_threshold, args.refractory))


if __name__ == "__main__":
    main()
