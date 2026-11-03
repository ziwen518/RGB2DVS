from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def diagnose(folder: Path) -> dict:
    paths = sorted(folder.glob("*.png"))
    if len(paths) < 2:
        raise ValueError("need at least two generated frames")
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths]
    arr = np.stack(frames)
    diffs = np.abs(arr[1:] - arr[:-1])
    gray = arr.mean(axis=-1)
    log_diff = np.abs(np.log(gray[1:] + 1.0e-3) - np.log(gray[:-1] + 1.0e-3))
    return {
        "frames": len(paths),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "mean_frame_delta": float(diffs.mean()),
        "p95_frame_delta": float(np.percentile(diffs, 95)),
        "mean_log_intensity_delta": float(log_diff.mean()),
        "p95_log_intensity_delta": float(np.percentile(log_diff, 95)),
        "mean_brightness": float(gray.mean()),
        "brightness_std_over_time": float(gray.mean(axis=(1, 2)).std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.input)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
