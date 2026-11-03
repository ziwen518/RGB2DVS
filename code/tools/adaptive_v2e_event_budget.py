from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_v2e_from_rgb_video import simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-events", type=int, default=600_000)
    args = parser.parse_args()
    candidates = [(t, r) for t in (0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30)
                  for r in (0, 1, 2, 4, 6)]
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for video in sorted(args.videos.glob("sample_*_video")):
        trials = []
        for threshold, refractory in candidates:
            trial = args.output / f"_trial_{video.name}_{threshold}_{refractory}.npz"
            metrics = simulate(video, trial, threshold, refractory)
            trials.append((abs(metrics["events"] - args.target_events), threshold, refractory, metrics["events"], trial))
        _, threshold, refractory, _, trial = min(trials, key=lambda row: row[0])
        final = args.output / f"{video.name}_adaptive.npz"
        trial.replace(final)
        rows.append({"video": video.name, "contrast_threshold": threshold,
                     "refractory": refractory, "target_events": args.target_events,
                     **simulate(video, final, threshold, refractory)})
        for _, _, _, _, path in trials:
            if path.exists() and path != final:
                path.unlink()
    (args.output / "adaptive_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
