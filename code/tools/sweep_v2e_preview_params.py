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
    args = parser.parse_args()
    rows = []
    for threshold in (0.06, 0.08, 0.10, 0.12, 0.16, 0.20):
        for refractory in (0, 1, 2, 4):
            per_sample = []
            for video in sorted(args.videos.glob("sample_*_video")):
                event_path = args.output.parent / f"_sweep_{video.name}_{threshold}_{refractory}.npz"
                per_sample.append(simulate(video, event_path, threshold, refractory))
                event_path.unlink(missing_ok=True)
            rows.append({
                "contrast_threshold": threshold,
                "refractory": refractory,
                "events_mean": float(np.mean([x["events"] for x in per_sample])),
                "events_min": int(min(x["events"] for x in per_sample)),
                "events_max": int(max(x["events"] for x in per_sample)),
                "rate_mean": float(np.mean([x["event_rate_per_frame"] for x in per_sample])),
                "positive_fraction_mean": float(np.mean([x["positive_fraction"] for x in per_sample])),
            })
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
