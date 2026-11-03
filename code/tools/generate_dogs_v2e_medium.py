from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image


def render_frames(image: Image.Image, frames: int, amplitude: float) -> np.ndarray:
    image = image.convert("L")
    width, height = image.size
    result = []
    padded = Image.new("L", (width * 3, height * 3))
    for yy in range(3):
        for xx in range(3):
            padded.paste(image, (xx * width, yy * height))
    for index in range(frames):
        phase = 2.0 * math.pi * index / max(frames - 1, 1)
        dx = amplitude * width * math.sin(phase)
        dy = 0.5 * amplitude * height * math.sin(phase + math.pi / 2.0)
        scale = 1.0 + 0.35 * amplitude * math.sin(phase)
        crop_w = max(2, int(round(width / scale)))
        crop_h = max(2, int(round(height / scale)))
        left = int(round(width + width / 2.0 + dx - crop_w / 2.0))
        top = int(round(height + height / 2.0 + dy - crop_h / 2.0))
        frame = padded.crop((left, top, left + crop_w, top + crop_h))
        frame = frame.resize((width, height), Image.Resampling.BICUBIC)
        result.append(np.asarray(frame, dtype=np.float32) / 255.0)
    return np.stack(result)


def simulate(frames: np.ndarray, target_events: int, refractory: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    log_frames = np.log(np.maximum(frames, 1.0e-3))
    delta = np.abs(log_frames[1:] - log_frames[:-1])
    # Select a per-sample contrast threshold from the observed motion energy.
    threshold = float(np.quantile(delta, max(0.5, 1.0 - target_events / max(delta.size, 1) * 1.8)))
    threshold = float(np.clip(threshold, 0.04, 0.30))
    previous = log_frames[0]
    last_event = np.full(previous.shape, -10**9, dtype=np.int32)
    xs, ys, ts, ps = [], [], [], []
    for frame_index, current in enumerate(log_frames[1:], start=1):
        change = current - previous
        magnitude = np.abs(change)
        polarity = (change >= 0).astype(np.uint8)
        count = np.clip(np.floor(magnitude / threshold).astype(np.int32), 0, 4)
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
    return (np.asarray(xs, dtype=np.int16), np.asarray(ys, dtype=np.int16),
            np.asarray(ts, dtype=np.int64), np.asarray(ps, dtype=np.uint8), threshold)


def process_one(source: Path, output: Path, label: int, source_id: str, frames: int, target_events: int) -> dict:
    rgb = Image.open(source).convert("RGB")
    video = render_frames(rgb, frames, 0.02)
    x, y, t, p, threshold = simulate(video, target_events, refractory=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, t=t, p=p, label=np.int64(label),
                        source=np.asarray(source_id), trajectory=np.asarray("rcls"),
                        simulator=np.asarray("v2e_proxy_highfps"), frames=np.int64(frames),
                        contrast_threshold=np.float32(threshold), refractory=np.int64(2))
    return {"id": source_id, "events": int(len(x)), "height": int(video.shape[1]),
            "width": int(video.shape[2]), "contrast_threshold": threshold,
            "positive_fraction": float(p.mean()) if len(p) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-event-root", type=Path)
    parser.add_argument("--source-rgb-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--target-events", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids = list(manifest["train_ids"] + manifest["validation_ids"] + manifest["test_ids"])
    if args.limit is not None: ids = ids[:args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    ids = ids[args.shard_index::args.shard_count]
    classes = sorted({sid.split("/")[1] for sid in ids})
    labels = {name: index for index, name in enumerate(classes)}
    rgb_root = args.source_rgb_root or (args.root / "data/dogs_full_staging/rgb")
    data_root = args.source_event_root or (args.root / "data/dogs")
    records = []
    for index, sid in enumerate(ids, start=1):
        source_npz = data_root / sid
        with np.load(source_npz, allow_pickle=False) as payload:
            source = str(np.asarray(payload["source"]).reshape(-1)[0])
        rgb_path = rgb_root / source
        split = sid.split("/", 1)[0]
        out = args.output_root / split / sid.split("/", 1)[1]
        out = out.with_suffix(".npz")
        if args.skip_existing and out.exists():
            if index % 25 == 0 or index == len(ids):
                print(json.dumps({"processed": index, "total": len(ids), "skipped_existing": True}), flush=True)
            continue
        records.append(process_one(rgb_path, out, labels[sid.split("/")[1]], sid, args.frames, args.target_events))
        if index % 25 == 0 or index == len(ids):
            print(json.dumps({"processed": index, "total": len(ids), "last_events": records[-1]["events"]}), flush=True)
    metrics_name = (
        "generation_metrics.json" if args.shard_count == 1
        else f"generation_metrics.shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json"
    )
    (args.output_root / metrics_name).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(records), "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
