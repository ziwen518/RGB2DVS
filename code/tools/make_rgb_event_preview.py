from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_single_rgb_motion_video import render_motion
from simulate_v2e_from_rgb_video import simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen = json.loads(args.ids.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (len(chosen) * 256, 768), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, sid in enumerate(chosen):
        with np.load(args.data_root / sid, allow_pickle=False) as payload:
            source = str(np.asarray(payload["source"]).reshape(-1)[0])
        image = Image.open(args.rgb_root / source.replace("train/", "").replace("test/", "")).convert("RGB")
        frames = render_motion(image, frames=128, amplitude=0.02)
        video_dir = args.output / f"sample_{index:02d}_video"
        video_dir.mkdir(exist_ok=True)
        for frame_index, frame in enumerate(frames):
            frame.save(video_dir / f"{frame_index:04d}.png")
        event_path = args.output / f"sample_{index:02d}_events.npz"
        metrics = simulate(video_dir, event_path, contrast_threshold=0.10, refractory=1)
        image.thumbnail((256, 256)); sheet.paste(image, (index * 256, 0))
        strip = Image.new("RGB", (256, 256))
        for step in range(16):
            strip.paste(frames[step * 8].resize((64, 64)), ((step % 4) * 64, (step // 4) * 64))
        sheet.paste(strip, (index * 256, 256))
        with np.load(event_path, allow_pickle=False) as events:
            event_image = np.zeros((events["height"], events["width"], 3), dtype=np.uint8)
            for channel, color in [(1, 0), (0, 2)]:
                mask = events["p"] == channel
                flat = np.asarray(events["y"][mask], dtype=np.int64) * int(events["width"]) + np.asarray(events["x"][mask], dtype=np.int64)
                flat = np.clip(flat, 0, int(events["height"]) * int(events["width"]) - 1)
                event_image[..., color] = np.minimum(255, np.bincount(
                    flat,
                    minlength=events["height"] * events["width"],
                ).reshape(events["height"], events["width"]) * 12)
        sheet.paste(Image.fromarray(event_image).resize((256, 256), Image.Resampling.NEAREST), (index * 256, 512))
        draw.text((index * 256 + 4, 742), f"{index + 1}: {metrics['events']} events", fill=(0, 0, 0))
        records.append({"id": sid, **metrics})
    sheet.save(args.output / "rgb_video_event_preview_5.png")
    (args.output / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(args.output / "rgb_video_event_preview_5.png")


if __name__ == "__main__":
    main()
