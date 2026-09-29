from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_generator_module():
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import generate_motion_events as generator

    return generator


def render_batch(base: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = base.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=base.device),
        torch.linspace(-1.0, 1.0, width, device=base.device),
        indexing="ij",
    )
    base_grid = torch.stack([xs, ys], dim=-1)
    displacement = torch.stack(
        [
            -2.0 * offsets[:, 0] / max(1, width - 1),
            -2.0 * offsets[:, 1] / max(1, height - 1),
        ],
        dim=-1,
    )
    frames = base.unsqueeze(1).expand(-1, len(offsets), -1, -1, -1)
    grid = base_grid.view(1, 1, height, width, 2) + displacement.view(
        1, len(offsets), 1, 1, 2
    )
    grid = grid.expand(batch, -1, -1, -1, -1)
    return F.grid_sample(
        frames.reshape(batch * len(offsets), 3, height, width),
        grid.reshape(batch * len(offsets), height, width, 2),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    ).reshape(batch, len(offsets), 3, height, width)


def events_batch(frames: torch.Tensor, threshold: float, frame_us: int) -> list[dict[str, np.ndarray]]:
    log_luminance = (
        0.299 * frames[:, :, 0]
        + 0.587 * frames[:, :, 1]
        + 0.114 * frames[:, :, 2]
    ).clamp_min(1.0e-3).log()
    residual = torch.zeros_like(log_luminance[:, 0])
    count_frames: list[np.ndarray] = []
    for frame_index in range(1, frames.shape[1]):
        total = residual + log_luminance[:, frame_index] - log_luminance[:, frame_index - 1]
        count = torch.zeros_like(total, dtype=torch.int64)
        positive = total >= threshold
        negative = total <= -threshold
        count[positive] = torch.floor(total[positive] / threshold).to(torch.int64)
        count[negative] = torch.ceil(total[negative] / threshold).to(torch.int64)
        residual = total - count.to(total.dtype) * threshold
        count_frames.append(count.cpu().numpy())

    output: list[dict[str, np.ndarray]] = []
    for batch_index in range(frames.shape[0]):
        xs, ys, ts, ps = [], [], [], []
        for frame_index, counts in enumerate(count_frames, start=1):
            count = counts[batch_index]
            yy, xx = np.where(count != 0)
            signed_count = count[yy, xx]
            repeats = np.abs(signed_count).astype(np.int64)
            if not len(repeats):
                continue
            total_events = int(repeats.sum())
            ends = np.cumsum(repeats)
            starts = ends - repeats
            repeated_starts = np.repeat(starts, repeats)
            repeated_counts = np.repeat(repeats, repeats)
            ordinal = np.arange(total_events, dtype=np.int64) - repeated_starts + 1
            fraction = ordinal / (repeated_counts + 1.0)
            frame_t = np.rint((frame_index - 1 + fraction) * frame_us).astype(np.int64)
            frame_x = np.repeat(xx.astype(np.int16), repeats)
            frame_y = np.repeat(yy.astype(np.int16), repeats)
            frame_p = np.repeat((signed_count > 0).astype(np.uint8), repeats)
            order = np.argsort(frame_t, kind="stable")
            xs.append(frame_x[order])
            ys.append(frame_y[order])
            ts.append(frame_t[order])
            ps.append(frame_p[order])
        if not xs:
            output.append(
                {
                    "x": np.empty(0, dtype=np.int16),
                    "y": np.empty(0, dtype=np.int16),
                    "t": np.empty(0, dtype=np.int64),
                    "p": np.empty(0, dtype=np.uint8),
                }
            )
        else:
            output.append(
                {
                    "x": np.concatenate(xs),
                    "y": np.concatenate(ys),
                    "t": np.concatenate(ts),
                    "p": np.concatenate(ps),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--class-name", action="append")
    args = parser.parse_args()
    generator = load_generator_module()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU batch generator requires CUDA")
    all_classes = sorted(path.name for path in (args.source_root / args.split).iterdir() if path.is_dir())
    selected = set(args.class_name or all_classes)
    classes = [name for name in all_classes if name in selected]
    if not classes:
        raise RuntimeError("no classes selected")
    label_by_class = {name: index for index, name in enumerate(all_classes)}
    offsets = torch.tensor(generator.rcls_offsets(6, 10.0, 5), dtype=torch.float32, device=device)
    output_split = args.output_root / args.split
    total = 0
    for label, class_name in enumerate(classes):
        sources = sorted(
            path
            for path in (args.source_root / args.split / class_name).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if args.max_samples_per_class is not None:
            sources = sources[: args.max_samples_per_class]
        class_output = output_split / class_name
        class_output.mkdir(parents=True, exist_ok=True)
        pending_sources = [
            source for source in sources
            if not (class_output / source.relative_to(args.source_root / args.split).with_suffix(".npz").name).exists()
        ]
        for start in range(0, len(pending_sources), args.batch_size):
            batch_sources = pending_sources[start : start + args.batch_size]
            bases = torch.stack(
                [generator.make_full_bleed(generator.load_image(path, 96), 128) for path in batch_sources]
            ).to(device)
            with torch.no_grad():
                frames = render_batch(bases, offsets)
                event_batch = events_batch(frames, 0.15, 10000)
            pending = [(source, event) for source, event in zip(batch_sources, event_batch)
                       if not (class_output / source.relative_to(args.source_root / args.split).with_suffix(".npz").name).exists()]
            for source, event in pending:
                relative = source.relative_to(args.source_root / args.split).with_suffix(".npz")
                target = class_output / relative.name
                np.savez_compressed(
                    target,
                    **event,
                    label=np.int64(label_by_class[class_name]),
                    source=np.asarray(str(source.relative_to(args.source_root)).replace("\\", "/")),
                    trajectory=np.asarray("rcls"),
                )
                total += 1
            print(json.dumps({"split": args.split, "class": class_name, "completed": total}), flush=True)
    print(json.dumps({"completed": total, "split": args.split, "classes": len(classes)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
