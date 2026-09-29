"""Generate paired RGB/RCLS event samples from an image-folder dataset.

The default trajectory follows the CIFAR10-DVS acquisition protocol: six
repetitions of a four-path, 45-degree closed loop. Images are rendered
full-bleed with reflection sampling, so motion does not expose a synthetic
high-contrast rectangular canvas border.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--class-name", help="Generate only one class directory; useful for parallel jobs.")
    p.add_argument("--class-names", nargs="+", help="Generate a list of class directories in one worker.")
    p.add_argument("--sensor-size", type=int, default=128)
    p.add_argument("--object-size", type=int, default=96)
    p.add_argument("--trajectory", choices=("rcls", "diagonal"), default="rcls")
    p.add_argument("--interpolation", type=int, default=8)
    p.add_argument("--trajectory-points", type=int, default=8)
    p.add_argument("--rcls-loops", type=int, default=6)
    p.add_argument("--rcls-amplitude", type=float, default=10.0)
    p.add_argument("--rcls-steps-per-path", type=int, default=5)
    p.add_argument("--frame-us", type=int, default=10000)
    p.add_argument("--contrast-threshold", type=float, default=0.15)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--save-frames", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_image(path: Path, object_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(object_size / max(width, 1), object_size / max(height, 1))
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    value = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return value


def make_base(image: torch.Tensor, sensor_size: int, object_size: int) -> torch.Tensor:
    image = load_image_from_tensor(image, object_size)
    canvas = torch.full((3, sensor_size, sensor_size), 0.02, dtype=torch.float32)
    h, w = image.shape[-2:]
    canvas[:, 8 : 8 + h, 8 : 8 + w] = image
    return canvas


def make_full_bleed(image: torch.Tensor, sensor_size: int) -> torch.Tensor:
    """Resize to cover the sensor and center-crop without adding a frame."""
    h, w = image.shape[-2:]
    scale = max(sensor_size / max(w, 1), sensor_size / max(h, 1))
    out_h = max(sensor_size, round(h * scale))
    out_w = max(sensor_size, round(w * scale))
    resized = F.interpolate(
        image.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    top = (out_h - sensor_size) // 2
    left = (out_w - sensor_size) // 2
    return resized[:, top : top + sensor_size, left : left + sensor_size]


def load_image_from_tensor(image: torch.Tensor, object_size: int) -> torch.Tensor:
    h, w = image.shape[-2:]
    scale = min(object_size / max(w, 1), object_size / max(h, 1))
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def shift(image: torch.Tensor, dx: float, dy: float, padding_mode: str = "border") -> torch.Tensor:
    _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
    )
    # Subtracting the displacement makes positive dx/dy move content right/down.
    grid = torch.stack(
        [xs - 2.0 * dx / max(1, w - 1), ys - 2.0 * dy / max(1, h - 1)], dim=-1
    ).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    ).squeeze(0)


def render_offsets(image: torch.Tensor, offsets: list[tuple[float, float]], padding_mode: str) -> torch.Tensor:
    """Render all trajectory states in one grid-sample call."""
    _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij"
    )
    base_grid = torch.stack([xs, ys], dim=-1)
    displacement = torch.tensor(
        [[-2.0 * dx / max(1, w - 1), -2.0 * dy / max(1, h - 1)] for dx, dy in offsets],
        dtype=image.dtype,
    ).view(-1, 1, 1, 2)
    grid = base_grid.unsqueeze(0) + displacement
    source = image.unsqueeze(0).expand(len(offsets), -1, -1, -1)
    return F.grid_sample(
        source, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )


def continuous_frames(base: torch.Tensor, interpolation: int) -> torch.Tensor:
    anchors = ((0.0, 0.0), (8.0, 8.0), (16.0, 16.0), (24.0, 24.0))
    frames = []
    for index in range(len(anchors) - 1):
        start = anchors[index]
        end = anchors[index + 1]
        for step in range(interpolation):
            alpha = step / float(interpolation)
            dx = start[0] * (1.0 - alpha) + end[0] * alpha
            dy = start[1] * (1.0 - alpha) + end[1] * alpha
            frames.append(shift(base, dx, dy))
    frames.append(shift(base, *anchors[-1]))
    return torch.stack(frames)


def rcls_offsets(loops: int = 6, amplitude: float = 10.0, steps_per_path: int = 5) -> list[tuple[float, float]]:
    """CIFAR10-DVS RCLS diamond: four 50-ms paths at 45 degrees.

    With the default five 10-ms steps per path, each x/y velocity component is
    200 px/s, matching Table 2 of Li et al., Frontiers in Neuroscience 2017.
    """
    if loops < 1 or steps_per_path < 1 or amplitude <= 0:
        raise ValueError("RCLS loops, steps and amplitude must be positive")
    vertices = (
        (-amplitude, 0.0),
        (0.0, -amplitude),
        (amplitude, 0.0),
        (0.0, amplitude),
        (-amplitude, 0.0),
    )
    offsets = [vertices[0]]
    for _ in range(loops):
        for path_index in range(4):
            start, end = vertices[path_index], vertices[path_index + 1]
            for step_index in range(1, steps_per_path + 1):
                alpha = step_index / float(steps_per_path)
                offsets.append(
                    (
                        start[0] * (1.0 - alpha) + end[0] * alpha,
                        start[1] * (1.0 - alpha) + end[1] * alpha,
                    )
                )
    return offsets


def frames_to_events(frames: torch.Tensor, threshold: float, frame_us: int) -> dict[str, np.ndarray]:
    rgb = frames.clamp_min(1.0e-3)
    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    log_luminance = luminance.log()
    residual = torch.zeros_like(log_luminance[0])
    xs, ys, ts, ps = [], [], [], []
    for frame_index in range(1, len(frames)):
        total = residual + log_luminance[frame_index] - log_luminance[frame_index - 1]
        count = torch.zeros_like(total, dtype=torch.int64)
        positive = total >= threshold
        negative = total <= -threshold
        count[positive] = torch.floor(total[positive] / threshold).to(torch.int64)
        count[negative] = torch.ceil(total[negative] / threshold).to(torch.int64)
        residual = total - count.to(total.dtype) * threshold
        yy, xx = torch.where(count != 0)
        signed_count = count[yy, xx].cpu().numpy()
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
        frame_x = np.repeat(xx.cpu().numpy().astype(np.int16), repeats)
        frame_y = np.repeat(yy.cpu().numpy().astype(np.int16), repeats)
        frame_p = np.repeat((signed_count > 0).astype(np.uint8), repeats)
        order = np.argsort(frame_t, kind="stable")
        xs.append(frame_x[order])
        ys.append(frame_y[order])
        ts.append(frame_t[order])
        ps.append(frame_p[order])
    if not xs:
        return {
            "x": np.empty(0, dtype=np.int16), "y": np.empty(0, dtype=np.int16),
            "t": np.empty(0, dtype=np.int64), "p": np.empty(0, dtype=np.uint8),
        }
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "t": np.concatenate(ts),
        "p": np.concatenate(ps),
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root) / args.split
    output_root = Path(args.output_root) / args.split
    classes = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    if args.class_names is not None:
        unknown = sorted(set(args.class_names) - set(classes))
        if unknown:
            raise ValueError(f"unknown class directories: {unknown}")
        classes = list(args.class_names)
    elif args.class_name is not None:
        if args.class_name not in classes:
            raise ValueError(f"unknown class directory: {args.class_name}")
        classes = [args.class_name]
    if not classes:
        raise FileNotFoundError(f"no class directories under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = 0
    for label, class_name in enumerate(classes):
        files = sorted(path for path in (source_root / class_name).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if args.max_samples is not None and len(files) > args.max_samples:
            files = [files[i] for i in rng.permutation(len(files))[: args.max_samples]]
            files.sort()
        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        for source in files:
            image = load_image(source, args.object_size)
            if args.trajectory == "rcls":
                base = make_full_bleed(image, args.sensor_size)
                offsets = rcls_offsets(
                    args.rcls_loops, args.rcls_amplitude, args.rcls_steps_per_path
                )
                frames = render_offsets(base, offsets, padding_mode="reflection")
            else:
                base = make_base(image, args.sensor_size, args.object_size)
                frames = continuous_frames(base, args.interpolation)
            event = frames_to_events(frames, args.contrast_threshold, args.frame_us)
            relative = source.relative_to(source_root).with_suffix(".npz")
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **event,
                "label": np.int64(label),
                "source": np.asarray(str(source.relative_to(Path(args.source_root))).replace("\\", "/")),
                "trajectory": np.asarray(args.trajectory),
            }
            if args.save_frames:
                payload["frames"] = np.rint(frames.numpy() * 255.0).astype(np.uint8)
            np.savez_compressed(target, **payload)
            total += 1
        print(json.dumps({"class": class_name, "label": label, "samples": len(files)}), flush=True)
    if args.class_name is None and args.class_names is None:
        manifest = {
            "source_root": str(Path(args.source_root)),
            "output_root": str(Path(args.output_root)),
            "split": args.split,
            "classes": {name: i for i, name in enumerate(classes)},
            "sensor_size": args.sensor_size,
            "object_size": args.object_size,
            "trajectory": args.trajectory,
            "border_mode": "full_bleed_reflection" if args.trajectory == "rcls" else "dark_canvas",
            "rcls_vertices": [[-args.rcls_amplitude, 0], [0, -args.rcls_amplitude], [args.rcls_amplitude, 0], [0, args.rcls_amplitude], [-args.rcls_amplitude, 0]],
            "rcls_loops": args.rcls_loops,
            "rcls_steps_per_path": args.rcls_steps_per_path,
            "rcls_path_ms": args.rcls_steps_per_path * args.frame_us / 1000.0,
            "interpolation": args.interpolation,
            "trajectory_points": args.trajectory_points,
            "frame_us": args.frame_us,
            "contrast_threshold": args.contrast_threshold,
            "save_frames": args.save_frames,
            "samples": total,
        }
        manifest_path = Path(args.output_root) / f"motion_event_manifest_{args.split}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"samples": total, "manifest": str(manifest_path)}))
    else:
        print(json.dumps({"class_name": args.class_name, "samples": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
