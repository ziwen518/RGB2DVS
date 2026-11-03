from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


POSITIVE = np.asarray([220, 48, 48], dtype=np.uint8)
NEGATIVE = np.asarray([38, 92, 205], dtype=np.uint8)
BACKGROUND = np.asarray([250, 250, 250], dtype=np.uint8)
PANEL_SIZE = 512


def load_npz_events(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        events = np.empty(
            len(payload["x"]),
            dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)],
        )
        for field in events.dtype.names:
            events[field] = payload[field]
    return events


def load_aedat2_events(path: Path) -> np.ndarray:
    """Read the AEDAT2 DVS128 layout used by the official CIFAR10-DVS files."""
    payload = path.read_bytes()
    offset = 0
    while offset < len(payload) and payload[offset : offset + 1] == b"#":
        newline = payload.find(b"\n", offset)
        if newline < 0:
            raise ValueError(f"unterminated AEDAT header: {path}")
        offset = newline + 1
    raw = np.frombuffer(payload, dtype=">u4", offset=offset)
    if raw.size % 2:
        raw = raw[:-1]
    packets = raw.reshape(-1, 2)
    address = packets[:, 0]
    events = np.empty(
        len(packets),
        dtype=[("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.uint8)],
    )
    events["x"] = (address >> 8) & 0x7F
    events["y"] = (address >> 1) & 0x7F
    events["t"] = packets[:, 1].astype(np.int64)
    events["p"] = address & 1
    # AEDAT streams can contain non-polarity packets; retain only valid DVS128 data.
    valid = (
        (events["x"] >= 0) & (events["x"] < 128)
        & (events["y"] >= 0) & (events["y"] < 128)
    )
    events = events[valid]
    if not len(events) or np.any(np.diff(events["t"]) < 0):
        raise ValueError(f"invalid AEDAT2 event stream: {path}")
    return events


def render_window(
    events: np.ndarray, start: int, end: int, height: int, width: int,
) -> Image.Image:
    selected = events[(events["t"] >= start) & (events["t"] < end)]
    image = np.broadcast_to(BACKGROUND.astype(np.float32), (height, width, 3)).copy()
    if len(selected):
        x = np.clip(selected["x"].astype(np.int64), 0, width - 1)
        y = np.clip(selected["y"].astype(np.int64), 0, height - 1)
        flat = y * width + x
        positive = np.bincount(
            flat[selected["p"] > 0], minlength=height * width
        ).reshape(height, width)
        negative = np.bincount(
            flat[selected["p"] == 0], minlength=height * width
        ).reshape(height, width)
        # Preserve both polarities instead of cancelling repeated ON/OFF events.
        pos = np.log1p(positive.astype(np.float32))
        neg = np.log1p(negative.astype(np.float32))
        pos_active = pos[pos > 0]
        neg_active = neg[neg > 0]
        pos_scale = float(np.percentile(pos_active, 99.0)) if pos_active.size else 1.0
        neg_scale = float(np.percentile(neg_active, 99.0)) if neg_active.size else 1.0
        pos_alpha = np.power(np.clip(pos / max(pos_scale, 1.0e-6), 0.0, 1.0), 0.6)
        neg_alpha = np.power(np.clip(neg / max(neg_scale, 1.0e-6), 0.0, 1.0), 0.6)
        pos_alpha = pos_alpha[..., None]
        neg_alpha = neg_alpha[..., None]
        image = image * (1.0 - neg_alpha) + NEGATIVE.astype(np.float32) * neg_alpha
        image = image * (1.0 - pos_alpha) + POSITIVE.astype(np.float32) * pos_alpha
    return Image.fromarray(image.round().astype(np.uint8)), int(len(selected))


def select_active_consecutive_edges(events: np.ndarray, bins: int) -> np.ndarray:
    """Pick a short, dense authentic interval and split it into adjacent windows."""
    start = int(events["t"].min())
    end = int(events["t"].max()) + 1
    fine_bins = max(120, bins * 20)
    fine_edges = np.linspace(start, end, fine_bins + 1).astype(np.int64)
    fine_counts, _ = np.histogram(events["t"], bins=fine_edges)
    block_counts = np.convolve(fine_counts, np.ones(bins, dtype=np.int64), mode="valid")
    block_start = int(np.argmax(block_counts))
    return fine_edges[block_start : block_start + bins + 1]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def add_label(image: Image.Image, text: str, height: int = 56) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), text, fill=(25, 25, 25), font=load_font(22))
    return canvas


def make_sequence(
    rgb_path: Path,
    event_path: Path,
    output: Path,
    bins: int,
) -> dict[str, object]:
    events = load_npz_events(event_path)
    rgb = Image.open(rgb_path).convert("RGB")
    width = max(int(events["x"].max()) + 1, rgb.width)
    height = max(int(events["y"].max()) + 1, rgb.height)
    rgb_panel = rgb.resize((height, height), Image.Resampling.NEAREST)
    rgb_panel.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST).save(
        output / "rcls_rgb.png"
    )

    edges = select_active_consecutive_edges(events, bins)
    panels = [add_label(rgb_panel.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST), "Paired CIFAR-10 RGB")]
    counts: list[int] = []
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        frame, count = render_window(events, int(start), int(end), height, width)
        frame.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST).save(
            output / f"rcls_event_t{index:02d}.png"
        )
        counts.append(count)
        panel = frame.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST)
        panels.append(add_label(panel, f"Event window {index}/{bins} | {count:,} events"))

    gap = 12
    sheet = Image.new(
        "RGB", (sum(panel.width for panel in panels) + gap * (len(panels) - 1), panels[0].height),
        (238, 238, 238),
    )
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    sheet.save(output / "cifar10_rcls_rgb_event_sequence.png")
    return {
        "rgb_source": str(rgb_path),
        "event_source": str(event_path),
        "event_count": int(len(events)),
        "time_range": [int(events["t"].min()), int(events["t"].max())],
        "visualized_time_range": [int(edges[0]), int(edges[-1] - 1)],
        "window_edges": edges.astype(int).tolist(),
        "window_counts": counts,
        "note": "Real CIFAR-10 RGB paired with its generated RCLS event stream; not official CIFAR10-DVS.",
    }


def make_real_dvs(
    event_path: Path,
    output: Path,
    bins: int,
) -> dict[str, object]:
    events = load_aedat2_events(event_path)
    width = max(128, int(events["x"].max()) + 1)
    height = max(128, int(events["y"].max()) + 1)
    # Use the temporal middle where the moving display usually has strongest structure.
    all_edges = np.linspace(int(events["t"].min()), int(events["t"].max()) + 1, bins + 1).astype(np.int64)
    frames: list[tuple[Image.Image, int]] = [
        render_window(events, int(start), int(end), height, width)
        for start, end in zip(all_edges[:-1], all_edges[1:])
    ]
    best_index = int(np.argmax([count for _, count in frames]))
    best, best_count = frames[best_index]
    best.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST).save(
        output / "real_cifar10_dvs_event.png"
    )

    panels = []
    for index, (frame, count) in enumerate(frames, start=1):
        frame.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST).save(
            output / f"real_dvs_event_t{index:02d}.png"
        )
        panels.append(add_label(
            frame.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST),
            f"Real DVS window {index}/{bins} | {count:,} events",
        ))
    gap = 12
    sheet = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), panels[0].height), (238, 238, 238))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    sheet.save(output / "real_cifar10_dvs_sequence.png")
    return {
        "event_source": str(event_path),
        "event_count": int(len(events)),
        "time_range": [int(events["t"].min()), int(events["t"].max())],
        "window_counts": [count for _, count in frames],
        "selected_window": best_index + 1,
        "selected_window_events": best_count,
        "note": "Original official CIFAR10-DVS AEDAT event stream; red=ON, blue=OFF.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcls-rgb", type=Path, required=True)
    parser.add_argument("--rcls-events", type=Path, required=True)
    parser.add_argument("--real-dvs-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=6)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("bins must be at least 2")
    for path in (args.rcls_rgb, args.rcls_events, args.real_dvs_events):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "rcls": make_sequence(args.rcls_rgb, args.rcls_events, args.output, args.bins),
        "real_cifar10_dvs": make_real_dvs(args.real_dvs_events, args.output, args.bins),
        "visual_encoding": {"ON_positive": POSITIVE.tolist(), "OFF_negative": NEGATIVE.tolist()},
        "visual_normalization": "independent per-window log1p ON/OFF counts, per-polarity 99th-percentile scale",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
