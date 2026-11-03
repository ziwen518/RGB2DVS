from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def render_motion(image: Image.Image, frames: int = 128, amplitude: float = 0.02) -> list[Image.Image]:
    """Generate a deterministic, low-amplitude camera-motion sequence.

    This is intentionally geometry-controlled: it does not hallucinate texture
    or alter object identity, making the sequence safe for paired event tests.
    """
    image = image.convert("RGB")
    width, height = image.size
    output = []
    for index in range(frames):
        phase = 2.0 * np.pi * index / max(frames - 1, 1)
        u = smoothstep(index / max(frames - 1, 1))
        dx = amplitude * width * np.sin(phase)
        dy = 0.5 * amplitude * height * np.sin(phase + np.pi / 2.0)
        scale = 1.0 + 0.35 * amplitude * np.sin(phase)
        crop_w = max(2, int(round(width / scale)))
        crop_h = max(2, int(round(height / scale)))
        cx = width / 2.0 + dx
        cy = height / 2.0 + dy
        left = int(round(cx - crop_w / 2.0))
        top = int(round(cy - crop_h / 2.0))
        # Reflect padding avoids introducing black borders/discontinuities.
        padded = Image.new("RGB", (width * 3, height * 3))
        for yy in range(3):
            for xx in range(3):
                padded.paste(image, (xx * width, yy * height))
        frame = padded.crop((left + width, top + height, left + width + crop_w, top + height + crop_h))
        frame = frame.resize((width, height), Image.Resampling.BICUBIC)
        if u < 0.02 or u > 0.98:
            frame = frame.filter(ImageFilter.GaussianBlur(radius=0.15))
        output.append(frame)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--amplitude", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = render_motion(Image.open(args.input), args.frames, args.amplitude)
    for index, frame in enumerate(frames):
        frame.save(args.output / f"{index:04d}.png")
    print({"input": str(args.input), "output": str(args.output), "frames": len(frames), "size": frames[0].size})


if __name__ == "__main__":
    main()
