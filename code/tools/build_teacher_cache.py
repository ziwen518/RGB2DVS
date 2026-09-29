from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT.parent / 'core'))
from data import make_event_dataset
from teacher_inputs import grayscale, make_teacher_input, sobel

sys.path.insert(0, str(ROOT))
from generate_motion_events import load_image, make_full_bleed, render_offsets, rcls_offsets  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_teacher(device: torch.device):
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    return torch.hub.load(str(repo), "dinov2_vits14", source="local").eval().to(device)


@torch.no_grad()
def encode_prefixes(teacher, events: torch.Tensor, device: torch.device) -> torch.Tensor:
    prefixes = events.cumsum(dim=1)
    outputs = []
    for t in range(prefixes.shape[1]):
        image = prefixes[:, t].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        image = (image.to(device) - MEAN.to(device)) / STD.to(device)
        outputs.append(F.normalize(teacher.forward_features(image)["x_norm_clstoken"].float(), dim=-1).cpu())
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def _load_rgb_images(paths: list[Path], size: int = 224) -> torch.Tensor:
    images = []
    for path in paths:
        image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
        images.append(image.permute(2, 0, 1).div_(255.0))
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) == 1:
        return F.interpolate(torch.stack(images), size=(size, size), mode="bicubic", align_corners=False)
    return torch.stack([
        F.interpolate(image.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False).squeeze(0)
        for image in images
    ])


@torch.no_grad()
def _load_images(paths: list[Path], device: torch.device, mode: str) -> torch.Tensor:
    image = make_teacher_input(_load_rgb_images(paths).to(device), mode)
    return (image - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def encode_rgb(teacher, paths: list[Path], device: torch.device, mode: str = "rgb") -> torch.Tensor:
    image = _load_images(paths, device, mode)
    feature = teacher.forward_features(image)["x_norm_clstoken"].float()
    return F.normalize(feature, dim=-1).cpu()


@torch.no_grad()
def encode_rgb_patch(teacher, paths: list[Path], device: torch.device, grid: int = 8, mode: str = "rgb") -> torch.Tensor:
    """Encode DINO patch tokens and pool its 16x16 grid to the SNN grid."""
    image = _load_images(paths, device, mode)
    tokens = teacher.forward_features(image)["x_norm_patchtokens"].float()
    side = int(tokens.shape[1] ** 0.5)
    if side * side != tokens.shape[1]:
        raise ValueError(f"DINO patch token count is not square: {tokens.shape[1]}")
    tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
    tokens = F.adaptive_avg_pool2d(tokens, (grid, grid))
    tokens = tokens.flatten(2).transpose(1, 2)
    return F.normalize(tokens, dim=-1).half().cpu()


@torch.no_grad()
def encode_texture_targets(
    paths: list[Path],
    points: int = 8,
    grid: int = 8,
    trajectory: bool = False,
) -> torch.Tensor:
    """Build spatial Sobel targets aligned with the student's patch grid."""
    if trajectory:
        frames = _rcls_frames(paths, points)
        batch, steps = frames.shape[:2]
        flat = frames.reshape(batch * steps, *frames.shape[2:])
        texture = sobel(grayscale(flat))
        texture = F.adaptive_avg_pool2d(texture, (grid, grid))
        return texture.reshape(batch, steps, grid * grid).half().cpu()
    else:
        base = _load_rgb_images(paths)
        texture = sobel(grayscale(base))
        texture = F.adaptive_avg_pool2d(texture, (grid, grid)).flatten(1)
        return texture.unsqueeze(1).repeat(1, points, 1).half().cpu()


@torch.no_grad()
def _rcls_frames(paths: list[Path], points: int = 8) -> torch.Tensor:
    """Recreate the exact RGB states used by the RCLS event generator."""
    if points < 2:
        raise ValueError("trajectory_points must be at least 2")
    batches = []
    for path in paths:
        image = load_image(path, object_size=96)
        base = make_full_bleed(image, sensor_size=128)
        states = render_offsets(base, rcls_offsets(), padding_mode="reflection")
        indices = torch.linspace(0, states.shape[0] - 1, points).round().long()
        batches.append(states[indices])
    return torch.stack(batches)


@torch.no_grad()
def encode_rgb_trajectory(teacher, paths: list[Path], device: torch.device, points: int = 8, mode: str = "rgb") -> torch.Tensor:
    """Encode RGB states sampled from the exact RCLS event trajectory."""
    frames = _rcls_frames(paths, points)
    outputs = []
    for t in range(frames.shape[1]):
        image = F.interpolate(frames[:, t], size=(224, 224), mode="bicubic", align_corners=False)
        image = (make_teacher_input(image.to(device), mode) - MEAN.to(device)) / STD.to(device)
        feature = teacher.forward_features(image)["x_norm_clstoken"].float()
        outputs.append(F.normalize(feature, dim=-1).cpu())
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def encode_rgb_patch_trajectory(teacher, paths: list[Path], device: torch.device, points: int = 8, grid: int = 8, mode: str = "rgb") -> torch.Tensor:
    """Encode the same RGB motion trajectory as spatial DINO patch targets."""
    frames = _rcls_frames(paths, points)
    outputs = []
    for t in range(frames.shape[1]):
        image = F.interpolate(frames[:, t], size=(224, 224), mode="bicubic", align_corners=False)
        image = (make_teacher_input(image.to(device), mode) - MEAN.to(device)) / STD.to(device)
        tokens = teacher.forward_features(image)["x_norm_patchtokens"].float()
        side = int(tokens.shape[1] ** 0.5)
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        tokens = F.adaptive_avg_pool2d(tokens, (grid, grid)).flatten(2).transpose(1, 2)
        outputs.append(F.normalize(tokens, dim=-1).half().cpu())
    return torch.stack(outputs, dim=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="motion", choices=("motion", "ncaltech101", "cifar10dvs"))
    p.add_argument("--event-root", required=True)
    p.add_argument("--rgb-root", help="Original RGB root; required for motion dataset")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--trajectory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trajectory-points", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--patch-grid", type=int, default=8)
    p.add_argument("--cache-patches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher-input-mode", choices=("rgb", "gray", "sobel", "canny", "gray_sobel"), default="rgb")
    p.add_argument("--source-from-relative-path", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_event_dataset(
        args.dataset, args.event_root, train=args.split == "train",
        steps=args.trajectory_points, size=48, two_views=False,
        max_samples=args.max_samples,
        frame_mode="count" if args.dataset == "motion" else "count",
    )
    teacher = load_teacher(device)
    ids, prefixes, patches, textures, labels = [], [], [], [], []
    if args.dataset == "motion":
        if not args.rgb_root:
            raise ValueError("--rgb-root is required for motion teacher cache")
        # Reading MotionEvents.__getitem__ would rasterize every large RCLS
        # stream even though the RGB teacher only needs the exact paired path.
        for start in range(0, len(dataset.samples), args.batch_size):
            records = dataset.samples[start : start + args.batch_size]
            sample_ids = [
                str(path.relative_to(dataset.root)).replace("\\", "/")
                for path, _, _ in records
            ]
            ids.extend(sample_ids)
            rgb_paths = []
            for path, _, _ in records:
                if args.source_from_relative_path:
                    source = str(path.relative_to(dataset.root).with_suffix(".png"))
                else:
                    with np.load(path, allow_pickle=False) as event_file:
                        source = str(event_file["source"])
                rgb_path = Path(args.rgb_root) / source
                if not rgb_path.exists():
                    raise FileNotFoundError(f"paired RGB image is missing: {rgb_path}")
                rgb_paths.append(rgb_path)
            if args.trajectory:
                prefixes.append(
                    encode_rgb_trajectory(teacher, rgb_paths, device, args.trajectory_points, args.teacher_input_mode)
                )
                if args.cache_patches:
                    patches.append(
                        encode_rgb_patch_trajectory(teacher, rgb_paths, device, args.trajectory_points, args.patch_grid, args.teacher_input_mode)
                    )
            else:
                prefixes.append(
                    encode_rgb(teacher, rgb_paths, device, args.teacher_input_mode)
                    .unsqueeze(1)
                    .repeat(1, args.trajectory_points, 1)
                )
                if args.cache_patches:
                    patches.append(
                        encode_rgb_patch(teacher, rgb_paths, device, args.patch_grid, args.teacher_input_mode)
                        .unsqueeze(1)
                        .repeat(1, args.trajectory_points, 1, 1)
                    )
            textures.append(
                encode_texture_targets(
                    rgb_paths,
                    args.trajectory_points,
                    args.patch_grid,
                    args.trajectory,
                )
            )
            labels.append(torch.tensor([label for _, label, _ in records]))
            if len(ids) % 2048 == 0 or len(ids) == len(dataset.samples):
                print(json.dumps({"cache_samples": len(ids), "total": len(dataset.samples)}), flush=True)
    else:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        for events, y, _, sample_ids in loader:
            ids.extend(str(sample_id) for sample_id in sample_ids)
            prefixes.append(encode_prefixes(teacher, events, device))
            labels.append(torch.as_tensor(y))
    payload = {"ids": ids, "prefix": torch.cat(prefixes), "labels": torch.cat(labels), "args": vars(args)}
    if patches:
        payload["patches"] = torch.cat(patches)
    if textures:
        payload["texture"] = torch.cat(textures)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({"device": str(device), "samples": len(ids), "shape": list(payload["prefix"].shape), "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
