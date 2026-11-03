"""Build label-free observable patch targets for the paired CIFAR event set.

Only train IDs estimate the event-to-RGB relay. Validation targets may be
emitted for model selection, while held-out test samples are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-grid", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def load_teacher() -> torch.nn.Module:
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    return torch.hub.load(str(repo), "dinov2_vits14", source="local").eval()


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = F.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    return (image - MEAN[:, None, None]) / STD[:, None, None]


def load_rgb(path: Path) -> torch.Tensor:
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
    return normalize_image(torch.from_numpy(value).permute(2, 0, 1).div_(255.0))


def load_event(path: Path) -> torch.Tensor:
    frames = torch.from_numpy(np.load(path, allow_pickle=False)).float().div_(255.0)
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError(f"expected [T,2,H,W] at {path}, got {tuple(frames.shape)}")
    on, off = frames[:, 0].mean(0), frames[:, 1].mean(0)
    return normalize_image(torch.stack((on, off, 0.5 * (on + off))))


def patch_tokens(features: dict[str, torch.Tensor], grid: int) -> torch.Tensor:
    tokens = features["x_norm_patchtokens"].float()
    side = int(tokens.shape[1] ** 0.5)
    if side * side != tokens.shape[1]:
        raise ValueError("DINO patch-token count is not square")
    feature_map = tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], side, side
    )
    pooled = F.adaptive_avg_pool2d(feature_map, (grid, grid))
    return F.normalize(pooled.flatten(2).transpose(1, 2), dim=-1)


def observable_patch_targets(
    event: torch.Tensor, rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate RGB patch relations by event-side sample observability.

    The downstream spectral-affinity loss is permutation invariant. Therefore
    neither this cache nor the Student objective assumes aligned patch indices.
    """
    if event.shape != rgb.shape or event.ndim != 3:
        raise ValueError(f"cache mismatch: event={tuple(event.shape)}, rgb={tuple(rgb.shape)}")
    event_summary = F.normalize(event.float().mean(dim=1), dim=-1)
    rgb_summary = F.normalize(rgb.float().mean(dim=1), dim=-1)
    confidence = F.cosine_similarity(event_summary, rgb_summary, dim=-1).clamp(0, 1)
    return F.normalize(rgb.float(), dim=-1), confidence[:, None].expand(-1, rgb.shape[1])


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.patch_grid <= 0:
        raise ValueError("batch size and patch grid must be positive")
    split_bytes = args.split_manifest.read_bytes()
    split = json.loads(split_bytes.decode("utf-8-sig"))
    train_ids = list(map(str, split["train_ids"]))
    validation_ids = list(map(str, split["validation_ids"]))
    test_ids = list(map(str, split.get("test_ids", [])))
    groups = tuple(map(set, (train_ids, validation_ids, test_ids)))
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("train, validation, and test IDs must be disjoint")

    # The held-out IDs are audited for overlap but deliberately not dereferenced.
    target_ids = train_ids
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = load_teacher().to(device)
    event_cache, rgb_cache = [], []
    for start in range(0, len(target_ids), args.batch_size):
        sample_ids = target_ids[start : start + args.batch_size]
        event_images, rgb_images = [], []
        for sample_id in sample_ids:
            relative = Path(sample_id)
            event_path = args.frame_cache / relative.with_suffix(".npy")
            rgb_path = args.rgb_root / relative.with_suffix(".png")
            if not event_path.is_file() or not rgb_path.is_file():
                raise FileNotFoundError(f"missing pair: event={event_path}, rgb={rgb_path}")
            event_images.append(load_event(event_path))
            rgb_images.append(load_rgb(rgb_path))
        combined = torch.cat((torch.stack(event_images), torch.stack(rgb_images))).to(device)
        tokens = patch_tokens(teacher.forward_features(combined), args.patch_grid).cpu().half()
        event_cache.append(tokens[: len(sample_ids)])
        rgb_cache.append(tokens[len(sample_ids) :])
        print(json.dumps({
            "extracted": min(start + len(sample_ids), len(target_ids)),
            "total_train": len(target_ids), "validation_loaded": False,
            "test_loaded": False,
        }), flush=True)

    event_tokens, rgb_tokens = torch.cat(event_cache), torch.cat(rgb_cache)
    prediction, confidence = observable_patch_targets(event_tokens, rgb_tokens)
    payload = {
        "ids": target_ids,
        "tokens": prediction.half(),
        "confidence": confidence.half(),
        "fit_ids": train_ids,
        "validation_ids": validation_ids,
        "excluded_test_ids": test_ids,
        "labels_used": False,
        "class_prototypes_used": False,
        "test_used_for_estimation": False,
        "test_target_emitted": False,
        "validation_used_for_estimation": False,
        "validation_target_emitted": False,
        "fit_only_estimation": True,
        "permutation_invariant_target": True,
        "source": "train-only RGB-DINO patch affinity gated by event-DINO observability; no pixel alignment",
        "patch_grid": args.patch_grid,
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output), "fit": len(train_ids),
        "excluded_validation": len(validation_ids), "excluded_test": len(test_ids),
        "tokens": list(payload["tokens"].shape),
        "mean_confidence": float(confidence.mean()), "labels_used": False,
        "validation_target_emitted": False, "test_used_for_estimation": False,
        "test_target_emitted": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
