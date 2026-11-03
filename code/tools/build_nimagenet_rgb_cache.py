"""Build RGB DINOv2 prefix/patch caches for paired N-ImageNet-mini samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def load_teacher() -> torch.nn.Module:
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    return torch.hub.load(str(repo), "dinov2_vits14", source="local").eval()


def load_images(paths: list[Path], device: torch.device) -> torch.Tensor:
    values = []
    for path in paths:
        image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
        values.append(image.permute(2, 0, 1).div_(255.0))
    batch = torch.stack([
        F.interpolate(value.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze(0)
        for value in values
    ])
    return (batch.to(device) - MEAN.to(device).view(1, 3, 1, 1)) / STD.to(device).view(1, 3, 1, 1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-grid", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text(encoding="utf-8-sig"))
    ids = [str(x) for group in ("train", "validation", "test") for x in manifest.get(group, [])]
    pairs = json.loads((args.root / "manifests" / "paired_all.json").read_text(encoding="utf-8"))
    by_event = {str(item["event_path"]): item for item in pairs}
    rgb_paths, labels = [], []
    for sample_id in ids:
        item = by_event.get(sample_id)
        if item is None:
            raise KeyError(f"missing paired RGB for {sample_id}")
        rgb_paths.append(args.root / item["rgb_path"])
        labels.append(int(item["class_id"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = load_teacher().to(device)
    prefixes, patches = [], []
    for start in range(0, len(ids), args.batch_size):
        features = teacher.forward_features(load_images(rgb_paths[start:start + args.batch_size], device))
        prefixes.append(F.normalize(features["x_norm_clstoken"].float(), dim=-1).cpu().unsqueeze(1).repeat(1, 16, 1))
        tokens = features["x_norm_patchtokens"].float()
        side = int(tokens.shape[1] ** 0.5)
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        tokens = F.adaptive_avg_pool2d(tokens, (args.patch_grid, args.patch_grid))
        patches.append(F.normalize(tokens.flatten(2).transpose(1, 2), dim=-1).half().cpu().unsqueeze(1).repeat(1, 16, 1, 1))
        if start == 0 or start + len(prefixes[-1]) == len(ids):
            print(json.dumps({"rgb_cache_samples": min(start + len(prefixes[-1]), len(ids)), "total": len(ids)}), flush=True)
    payload = {
        "ids": ids, "prefix": torch.cat(prefixes), "patches": torch.cat(patches),
        "labels": torch.tensor(labels, dtype=torch.long),
        "source": "N-ImageNet-mini paired RGB -> frozen DINOv2; labels retained for evaluation only",
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(ids), "prefix": list(payload["prefix"].shape), "patches": list(payload["patches"].shape)}), flush=True)


if __name__ == "__main__":
    main()
