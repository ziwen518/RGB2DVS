"""Extract a frozen DINO representation from N-Caltech event frames.

This is an event-native relay input, not an RGB teacher: polarity channels are
aggregated over the event frames and mapped to three pseudo-image channels.
The resulting cache is used only to fit the event->RGB observable relay.
"""
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


def event_image(path: Path) -> torch.Tensor:
    frames = torch.from_numpy(np.load(path, allow_pickle=False)).float() / 255.0
    # [T,2,H,W] -> polarity-aware pseudo RGB; preserve positive/negative mass.
    on = frames[:, 0].mean(0)
    off = frames[:, 1].mean(0)
    image = torch.stack((on, off, (on + off).mul(0.5)), dim=0)
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return (image - MEAN.view(3, 1, 1)) / STD.view(3, 1, 1)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--frame-root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-grid", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    manifest = json.loads(args.split.read_text(encoding="utf-8"))
    ids = [str(x) for group in ("train", "validation", "test") for x in manifest.get(group, [])]
    pairs = json.loads((args.root / "manifests" / "paired_all.json").read_text(encoding="utf-8"))
    labels = {str(p["event_path"]): int(p["class_id"]) for p in pairs}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = load_teacher().to(device)
    prefixes, patches = [], []
    for start in range(0, len(ids), args.batch_size):
        batch = []
        for sample_id in ids[start : start + args.batch_size]:
            rel_path = Path(sample_id)
            # Frame cache is rooted at the class directory, while IDs retain
            # the dataset-level ``events/`` prefix.
            if rel_path.parts and rel_path.parts[0].lower() == "events":
                rel_path = Path(*rel_path.parts[1:])
            rel = rel_path.with_suffix(".npy")
            batch.append(event_image(args.frame_root / rel))
        output = teacher.forward_features(torch.stack(batch).to(device))
        prefixes.append(F.normalize(output["x_norm_clstoken"].float(), dim=-1).cpu().unsqueeze(1).repeat(1, 16, 1))
        tokens = output["x_norm_patchtokens"].float()
        side = int(tokens.shape[1] ** 0.5)
        tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        tokens = F.adaptive_avg_pool2d(tokens, (args.patch_grid, args.patch_grid))
        patches.append(F.normalize(tokens.flatten(2).transpose(1, 2), dim=-1).half().cpu().unsqueeze(1).repeat(1, 16, 1, 1))
        if start == 0 or start + len(batch) == len(ids):
            print(json.dumps({"event_cache_samples": min(start + len(batch), len(ids)), "total": len(ids)}), flush=True)
    payload = {
        "ids": ids,
        "prefix": torch.cat(prefixes),
        "patches": torch.cat(patches),
        "labels": torch.tensor([labels.get(x, -1) for x in ids], dtype=torch.long),
        "source": "N-Caltech event polarity aggregate -> frozen DINO; no labels used",
    }
    if (payload["labels"] < 0).any():
        raise ValueError("event cache contains IDs without labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(ids), "prefix": list(payload["prefix"].shape)}), flush=True)


if __name__ == "__main__":
    main()
