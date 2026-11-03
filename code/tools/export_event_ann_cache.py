from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_event_ann_teacher import EventDINOBridge
try:
    from isgd.data import make_event_dataset
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
    from data import make_event_dataset


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--frame-mode", default="count_global")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = args.split == "train"
    dataset = make_event_dataset(
        "motion", args.data_root, train, args.steps, args.size, two_views=False,
        max_samples=args.max_samples, frame_mode=args.frame_mode, preload_frames=True,
    )
    payload = torch.load(args.checkpoint, map_location="cpu")
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local")
    model = EventDINOBridge(
        backbone, args.steps, len(dataset.class_names),
        channels=int(dataset[0][0].shape[1]),
    ).to(device)
    model.stem.load_state_dict(payload["stem"], strict=True)
    model.classifier.load_state_dict(payload["classifier"], strict=True)
    model.eval()
    ids, features, logits, labels = [], [], [], []
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    for events, target, _, sample_ids in loader:
        output = model(events.to(device, non_blocking=True))
        feature = F.normalize(output["feature"].float(), dim=-1).cpu()
        logit = output["logits"].float().cpu()
        ids.extend(str(sample_id) for sample_id in sample_ids)
        features.append(feature.unsqueeze(1).repeat(1, args.steps, 1))
        logits.append(logit)
        labels.append(torch.as_tensor(target))
    result = {
        "ids": ids,
        "prefix": torch.cat(features),
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "args": vars(args),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(result, output)
    print(json.dumps({"output": str(output), "samples": len(ids), "shape": list(result["prefix"].shape)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
