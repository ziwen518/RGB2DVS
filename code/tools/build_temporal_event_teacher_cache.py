from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / "train"
sys.path.insert(0, str(TRAIN))
from train_event_ann_teacher import EventDINOBridge

try:
    from isgd.data import make_event_dataset
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT.parent / "core"))
    from data import make_event_dataset


@torch.inference_mode()
def main() -> int:
    p = argparse.ArgumentParser(description="Export causal prefix-conditioned event-teacher features.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, required=True)
    p.add_argument("--frame-mode", default="count_global")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_event_dataset(
        "motion", args.data_root, args.split == "train", args.steps, args.size,
        two_views=False, max_samples=args.max_samples,
        frame_mode=args.frame_mode, preload_frames=True,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local")
    model = EventDINOBridge(
        backbone, args.steps, len(dataset.class_names),
        channels=int(dataset[0][0].shape[1]),
    ).to(device)
    model.stem.load_state_dict(payload["stem"], strict=True)
    model.classifier.load_state_dict(payload["classifier"], strict=True)
    model.eval()
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ids, labels, prefixes, logits = [], [], [], []
    for events, target, _, sample_ids in loader:
        events = events.to(device, non_blocking=True)
        batch_features = []
        for step in range(args.steps):
            # Causal prefix: later bins are hidden, while the original temporal
            # binning and polarity channels remain unchanged for the prefix.
            prefix_events = events.clone()
            if step + 1 < args.steps:
                prefix_events[:, step + 1:] = 0
            output = model(prefix_events)
            batch_features.append(output["feature"].float().cpu())
        prefixes.append(torch.stack(batch_features, dim=1))
        ids.extend(str(sample_id) for sample_id in sample_ids)
        labels.append(torch.as_tensor(target).long())
    result = {
        "ids": ids,
        "prefix": torch.cat(prefixes),
        "labels": torch.cat(labels),
        "args": vars(args),
        "source": "causal zero-future event prefixes through frozen EventDINOBridge",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(json.dumps({"output": str(output), "samples": len(ids), "shape": list(result["prefix"].shape)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
