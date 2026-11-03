from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


@torch.inference_mode()
def extract(model, loader, device):
    model.eval(); features, labels = [], []
    for events, target, *_ in loader:
        output = model(events.to(device, non_blocking=True))
        features.append(output["embedding"].float().cpu())
        labels.append(torch.as_tensor(target))
    return F.normalize(torch.cat(features), dim=-1), torch.cat(labels)


def knn(bank, bank_y, query, query_y, k=10):
    neighbors = (query @ bank.t()).topk(min(k, len(bank)), dim=-1).indices
    prediction = torch.mode(bank_y[neighbors], dim=-1).values
    return float(prediction.eq(query_y).float().mean())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--synthetic-root", type=Path, required=True)
    p.add_argument("--real-events-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--real-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    core = args.project_root / "code" / "core"
    sys.path.insert(0, str(core))
    import model as model_module
    from data import make_event_dataset
    from model import PureSpikeFormer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(args.gpu)
    model = PureSpikeFormer(
        in_channels=2, dim=int(cfg["dim"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), patch_size=int(cfg["patch_size"]),
        threshold=float(cfg["threshold"]),
        use_cls_token=bool(cfg.get("use_cls_token", False)), norm="bntt",
        image_size=int(cfg["size"]), temporal_readout="learned",
        local_stem=not bool(cfg.get("pyramid_stem", False)),
        pyramid_stem=bool(cfg.get("pyramid_stem", False)),
        use_positional_bias=True, attention_mode="normalized",
        signed_readout=bool(cfg.get("signed_readout", True)),
        multidepth_readout=bool(cfg.get("multidepth_readout", False)),
        multidepth_readout_layers=int(cfg.get("multidepth_readout_layers", 3)),
        block_spiking=cfg.get("block_spiking", "full_respike"),
        hybrid_attention=bool(cfg.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(cfg.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=cfg.get("qkv_temporal_mode", "standard"),
        continuous=bool(cfg.get("continuous_relaxation", False)),
        ternary_threshold=float(cfg.get("ternary_threshold", 0.5)),
        temporal_steps=int(cfg["steps"]),
        population_bins=int(cfg.get("population_bins", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    common_real = dict(
        steps=args.real_steps, size=int(cfg["size"]), two_views=False, seed=42,
    )
    real_train = make_event_dataset(
        "cifar10dvs", args.real_events_root, True, **common_real,
    )
    real_test = make_event_dataset(
        "cifar10dvs", args.real_events_root, False, **common_real,
    )
    synthetic_train = make_event_dataset(
        "motion", args.synthetic_root, True, steps=int(cfg["steps"]),
        size=int(cfg["size"]), two_views=False,
        frame_mode=cfg["frame_mode"], preload_frames=True,
    )
    loader = lambda ds: DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    synthetic_x, synthetic_y = extract(model, loader(synthetic_train), device)
    real_train_x, real_train_y = extract(model, loader(real_train), device)
    real_test_x, real_test_y = extract(model, loader(real_test), device)

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
    )
    probe.fit(real_train_x.numpy(), real_train_y.numpy())
    result = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["metrics"]["epoch"],
            "student_training_domain": "CIFAR10-RCLS-medium synthetic events",
            "student_frozen": True,
            "real_dataset": "CIFAR10-DVS",
            "real_steps": args.real_steps,
            "synthetic_bank": len(synthetic_train),
            "real_train_bank": len(real_train),
            "real_test": len(real_test),
        },
        "strict_cross_domain_knn": {
            str(k): knn(synthetic_x, synthetic_y, real_test_x, real_test_y, k)
            for k in (1, 5, 10, 20)
        },
        "frozen_real_dvs_knn": {
            str(k): knn(real_train_x, real_train_y, real_test_x, real_test_y, k)
            for k in (1, 5, 10, 20)
        },
        "frozen_real_dvs_linear_probe": float(
            probe.score(real_test_x.numpy(), real_test_y.numpy())
        ),
        "model_file": str(Path(model_module.__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
