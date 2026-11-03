from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from isgd.data import make_event_dataset
except ModuleNotFoundError:
    # The isolated workspace keeps the audited loader in code/core.
    CORE = ROOT.parent / "core"
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    from data import make_event_dataset


class EventDINOBridge(nn.Module):
    """Learn an event-native input adapter for a frozen RGB DINO backbone."""

    def __init__(self, backbone: nn.Module, steps: int, classes: int, channels: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.stem = nn.Sequential(
            nn.Conv2d(channels * steps, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),
        )
        self.classifier = nn.Linear(384, classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, events: torch.Tensor) -> dict[str, torch.Tensor]:
        image = torch.sigmoid(self.stem(events.flatten(1, 2)))
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        feature = self.backbone.forward_features((image - self.mean) / self.std)["x_norm_clstoken"].float()
        feature = F.normalize(feature, dim=-1)
        return {"feature": feature, "logits": self.classifier(feature)}


def relation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape[0] < 2:
        return student.new_zeros(())
    student_relation = F.normalize(student.float(), dim=-1) @ F.normalize(student.float(), dim=-1).t()
    teacher_relation = F.normalize(teacher.detach().float(), dim=-1) @ F.normalize(teacher.detach().float(), dim=-1).t()
    mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
    return F.smooth_l1_loss(student_relation[mask], teacher_relation[mask])


@torch.inference_mode()
def extract(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    features, labels = [], []
    for events, target, *_ in loader:
        output = model(events.to(device, non_blocking=True))
        features.append(output["feature"].cpu())
        labels.append(torch.as_tensor(target))
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def evaluate(model: nn.Module, bank_loader: DataLoader, query_loader: DataLoader, device: torch.device):
    bank, bank_y = extract(model, bank_loader, device)
    query, query_y = extract(model, query_loader, device)
    neighbors = (query @ bank.t()).topk(min(10, len(bank)), dim=-1).indices
    prediction = torch.mode(bank_y[neighbors], dim=-1).values
    knn = float((prediction == query_y).float().mean())
    return knn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=9000)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-mode", choices=("count", "count_global", "count_recency", "occupancy"), default="count_global")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = make_event_dataset(
        "motion", args.data_root, True, args.steps, args.size, two_views=False,
        max_samples=args.max_train_samples, frame_mode=args.frame_mode, preload_frames=True,
    )
    test_set = make_event_dataset(
        "motion", args.data_root, False, args.steps, args.size, two_views=False,
        max_samples=args.max_eval_samples, frame_mode=args.frame_mode, preload_frames=True,
    )
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    bank_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    payload = torch.load(args.teacher_cache, map_location="cpu")
    teacher_by_id = {
        str(sample_id): F.normalize(feature[-1].float(), dim=-1)
        for sample_id, feature in zip(payload["ids"], payload["prefix"])
    }
    missing = [
        str(path.relative_to(train_set.root)).replace("\\", "/")
        for path, _, _ in train_set.samples
        if str(path.relative_to(train_set.root)).replace("\\", "/") not in teacher_by_id
    ]
    if missing:
        raise KeyError(f"RGB teacher cache is missing {len(missing)} event samples")

    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local")
    channels = 4 if args.frame_mode == "count_recency" else 2
    core = EventDINOBridge(backbone, args.steps, len(train_set.class_names), channels=channels).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu")
        core.stem.load_state_dict(initial["stem"], strict=True)
        core.classifier.load_state_dict(initial["classifier"], strict=True)
    model: nn.Module = nn.DataParallel(core) if args.data_parallel and torch.cuda.device_count() > 1 else core
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter(); model.train()
        totals, classifications, features, relations = [], [], [], []
        for events, labels, _, sample_ids in train_loader:
            events = events.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            teacher = torch.stack([teacher_by_id[str(sample_id)] for sample_id in sample_ids]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                output = model(events)
                classification = F.cross_entropy(output["logits"], labels)
                feature = (1.0 - F.cosine_similarity(output["feature"], teacher, dim=-1)).mean()
                relation = relation_loss(output["feature"], teacher)
                loss = classification + args.feature_weight * feature + args.relation_weight * relation
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            totals.append(loss.detach()); classifications.append(classification.detach()); features.append(feature.detach()); relations.append(relation.detach())
        scheduler.step(); knn = evaluate(model, bank_loader, test_loader, device)
        row = {
            "epoch": epoch,
            "loss": float(torch.stack(totals).mean().cpu()),
            "classification_loss": float(torch.stack(classifications).mean().cpu()),
            "feature_loss": float(torch.stack(features).mean().cpu()),
            "relation_loss": float(torch.stack(relations).mean().cpu()),
            "knn_acc": knn,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row); print(json.dumps(row), flush=True)
        module = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint = {
            "stem": module.stem.state_dict(),
            "classifier": module.classifier.state_dict(),
            "metrics": row,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if knn > best:
            best = knn; torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
