"""Formal frozen-PureSpike FPN + Faster R-CNN evaluation on DSEC."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

from dsec_formal_common import (
    DET_CLASSES, DSECDetectionDataset, OperationEnergyMeter,
    assert_disjoint, frozen_token_grid, load_encoder,
)


class FrozenSpikeFPN(nn.Module):
    """Convert frozen 12x12 spike tokens into a three-level feature pyramid."""

    def __init__(self, encoder: nn.Module, temporal_steps: int, channels: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal_steps = int(temporal_steps)
        dim = int(encoder.dim)
        self.c3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(24, dim), nn.GELU(),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(24, dim), nn.GELU(),
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(24, dim), nn.GELU(),
        )
        self.fpn = FeaturePyramidNetwork([dim, dim, dim], channels)
        self.out_channels = channels
        self.last_spike_rate = 0.0
        self.last_tokens_binary = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        batch, channels, height, width = images.shape
        expected = self.temporal_steps * 2
        if channels != expected:
            raise ValueError(f"expected {expected} flattened event channels, got {channels}")
        events = images.reshape(batch, self.temporal_steps, 2, height, width)
        tokens, spike_rate, binary = frozen_token_grid(self.encoder, events)
        self.last_spike_rate = spike_rate
        self.last_tokens_binary = binary
        c3 = self.c3(tokens)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return self.fpn(OrderedDict((str(i), x) for i, x in enumerate((c3, c4, c5))))


def build_detector(encoder: nn.Module, steps: int) -> FasterRCNN:
    backbone = FrozenSpikeFPN(encoder, steps)
    anchors = AnchorGenerator(
        sizes=((4, 8, 12), (8, 16, 24), (16, 24, 36)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )
    roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2"], output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone, num_classes=DET_CLASSES + 1,
        rpn_anchor_generator=anchors, box_roi_pool=roi_pool,
        min_size=48, max_size=48,
        image_mean=[0.0] * (steps * 2), image_std=[1.0] * (steps * 2),
        rpn_pre_nms_top_n_train=1000, rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_train=500, rpn_post_nms_top_n_test=200,
        box_detections_per_img=100,
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, rb - lt), axis=-1)
    area_a = np.prod(np.maximum(0.0, a[:, 2:] - a[:, :2]), axis=-1)
    area_b = np.prod(np.maximum(0.0, b[:, 2:] - b[:, :2]), axis=-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


def average_precision(rows: list[tuple[float, bool]], gt_count: int) -> float:
    if not rows or gt_count <= 0:
        return 0.0
    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    true_positive = np.cumsum([int(hit) for _, hit in ordered], dtype=np.float64)
    false_positive = np.cumsum([int(not hit) for _, hit in ordered], dtype=np.float64)
    recall = true_positive / float(gt_count)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changed] - recall[changed - 1]) * precision[changed]))


@torch.inference_mode()
def validate(model, loader, device, encoder_meter, head_meter) -> dict[str, object]:
    model.eval()
    encoder_meter.reset()
    head_meter.reset()
    detections: list[list[tuple[float, bool]]] = [[] for _ in range(DET_CLASSES)]
    gt_counts = [0] * DET_CLASSES
    spike_rates: list[float] = []
    tokens_binary = True
    for images, targets, _ids in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(images)
        spike_rates.append(float(model.backbone.last_spike_rate))
        tokens_binary &= bool(model.backbone.last_tokens_binary)
        for prediction, target in zip(outputs, targets):
            predicted_boxes = prediction["boxes"].cpu().numpy()
            predicted_labels = prediction["labels"].cpu().numpy()
            predicted_scores = prediction["scores"].cpu().numpy()
            truth_boxes = target["boxes"].numpy()
            truth_labels = target["labels"].numpy()
            for label in range(1, DET_CLASSES + 1):
                truth = truth_boxes[truth_labels == label]
                gt_counts[label - 1] += len(truth)
                used = np.zeros(len(truth), dtype=bool)
                order = np.flatnonzero(predicted_labels == label)
                order = order[np.argsort(predicted_scores[order])[::-1]]
                for index in order:
                    overlaps = box_iou(predicted_boxes[index:index + 1], truth)[0]
                    hit = bool(len(overlaps) and overlaps.max() >= 0.5)
                    if hit:
                        match = int(overlaps.argmax())
                        if used[match]:
                            hit = False
                        else:
                            used[match] = True
                    detections[label - 1].append((float(predicted_scores[index]), hit))
    per_class = [
        average_precision(detections[index], gt_counts[index])
        for index in range(DET_CLASSES) if gt_counts[index] > 0
    ]
    return {
        "validation_ap50_proxy": float(np.mean(per_class)) if per_class else 0.0,
        "validation_ap50_per_present_class": per_class,
        "validation_gt_boxes": int(sum(gt_counts)),
        "validation_present_classes": len(per_class),
        "encoder_spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
        "tokens_binary": tokens_binary,
        "encoder_energy": encoder_meter.report(len(loader)),
        "detection_head_energy": head_meter.report(len(loader)),
    }


def training_regime(history: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [row for row in history if "validation_ap50_proxy" in row]
    if len(evaluated) < 3:
        return {"state": "insufficient_evidence", "evaluations": len(evaluated)}
    window = evaluated[-3:]
    metric = [float(row["validation_ap50_proxy"]) for row in window]
    loss = [float(row["train_loss"]) for row in window]
    delta = metric[-1] - metric[0]
    recent = metric[-1] - metric[-2]
    loss_reduction = (loss[0] - loss[-1]) / max(abs(loss[0]), 1e-8)
    if delta >= 0.005 and recent >= -0.002:
        state = "undertrained_improving"
    elif recent <= -0.005:
        state = "downstream_overfit"
    elif abs(delta) < 0.003 and loss_reduction < 0.01:
        state = "converged_plateau"
    elif loss_reduction >= 0.02 and delta < 0.003:
        state = "representation_or_objective_limited"
    else:
        state = "ambiguous_slow_progress"
    return {
        "state": state, "evaluation_epochs": [row["epoch"] for row in window],
        "metric_delta": delta, "recent_metric_delta": recent,
        "relative_loss_reduction": loss_reduction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke and (args.epochs != 36 or args.eval_every != 5):
        raise ValueError("formal DSEC detection requires 36 epochs and eval-every=5")
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    train = DSECDetectionDataset(
        args.data_root, args.annotations, args.split_manifest, "train"
    )
    validation = DSECDetectionDataset(
        args.data_root, args.annotations, args.split_manifest, "validation"
    )
    assert_disjoint(train, validation)
    if args.smoke:
        train.samples, train.event_indices = train.samples[:4], train.event_indices[:4]
        validation.samples, validation.event_indices = (
            validation.samples[:4], validation.event_indices[:4]
        )
    collate = lambda rows: tuple(zip(*rows))
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
    )
    encoder = load_encoder(args.checkpoint, device)
    steps = int(encoder.formal_checkpoint_args["steps"])
    model = build_detector(encoder, steps).to(device)
    encoder_meter = OperationEnergyMeter(encoder)
    head_meter = OperationEnergyMeter(
        nn.ModuleList([model.backbone.c3, model.backbone.c4, model.backbone.c5,
                       model.backbone.fpn, model.rpn, model.roi_heads])
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_metric, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        components: dict[str, float] = {}
        for images, targets, _ids in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            for key, value in losses.items():
                components[key] = components.get(key, 0.0) + float(value.detach())
        row: dict[str, object] = {
            "epoch": epoch, "train_loss": total / max(len(train_loader), 1),
            "loss_components": {
                key: value / max(len(train_loader), 1)
                for key, value in components.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples": len(train), "validation_samples": len(validation),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row.update(validate(model, validation_loader, device, encoder_meter, head_meter))
            if not row["tokens_binary"]:
                raise RuntimeError("formal DSEC detection received non-binary Student tokens")
            row["training_regime"] = training_regime([*history, row])
            metric = float(row["validation_ap50_proxy"])
            if metric > best_metric:
                best_metric, best_epoch = metric, epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "metrics": row},
                    args.output_dir / "best_faster_rcnn.pt",
                )
        history.append(row)
        (args.output_dir / "detection_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        scheduler.step()
    result = {
        "protocol": "frozen_purespike_fpn_faster_rcnn_36e",
        "formal": not args.smoke, "proxy_reason": "reduced DSEC event subset",
        "student_labels_used": False,
        "labels_used_for_detection_head_only": True,
        "epochs": args.epochs, "validation_every_epochs": args.eval_every,
        "train_samples": len(train), "validation_samples": len(validation),
        "best_validation_ap50_proxy": best_metric, "best_epoch": best_epoch,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "detection_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint": str(args.checkpoint), "history": history,
    }
    (args.output_dir / "detection_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    encoder_meter.close()
    head_meter.close()
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

