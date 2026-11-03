"""Formal frozen-PureSpike DeepLabV3+ semantic segmentation on DSEC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dsec_formal_common import (
    SEG_CLASSES, SEG_IGNORE, DSECSegmentationDataset, OperationEnergyMeter,
    assert_disjoint, frozen_token_grid, load_encoder,
)


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, in_channels, 3, padding=dilation, dilation=dilation,
                groups=in_channels, bias=False,
            ),
            nn.GroupNorm(16, in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels), nn.GELU(),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels), nn.GELU(),
        )
        self.dilated = nn.ModuleList(
            SeparableConv(in_channels, out_channels, dilation)
            for dilation in (2, 4, 6)
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels), nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.context(x)
        context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
        branches = [self.point(x), *(branch(x) for branch in self.dilated), context]
        return self.project(torch.cat(branches, dim=1))


class DeepLabV3Plus(nn.Module):
    """ASPP high-level context plus a low-level spike-token decoder."""

    def __init__(self, channels: int, classes: int = SEG_CLASSES) -> None:
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(channels, 48, 1, bias=False),
            nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(24, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(24, channels), nn.GELU(),
        )
        self.aspp = ASPP(channels, 256)
        self.decoder = nn.Sequential(
            SeparableConv(304, 256), SeparableConv(256, 256),
            nn.Conv2d(256, classes, 1),
        )

    def forward(self, tokens: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        low = self.low(tokens)
        high = self.aspp(self.high(tokens))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([low, high], dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def ohem_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, fraction: float,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, ignore_index=SEG_IGNORE, reduction="none")
    valid = losses[targets.ne(SEG_IGNORE)]
    if not len(valid):
        return losses.sum() * 0.0
    count = max(1, int(np.ceil(fraction * len(valid))))
    return valid.topk(count, sorted=False).values.mean()


@torch.inference_mode()
def validate(encoder, decoder, loader, device, encoder_meter, head_meter) -> dict[str, object]:
    encoder.eval()
    decoder.eval()
    encoder_meter.reset()
    head_meter.reset()
    confusion = np.zeros((SEG_CLASSES, SEG_CLASSES), dtype=np.int64)
    spike_rates: list[float] = []
    tokens_binary = True
    for events, masks, _ids in loader:
        events = events.to(device, non_blocking=True)
        tokens, spike_rate, binary = frozen_token_grid(encoder, events)
        prediction = decoder(tokens, tuple(masks.shape[-2:])).argmax(1).cpu().numpy()
        truth = masks.numpy()
        valid = truth != SEG_IGNORE
        pair = SEG_CLASSES * truth[valid].astype(np.int64) + prediction[valid]
        confusion += np.bincount(
            pair, minlength=SEG_CLASSES * SEG_CLASSES
        ).reshape(SEG_CLASSES, SEG_CLASSES)
        spike_rates.append(spike_rate)
        tokens_binary &= binary
    union = confusion.sum(1) + confusion.sum(0) - np.diag(confusion)
    iou = np.diag(confusion) / np.maximum(union, 1)
    present = confusion.sum(1) > 0
    return {
        "validation_miou_19class": float(iou.mean()),
        "validation_miou_present": float(iou[present].mean()) if present.any() else 0.0,
        "validation_present_classes": int(present.sum()),
        "validation_iou_per_class": iou.tolist(),
        "encoder_spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
        "tokens_binary": tokens_binary,
        "encoder_energy": encoder_meter.report(len(loader)),
        "segmentation_head_energy": head_meter.report(len(loader)),
    }


def training_regime(history: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [row for row in history if "validation_miou_19class" in row]
    if len(evaluated) < 3:
        return {"state": "insufficient_evidence", "evaluations": len(evaluated)}
    window = evaluated[-3:]
    metric = [float(row["validation_miou_19class"]) for row in window]
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ohem-fraction", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke and (args.epochs != 100 or args.eval_every != 5):
        raise ValueError("formal DSEC segmentation requires 100 epochs and eval-every=5")
    if not 0.0 < args.ohem_fraction <= 1.0:
        raise ValueError("ohem-fraction must be in (0, 1]")
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    train = DSECSegmentationDataset(args.data_root, args.annotations, "train")
    validation = DSECSegmentationDataset(args.data_root, args.annotations, "validation")
    assert_disjoint(train, validation)
    if args.smoke:
        train.samples, train.event_indices = train.samples[:4], train.event_indices[:4]
        validation.samples, validation.event_indices = (
            validation.samples[:4], validation.event_indices[:4]
        )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    encoder = load_encoder(args.checkpoint, device)
    decoder = DeepLabV3Plus(int(encoder.dim)).to(device)
    encoder_meter = OperationEnergyMeter(encoder)
    head_meter = OperationEnergyMeter(decoder)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_metric, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        encoder.eval()
        decoder.train()
        total = 0.0
        for events, masks, _ids in train_loader:
            events = events.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)
            tokens, _spike_rate, binary = frozen_token_grid(encoder, events)
            if not binary:
                raise RuntimeError("formal DSEC segmentation received non-binary Student tokens")
            logits = decoder(tokens, tuple(masks.shape[-2:]))
            loss = ohem_cross_entropy(logits, masks, args.ohem_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        row: dict[str, object] = {
            "epoch": epoch, "train_loss": total / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples": len(train), "validation_samples": len(validation),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row.update(validate(
                encoder, decoder, validation_loader, device, encoder_meter, head_meter
            ))
            if not row["tokens_binary"]:
                raise RuntimeError("formal DSEC segmentation received non-binary Student tokens")
            row["training_regime"] = training_regime([*history, row])
            metric = float(row["validation_miou_19class"])
            if metric > best_metric:
                best_metric, best_epoch = metric, epoch
                torch.save(
                    {"decoder": decoder.state_dict(), "epoch": epoch, "metrics": row},
                    args.output_dir / "best_deeplabv3plus.pt",
                )
        history.append(row)
        (args.output_dir / "segmentation_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        scheduler.step()
    result = {
        "protocol": "frozen_purespike_deeplabv3plus_100e",
        "formal": not args.smoke,
        "proxy_reason": "semantic-mask intersection of reduced DSEC event subset",
        "student_labels_used": False,
        "labels_used_for_segmentation_head_only": True,
        "epochs": args.epochs, "validation_every_epochs": args.eval_every,
        "train_samples": len(train), "validation_samples": len(validation),
        "best_validation_miou_19class": best_metric, "best_epoch": best_epoch,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "segmentation_parameters": sum(p.numel() for p in decoder.parameters()),
        "checkpoint": str(args.checkpoint), "history": history,
    }
    (args.output_dir / "segmentation_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    encoder_meter.close()
    head_meter.close()
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

