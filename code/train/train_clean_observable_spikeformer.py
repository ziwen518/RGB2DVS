from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class IndexedTargets(Dataset):
    def __init__(
        self, dataset, indices: list[int], targets: dict[str, torch.Tensor] | None,
        augment=None, augment_strength: str = "weak", include_label: bool = False,
    ):
        self.dataset = dataset
        self.indices = indices
        self.targets = targets
        self.augment = augment
        self.augment_strength = augment_strength
        self.include_label = include_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        event, label, _, sample_id = self.dataset[self.indices[index]]
        if self.augment is not None:
            event = self.augment(event, self.augment_strength)
        target = torch.empty(0) if self.targets is None else self.targets[str(sample_id)]
        temporal = torch.empty(0)
        if isinstance(target, tuple):
            if len(target) == 3:
                target, confidence, temporal = target
            else:
                target, confidence = target
        else:
            confidence = torch.tensor(1.0)
        if self.include_label:
            return event, label, target, str(sample_id)
        return event, target, confidence, temporal, str(sample_id)


class OperationEnergyMeter:
    """Estimate sparse SynOps and a technology-normalized energy proxy."""

    def __init__(self, model: nn.Module) -> None:
        self.rows: dict[str, dict[str, float]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name: str, module: nn.Module):
        def hook(_module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            x = inputs[0].detach()
            batch = max(1, int(output.shape[0]))
            if isinstance(module, nn.Linear):
                kernel_ops = module.in_features
            else:
                kh, kw = module.kernel_size
                kernel_ops = kh * kw * (module.in_channels // module.groups)
            dense = float(output[0].numel() * kernel_ops)
            activity = float(x.float().ne(0).float().mean())
            row = self.rows.setdefault(name, {"calls": 0.0, "dense": 0.0, "synops": 0.0, "activity": 0.0})
            row["calls"] += 1.0
            # output[0] is one sample, so each hook call contributes a
            # per-sample operation count regardless of training batch size.
            row["dense"] += dense
            row["synops"] += dense * activity
            row["activity"] += activity
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.rows.clear()

    def report(self, batches: int = 1) -> dict[str, object]:
        # Hooks use output[0], so every row is already normalized to one
        # representative sample per batch.
        batches = max(1, int(batches))
        dense = sum(row["dense"] for row in self.rows.values()) / batches
        synops = sum(row["synops"] for row in self.rows.values()) / batches
        calls = sum(row["calls"] for row in self.rows.values()) / batches
        # 45nm proxy used only for relative comparisons, not wall-clock power.
        ac_pj, mac_pj = 0.9, 4.6
        return {
            "synops_per_sample": float(synops),
            "dense_ops_per_sample": float(dense),
            "activity_fraction": float(synops / max(dense, 1.0)),
            "estimated_energy_pj": float(synops * ac_pj),
            "dense_mac_energy_pj": float(dense * mac_pj),
            "estimated_energy_ratio": float((synops * ac_pj) / max(dense * mac_pj, 1.0)),
            "hook_calls_per_sample": int(calls),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="motion", choices=("motion", "ncaltech101", "nimagenet", "dsec", "cifar10dvs"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--spatial-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-name", default="none")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--extension-epochs", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--frame-mode", default="count_global")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--population-bins", type=int, default=1)
    parser.add_argument(
        "--block-spiking", choices=("full_respike", "compact_residual"),
        default="full_respike",
    )
    parser.add_argument("--hybrid-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hybrid-attention-suffix", type=int, default=0)
    parser.add_argument(
        "--qkv-temporal-mode",
        choices=(
            "standard", "role_decay", "causal_kv",
            "temporal_context", "event_gate", "membrane_aware",
            "multiscale", "frequency", "delta", "motion_guided",
            "state_fusion", "mesa",
        ),
        default="standard",
    )
    parser.add_argument("--continuous-relaxation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ternary-threshold", type=float, default=0.5)
    parser.add_argument("--dino-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dino-layer-map", choices=("first", "uniform", "last"), default="first",
    )
    parser.add_argument("--freeze-dino-epochs", type=int, default=0)
    parser.add_argument("--signed-readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event-prompt", action=argparse.BooleanOptionalAction, default=False,
                        help="enable Event-Conditioned Signed Prompt adapter")
    parser.add_argument("--prompt-strength", type=float, default=0.7)
    parser.add_argument(
        "--local-structure-mixer", action=argparse.BooleanOptionalAction,
        default=False,
        help="event-observable sparse multi-scale local relay in each spike block",
    )
    parser.add_argument(
        "--local-mixer-dilations", type=int, nargs="+", default=[1, 2],
        help="positive spatial dilation rates for the local relay",
    )
    parser.add_argument("--multidepth-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multidepth-readout-layers", type=int, default=3)
    parser.add_argument(
        "--hierarchical-patch-affinity", action=argparse.BooleanOptionalAction,
        default=False,
        help="apply observable patch-affinity to the last multidepth spike-token layers",
    )
    parser.add_argument("--membrane-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-cls-token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pyramid-stem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--protected-core", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--augment-strength",
        choices=("weak", "phase", "sensor", "sensor_calibrated", "dogs_sparse_calibrated", "log_dual_threshold"),
        default="weak",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes; use 2-4 for cached DSEC frames")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--pretrained-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--observable-weight", type=float, default=2.0)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument("--temporal-target-cache", type=Path)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--temporal-rate-weight", type=float, default=0.0)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument(
        "--conditional-geometry", action=argparse.BooleanOptionalAction, default=False,
        help="weight pairwise geometry by label-free observable confidence",
    )
    parser.add_argument("--neighborhood-weight", type=float, default=0.0)
    parser.add_argument("--neighborhood-temperature", type=float, default=0.1)
    parser.add_argument("--spatial-weight", type=float, default=0.0)
    parser.add_argument("--patch-affinity-weight", type=float, default=0.0)
    parser.add_argument(
        "--object-slot-weight", type=float, default=0.0,
        help="label-free event-observable semantic-slot transport loss",
    )
    parser.add_argument("--object-slots", type=int, default=8)
    parser.add_argument("--saliency-patch-weight", type=float, default=0.0,
                        help="label-free event-saliency weighted patch affinity")
    parser.add_argument(
        "--patch-affinity-mode", choices=("indexed", "spectral"),
        default="indexed",
        help="indexed is the legacy token-order loss; spectral is permutation-invariant",
    )
    parser.add_argument(
        "--spread-weight", type=float, default=0.0,
        help="label-free embedding variance/decorrelation regularizer",
    )
    parser.add_argument("--spread-target-std", type=float, default=0.03)
    parser.add_argument("--spatial-grid", type=int, default=6)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-fit", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def stratified_limit(indices: list[int], dataset, limit: int | None) -> list[int]:
    if limit is None or limit >= len(indices):
        return indices
    labels = sorted({dataset.samples[index][1] for index in indices})
    per_class, remainder = divmod(limit, len(labels))
    selected: list[int] = []
    for position, label in enumerate(labels):
        quota = per_class + int(position < remainder)
        group = [index for index in indices if dataset.samples[index][1] == label]
        selected.extend(group[:quota])
    return selected


def geometry_loss(
    student: torch.Tensor, teacher: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(student) < 2:
        return student.new_zeros(())
    student_relation = F.normalize(student.float(), dim=-1) @ F.normalize(student.float(), dim=-1).t()
    teacher_relation = F.normalize(teacher.float(), dim=-1) @ F.normalize(teacher.float(), dim=-1).t()
    mask = ~torch.eye(len(student), dtype=torch.bool, device=student.device)
    error = F.smooth_l1_loss(
        student_relation[mask], teacher_relation.detach()[mask], reduction="none",
    )
    if confidence is None:
        return error.mean()
    confidence = confidence.float().flatten().clamp(0.0, 1.0)
    pair_weight = (confidence[:, None] * confidence[None, :])[mask]
    return (error * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def neighborhood_loss(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Distill each sample's unlabeled soft-neighbor distribution."""
    if len(student) < 2:
        return student.new_zeros(())
    student_similarity = F.normalize(student.float(), dim=-1) @ F.normalize(student.float(), dim=-1).t()
    teacher_similarity = F.normalize(teacher.float(), dim=-1) @ F.normalize(teacher.float(), dim=-1).t()
    diagonal = torch.eye(len(student), dtype=torch.bool, device=student.device)
    student_logits = (student_similarity / temperature).masked_fill(diagonal, -1.0e4)
    teacher_logits = (teacher_similarity / temperature).masked_fill(diagonal, -1.0e4)
    with torch.no_grad():
        teacher_distribution = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1), teacher_distribution,
        reduction="batchmean",
    ) * (temperature ** 2)


def spatial_geometry_loss(
    student_steps: torch.Tensor, teacher_tokens: torch.Tensor, grid: int,
) -> torch.Tensor:
    """Align spatial relations while leaving ANN/SNN feature coordinates free."""
    student = student_steps.float().mean(dim=1)
    teacher = teacher_tokens.float()
    student_side = math.isqrt(student.shape[1])
    teacher_side = math.isqrt(teacher.shape[1])
    if student_side ** 2 != student.shape[1] or teacher_side ** 2 != teacher.shape[1]:
        raise ValueError("spatial token counts must form square grids")

    def relation(tokens: torch.Tensor, side: int) -> torch.Tensor:
        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], side, side
        )
        pooled = F.adaptive_avg_pool2d(feature_map, (grid, grid)).flatten(2).transpose(1, 2)
        pooled = pooled - pooled.mean(dim=1, keepdim=True)
        pooled = F.normalize(pooled, dim=-1)
        return pooled @ pooled.transpose(-2, -1)

    return F.smooth_l1_loss(
        relation(student, student_side), relation(teacher, teacher_side).detach()
    )


def patch_affinity_loss(
    student_steps: torch.Tensor, teacher_tokens: torch.Tensor,
    confidence: torch.Tensor | None = None, mode: str = "indexed",
) -> torch.Tensor:
    """Match patch affinity without requiring pixel-level correspondence.

    The legacy indexed mode is retained for checkpoint-compatible baselines.
    Spectral mode compares permutation-invariant affinity summaries, avoiding
    the hidden assumption that event and RGB patch indices are aligned.
    """
    student = student_steps.float().mean(dim=1)
    side = math.isqrt(student.shape[1])
    target_side = math.isqrt(teacher_tokens.shape[1])
    if side * side != student.shape[1] or target_side * target_side != teacher_tokens.shape[1]:
        raise ValueError("patch token counts must form square grids")
    if side != target_side:
        student = F.adaptive_avg_pool2d(
            student.transpose(1, 2).reshape(student.shape[0], student.shape[2], side, side),
            (target_side, target_side),
        ).flatten(2).transpose(1, 2)
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher_tokens.float(), dim=-1).detach()
    student_relation = student @ student.transpose(-2, -1)
    teacher_relation = teacher @ teacher.transpose(-2, -1)
    if mode == "spectral":
        # Sorting is permutation-invariant but substantially cheaper and more
        # stable to differentiate than a batched eigendecomposition.
        diagonal = torch.eye(student.shape[1], dtype=torch.bool, device=student.device)
        student_values = student_relation.masked_fill(diagonal, 0.0)
        teacher_values = teacher_relation.masked_fill(diagonal, 0.0).detach()
        student_spectrum = torch.sort(student_values.flatten(1), dim=-1).values
        teacher_spectrum = torch.sort(teacher_values.flatten(1), dim=-1).values
        student_degree = torch.sort(student_values.mean(dim=-1), dim=-1).values
        teacher_degree = torch.sort(teacher_values.mean(dim=-1), dim=-1).values
        loss = (
            F.smooth_l1_loss(student_spectrum, teacher_spectrum)
            + 0.5 * F.smooth_l1_loss(student_degree, teacher_degree)
        )
        if confidence is not None:
            loss = loss * confidence.float().mean().clamp_min(0.05)
        return loss
    if mode != "indexed":
        raise ValueError(f"unsupported patch affinity mode: {mode}")
    mask = ~torch.eye(student.shape[1], dtype=torch.bool, device=student.device)
    error = F.smooth_l1_loss(student_relation, teacher_relation, reduction="none")
    if confidence is not None:
        weight = confidence.float()[:, :, None] * confidence.float()[:, None, :]
        error = error * weight
    return error[..., mask].mean()


def saliency_patch_affinity_loss(
    student_steps: torch.Tensor, teacher_tokens: torch.Tensor,
    event_tokens: torch.Tensor,
) -> torch.Tensor:
    """Weight local affinity by label-free event activity.

    Event activity is detached before weighting, so this term only selects
    observable regions and cannot leak class labels into the Student loss.
    """
    student = student_steps.float().mean(dim=1)
    side = math.isqrt(student.shape[1])
    target_side = math.isqrt(teacher_tokens.shape[1])
    if side * side != student.shape[1] or target_side * target_side != teacher_tokens.shape[1]:
        raise ValueError("patch token counts must form square grids")
    if side != target_side:
        student = F.adaptive_avg_pool2d(
            student.transpose(1, 2).reshape(student.shape[0], student.shape[2], side, side),
            (target_side, target_side),
        ).flatten(2).transpose(1, 2)
    event_tokens = event_tokens.float()
    if event_tokens.shape[1] != target_side * target_side:
        event_tokens = F.adaptive_avg_pool2d(
            event_tokens.transpose(1, 2).reshape(event_tokens.shape[0], event_tokens.shape[2], side, side),
            (target_side, target_side),
        ).flatten(2).transpose(1, 2)
    saliency = event_tokens.detach().abs().mean(dim=-1)
    saliency = saliency / saliency.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher_tokens.float(), dim=-1).detach()
    student_relation = student @ student.transpose(-2, -1)
    teacher_relation = teacher @ teacher.transpose(-2, -1)
    mask = ~torch.eye(target_side * target_side, dtype=torch.bool, device=student.device)
    pair_weight = (saliency[:, :, None] * saliency[:, None, :]).clamp(0.25, 4.0)
    error = F.smooth_l1_loss(student_relation, teacher_relation, reduction="none")
    return (error * pair_weight * mask).sum() / (pair_weight * mask).sum().clamp_min(1.0)


def event_observable_slot_loss(
    student_steps: torch.Tensor,
    teacher_tokens: torch.Tensor,
    teacher_confidence: torch.Tensor | None,
    slots: int = 8,
) -> torch.Tensor:
    """Match unordered event-observable prototypes and their relations.

    Teacher patches are a fit-split-only observable relay, not raw RGB tokens.
    Both sides are sorted by label-free saliency and pooled into semantic slots,
    so no pixel correspondence or detection/class label enters the objective.
    """
    student = student_steps.float().mean(dim=1)
    teacher = teacher_tokens.float().detach()
    slot_count = max(2, min(int(slots), student.shape[1], teacher.shape[1]))

    student_saliency = student_steps.float().mean(dim=(1, 3))
    student_order = student_saliency.argsort(dim=-1, descending=True)
    student = torch.gather(
        student, 1, student_order.unsqueeze(-1).expand_as(student)
    )
    if teacher_confidence is None:
        teacher_saliency = teacher.norm(dim=-1)
    else:
        teacher_saliency = teacher_confidence.float().detach()
    teacher_order = teacher_saliency.argsort(dim=-1, descending=True)
    teacher = torch.gather(
        teacher, 1, teacher_order.unsqueeze(-1).expand_as(teacher)
    )
    ordered_confidence = torch.gather(teacher_saliency, 1, teacher_order)

    def pool_slots(tokens: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(
            tokens.transpose(1, 2), slot_count
        ).transpose(1, 2)
        return F.normalize(pooled, dim=-1)

    student_slots = pool_slots(student)
    teacher_slots = pool_slots(teacher)
    slot_confidence = F.adaptive_avg_pool1d(
        ordered_confidence.unsqueeze(1), slot_count
    ).squeeze(1).clamp(0.05, 1.0)
    alignment = 1.0 - F.cosine_similarity(student_slots, teacher_slots, dim=-1)
    alignment = (alignment * slot_confidence).sum() / slot_confidence.sum().clamp_min(1.0e-6)

    student_relation = student_slots @ student_slots.transpose(-2, -1)
    teacher_relation = (teacher_slots @ teacher_slots.transpose(-2, -1)).detach()
    diagonal = ~torch.eye(slot_count, dtype=torch.bool, device=student.device)
    pair_confidence = slot_confidence[:, :, None] * slot_confidence[:, None, :]
    relation_error = F.smooth_l1_loss(
        student_relation, teacher_relation, reduction="none"
    )
    relation = (
        relation_error * pair_confidence * diagonal
    ).sum() / (pair_confidence * diagonal).sum().clamp_min(1.0)
    return alignment + relation


def representation_spread_loss(
    embedding: torch.Tensor, target_std: float = 0.03,
) -> torch.Tensor:
    """Prevent unlabeled feature collapse without using class labels."""
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())
    centered = embedding.float() - embedding.float().mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.square().mean(dim=0) + 1.0e-4)
    variance = F.relu(float(target_std) - std).square().mean()
    covariance = centered.t() @ centered / float(embedding.shape[0])
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return variance + 0.05 * off_diagonal.square().mean()


def classify_training_regime(rows: list[dict[str, object]]) -> dict[str, object]:
    """Separate an under-trained curve from a plateau or objective mismatch.

    A single validation point is never enough evidence. The decision uses the
    latest three five-epoch evaluations, representation health, loss progress,
    and whether the learning-rate schedule still has room to move.
    """
    evaluated = [row for row in rows if row.get("signed_knn") is not None]
    if len(evaluated) < 3:
        return {
            "state": "insufficient_evidence",
            "evaluations": len(evaluated),
            "required_evaluations": 3,
            "recommendation": "continue_to_next_validation",
        }
    window = evaluated[-3:]
    metric = [float(row["signed_knn"]) for row in window]
    losses = [float(row["loss"]) for row in window]
    ranks = [float(row.get("effective_rank", 0.0)) for row in window]
    stds = [float(row.get("embedding_std", 0.0)) for row in window]
    learning_rates = [
        max(float(value) for value in row.get("learning_rates", [0.0]))
        for row in window
    ]
    metric_delta = metric[-1] - metric[0]
    recent_delta = metric[-1] - metric[-2]
    relative_loss_reduction = (
        (losses[0] - losses[-1]) / max(abs(losses[0]), 1.0e-8)
    )
    representation_healthy = ranks[-1] >= 0.85 * max(ranks[0], 1.0) and stds[-1] >= 0.01
    schedule_exhausted = learning_rates[-1] <= 0.05 * max(learning_rates[0], 1.0e-12)

    if not representation_healthy:
        state = "representation_collapse"
        recommendation = "stop_and_fix_representation"
    elif metric_delta >= 0.005 and recent_delta >= -0.002:
        state = "undertrained_improving"
        recommendation = "continue_training"
    elif metric_delta <= -0.002 and relative_loss_reduction >= 0.01:
        state = "objective_mismatch_or_overfit"
        recommendation = "stop_extension_and_change_objective_or_architecture"
    elif abs(metric_delta) < 0.005 and relative_loss_reduction < 0.01:
        state = "converged_plateau"
        recommendation = "stop_extension_and_change_capacity_or_objective"
    elif schedule_exhausted and metric_delta < 0.005:
        state = "schedule_exhausted_plateau"
        recommendation = "do_not_add_epochs_without_new_schedule_or_architecture"
    else:
        state = "ambiguous_slow_progress"
        recommendation = "continue_one_validation_interval_only"
    return {
        "state": state,
        "recommendation": recommendation,
        "evaluation_epochs": [int(row["epoch"]) for row in window],
        "metric": "signed_knn",
        "metric_delta": metric_delta,
        "recent_metric_delta": recent_delta,
        "relative_loss_reduction": relative_loss_reduction,
        "effective_rank_delta": ranks[-1] - ranks[0],
        "embedding_std_delta": stds[-1] - stds[0],
        "representation_healthy": representation_healthy,
        "schedule_exhausted": schedule_exhausted,
    }


def loss_weights(args: argparse.Namespace, epoch: int) -> tuple[float, float]:
    if not args.curriculum:
        return args.observable_weight, args.geometry_weight
    progress = epoch / args.epochs
    if progress <= 0.2:
        return 5.0, 2.0
    if progress <= 0.6:
        return 3.0, 1.0
    return 1.0, 0.5


def set_dino_block_weights_trainable(model: nn.Module, trainable: bool) -> int:
    suffixes = (
        "attn.qkv.weight", "attn.proj.weight",
        "mlp.fc1.weight", "mlp.fc2.weight",
    )
    changed = 0
    for name, parameter in model.named_parameters():
        if name.startswith("blocks.") and name.endswith(suffixes):
            parameter.requires_grad_(trainable)
            changed += 1
    return changed


@torch.no_grad()
def initialize_blocks_from_dino(
    student: nn.Module, teacher: nn.Module, layer_map: str = "first",
) -> tuple[int, list[int]]:
    """Copy shape-compatible ViT weights; spiking dynamics remain Student-owned."""
    source = teacher.state_dict()
    destination = student.state_dict()
    teacher_depth = len(teacher.blocks)
    student_depth = len(student.blocks)
    if student_depth > teacher_depth:
        raise ValueError("Student depth cannot exceed DINO depth for initialization")
    if layer_map == "first":
        source_blocks = list(range(student_depth))
    elif layer_map == "last":
        source_blocks = list(range(teacher_depth - student_depth, teacher_depth))
    elif layer_map == "uniform":
        source_blocks = torch.linspace(
            0, teacher_depth - 1, student_depth
        ).round().to(torch.int64).tolist()
    else:
        raise ValueError(f"unsupported DINO layer map: {layer_map}")
    copied = 0
    for student_index, teacher_index in enumerate(source_blocks):
        for suffix in (
            "attn.qkv.weight", "attn.proj.weight",
            "mlp.fc1.weight", "mlp.fc2.weight",
        ):
            source_key = f"blocks.{teacher_index}.{suffix}"
            destination_key = f"blocks.{student_index}.{suffix}"
            if (
                source_key in source and destination_key in destination
                and source[source_key].shape == destination[destination_key].shape
            ):
                destination[destination_key].copy_(source[source_key])
                copied += 1
    student.load_state_dict(destination)
    return copied, source_blocks


@torch.inference_mode()
def extract(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    signed, core, labels = [], [], []
    spike_rate = 0.0
    batches = 0
    for events, target, _, _ in loader:
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(events.to(device, non_blocking=True))
        signed.append(output["embedding"].float().cpu())
        core.append(output["core_embedding"].float().cpu())
        labels.append(torch.as_tensor(target))
        spike_rate += float(output["spike_rate"])
        batches += 1
    return torch.cat(signed), torch.cat(core), torch.cat(labels), spike_rate / max(batches, 1)


def knn(bank: torch.Tensor, bank_y: torch.Tensor, query: torch.Tensor, query_y: torch.Tensor, k: int = 10) -> float:
    neighbors = (F.normalize(query, dim=-1) @ F.normalize(bank, dim=-1).t()).topk(min(k, len(bank)), dim=-1).indices
    prediction = torch.mode(bank_y[neighbors], dim=-1).values
    return float(prediction.eq(query_y).float().mean())


@torch.inference_mode()
def evaluate(model, fit_loader, validation_loader, device, knn_k: int = 10):
    fit, fit_core, fit_y, spike_rate = extract(model, fit_loader, device)
    validation, validation_core, validation_y, _ = extract(model, validation_loader, device)
    centered = fit - fit.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probability = singular / singular.sum().clamp_min(1.0e-8)
    temporal_knn = knn(fit, fit_y, validation, validation_y, k=knn_k)
    uniform_knn = knn(fit_core, fit_y, validation_core, validation_y, k=knn_k)
    return {
        "temporal_knn": temporal_knn,
        "uniform_knn": uniform_knn,
        # Compatibility aliases for older analysis scripts.
        "signed_knn": temporal_knn,
        "core_knn": uniform_knn,
        "knn_k": int(knn_k),
        "embedding_std": float(fit.std(dim=0).mean()),
        "effective_rank": float(torch.exp(-(probability * probability.clamp_min(1.0e-8).log()).sum())),
        "spike_rate": spike_rate,
    }


def main() -> int:
    args = parse_args()
    if args.target_cache is None:
        raise ValueError("--target-cache is required for label-free distillation")
    if args.epochs <= 0 or args.extension_epochs < 0:
        raise ValueError("--epochs must be positive and --extension-epochs non-negative")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if args.hierarchical_patch_affinity and not args.multidepth_readout:
        raise ValueError("hierarchical patch affinity requires --multidepth-readout")
    if args.hierarchical_patch_affinity and args.patch_affinity_weight <= 0:
        raise ValueError("hierarchical patch affinity requires positive --patch-affinity-weight")
    if args.extension_epochs > 0 and args.early_stop_patience == 0:
        raise ValueError("extended training requires --early-stop-patience")
    if args.neighborhood_weight < 0 or args.neighborhood_temperature <= 0:
        raise ValueError("invalid neighborhood loss weight/temperature")
    if not 0.0 <= args.confidence_floor <= 1.0:
        raise ValueError("--confidence-floor must be in [0, 1]")
    if args.protected_core and not args.signed_readout:
        raise ValueError("--protected-core requires signed readout")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Resolve the isolated rgb2dvs layout before any legacy project modules.
    trainer_dir = Path(__file__).resolve().parent
    core_dir = trainer_dir.parent / "core"
    if core_dir.is_dir():
        sys.path.insert(0, str(core_dir))
    sys.path.insert(1, str(trainer_dir))
    sys.path.insert(2, str(args.project_root))
    from data import augment_events, make_event_dataset
    from model import PureSpikeFormer
    print(json.dumps({"model_file": str(Path(sys.modules["model"].__file__).resolve())}), flush=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cpu")
    dataset_kwargs = dict(steps=args.steps, size=args.size, two_views=False)
    if args.dataset_name in {"motion", "ncaltech101", "nimagenet", "dsec"}:
        dataset_kwargs.update(frame_mode=args.frame_mode, preload_frames=True)
        # Preserve the historical N-Caltech cache when callers omit an
        # explicit event representation; alternate modes get isolated caches.
        if args.dataset_name == "ncaltech101" and args.frame_mode == "count_global":
            dataset_kwargs["frame_mode"] = "count"
    if args.dataset_name in {"ncaltech101", "nimagenet"}:
        dataset_kwargs["use_all"] = True
    dataset = make_event_dataset(args.dataset_name, args.data_root, True, **dataset_kwargs)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    fit_ids = set(map(str, manifest["train_ids"]))
    validation_ids = set(map(str, manifest["validation_ids"]))
    if args.dataset_name == "dsec":
        id_to_index = {
            f"train/{sample[0]}/{sample[1]:06d}": index
            for index, sample in enumerate(dataset.samples)
        }
    else:
        id_to_index = {
            str(sample[0].relative_to(dataset.root)).replace("\\", "/"): index
            for index, sample in enumerate(dataset.samples)
        }
    missing = (fit_ids | validation_ids) - set(id_to_index)
    if missing:
        raise KeyError(f"dataset is missing {len(missing)} split IDs")
    # The Student fit subset is selected without consulting class labels.
    fit_indices = [id_to_index[key] for key in manifest["train_ids"]]
    if args.max_fit is not None:
        fit_indices = fit_indices[:args.max_fit]
    validation_indices = stratified_limit([id_to_index[key] for key in manifest["validation_ids"]], dataset, args.max_validation)

    targets = None
    if args.target_cache is not None:
        payload = torch.load(args.target_cache, map_location="cpu", weights_only=False)
        values = payload.get("transported", payload.get("observable"))
        if values is None:
            raise KeyError("target cache has neither transported nor observable features")
        confidence = payload.get("observable_confidence")
        if confidence is not None and len(confidence) != len(values):
            raise ValueError("observable confidence length differs from target cache")
        targets = {
            str(key): (
                F.normalize(value.float(), dim=-1),
                torch.as_tensor(confidence[index]).float().clamp(0.0, 1.0),
            ) if confidence is not None else F.normalize(value.float(), dim=-1)
            for index, (key, value) in enumerate(zip(payload["ids"], values))
        }
        if args.dataset_name == "dsec":
            used_ids = {f"train/{dataset.samples[index][0]}/{dataset.samples[index][1]:06d}" for index in fit_indices}
        else:
            used_ids = {str(dataset.samples[index][0].relative_to(dataset.root)).replace("\\", "/") for index in fit_indices}
        if not used_ids.issubset(targets):
            raise KeyError("observable cache does not cover the fit split")

    temporal_targets = None
    if args.temporal_target_cache is not None:
        temporal_payload = torch.load(
            args.temporal_target_cache, map_location="cpu", weights_only=False,
        )
        temporal_values = temporal_payload["temporal_targets"]
        temporal_targets = {
            str(key): F.normalize(value.float(), dim=-1)
            for key, value in zip(temporal_payload["ids"], temporal_values)
        }
        if args.dataset_name == "dsec":
            used_ids = {f"train/{dataset.samples[index][0]}/{dataset.samples[index][1]:06d}" for index in fit_indices}
        else:
            used_ids = {str(dataset.samples[index][0].relative_to(dataset.root)).replace("\\", "/") for index in fit_indices}
        if not used_ids.issubset(temporal_targets):
            raise KeyError("temporal target cache does not cover the fit split")
        if temporal_values.ndim != 3 or temporal_values.shape[1] != args.steps:
            raise ValueError("temporal target cache must have shape [N, steps, dim]")

    if temporal_targets is not None:
        for sample_id in list(targets or {}):
            base = targets[sample_id]
            confidence = base[1] if isinstance(base, tuple) else torch.tensor(1.0)
            targets[sample_id] = (
                base[0] if isinstance(base, tuple) else base,
                confidence,
                temporal_targets[sample_id],
            )

    spatial_targets = None
    spatial_confidence = None
    if args.spatial_cache is not None:
        spatial_payload = torch.load(
            args.spatial_cache, map_location="cpu", weights_only=False
        )
        spatial_targets = {
            str(key): value.float()
            for key, value in zip(spatial_payload["ids"], spatial_payload["tokens"])
        }
        if spatial_payload.get("confidence") is not None:
            spatial_confidence = {
                str(key): value.float()
                for key, value in zip(spatial_payload["ids"], spatial_payload["confidence"])
            }
        if args.dataset_name == "dsec":
            used_ids = {f"train/{dataset.samples[index][0]}/{dataset.samples[index][1]:06d}" for index in fit_indices}
        else:
            used_ids = {str(dataset.samples[index][0].relative_to(dataset.root)).replace("\\", "/") for index in fit_indices}
        if not used_ids.issubset(spatial_targets):
            raise KeyError("spatial cache does not cover the fit split")

    fit = IndexedTargets(
        dataset, fit_indices, targets, augment_events if args.augment else None,
        augment_strength=args.augment_strength,
    )
    # Labels enter only these post-training representation-evaluation loaders.
    fit_evaluation = IndexedTargets(dataset, fit_indices, targets, include_label=True)
    validation = IndexedTargets(dataset, validation_indices, targets, include_label=True)
    loader_workers = max(0, int(args.workers))
    train_loader = DataLoader(fit, args.batch_size, shuffle=True, drop_last=False, num_workers=loader_workers, pin_memory=True, persistent_workers=loader_workers > 0)
    fit_loader = DataLoader(fit_evaluation, args.batch_size * 2, shuffle=False, num_workers=loader_workers, pin_memory=True, persistent_workers=loader_workers > 0)
    validation_loader = DataLoader(validation, args.batch_size * 2, shuffle=False, num_workers=loader_workers, pin_memory=True, persistent_workers=loader_workers > 0)
    input_channels = int(dataset[0][0].shape[1])
    model = PureSpikeFormer(
        in_channels=input_channels, dim=args.dim, depth=args.depth, heads=args.heads,
        patch_size=args.patch_size, threshold=args.threshold,
        use_cls_token=args.use_cls_token, norm="bntt",
        image_size=args.size, temporal_readout="learned",
        local_stem=not args.pyramid_stem, pyramid_stem=args.pyramid_stem,
        use_positional_bias=True, attention_mode="normalized",
        signed_readout=args.signed_readout,
        multidepth_readout=args.multidepth_readout,
        multidepth_readout_layers=args.multidepth_readout_layers,
        membrane_readout=args.membrane_readout,
        block_spiking=args.block_spiking,
        hybrid_attention=args.hybrid_attention,
        hybrid_attention_suffix=args.hybrid_attention_suffix,
        qkv_temporal_mode=args.qkv_temporal_mode,
        continuous=args.continuous_relaxation,
        ternary_threshold=args.ternary_threshold,
        temporal_steps=args.steps,
        population_bins=args.population_bins,
        event_prompt=args.event_prompt,
        prompt_strength=args.prompt_strength,
        local_structure_mixer=args.local_structure_mixer,
        local_mixer_dilations=tuple(args.local_mixer_dilations),
    ).to(device)
    copied_dino_tensors = 0
    dino_source_blocks: list[int] = []
    if args.dino_init:
        repository = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        dino = torch.hub.load(str(repository), "dinov2_vits14", source="local").eval().to(device)
        copied_dino_tensors, dino_source_blocks = initialize_blocks_from_dino(
            model, dino, args.dino_layer_map
        )
        del dino
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frozen_dino_tensors = 0
    if args.freeze_dino_epochs > 0:
        frozen_dino_tensors = set_dino_block_weights_trainable(model, False)
    pretrained_suffixes = (
        "attn.qkv.weight", "attn.proj.weight",
        "mlp.fc1.weight", "mlp.fc2.weight",
    )
    pretrained_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("blocks.") and name.endswith(pretrained_suffixes)
    ]
    pretrained_ids = {id(parameter) for parameter in pretrained_parameters}
    adaptation_parameters = [
        parameter for parameter in model.parameters()
        if id(parameter) not in pretrained_ids
    ]
    pretrained_lr = args.lr if args.pretrained_lr is None else args.pretrained_lr
    optimizer = torch.optim.AdamW([
        {"params": pretrained_parameters, "lr": pretrained_lr},
        {"params": adaptation_parameters, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    # Decay over the guaranteed training phase, then fine-tune at 1% LR while
    # validation is still improving. This avoids cosine LR rising after T_max.
    def lr_multiplier(step: int) -> float:
        progress = min(step, args.epochs) / args.epochs
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    energy_meter = OperationEnergyMeter(model)
    # BF16 retains FP32-like exponent range; final embeddings and losses stay FP32.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    best_epoch = 0
    evaluations_without_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        history_path = args.resume.parent / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            evaluated = [row for row in history if "signed_knn" in row]
            if evaluated:
                best = max(float(row["signed_knn"]) for row in evaluated)
                best_epoch = max(int(row["epoch"]) for row in evaluated if float(row["signed_knn"]) == best)
            start_epoch = int(history[-1]["epoch"]) + 1 if history else 1
    max_epochs = args.epochs + args.extension_epochs
    print(json.dumps({
        "epoch": 0, "copied_dino_tensors": copied_dino_tensors,
        "frozen_dino_tensors": frozen_dino_tensors,
        "evaluation_deferred": True, "input_channels": input_channels,
        "pretrained_lr": pretrained_lr, "adaptation_lr": args.lr,
        "pretrained_parameter_tensors": len(pretrained_parameters),
        "dino_source_blocks": dino_source_blocks,
        "minimum_epochs": args.epochs, "maximum_epochs": max_epochs,
        "early_stop_patience_evaluations": args.early_stop_patience,
        "label_free_training": True,
        "resume": str(args.resume) if args.resume else None,
    }), flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        if args.freeze_dino_epochs > 0 and epoch == args.freeze_dino_epochs + 1:
            set_dino_block_weights_trainable(model, True)
        model.train(); started = time.perf_counter()
        observable_weight, geometry_weight = loss_weights(args, epoch)
        sums = {
            "loss": 0.0, "alignment": 0.0,
            "geometry": 0.0, "neighborhood": 0.0, "spatial": 0.0,
            "patch_affinity": 0.0,
            "object_slot": 0.0,
            "spread": 0.0,
            "temporal_alignment": 0.0,
            "temporal_rate_alignment": 0.0,
            "prompt_rate": 0.0, "route_rate": 0.0,
        }
        batches = 0
        energy_meter.reset()
        for batch_index, (events, teacher, confidence, temporal_teacher, sample_ids) in enumerate(train_loader):
            events = events.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            confidence = confidence.to(device, non_blocking=True).float()
            confidence = args.confidence_floor + (1.0 - args.confidence_floor) * confidence
            temporal_teacher = temporal_teacher.to(device, non_blocking=True).float()
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                output = model(
                    events,
                    return_tokens=(spatial_targets is not None)
                    or args.saliency_patch_weight > 0
                    or (batch_index == 0 and epoch == 1),
                    return_token_layers=(
                        args.multidepth_readout_layers
                        if args.hierarchical_patch_affinity else 0
                    ),
                )
            embedding = output["embedding"].float()
            core_embedding = output["core_embedding"].float()
            alignment = embedding.new_zeros(())
            geometry = embedding.new_zeros(())
            neighborhood = embedding.new_zeros(())
            spatial = embedding.new_zeros(())
            patch_affinity = embedding.new_zeros(())
            object_slot = embedding.new_zeros(())
            spread = representation_spread_loss(
                embedding, args.spread_target_std,
            ) if args.spread_weight > 0 else embedding.new_zeros(())
            alignment_per_sample = 1.0 - F.cosine_similarity(
                embedding, teacher.float(), dim=-1,
            )
            alignment = (alignment_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-6)
            geometry_embedding = core_embedding if args.protected_core else embedding
            geometry = geometry_loss(
                geometry_embedding, teacher.float(),
                confidence if args.conditional_geometry else None,
            )
            temporal_alignment = embedding.new_zeros(())
            temporal_rate_alignment = embedding.new_zeros(())
            if args.temporal_weight > 0 and temporal_teacher.numel() > 0:
                # Match continuous teacher semantics to signed spike rates,
                # not to normalized vectors. The small target scale reflects
                # sparse event communication and avoids zero-vector cosine
                # gradients in early prefixes.
                rate_target = 0.1 * torch.tanh(4.0 * temporal_teacher)
                raw = output["raw_trajectory"].float()
                temporal_alignment = F.smooth_l1_loss(
                    raw[:, -min(4, raw.shape[1]):],
                    rate_target[:, -min(4, raw.shape[1]):],
                )
            if args.temporal_rate_weight > 0 and temporal_teacher.numel() > 0:
                teacher_direction = temporal_teacher.float().clamp(-1.0, 1.0)
                positive_target = 0.05 + 0.15 * torch.sigmoid(4.0 * teacher_direction)
                negative_target = 0.05 + 0.15 * torch.sigmoid(-4.0 * teacher_direction)
                rate_target = torch.cat([positive_target, negative_target], dim=-1)
                pred_rate = output["semantic_rate_trajectory"].float()
                temporal_rate_alignment = F.smooth_l1_loss(
                    pred_rate[:, -min(4, pred_rate.shape[1]):],
                    rate_target[:, -min(4, rate_target.shape[1]):],
                )
            if args.neighborhood_weight > 0:
                neighborhood = neighborhood_loss(
                    embedding, teacher.float(), args.neighborhood_temperature,
                )
            if spatial_targets is not None and (args.spatial_weight > 0 or args.patch_affinity_weight > 0):
                teacher_spatial = torch.stack([
                    spatial_targets[str(sample_id)] for sample_id in sample_ids
                ]).to(device, non_blocking=True)
                if args.spatial_weight > 0:
                    spatial = spatial_geometry_loss(
                        output["tokens"], teacher_spatial, args.spatial_grid
                    )
                if args.patch_affinity_weight > 0:
                    teacher_confidence = None
                    if spatial_confidence is not None:
                        teacher_confidence = torch.stack([
                            spatial_confidence[str(sample_id)] for sample_id in sample_ids
                        ]).to(device, non_blocking=True)
                    if args.hierarchical_patch_affinity:
                        token_pyramid = output["token_pyramid"]
                        layer_weights = torch.linspace(
                            1.0, 2.0, token_pyramid.shape[1],
                            device=token_pyramid.device,
                        )
                        layer_weights = layer_weights / layer_weights.sum()
                        patch_affinity = sum(
                            weight * patch_affinity_loss(
                                layer_tokens, teacher_spatial,
                                teacher_confidence, mode=args.patch_affinity_mode,
                            )
                            for weight, layer_tokens in zip(
                                layer_weights, token_pyramid.unbind(dim=1)
                            )
                        )
                    else:
                        patch_affinity = patch_affinity_loss(
                            output["tokens"], teacher_spatial, teacher_confidence,
                            mode=args.patch_affinity_mode,
                        )
            if args.saliency_patch_weight > 0 and spatial_targets is not None:
                teacher_spatial = torch.stack([
                    spatial_targets[str(sample_id)] for sample_id in sample_ids
                ]).to(device, non_blocking=True)
                patch_affinity = patch_affinity + args.saliency_patch_weight * saliency_patch_affinity_loss(
                    output["tokens"], teacher_spatial, output["tokens"].mean(dim=1)
                )
            if args.object_slot_weight > 0:
                if spatial_targets is None:
                    raise ValueError("--object-slot-weight requires --spatial-cache")
                teacher_spatial = torch.stack([
                    spatial_targets[str(sample_id)] for sample_id in sample_ids
                ]).to(device, non_blocking=True)
                teacher_confidence = None
                if spatial_confidence is not None:
                    teacher_confidence = torch.stack([
                        spatial_confidence[str(sample_id)] for sample_id in sample_ids
                    ]).to(device, non_blocking=True)
                object_slot = event_observable_slot_loss(
                    output["tokens"], teacher_spatial,
                    teacher_confidence, slots=args.object_slots,
                )
            loss = (
                observable_weight * alignment + geometry_weight * geometry
                + args.temporal_weight * temporal_alignment
                + args.temporal_rate_weight * temporal_rate_alignment
                + args.neighborhood_weight * neighborhood
                + args.spatial_weight * spatial
                + args.patch_affinity_weight * patch_affinity
                + args.object_slot_weight * object_slot
                + args.spread_weight * spread
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if epoch == 1 and batch_index == 0:
                tokens = output["tokens"]
                if not args.continuous_relaxation and not torch.all((tokens == 0) | (tokens == 1)):
                    raise RuntimeError("inter-block activations are not binary")
                if not np.isfinite(grad_norm) or grad_norm <= 0.0:
                    raise RuntimeError(f"invalid Student gradient norm: {grad_norm}")
                print(json.dumps({"contract_check": "passed", "student_grad_norm": grad_norm, "tokens_binary": not args.continuous_relaxation}), flush=True)
            scaler.step(optimizer); scaler.update()
            batches += 1
            for key, value in (
                ("loss", loss), ("alignment", alignment),
                ("geometry", geometry), ("neighborhood", neighborhood),
                ("spatial", spatial), ("temporal_alignment", temporal_alignment),
                ("temporal_rate_alignment", temporal_rate_alignment),
                ("patch_affinity", patch_affinity),
                ("object_slot", object_slot),
                ("spread", spread),
                ("prompt_rate", output.get("prompt_rate", embedding.new_zeros(()))),
                ("route_rate", output.get("route_rate", embedding.new_zeros(()))),
            ):
                sums[key] += float(value.detach())
        scheduler.step()
        train_seconds = time.perf_counter() - started
        energy_metrics = energy_meter.report(batches)
        should_evaluate = epoch % args.eval_every == 0 or epoch == max_epochs
        evaluation_started = time.perf_counter()
        metrics = (
            evaluate(model, fit_loader, validation_loader, device, 10)
            if should_evaluate else {}
        )
        eval_seconds = time.perf_counter() - evaluation_started if should_evaluate else 0.0
        row = {
            "epoch": epoch,
            "train_classifier_acc": None,
            "ce_weight": 0.0, "observable_weight": observable_weight,
            "geometry_weight": geometry_weight,
            "neighborhood_weight": args.neighborhood_weight,
            "neighborhood_temperature": args.neighborhood_temperature,
            "confidence_floor": args.confidence_floor,
            "conditional_geometry": args.conditional_geometry,
            "temporal_weight": args.temporal_weight,
            "temporal_rate_weight": args.temporal_rate_weight,
            "qkv_temporal_mode": args.qkv_temporal_mode,
            "multidepth_readout": args.multidepth_readout,
            "multidepth_readout_layers": args.multidepth_readout_layers,
            "hierarchical_patch_affinity": args.hierarchical_patch_affinity,
            "dino_weights_frozen": epoch <= args.freeze_dino_epochs,
            **{key: value / max(batches, 1) for key, value in sums.items()},
            **metrics, "train_seconds": train_seconds,
            "eval_seconds": eval_seconds, "seconds": time.perf_counter() - started,
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
            "supervised_loss_enabled": False, "labels_used": False,
            "labels_used_for_evaluation_only": True,
            "teacher_used_at_inference": False,
            "continuous_postprocessor": False, "mode": "observable",
            "target_name": args.target_name,
            "spatial_weight": args.spatial_weight,
            "patch_affinity_weight": args.patch_affinity_weight,
            "patch_affinity_mode": args.patch_affinity_mode,
            "object_slot_weight": args.object_slot_weight,
            "object_slots": args.object_slots,
            "saliency_patch_weight": args.saliency_patch_weight,
            "event_prompt": args.event_prompt,
            "prompt_strength": args.prompt_strength,
            "local_structure_mixer": args.local_structure_mixer,
            "local_mixer_dilations": args.local_mixer_dilations,
            "spread_weight": args.spread_weight,
            "spread_target_std": args.spread_target_std,
            "protected_core": args.protected_core,
            "population_bins": args.population_bins,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "energy": energy_metrics,
        }
        if should_evaluate:
            row["training_regime"] = classify_training_regime([*history, row])
        history.append(row); print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "args": vars(args), "metrics": row,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if should_evaluate and metrics["signed_knn"] > best:
            best = metrics["signed_knn"]
            best_epoch = epoch
            evaluations_without_improvement = 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        elif should_evaluate:
            evaluations_without_improvement += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        if (
            epoch >= args.epochs and args.early_stop_patience > 0
            and evaluations_without_improvement >= args.early_stop_patience
        ):
            print(json.dumps({
                "early_stop": True, "epoch": epoch, "best_epoch": best_epoch,
                "evaluations_without_improvement": evaluations_without_improvement,
            }), flush=True)
            break
    print(json.dumps({
        "best_validation_signed_knn": best, "best_epoch": best_epoch,
        "completed_epoch": history[-1]["epoch"], "test_used": False,
    }), flush=True)
    energy_meter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
