"""End-to-end Next-Gen PureSpikeFormer trainer with dense patch affinity, masked event completion, and EMA.

Unifies Stage 2 (`patch_size=4`, 12x12 tokens) and Stage 3 (`patch_size=3`, 16x16 tokens 1-to-1 with
EventProxy-DINOv2 + energy-compensated spatial block masking) under strict U1 label-free rules:
1. DINOv2 Semantic Late/Uniform Block Initialization (`--dino-blocks 3 6 9 11` by default).
2. Next-Gen Pure-Spike Architecture (`local_structure_mixer=True`, `multidepth_readout=True`, live BNTT).
3. Canonical-pose (`p_flip=0.0`) variance-preserving integer shifts + optional energy-preserving patch
   masking (`--drop-ratio`), paired into `[x_clean, x_aug]` for balanced live BNTT statistics.
4. Coordinate-free spatially centered saliency patch affinity (`centered_saliency_patch_affinity`) +
   dual-temperature full-bank `global_neighbor_kl` + batch-centered mid-layer relational KL.
5. Student EMA + exact clean-train population BNTT calibration (`recalibrate_bn_on_train`, `labels_used=False`).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
for sub in ("core", "train", "eval"):
    p = str(ROOT_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from model import PureSpikeFormer  # noqa: E402
from multidepth_spectral_readout import (  # noqa: E402
    block_whiten_pca384,
    cosine_linear_probe,
    evaluate_all_modes,
    knn_top1,
    labels_for_split,
    linear_probe,
    load_frames_u8,
    load_student_split,
    ridge_eval,
    sha256_file,
    whiten_block,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def initialize_from_dino_blocks(
    student: PureSpikeFormer,
    dino_ckpt_path: Path,
    source_blocks: tuple[int, ...] = (3, 6, 9, 11),
) -> int:
    """Copy shape-compatible attention/MLP weights from selected DINOv2 blocks."""
    state = torch.load(dino_ckpt_path, map_location="cpu", weights_only=False)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    dest = student.state_dict()
    suffixes = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
    copied = 0
    for s_idx, t_idx in enumerate(source_blocks):
        for suf in suffixes:
            sk = f"blocks.{t_idx}.{suf}"
            dk = f"blocks.{s_idx}.{suf}"
            if sk in state and dk in dest and state[sk].shape == dest[dk].shape:
                dest[dk].copy_(state[sk])
                copied += 1
    student.load_state_dict(dest, strict=True)
    return copied


@torch.no_grad()
def recalibrate_bn_on_train(
    model: nn.Module,
    x_tr_u8: torch.Tensor,
    device: torch.device,
    bs: int = 128,
) -> None:
    """Compute exact population BNTT statistics strictly on clean `train_ids` (`no_grad`)."""
    saved_momenta: list[tuple[torch.nn.modules.batchnorm._BatchNorm, float | None]] = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            saved_momenta.append((m, m.momentum))
            m.reset_running_stats()
            m.momentum = None
    model.train()
    use_amp = device.type == "cuda"
    for i in range(0, len(x_tr_u8), bs):
        x = x_tr_u8[i : i + bs].to(device, non_blocking=True).float() / 255.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(x)
    for m, mom in saved_momenta:
        m.momentum = mom
    model.eval()


def geometry_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if len(student) < 2:
        return student.new_zeros(())
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    mask = ~torch.eye(len(student), dtype=torch.bool, device=student.device)
    return F.smooth_l1_loss((s @ s.t())[mask], (t @ t.t())[mask])


def batch_neighborhood_kl(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.10
) -> torch.Tensor:
    if len(student) < 4:
        return student.new_zeros(())
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    diag = torch.eye(len(student), dtype=torch.bool, device=student.device)
    s_logits = (s @ s.t() / temperature).masked_fill(diag, -1.0e4)
    with torch.no_grad():
        t_probs = F.softmax((t @ t.t() / temperature).masked_fill(diag, -1.0e4), dim=-1)
    return F.kl_div(F.log_softmax(s_logits, dim=-1), t_probs, reduction="batchmean")


def center_norm_batch(x: torch.Tensor) -> torch.Tensor:
    xn = F.normalize(x.float(), dim=-1)
    return F.normalize(xn - xn.mean(dim=0, keepdim=True), dim=-1)


def global_neighbor_kl(
    student: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_bank: torch.Tensor,
    indices: torch.Tensor,
    temperature_s: float = 0.10,
    temperature_t: float = 0.08,
) -> torch.Tensor:
    s = F.normalize(student.float(), dim=-1)
    keys = F.normalize(teacher_bank.detach().float(), dim=-1)
    idx = indices.to(s.device).long().view(-1, 1)
    logits = (s @ keys.t() / temperature_s).scatter(1, idx, -1.0e4)
    with torch.no_grad():
        t = F.normalize(teacher_query.detach().float(), dim=-1)
        t_logits = (t @ keys.t() / temperature_t).scatter(1, idx, -1.0e4)
        target = F.softmax(t_logits, dim=-1)
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def centered_saliency_patch_affinity(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    sal_weights: torch.Tensor,
) -> torch.Tensor:
    """Coordinate-free spatially centered patch affinity weighted by detached event saliency."""
    s = student_tokens.float()
    t = teacher_tokens.detach().float()
    s_side = math.isqrt(s.shape[1])
    t_side = math.isqrt(t.shape[1])
    if s_side * s_side != s.shape[1] or t_side * t_side != t.shape[1]:
        raise ValueError("Patch token counts must form square grids.")
    sal = sal_weights.detach().float().reshape(s.shape[0], s.shape[1])
    if s_side != t_side:
        s = (
            F.adaptive_avg_pool2d(
                s.transpose(1, 2).reshape(s.shape[0], s.shape[2], s_side, s_side), (t_side, t_side)
            )
            .flatten(2)
            .transpose(1, 2)
        )
        sal = F.adaptive_avg_pool2d(sal.reshape(s.shape[0], 1, s_side, s_side), (t_side, t_side)).reshape(
            s.shape[0], t_side * t_side
        )
    s = F.normalize(s - s.mean(dim=1, keepdim=True), dim=-1, eps=1e-4)
    t = F.normalize(t - t.mean(dim=1, keepdim=True), dim=-1, eps=1e-4)
    s_rel = s @ s.transpose(-2, -1)
    t_rel = t @ t.transpose(-2, -1)
    n_tok = s.shape[1]
    mask = ~torch.eye(n_tok, dtype=torch.bool, device=s.device)
    sal = sal / sal.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    pair_w = (sal[:, :, None] * sal[:, None, :]).clamp(0.25, 4.0)
    err = F.smooth_l1_loss(s_rel, t_rel, reduction="none")
    return (err * pair_w * mask).sum() / (pair_w * mask).sum().clamp_min(1.0)


def gpu_vp_shift_and_energy_mask(
    x: torch.Tensor,
    max_shift: int = 3,
    p_shift: float = 0.90,
    p_mask: float = 0.75,
    drop_ratio: float = 0.18,
    patch_size: int = 3,
) -> torch.Tensor:
    """Canonical-pose integer translation (`p_flip=0.0`) + energy-compensated patch masking."""
    out = x.clone()
    b, _, _, h, w = out.shape
    if max_shift > 0:
        for i in range(b):
            if random.random() < p_shift:
                dy = random.randint(-max_shift, max_shift)
                dx = random.randint(-max_shift, max_shift)
                if dy != 0 or dx != 0:
                    out[i] = torch.roll(out[i], shifts=(dy, dx), dims=(-2, -1))
    if p_mask > 0.0 and drop_ratio > 0.0:
        gh, gw = h // patch_size, w // patch_size
        apply_m = torch.rand(b, 1, 1, 1, 1, device=out.device) < p_mask
        keep_grid = (torch.rand(b, 1, 1, gh, gw, device=out.device) >= drop_ratio).float()
        mask_hw = keep_grid.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
        orig_e = out.sum(dim=(1, 2, 3, 4), keepdim=True)
        masked = out * mask_hw
        masked_e = masked.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
        scale = (orig_e / masked_e).clamp(1.0, 1.35)
        out = torch.where(apply_m, masked * scale, out)
    return out


def forward_pure_rich(
    model: PureSpikeFormer,
    events: torch.Tensor,
    sal_floor: float = 0.10,
    sal_blend: float = 0.35,
    need_membranes: bool = False,
) -> dict[str, torch.Tensor]:
    """Exact PureSpikeFormer forward pass exposing spatial `sem_patches`, `emb_sal`, and block spikes."""
    b, t_steps = events.shape[0], events.shape[1]
    p_size = int(model.patch.proj.stride[0])
    ev_map = F.avg_pool2d(
        events.sum(dim=(1, 2)).unsqueeze(1), kernel_size=p_size, stride=p_size
    ).reshape(b, 1, -1, 1)
    sal_w = (ev_map + sal_floor) / (ev_map + sal_floor).sum(dim=2, keepdim=True)

    patch_m, local_m, sem_m = None, None, None
    states: list[dict[str, object] | None] = [None] * len(model.blocks)
    b_steps: list[list[torch.Tensor]] = [[] for _ in range(len(model.blocks))]
    mem_steps: list[list[torch.Tensor]] = [[] for _ in range(len(model.blocks))]
    sem_patch_steps: list[torch.Tensor] = []
    trajectory: list[torch.Tensor] = []

    for t in range(t_steps):
        x, patch_m, local_m, _ = model.patch(events[:, t], patch_m, local_m, t)
        readout_sources: list[torch.Tensor] = []
        for i, block in enumerate(model.blocks):
            x, states[i], _ = block(x, states[i], t)
            b_steps[i].append(x)
            if model.multidepth_readout and i >= len(model.blocks) - model.multidepth_readout_layers:
                readout_sources.append(x)
            if need_membranes:
                mem_steps[i].append(states[i]["mr"].mean(dim=1))

        if model.multidepth_readout:
            weights = F.softmax(model.multidepth_logits, dim=0).to(x)
            sem_cur = sum(w * proj(src) for w, proj, src in zip(weights, model.semantic_projs, readout_sources))
        else:
            sem_cur = model.semantic_proj(x)
        sem_cur = model.semantic_norm(sem_cur, t)
        sem_spk, sem_m = model.semantic_lif(sem_cur, sem_m)
        pos, neg = sem_spk.chunk(2, dim=-1)
        sem_step = pos - neg
        sem_patch_steps.append(sem_step)
        trajectory.append(sem_step.mean(dim=1))

    step_embeddings = torch.stack(trajectory, dim=1).float()
    sem_patch_all = torch.stack(sem_patch_steps, dim=1).float()

    if model.temporal_readout == "learned" and model.temporal_logits is not None:
        w_t = F.softmax(model.temporal_logits[:t_steps], dim=0).float()
        emb_unif = F.normalize((step_embeddings * w_t.view(1, -1, 1)).sum(dim=1), dim=-1)
        sem_patches = (sem_patch_all * w_t.view(1, -1, 1, 1)).sum(dim=1)
    else:
        emb_unif = F.normalize(step_embeddings.mean(dim=1), dim=-1)
        sem_patches = sem_patch_all.mean(dim=1)

    emb_sal = F.normalize((sem_patches * sal_w.squeeze(1).float()).sum(dim=1), dim=-1)
    emb_blend = F.normalize(emb_unif + sal_blend * emb_sal, dim=-1)

    tp_mean = torch.stack([torch.stack(s, dim=1).float().mean(dim=1) for s in b_steps], dim=1)
    layers_unif = F.normalize(tp_mean.mean(dim=2), dim=-1)
    core_unif = layers_unif[:, -1]
    tp_sal = F.normalize((tp_mean * sal_w.float()).sum(dim=2), dim=-1)

    out = {
        "embedding": emb_unif,
        "emb_sal": emb_sal,
        "emb_blend": emb_blend,
        "sem_patches": sem_patches,
        "tp_mean": tp_mean,
        "layers_unif": layers_unif,
        "core_unif": core_unif,
        "tp_sal": tp_sal,
        "sal_w": sal_w.float(),
    }
    if need_membranes:
        out["memb_unif"] = torch.stack(
            [F.normalize(torch.stack(s, dim=1).float().mean(dim=1), dim=-1) for s in mem_steps], dim=1
        )
    return out


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Next-Gen PureSpikeFormer with dense spatial patch distillation & masked event completion."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dense-cache", type=Path, required=True, help="Path to train-only dense teacher cache (.pt)")
    parser.add_argument("--dino-ckpt", type=Path, required=True, help="Path to pretrained DINOv2 ViT-S/14 weights (.pth)")
    parser.add_argument("--split-manifest", type=Path, default=Path("data/teacher_cache/ncaltech101/student_split.json"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/ncaltech101_rgb_event/frames_ncaltech101_t16_s48"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=3, choices=(3, 4))
    parser.add_argument("--dino-blocks", type=int, nargs=4, default=[3, 6, 9, 11])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--unique-batch-size", type=int, default=18)
    parser.add_argument("--lr", type=float, default=9.0e-4)
    parser.add_argument("--pretrained-lr", type=float, default=1.3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--drop-ratio", type=float, default=0.18, help="Set 0.0 for Stage 2, 0.18 for Stage 3 masked completion")
    parser.add_argument("--mem-fraction", type=float, default=0.80)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else args.project_root / p

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        if 0.0 < args.mem_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    seed_all(args.seed)

    tr_ids, va_ids, _ = load_student_split(resolve(args.split_manifest))
    y_tr, y_va, _ = labels_for_split(tr_ids, va_ids)
    x_tr_u8 = load_frames_u8(resolve(args.frames_root), tr_ids)
    x_va_u8 = load_frames_u8(resolve(args.frames_root), va_ids)

    dcache = torch.load(resolve(args.dense_cache), map_location="cpu", weights_only=False)
    assert dcache["training_labels_used"] is False
    assert list(dcache["train_ids"]) == tr_ids
    t_train = F.normalize(dcache["transported"].float(), dim=-1).to(device)
    cls_11_train = F.normalize(dcache["cls_layers"][:, 2].float(), dim=-1).to(device)
    if args.patch_size == 3:
        p12_target = dcache["patch_12"].half()
    else:
        p12_raw = dcache["patch_12"].float().transpose(1, 2).reshape(len(tr_ids), 384, 16, 16)
        p12_target = F.normalize(
            F.adaptive_avg_pool2d(p12_raw, (12, 12)).flatten(2).transpose(1, 2), dim=-1
        ).half()
    p10_12x12 = dcache["patch_10_12x12"].half()
    del dcache

    model_kwargs = {
        "in_channels": 2,
        "dim": 384,
        "depth": 4,
        "heads": 6,
        "patch_size": args.patch_size,
        "threshold": 0.5,
        "norm": "bntt",
        "image_size": 48,
        "temporal_readout": "learned",
        "local_stem": True,
        "use_positional_bias": True,
        "attention_mode": "normalized",
        "signed_readout": True,
        "multidepth_readout": True,
        "multidepth_readout_layers": 3,
        "local_structure_mixer": True,
        "local_mixer_dilations": [1, 2],
        "temporal_steps": 16,
    }
    model = PureSpikeFormer(**model_kwargs).to(device)
    dino_blocks = tuple(int(b) for b in args.dino_blocks)
    initialize_from_dino_blocks(model, resolve(args.dino_ckpt), source_blocks=dino_blocks)

    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    dino_suffix = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
    pre_params = [p for n, p in model.named_parameters() if n.startswith("blocks.") and n.endswith(dino_suffix)]
    pre_ids = {id(p) for p in pre_params}
    adapt_params = [p for p in model.parameters() if id(p) not in pre_ids]

    optimizer = torch.optim.AdamW(
        [{"params": pre_params, "lr": args.pretrained_lr}, {"params": adapt_params, "lr": args.lr}],
        weight_decay=0.01,
    )

    def lr_lambda(ep_idx: int) -> float:
        if ep_idx < args.warmup_epochs:
            return 0.15 + 0.85 * (ep_idx / float(max(1, args.warmup_epochs)))
        prog = (ep_idx - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    n_train = len(x_tr_u8)

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed * 100 + ep))
        loss_sum, steps = 0.0, 0
        for bi in range(0, n_train, args.unique_batch_size):
            idx = perm[bi : bi + args.unique_batch_size]
            bs_cur = len(idx)
            if bs_cur < 6:
                continue
            idx_dev = idx.to(device)
            x_v1 = x_tr_u8[idx].to(device, non_blocking=True).float() / 255.0
            x_v2 = gpu_vp_shift_and_energy_mask(
                x_v1,
                max_shift=args.max_shift,
                p_shift=0.90,
                p_mask=0.75 if args.drop_ratio > 0 else 0.0,
                drop_ratio=args.drop_ratio,
                patch_size=args.patch_size,
            )
            x_cat = torch.cat([x_v1, x_v2], dim=0)
            tgt = t_train[idx_dev]
            tgt_c11 = cls_11_train[idx_dev]
            p12_b = p12_target[idx].to(device, non_blocking=True).float()
            p10_b = p10_12x12[idx].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = forward_pure_rich(model, x_cat, need_membranes=False)

            eu1, eu2 = out["embedding"][:bs_cur], out["embedding"][bs_cur:]
            eb1, eb2 = out["emb_blend"][:bs_cur], out["emb_blend"][bs_cur:]
            core1, core2 = out["core_unif"][:bs_cur], out["core_unif"][bs_cur:]
            sem_p1, tp1, sal_w1 = out["sem_patches"][:bs_cur], out["tp_mean"][:bs_cur], out["sal_w"][:bs_cur]
            tp_sal1, lay1 = out["tp_sal"][:bs_cur], out["layers_unif"][:bs_cur]

            align = 0.5 * (
                0.65 * (1.0 - F.cosine_similarity(eb1, tgt, dim=-1)).mean()
                + 0.35 * (1.0 - F.cosine_similarity(eu1, tgt, dim=-1)).mean()
                + 0.65 * (1.0 - F.cosine_similarity(eb2, tgt, dim=-1)).mean()
                + 0.35 * (1.0 - F.cosine_similarity(eu2, tgt, dim=-1)).mean()
            )
            geom = 0.5 * (geometry_loss(eb1, tgt) + geometry_loss(eb2, tgt))
            nbr = 0.5 * (
                0.55 * global_neighbor_kl(eb1, tgt, t_train, idx_dev, 0.10, 0.08)
                + 0.45 * global_neighbor_kl(eb1, tgt, t_train, idx_dev, 0.07, 0.06)
                + 0.55 * global_neighbor_kl(eb2, tgt, t_train, idx_dev, 0.10, 0.08)
                + 0.45 * global_neighbor_kl(eb2, tgt, t_train, idx_dev, 0.07, 0.06)
            )
            inv = (
                0.55 * (1.0 - F.cosine_similarity(eb2, eb1.detach(), dim=-1)).mean()
                + 0.25 * (1.0 - F.cosine_similarity(eu2, eu1.detach(), dim=-1)).mean()
                + 0.20 * (1.0 - F.cosine_similarity(core2, core1.detach(), dim=-1)).mean()
            )
            p_aff = (
                0.55 * centered_saliency_patch_affinity(sem_p1, p12_b, sal_w1)
                + 0.30 * centered_saliency_patch_affinity(tp1[:, 3], p12_b, sal_w1)
                + 0.15 * centered_saliency_patch_affinity(tp1[:, 2], p10_b, sal_w1)
            )
            core_sal1 = F.normalize(core1 + 0.35 * tp_sal1[:, 3], dim=-1)
            l3_sal1 = F.normalize(lay1[:, 2] + 0.35 * tp_sal1[:, 2], dim=-1)
            mid = 0.65 * batch_neighborhood_kl(center_norm_batch(core_sal1), tgt, 0.10) + 0.35 * batch_neighborhood_kl(
                center_norm_batch(l3_sal1), tgt_c11, 0.10
            )

            w_aux = min(1.0, max(0.0, (ep - 1) / 4.0))
            w_paff = 0.70 if args.patch_size == 3 else 0.65
            loss = 2.2 * align + 1.0 * geom + 0.70 * nbr + 0.55 * inv + w_aux * (w_paff * p_aff + 0.30 * mid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ep > args.warmup_epochs:
                update_ema(ema_model, model, args.ema_decay)
            loss_sum += float(loss.item())
            steps += 1

        if ep == args.warmup_epochs:
            ema_model.load_state_dict(model.state_dict(), strict=True)
        scheduler.step()
        print(f"[Ep {ep:02d}/{args.epochs:02d}] ({time.time()-t_ep:.1f}s) loss={loss_sum/max(1,steps):.4f}", flush=True)

        if ep in {4, 8, 12, 16, 20, 24, 28, 31, 34, 36, 38, 40, args.epochs}:
            ev_raw = evaluate_all_modes(model, None, x_tr_u8, x_va_u8, y_tr, y_va, device, run_full_probes=(ep == args.epochs))
            q_knn = ev_raw["purespike_quad_dualsal_pca384"]["knn10_val"]
            e_knn = ev_raw["emb_384"]["knn10_val"]
            print(f"  [Eval Ep {ep:02d}] raw emb_384={e_knn*100:.2f}% | quad_dualsal_384={q_knn*100:.2f}%", flush=True)
            if ep == 4 or q_knn > getattr(main, "_best_ps", 0.0):
                main._best_ps = q_knn  # type: ignore[attr-defined]
                torch.save(
                    {"epoch": ep, "variant": "raw", "model": model.state_dict(), "model_kwargs": model_kwargs, "dino_source_blocks": list(dino_blocks), "labels_used": False},
                    out_dir / "best_purespike.pt",
                )

    recalibrate_bn_on_train(ema_model, x_tr_u8, device)
    torch.save(
        {"epoch": args.epochs, "variant": "ema", "model": ema_model.state_dict(), "model_kwargs": model_kwargs, "dino_source_blocks": list(dino_blocks), "labels_used": False},
        out_dir / "best_ema.pt",
    )
    torch.save(
        {"epoch": args.epochs, "variant": "last_raw", "model": model.state_dict(), "model_kwargs": model_kwargs, "dino_source_blocks": list(dino_blocks), "labels_used": False},
        out_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
