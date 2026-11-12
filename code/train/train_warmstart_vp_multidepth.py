"""Warm-start variance-preserving spatial invariance & multi-depth distillation trainer.

Fine-tunes a pretrained PureSpikeFormer checkpoint with:
1. Locked BatchNorm running statistics (`lock_bn_stats`) to preserve BNTT firing thresholds.
2. Exact 100% variance-preserving GPU spatial augmentation (`gpu_vp_augment`: integer pixel
   shifts +/-3px and horizontal flips) with sequential two-view gradient accumulation.
3. Two-view shift/flip cosine invariance + SEED-style full-training-bank neighborhood KL
   (`global_neighbor_kl`) + batch relational geometry + cosine alignment.
4. Differential learning rates (`1e-6` on DINO-initialized attention/MLP weights, `1e-5` on rest)
   and an optional detached multi-depth residual readout head (`DetachedMultiDepthHead`).
"""
from __future__ import annotations

import argparse
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
    DetachedMultiDepthHead,
    evaluate_all_modes,
    labels_for_split,
    load_frames_u8,
    load_student_split,
    sha256_file,
)
from train_clean_observable_spikeformer import geometry_loss  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lock_bn_stats(model: nn.Module) -> None:
    """Keep pretrained BatchNorm running mean/var fixed while leaving affine weights trainable."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buf.detach().cpu().clone()
        for name, buf in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def global_neighbor_kl(
    student: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_bank: torch.Tensor,
    indices: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SEED-style query-to-frozen-training-bank relational KL divergence (excluding self)."""
    assert len(teacher_bank) > 1 and temperature > 0
    s = F.normalize(student.float(), dim=-1)
    keys = F.normalize(teacher_bank.detach().float(), dim=-1)
    idx = indices.to(s.device).long().view(-1, 1)
    logits = (s @ keys.t() / temperature).scatter(1, idx, -1.0e4)
    with torch.no_grad():
        t = F.normalize(teacher_query.detach().float(), dim=-1)
        teacher_logits = (t @ keys.t() / temperature).scatter(1, idx, -1.0e4)
        target = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def gpu_vp_augment(
    x: torch.Tensor,
    max_shift: int = 3,
    p_flip: float = 0.5,
    p_shift: float = 0.85,
) -> torch.Tensor:
    """100% variance-preserving integer spatial translation and horizontal flip on GPU."""
    batch_size = x.shape[0]
    out = x.clone()
    flip_mask = torch.rand(batch_size, device=x.device) < p_flip
    if flip_mask.any():
        out[flip_mask] = torch.flip(out[flip_mask], dims=[-1])
    if max_shift > 0:
        for i in range(batch_size):
            if random.random() < p_shift:
                dy = random.randint(-max_shift, max_shift)
                dx = random.randint(-max_shift, max_shift)
                if dy != 0 or dx != 0:
                    out[i] = torch.roll(out[i], shifts=(dy, dx), dims=(-2, -1))
    return out


def atomic_json_dump(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start PureSpikeFormer with variance-preserving shift invariance & multi-depth readout."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-ckpt",
        type=Path,
        default=Path("iclr_execution_20260907/runs/full_ncaltech101_learned_proxy_s42/best.pt"),
        help="Path to pretrained PureSpikeFormer checkpoint (.pt)",
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path("iclr_execution_20260907/ncal_event_proxy.pt"),
        help="Path to train-only transported teacher target cache (.pt)",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/teacher_cache/ncaltech101/student_split.json"),
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("data/ncaltech101_rgb_event/frames_ncaltech101_t16_s48"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr-dino", type=float, default=1e-6)
    parser.add_argument("--lr-rest", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument(
        "--p-flip",
        type=float,
        default=0.0,
        help="Horizontal flip probability (0.0 preserves canonical N-Caltech101 horizontal orientation)",
    )
    parser.add_argument("--mem-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else args.project_root / p

    source_ckpt = resolve(args.source_ckpt)
    target_cache = resolve(args.target_cache)
    split_path = resolve(args.split_manifest)
    frames_root = resolve(args.frames_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        if 0.0 < args.mem_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    seed_all(args.seed)

    tr_ids, va_ids, _ = load_student_split(split_path)
    y_tr, y_va, _ = labels_for_split(tr_ids, va_ids)
    x_tr_u8 = load_frames_u8(frames_root, tr_ids)
    x_va_u8 = load_frames_u8(frames_root, va_ids)

    cache_obj = torch.load(target_cache, map_location="cpu", weights_only=False)
    cache_index = {str(s): i for i, s in enumerate(cache_obj["ids"])}
    t_train = F.normalize(cache_obj["transported"][[cache_index[s] for s in tr_ids]].float(), dim=-1).to(device)
    del cache_obj

    ckpt = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    model_kwargs = ckpt.get("model_kwargs") or ckpt["config"]["model_kwargs"]
    model = PureSpikeFormer(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    initial_bn = bn_buffers(model)

    head = DetachedMultiDepthHead(dim=model_kwargs.get("dim", 384), n_sources=7).to(device)

    dino_suffix = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
    pre_params = [p for n, p in model.named_parameters() if n.startswith("blocks.") and n.endswith(dino_suffix)]
    pre_ids = {id(p) for p in pre_params}
    rest_params = [p for p in model.parameters() if id(p) not in pre_ids]

    opt_model = torch.optim.AdamW(
        [
            {"params": pre_params, "lr": args.lr_dino},
            {"params": rest_params, "lr": args.lr_rest},
        ],
        weight_decay=0.01,
    )
    sched_model = torch.optim.lr_scheduler.LambdaLR(
        opt_model,
        lambda e: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(e, 15) / 15.0)),
    )

    opt_head = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=0.01)
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=args.epochs, eta_min=1e-5)

    history: list[dict[str, object]] = []
    best_emb_knn = 0.0
    best_purespike_knn = 0.0
    best_all9_knn = 0.0
    n_train = len(x_tr_u8)

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        lock_bn_stats(model)
        head.train()

        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed * 100 + ep))
        sums = {"loss_m": 0.0, "loss_h": 0.0, "align": 0.0, "inv": 0.0, "nbr": 0.0}
        steps = 0

        for bi in range(0, n_train, args.batch_size):
            idx = perm[bi : bi + args.batch_size]
            if len(idx) < 4:
                continue
            idx_dev = idx.to(device)
            x_clean = x_tr_u8[idx].to(device, non_blocking=True).float() / 255.0
            x_v1 = gpu_vp_augment(x_clean, max_shift=0, p_flip=args.p_flip, p_shift=0.0)
            x_v2 = gpu_vp_augment(x_clean, max_shift=args.max_shift, p_flip=args.p_flip, p_shift=0.85)
            tgt = t_train[idx_dev]

            opt_model.zero_grad(set_to_none=True)
            opt_head.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out1 = model(x_v1, return_layers=True, return_membranes=True)
            emb1 = out1["embedding"].float()
            emb1_det = emb1.detach()
            fused1, _ = head(out1)

            loss1_m = 0.5 * (
                2.0 * (1.0 - F.cosine_similarity(emb1, tgt, dim=-1)).mean()
                + 1.0 * geometry_loss(emb1, tgt)
                + 0.5 * global_neighbor_kl(emb1, tgt, t_train, idx_dev, 0.1)
            )
            loss1_m.backward()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out2 = model(x_v2, return_layers=True, return_membranes=True)
            emb2 = out2["embedding"].float()
            fused2, _ = head(out2)

            align2 = (1.0 - F.cosine_similarity(emb2, tgt, dim=-1)).mean()
            geom2 = geometry_loss(emb2, tgt)
            nbr2 = global_neighbor_kl(emb2, tgt, t_train, idx_dev, 0.1)
            inv2 = (1.0 - F.cosine_similarity(emb2, emb1_det, dim=-1)).mean()

            loss2_m = 0.5 * (2.0 * align2 + 1.0 * geom2 + 0.5 * nbr2) + 0.5 * inv2
            loss2_m.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_model.step()

            loss_h = (
                1.5
                * (
                    (1.0 - F.cosine_similarity(fused1, tgt, dim=-1)).mean()
                    + (1.0 - F.cosine_similarity(fused2, tgt, dim=-1)).mean()
                )
                + 0.5 * (geometry_loss(fused1, tgt) + geometry_loss(fused2, tgt))
                + 0.5
                * (
                    global_neighbor_kl(fused1, tgt, t_train, idx_dev, 0.1)
                    + global_neighbor_kl(fused2, tgt, t_train, idx_dev, 0.1)
                )
                + 0.5 * (1.0 - F.cosine_similarity(fused1, fused2, dim=-1)).mean()
            )
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt_head.step()

            sums["loss_m"] += float((loss1_m + loss2_m).item())
            sums["loss_h"] += float(loss_h.item())
            sums["align"] += float(align2.item())
            sums["inv"] += float(inv2.item())
            sums["nbr"] += float(nbr2.item())
            steps += 1

        sched_model.step()
        sched_head.step()
        cur_bn = bn_buffers(model)
        assert all(torch.equal(v, cur_bn[k]) for k, v in initial_bn.items()), "Locked BN buffers changed!"

        avg = {k: v / max(1, steps) for k, v in sums.items()}
        run_full = ep in (5, 8, 10, 12) or ep == args.epochs
        ev = evaluate_all_modes(
            model=model,
            head=head,
            x_tr_u8=x_tr_u8,
            x_va_u8=x_va_u8,
            y_tr=y_tr,
            y_va=y_va,
            device=device,
            run_full_probes=run_full,
        )
        dt = time.time() - t_ep
        print(
            f"\n[Epoch {ep:02d}/{args.epochs:02d}] ({dt:.1f}s) "
            f"loss_m={avg['loss_m']:.4f} align={avg['align']:.4f} inv={avg['inv']:.4f} nbr={avg['nbr']:.4f}",
            flush=True,
        )
        for k, v in ev.items():
            extra = f" | Ridge={v['ridge_val']*100:.2f}% | LP={v['lp_val']*100:.2f}%" if "ridge_val" in v else ""
            print(
                f"  {k:20s}: kNN={v['knn10_val']*100:.2f}% ({v['knn10_val_correct']}/900) "
                f"| LOO={v['knn10_train_loo']*100:.2f}%{extra}",
                flush=True,
            )

        history.append({"epoch": ep, "wall_s": dt, "losses": avg, "eval": ev})
        atomic_json_dump(
            {
                "args": vars(args),
                "source_sha256": sha256_file(source_ckpt),
                "history": history,
            },
            out_dir / "history_round4_full.json",
        )

        state_payload = {
            "epoch": int(ckpt.get("epoch", 39)) + ep,
            "round4_epoch": ep,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optimizer_model": opt_model.state_dict(),
            "scheduler_model": sched_model.state_dict(),
            "model_kwargs": model_kwargs,
            "eval": ev,
        }
        torch.save(state_payload, out_dir / "last.pt")

        if float(ev["emb_384"]["knn10_val"]) >= best_emb_knn:
            best_emb_knn = float(ev["emb_384"]["knn10_val"])
            torch.save(state_payload, out_dir / "best_emb.pt")

        if float(ev["purespike_pca384"]["knn10_val"]) >= best_purespike_knn:
            best_purespike_knn = float(ev["purespike_pca384"]["knn10_val"])
            torch.save(state_payload, out_dir / "best_purespike.pt")

        if float(ev["allblocks9_pca384"]["knn10_val"]) >= best_all9_knn:
            best_all9_knn = float(ev["allblocks9_pca384"]["knn10_val"])
            torch.save(state_payload, out_dir / "best_allblocks9.pt")


if __name__ == "__main__":
    main()
