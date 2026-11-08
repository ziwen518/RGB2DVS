"""Multi-depth spectral whitening and readout utilities for PureSpikeFormer.

Provides train-only unsupervised centering, block-wise spectral power-whitening,
and 384-d PCA compression for both pure binary spike-rate blocks (`purespike_pca384`)
and combined spike + residual membrane blocks (`allblocks9_pca384`), alongside the
detached multi-depth residual head (`DetachedMultiDepthHead`) and unified U1 probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from model import PureSpikeFormer  # noqa: E402


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_student_split(split_path: Path) -> tuple[list[str], list[str], int]:
    """Load train and validation sample IDs; test IDs are never returned."""
    data = json.loads(split_path.read_text(encoding="utf-8-sig"))
    train_ids = list(map(str, data["train_ids"]))
    val_ids = list(map(str, data["validation_ids"]))
    test_ids = list(map(str, data["test_ids"]))
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))
    return train_ids, val_ids, len(test_ids)


def label_of(sample_id: str) -> str:
    return sample_id.replace("\\", "/").split("/")[-2]


def labels_for_split(train_ids: list[str], val_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    classes = sorted({label_of(i) for i in train_ids})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = torch.tensor([class_to_idx[label_of(i)] for i in train_ids], dtype=torch.long)
    y_val = torch.tensor([class_to_idx[label_of(i)] for i in val_ids], dtype=torch.long)
    return y_train, y_val, classes


def resolve_frame_path(frames_root: Path, sample_id: str) -> Path:
    rel = PurePosixPath(sample_id)
    if rel.parts and rel.parts[0] == "events":
        rel = PurePosixPath(*rel.parts[1:])
    return frames_root.joinpath(*rel.parts).with_suffix(".npy")


def load_frames_u8(frames_root: Path, sample_ids: list[str]) -> torch.Tensor:
    arr = np.stack([np.load(resolve_frame_path(frames_root, sid), allow_pickle=False) for sid in sample_ids])
    if arr.shape[1:] != (16, 2, 48, 48) or arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames of shape [N, 16, 2, 48, 48], got {arr.shape} ({arr.dtype})")
    return torch.from_numpy(arr)


class DetachedMultiDepthHead(nn.Module):
    """Zero-initialized 384-d residual readout head over detached multi-depth features.

    Fuses normalized `[emb, core, L2, L3, M1, M2, M3]` while detaching intermediate
    tensors so auxiliary head gradients do not perturb mid-layer spiking thresholds.
    """

    def __init__(self, dim: int = 384, n_sources: int = 7) -> None:
        super().__init__()
        self.dim = dim
        self.n_sources = n_sources
        self.in_dim = dim * n_sources
        self.norm = nn.LayerNorm(self.in_dim)
        self.proj = nn.Linear(self.in_dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.3))

    def extract_concat_detached(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            F.normalize(out["embedding"].detach().float(), dim=-1),
            F.normalize(out["core_embedding"].detach().float(), dim=-1),
            F.normalize(out["layer_embeddings"][:, 2].detach().float(), dim=-1),
            F.normalize(out["layer_embeddings"][:, 3].detach().float(), dim=-1),
            F.normalize(out["membrane_embeddings"][:, 1].detach().float(), dim=-1),
            F.normalize(out["membrane_embeddings"][:, 2].detach().float(), dim=-1),
            F.normalize(out["membrane_embeddings"][:, 3].detach().float(), dim=-1),
        ]
        return torch.cat(parts, dim=-1)

    def forward(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cat_feat = self.extract_concat_detached(out)
        delta = self.proj(self.norm(cat_feat))
        emb = F.normalize(out["embedding"].detach().float(), dim=-1)
        fused = F.normalize(emb + torch.tanh(self.scale) * delta, dim=-1)
        return fused, cat_feat


@torch.no_grad()
def whiten_block(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    p: float = 0.30,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit centering and spectral power-whitening on `x_train` only and apply to `x_eval`."""
    x_train = F.normalize(x_train.float(), dim=-1)
    x_eval = F.normalize(x_eval.float(), dim=-1)
    mu = x_train.mean(dim=0, keepdim=True)
    x_centered = x_train - mu
    _, singular_values, vh = torch.linalg.svd(x_centered, full_matrices=False)
    scale = (singular_values.unsqueeze(0) + eps) ** (-p)
    transform = (vh.T * scale) @ vh
    z_train = F.normalize((x_train - mu) @ transform, dim=-1)
    z_eval = F.normalize((x_eval - mu) @ transform, dim=-1)
    return z_train, z_eval


@torch.no_grad()
def block_whiten_pca384(
    blocks: list[tuple[float, torch.Tensor, torch.Tensor]],
    pb: float = 0.30,
    pg: float = 0.35,
    out_dim: int = 384,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-block spectral whitening followed by train-only PCA compression to `out_dim`."""
    whitened_train, whitened_eval = [], []
    for weight, xt, xv in blocks:
        wt, wv = whiten_block(xt, xv, p=pb, eps=eps)
        whitened_train.append(weight * wt)
        whitened_eval.append(weight * wv)
    x_train = torch.cat(whitened_train, dim=-1)
    x_eval = torch.cat(whitened_eval, dim=-1)
    mu = x_train.mean(dim=0, keepdim=True)
    x_centered = x_train - mu
    _, singular_values, vh = torch.linalg.svd(x_centered, full_matrices=False)
    k = min(out_dim, vh.shape[0])
    proj = vh[:k].T / ((singular_values[:k].unsqueeze(0) + eps) ** pg)
    return (
        F.normalize((x_train - mu) @ proj, dim=-1),
        F.normalize((x_eval - mu) @ proj, dim=-1),
    )


@torch.no_grad()
def knn_top1(
    bank: torch.Tensor,
    bank_y: torch.Tensor,
    query: torch.Tensor,
    query_y: torch.Tensor,
    k: int = 10,
    exclude_self: bool = False,
) -> tuple[float, torch.Tensor]:
    bank = F.normalize(bank.float(), dim=-1)
    query = F.normalize(query.float(), dim=-1)
    sim = query @ bank.t()
    if exclude_self:
        sim.fill_diagonal_(-2.0)
    top_indices = sim.topk(min(k, len(bank)), dim=-1).indices
    pred = torch.mode(bank_y[top_indices], dim=-1).values
    return float(pred.eq(query_y).float().mean()), pred


def ridge_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> tuple[float, dict[str, object]]:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold

    x_tr = F.normalize(train_x.float(), dim=-1).cpu().numpy()
    x_va = F.normalize(val_x.float(), dim=-1).cpu().numpy()
    y_tr = train_y.cpu().numpy()
    y_va = val_y.cpu().numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3872026)
    best_alpha, best_cv, cv_scores = None, -1.0, {}
    for alpha in alphas:
        fold_scores = []
        for tr_idx, te_idx in skf.split(x_tr, y_tr):
            clf = RidgeClassifier(alpha=alpha).fit(x_tr[tr_idx], y_tr[tr_idx])
            fold_scores.append(float((clf.predict(x_tr[te_idx]) == y_tr[te_idx]).mean()))
        cv_scores[str(alpha)] = float(np.mean(fold_scores))
        if cv_scores[str(alpha)] > best_cv:
            best_cv, best_alpha = cv_scores[str(alpha)], alpha
    clf = RidgeClassifier(alpha=best_alpha).fit(x_tr, y_tr)
    val_acc = float((clf.predict(x_va) == y_va).mean())
    return val_acc, {"alpha": best_alpha, "train_cv_acc": best_cv, "cv_by_alpha": cv_scores}


def linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 3872026,
    device: str | torch.device = "cpu",
) -> tuple[float, dict[str, object]]:
    """Fixed-budget 100-epoch linear probe with train-only standardization."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mean = train_x.float().mean(dim=0)
    std = train_x.float().std(dim=0).clamp_min(1e-6)
    x_tr = ((train_x.float() - mean) / std).clamp(-10, 10).to(device)
    x_va = ((val_x.float() - mean) / std).clamp(-10, 10).to(device)
    y_tr = train_y.to(device)
    clf_head = nn.Linear(x_tr.shape[1], int(train_y.max()) + 1).to(device)
    opt = torch.optim.AdamW(clf_head.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(x_tr), generator=gen).to(device)
        for i in range(0, len(x_tr), 256):
            idx = perm[i : i + 256]
            loss = F.cross_entropy(clf_head(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = float(clf_head(x_va).argmax(dim=-1).cpu().eq(val_y).float().mean())
            history.append(acc)
    return history[-1], {"epochs": epochs, "lr": lr, "final_val_acc": history[-1]}


def cosine_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    temp: float = 16.0,
    wd: float = 1e-2,
    lr: float = 1e-2,
    epochs: int = 150,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[float, dict[str, object]]:
    """Centroid-initialized temperature-scaled cosine classifier on unit-sphere features."""
    torch.manual_seed(seed)
    num_classes = int(train_y.max().item()) + 1
    x_tr = F.normalize(train_x.to(device).float(), dim=-1)
    x_va = F.normalize(val_x.to(device).float(), dim=-1)
    y_tr = train_y.to(device)
    y_va = val_y.to(device)

    weight = nn.Parameter(torch.zeros(num_classes, x_tr.shape[1], device=device))
    with torch.no_grad():
        for c in range(num_classes):
            mask = y_tr == c
            if mask.any():
                weight.data[c] = F.normalize(x_tr[mask].mean(dim=0), dim=-1)
    opt = torch.optim.AdamW([weight], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        logits = temp * (x_tr @ F.normalize(weight, dim=-1).t())
        loss = F.cross_entropy(logits, y_tr, label_smoothing=0.1)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = (temp * (x_va @ F.normalize(weight, dim=-1).t())).argmax(dim=-1)
        val_acc = float((pred == y_va).float().mean().item())
    return val_acc, {"epochs": epochs, "temp": temp, "lr": lr, "final_val_acc": val_acc}


def evaluate_all_modes(
    model: PureSpikeFormer,
    head: DetachedMultiDepthHead | None,
    x_tr_u8: torch.Tensor,
    x_va_u8: torch.Tensor,
    y_tr: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    z_va_teacher: torch.Tensor | None = None,
    run_full_probes: bool = True,
    saliency_floor: float = 0.10,
    saliency_alpha: float = 0.35,
) -> dict[str, dict[str, object]]:
    """Extract multi-depth features and evaluate all 384-d representation modes on U1."""
    model.eval()
    if head is not None:
        head.eval()

    @torch.no_grad()
    def collect(x_u8: torch.Tensor, bs: int = 128) -> dict[str, torch.Tensor]:
        emb_l, core_l, fused_l, lay_l, mem_l, sal_l = [], [], [], [], [], []
        for i in range(0, len(x_u8), bs):
            x = x_u8[i : i + bs].to(device, non_blocking=True).float() / 255.0
            ev_map = F.avg_pool2d(x.sum(dim=(1, 2)).unsqueeze(1), kernel_size=4, stride=4).reshape(x.shape[0], 1, -1, 1)
            sal_w = (ev_map + saliency_floor) / (ev_map + saliency_floor).sum(dim=2, keepdim=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x, return_layers=True, return_membranes=True, return_token_layers=4)
            if head is not None:
                fused, _ = head(out)
                fused_l.append(fused.float().cpu())
            tp = out["token_pyramid"].float().mean(dim=2)
            sal_pool = F.normalize((tp * sal_w).sum(dim=2), dim=-1)
            emb_l.append(out["embedding"].float().cpu())
            core_l.append(out["core_embedding"].float().cpu())
            lay_l.append(out["layer_embeddings"].float().cpu())
            mem_l.append(out["membrane_embeddings"].float().cpu())
            sal_l.append(sal_pool.float().cpu())
        res_dict = {
            "emb": torch.cat(emb_l, dim=0),
            "core": torch.cat(core_l, dim=0),
            "layers": torch.cat(lay_l, dim=0),
            "memb": torch.cat(mem_l, dim=0),
            "sal": torch.cat(sal_l, dim=0),
        }
        if fused_l:
            res_dict["fused"] = torch.cat(fused_l, dim=0)
        return res_dict

    tr_c = collect(x_tr_u8)
    va_c = collect(x_va_u8)

    tr_emb_w, va_emb_w = whiten_block(tr_c["emb"], va_c["emb"], p=0.45)

    spk_blocks = [
        (1.2, tr_c["emb"], va_c["emb"]),
        (1.0, tr_c["core"], va_c["core"]),
        (0.6, tr_c["layers"][:, 1], va_c["layers"][:, 1]),
        (0.8, tr_c["layers"][:, 2], va_c["layers"][:, 2]),
        (1.0, tr_c["layers"][:, 3], va_c["layers"][:, 3]),
    ]
    tr_spk384, va_spk384 = block_whiten_pca384(spk_blocks, pb=0.30, pg=0.35)

    spk_sal_blocks = [
        (1.2, tr_c["emb"], va_c["emb"]),
        (
            1.0,
            F.normalize(tr_c["core"] + saliency_alpha * tr_c["sal"][:, 3], dim=-1),
            F.normalize(va_c["core"] + saliency_alpha * va_c["sal"][:, 3], dim=-1),
        ),
        (
            0.6,
            F.normalize(tr_c["layers"][:, 1] + saliency_alpha * tr_c["sal"][:, 0], dim=-1),
            F.normalize(va_c["layers"][:, 1] + saliency_alpha * va_c["sal"][:, 0], dim=-1),
        ),
        (
            0.8,
            F.normalize(tr_c["layers"][:, 2] + saliency_alpha * tr_c["sal"][:, 1], dim=-1),
            F.normalize(va_c["layers"][:, 2] + saliency_alpha * va_c["sal"][:, 1], dim=-1),
        ),
        (
            1.0,
            F.normalize(tr_c["layers"][:, 3] + saliency_alpha * tr_c["sal"][:, 2], dim=-1),
            F.normalize(va_c["layers"][:, 3] + saliency_alpha * va_c["sal"][:, 2], dim=-1),
        ),
    ]
    tr_sal384, va_sal384 = block_whiten_pca384(spk_sal_blocks, pb=0.30, pg=0.35)

    all9_blocks = spk_blocks + [
        (0.6, tr_c["memb"][:, 0], va_c["memb"][:, 0]),
        (0.8, tr_c["memb"][:, 1], va_c["memb"][:, 1]),
        (1.0, tr_c["memb"][:, 2], va_c["memb"][:, 2]),
        (1.0, tr_c["memb"][:, 3], va_c["memb"][:, 3]),
    ]
    tr_all9_384, va_all9_384 = block_whiten_pca384(all9_blocks, pb=0.30, pg=0.35)

    modes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "emb_384": (tr_c["emb"], va_c["emb"]),
        "emb_whiten_384": (tr_emb_w, va_emb_w),
    }
    if "fused" in tr_c:
        modes["bridged_384"] = (tr_c["fused"], va_c["fused"])
    modes["purespike_pca384"] = (tr_spk384, va_spk384)
    modes["purespike_sal_pca384"] = (tr_sal384, va_sal384)
    modes["allblocks9_pca384"] = (tr_all9_384, va_all9_384)

    results: dict[str, dict[str, object]] = {}
    for m_name, (x_tr, x_va) in modes.items():
        x_tr = F.normalize(x_tr.float(), dim=-1)
        x_va = F.normalize(x_va.float(), dim=-1)
        knn, _ = knn_top1(x_tr, y_tr, x_va, y_va, k=10)
        loo, _ = knn_top1(x_tr, y_tr, x_tr, y_tr, k=10, exclude_self=True)
        entry: dict[str, object] = {
            "knn10_val": knn,
            "knn10_val_correct": int(round(knn * len(y_va))),
            "knn10_train_loo": loo,
            "gap": loo - knn,
        }
        if z_va_teacher is not None and m_name in ("emb_384", "bridged_384"):
            entry["val_teacher_cos"] = float(F.cosine_similarity(x_va, z_va_teacher, dim=-1).mean())
        if run_full_probes and m_name in (
            "emb_384",
            "emb_whiten_384",
            "purespike_pca384",
            "purespike_sal_pca384",
            "allblocks9_pca384",
        ):
            r_acc, _ = ridge_eval(x_tr, y_tr, x_va, y_va)
            lp_acc, _ = linear_probe(x_tr, y_tr, x_va, y_va, device=device)
            clp_acc, _ = cosine_linear_probe(x_tr, y_tr, x_va, y_va, device=device)
            entry["ridge_val"] = r_acc
            entry["lp_val"] = lp_acc
            entry["cosine_lp_val"] = clp_acc
        results[m_name] = entry
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PureSpikeFormer with multi-depth spectral readout on U1.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to student checkpoint (.pt)")
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
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    split_path = args.split_manifest if args.split_manifest.is_absolute() else args.project_root / args.split_manifest
    frames_root = args.frames_root if args.frames_root.is_absolute() else args.project_root / args.frames_root
    device = torch.device(args.device)

    tr_ids, va_ids, _ = load_student_split(split_path)
    y_tr, y_va, _ = labels_for_split(tr_ids, va_ids)
    x_tr_u8 = load_frames_u8(frames_root, tr_ids)
    x_va_u8 = load_frames_u8(frames_root, va_ids)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_kwargs = ckpt.get("model_kwargs") or ckpt["config"]["model_kwargs"]
    model = PureSpikeFormer(**model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    head = None
    if "head" in ckpt:
        head = DetachedMultiDepthHead(dim=model_kwargs.get("dim", 384), n_sources=7).to(device).eval()
        head.load_state_dict(ckpt["head"], strict=True)

    metrics = evaluate_all_modes(
        model=model,
        head=head,
        x_tr_u8=x_tr_u8,
        x_va_u8=x_va_u8,
        y_tr=y_tr,
        y_va=y_va,
        device=device,
        run_full_probes=True,
    )
    for name, m in metrics.items():
        extra = f" | Ridge={m['ridge_val']*100:.2f}% | LP={m['lp_val']*100:.2f}%" if "ridge_val" in m else ""
        print(
            f"{name:22s}: kNN={m['knn10_val']*100:.2f}% ({m['knn10_val_correct']}/900) "
            f"| LOO={m['knn10_train_loo']*100:.2f}%{extra}"
        )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
