"""DINOv3 Figure 13 style dense patch-token PCA-to-RGB visualization for PureSpikeFormer.

Follows the qualitative dense-feature evaluation protocol of DINOv3 (Section 6.1.1, Fig. 13):
1. Extract dense spatial token maps from each backbone on held-out validation recordings
   (`test_ids` are never accessed; `labels_used=False`).
2. Subtract each model's static train-only spatial position carrier `E_{train}[f(x, y)]`
   computed strictly over `train_ids`, then L2-normalize each spatial token.
3. Project each dense token map to 3 principal components via PCA (`PC1, PC2, PC3`) and
   resolve PCA's 8 sign flips and 6 channel permutations (`48` discrete variants, DINOv3
   Sec. 6.1.1) against the teacher orientation for consistent cross-column semantics.
4. Upsample floating-point 3D PCA maps smoothly to display resolution and map robust
   percentiles directly to RGB, reporting `PC1/2/3` explained variance ratios per column.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

DEFAULT_DISPLAY_IDS = [
    "events/airplanes/image_0774.bin",
    "events/car_side/image_0093.bin",
    "events/Motorbikes/image_0428.bin",
    "events/Faces_easy/image_0087.bin",
    "events/butterfly/image_0034.bin",
    "events/watch/image_0153.bin",
]

PRETTY_ROW_LABELS = {
    "airplanes": "airplane",
    "car_side": "automobile",
    "Motorbikes": "motorbike",
    "Faces_easy": "face",
    "butterfly": "butterfly",
    "watch": "watch",
    "Leopards": "leopard",
    "ferry": "ferry",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2_normalize_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def resolve_frame_path(frames_root: Path, sample_id: str) -> Path:
    rel = PurePosixPath(sample_id.replace("\\", "/"))
    if rel.parts and rel.parts[0] == "events":
        rel = PurePosixPath(*rel.parts[1:])
    return frames_root.joinpath(*rel.parts).with_suffix(".npy")


def extract_bundle_on_gpu(args: argparse.Namespace) -> Path:
    import torch
    import torch.nn.functional as F
    from sklearn.decomposition import PCA

    root = args.project_root.resolve()
    for sub in ("code/core", "code/eval", "code/train", "ICLR/execution_20260907"):
        cand = root / sub
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))

    from model import PureSpikeFormer  # noqa: E402
    from multidepth_spectral_readout import (  # noqa: E402
        NextGenMultiDepthSpikingStudent,
        load_frames_u8,
        load_student_split,
    )
    from train_nextgen_dense_spikeformer import forward_pure_rich  # noqa: E402

    device = torch.device(args.device)
    split_path = root / args.split_manifest if not args.split_manifest.is_absolute() else args.split_manifest
    frames_root = root / args.frames_root if not args.frames_root.is_absolute() else args.frames_root
    rgb_root = root / args.rgb_root if not args.rgb_root.is_absolute() else args.rgb_root
    paired_manifest = root / args.paired_manifest if not args.paired_manifest.is_absolute() else args.paired_manifest

    train_ids, val_ids, _ = load_student_split(split_path)
    val_set = set(val_ids)
    display_ids = list(args.display_ids)
    for sid in display_ids:
        if sid not in val_set:
            raise ValueError(f"Display sample {sid!r} must belong strictly to validation_ids (never train/test).")

    paired_entries = json.loads(paired_manifest.read_text(encoding="utf-8-sig"))
    paired_map = {str(item["id"]): str(item["rgb_path"]) for item in paired_entries}

    x_train_u8 = load_frames_u8(frames_root, train_ids)
    x_disp_u8 = load_frames_u8(frames_root, display_ids)

    rgb_imgs = []
    for sid in display_ids:
        im = Image.open(rgb_root / paired_map[sid]).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        rgb_imgs.append(np.asarray(im, dtype=np.uint8))
    rgb_u8 = np.stack(rgb_imgs, axis=0)

    def forward_wrapper_sem_patches(student: NextGenMultiDepthSpikingStudent, x: torch.Tensor) -> torch.Tensor:
        b, t_steps = x.shape[0], x.shape[1]
        pm, lm = None, None
        states = [None] * len(student.base.blocks)
        b_steps = [[] for _ in range(len(student.base.blocks))]
        for t in range(t_steps):
            xt, pm, lm, _ = student.base.patch(x[:, t], pm, lm, t)
            for bi, blk in enumerate(student.base.blocks):
                xt, states[bi], _ = blk(xt, states[bi], t)
                b_steps[bi].append(xt)
        sem_m = None
        sem_steps = []
        g1 = 0.20 * torch.tanh(student.gate_b1)
        g2 = 0.30 * torch.tanh(student.gate_b2)
        g3 = 0.30 * torch.tanh(student.gate_b3)
        for t in range(t_steps):
            cur = (
                student.base.semantic_proj(b_steps[-1][t])
                + g1 * student.aux_proj_b1(b_steps[0][t])
                + g2 * student.aux_proj_b2(b_steps[1][t])
                + g3 * student.aux_proj_b3(b_steps[2][t])
            )
            cur = student.base.semantic_norm(cur, t)
            sp, sem_m = student.base.semantic_lif(cur, sem_m)
            pos, neg = sp.chunk(2, dim=-1)
            sem_steps.append(pos - neg)
        sp_all = torch.stack(sem_steps, dim=1).float()
        wt = F.softmax(student.base.temporal_logits[:t_steps], dim=0).view(1, t_steps, 1, 1)
        return (sp_all * wt).sum(dim=1)

    def compute_debiased_pca(train_sum: np.ndarray, n_train: int, disp_tokens: np.ndarray) -> dict[str, np.ndarray]:
        pos_mean = (train_sum / float(n_train))[None, :, :]
        debiased = l2_normalize_np(disp_tokens - pos_mean)
        proj_list, evr_list = [], []
        for i in range(len(debiased)):
            pca = PCA(n_components=3, random_state=42).fit(debiased[i])
            proj_list.append(pca.transform(debiased[i]))
            evr_list.append(pca.explained_variance_ratio_)
        return {
            "proj_3d": np.stack(proj_list, axis=0).astype(np.float32),
            "evr": np.stack(evr_list, axis=0).astype(np.float32),
            "pos_mean": pos_mean.squeeze(0).astype(np.float16),
        }

    bundle: dict[str, np.ndarray] = {
        "display_ids": np.array(display_ids),
        "events_u8": x_disp_u8.numpy(),
        "rgb_u8": rgb_u8,
        "train_count": np.array([len(train_ids)], dtype=np.int64),
    }
    provenance_models: dict[str, dict[str, object]] = {}

    # 1. EventProxy-DINOv2 teacher (if available)
    if args.include_teacher:
        import export_t374_eventproxy_teacher_depths_linux as tbase  # noqa: E402

        proxy = tbase.Proxy().to(device).eval()
        adap = torch.load(tbase.ADAPTER, map_location="cpu", weights_only=False)
        proxy.stem.load_state_dict(adap["stem"], strict=True)

        t_sum = np.zeros((256, 384), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(x_train_u8), args.batch_size):
                x = x_train_u8[i : i + args.batch_size].to(device, non_blocking=True).float() / 255.0
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    img = F.interpolate(
                        torch.sigmoid(proxy.stem(x.flatten(1, 2))), (224, 224), mode="bilinear", align_corners=False
                    )
                    img = (img - proxy.mean) / proxy.std
                    p12 = proxy.backbone.get_intermediate_layers(img, n=[11], return_class_token=False)[0]
                t_sum += p12.float().sum(dim=0).cpu().numpy().astype(np.float64)

            xd = x_disp_u8.to(device).float() / 255.0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                img_d = F.interpolate(
                    torch.sigmoid(proxy.stem(xd.flatten(1, 2))), (224, 224), mode="bilinear", align_corners=False
                )
                img_d = (img_d - proxy.mean) / proxy.std
                t_disp = proxy.backbone.get_intermediate_layers(img_d, n=[11], return_class_token=False)[0]
            t_disp_np = t_disp.float().cpu().numpy()

        t_res = compute_debiased_pca(t_sum, len(x_train_u8), t_disp_np)
        bundle["teacher_proj_3d"] = t_res["proj_3d"]
        bundle["teacher_evr"] = t_res["evr"]
        adap_path = Path(tbase.ADAPTER)
        adap_rel = adap_path.relative_to(root) if adap_path.is_relative_to(root) else adap_path
        provenance_models["teacher"] = {
            "adapter": str(adap_rel),
            "adapter_sha256": sha256_file(adap_path),
            "grid": [16, 16],
            "mean_pc123_pct": [round(float(v) * 100.0, 2) for v in t_res["evr"].mean(axis=0)],
        }

    # 2. Student checkpoints
    model_specs = [
        ("baseline", args.baseline_ckpt, False),
        ("warmstart", args.warmstart_ckpt, True),
        ("stage2", args.stage2_ckpt, False),
        ("stage3", args.stage3_ckpt, False),
    ]
    for key, rel_ckpt, is_wrapper in model_specs:
        if rel_ckpt is None:
            continue
        ckpt_path = root / rel_ckpt if not rel_ckpt.is_absolute() else rel_ckpt
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ck.get("model_kwargs") or ck["config"]["model_kwargs"]
        base = PureSpikeFormer(**model_kwargs).to(device).eval()
        if is_wrapper:
            student = NextGenMultiDepthSpikingStudent(base).to(device).eval()
            student.load_state_dict(ck["student"], strict=True)
        else:
            base.load_state_dict(ck["model"], strict=True)

        p_size = int(model_kwargs["patch_size"])
        side = int(model_kwargs["image_size"]) // p_size
        n_tokens = side * side
        s_sum = np.zeros((n_tokens, int(model_kwargs["dim"])), dtype=np.float64)

        with torch.no_grad():
            for i in range(0, len(x_train_u8), args.batch_size):
                x = x_train_u8[i : i + args.batch_size].to(device, non_blocking=True).float() / 255.0
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    sem_p = forward_wrapper_sem_patches(student, x) if is_wrapper else forward_pure_rich(base, x)["sem_patches"]
                s_sum += sem_p.float().sum(dim=0).cpu().numpy().astype(np.float64)

            xd = x_disp_u8.to(device).float() / 255.0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                s_disp = forward_wrapper_sem_patches(student, xd) if is_wrapper else forward_pure_rich(base, xd)["sem_patches"]
            s_disp_np = s_disp.float().cpu().numpy()

        s_res = compute_debiased_pca(s_sum, len(x_train_u8), s_disp_np)
        bundle[f"{key}_proj_3d"] = s_res["proj_3d"]
        bundle[f"{key}_evr"] = s_res["evr"]
        provenance_models[key] = {
            "checkpoint": str(rel_ckpt),
            "sha256": sha256_file(ckpt_path),
            "epoch": ck.get("epoch"),
            "grid": [side, side],
            "mean_pc123_pct": [round(float(v) * 100.0, 2) for v in s_res["evr"].mean(axis=0)],
        }
        print(f"[{key}] grid={side}x{side} mean PC1/2/3={provenance_models[key]['mean_pc123_pct']}")

    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.bundle_out, **bundle)
    prov_path = args.bundle_out.with_suffix(".provenance.json")
    prov = {
        "protocol": "DINOv3 Section 6.1.1 / Figure 13 dense patch PCA-to-RGB with train-only spatial position debiasing",
        "dataset": "N-Caltech101 (locked U1 student_split.json)",
        "train_samples_for_position_debiasing": len(train_ids),
        "display_split": "validation_ids_only",
        "test_split_accessed": False,
        "labels_used": False,
        "display_ids": display_ids,
        "models": provenance_models,
    }
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"Saved bundle to {args.bundle_out} and provenance to {prov_path}")
    return args.bundle_out


def render_event_panel(ev: np.ndarray, out_size: int = 224) -> np.ndarray:
    """Render [T, 2, H, W] event tensor as polarity-tinted copper/rose accumulation on dark background."""
    on = ev[:, 0].astype(np.float32).sum(axis=0)
    off = ev[:, 1].astype(np.float32).sum(axis=0)
    tot = on + off
    scale = max(float(np.percentile(tot, 99.2)), 1.0)
    on_n = np.clip(on / scale, 0.0, 1.0) ** 0.68
    off_n = np.clip(off / scale, 0.0, 1.0) ** 0.68
    bg = 0.035
    r = np.clip(bg + 0.90 * on_n + 0.50 * off_n, 0.0, 1.0)
    g = np.clip(bg + 0.56 * on_n + 0.40 * off_n, 0.0, 1.0)
    b = np.clip(bg + 0.48 * on_n + 0.86 * off_n, 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((out_size, out_size), Image.Resampling.BILINEAR))


def align_and_upsample_pca_rgb(
    proj_grid: np.ndarray,
    ref_std_16x16: np.ndarray | None,
    out_size: int = 224,
    allow_channel_perm: bool = True,
) -> np.ndarray:
    """Resolve PCA sign/channel ambiguity (DINOv3 Sec. 6.1.1) and smoothly upsample in float32."""
    import torch
    import torch.nn.functional as F

    side = proj_grid.shape[0]
    if ref_std_16x16 is None:
        border = np.zeros((side, side), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        signs = np.where(proj_grid[border].mean(axis=0) >= 0.0, 1.0, -1.0)
        aligned = proj_grid * signs[None, None, :]
    else:
        m_std = (proj_grid - proj_grid.mean(axis=(0, 1), keepdims=True)) / (
            proj_grid.std(axis=(0, 1), keepdims=True) + 1e-6
        )
        if side != 16:
            t_up = np.repeat(np.repeat(ref_std_16x16, 3, axis=0), 3, axis=1)
            t_ref = t_up.reshape(12, 4, 12, 4, 3).mean(axis=(1, 3))
        else:
            t_ref = ref_std_16x16
        perms = list(itertools.permutations((0, 1, 2))) if allow_channel_perm else [(0, 1, 2)]
        best_score = -1e18
        best_perm, best_signs = (0, 1, 2), (1.0, 1.0, 1.0)
        for perm in perms:
            for sg in itertools.product((-1.0, 1.0), repeat=3):
                cand = m_std[..., list(perm)] * np.array(sg, dtype=np.float32)[None, None, :]
                score = float(np.sum(cand * t_ref))
                if score > best_score:
                    best_score = score
                    best_perm, best_signs = perm, sg
        aligned = proj_grid[..., list(best_perm)] * np.array(best_signs, dtype=np.float32)[None, None, :]

    # Smooth float32 upsampling before percentile normalization prevents uint8 banding
    tensor = torch.from_numpy(aligned).permute(2, 0, 1).unsqueeze(0).float()
    upsampled = F.interpolate(tensor, size=(out_size, out_size), mode="bicubic", align_corners=False)
    up_np = upsampled.squeeze(0).permute(1, 2, 0).numpy()

    rgb = np.zeros_like(up_np)
    for ch in range(3):
        lo, hi = np.percentile(up_np[..., ch], [1.5, 98.5])
        rgb[..., ch] = np.clip((up_np[..., ch] - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return rgb


def render_figures_from_bundle(bundle_path: Path, out_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "none"

    data = np.load(bundle_path, allow_pickle=False)
    display_ids = [str(s) for s in data["display_ids"]]
    events_u8 = data["events_u8"]
    rgb_u8 = data["rgb_u8"]
    n_rows = len(display_ids)
    out_size = 224

    row_names = []
    for sid in display_ids:
        raw_cat = sid.replace("\\", "/").split("/")[-2]
        row_names.append(PRETTY_ROW_LABELS.get(raw_cat, raw_cat))

    # Precompute reference teacher standardized maps
    has_teacher = "teacher_proj_3d" in data
    teacher_refs: list[np.ndarray | None] = [None] * n_rows
    teacher_rgbs: list[np.ndarray] = []
    if has_teacher:
        for r in range(n_rows):
            t_raw = data["teacher_proj_3d"][r].reshape(16, 16, 3)
            border = np.zeros((16, 16), dtype=bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            t_signs = np.where(t_raw[border].mean(axis=0) >= 0.0, 1.0, -1.0)
            t_oriented = t_raw * t_signs[None, None, :]
            teacher_refs[r] = (t_oriented - t_oriented.mean(axis=(0, 1), keepdims=True)) / (
                t_oriented.std(axis=(0, 1), keepdims=True) + 1e-6
            )
            teacher_rgbs.append(align_and_upsample_pca_rgb(t_raw, None, out_size=out_size))

    def fmt_evr(key: str) -> str:
        evr = np.round(data[f"{key}_evr"].mean(axis=0) * 100.0).astype(int)
        return f"PC1/2/3 = {evr[0]}/{evr[1]}/{evr[2]} %"

    student_defs = [
        ("baseline", "Baseline Student\nepoch 39 (12×12)"),
        ("warmstart", "+ VP Warm-Start\nStage 1 (12×12)"),
        ("stage2", "+ Dense Affinity\nStage 2, ep 31 (12×12)"),
        ("stage3", "+ HiRes & Masked\nStage 3, ep 36 (16×16)"),
    ]
    student_rgbs: dict[str, list[np.ndarray]] = {}
    for key, _ in student_defs:
        maps = []
        for r in range(n_rows):
            raw = data[f"{key}_proj_3d"][r]
            side = int(round(math.sqrt(raw.shape[0])))
            maps.append(align_and_upsample_pca_rgb(raw.reshape(side, side, 3), teacher_refs[r], out_size=out_size))
        student_rgbs[key] = maps

    ev_panels = [render_event_panel(events_u8[r], out_size=out_size) for r in range(n_rows)]

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    def add_group_brackets_and_footer(
        fig,
        axes_top_row,
        groups: list[tuple[int, int, str, str]],
        footer_text: str,
    ) -> None:
        """Draw clean spanning group brackets above the top row and an explanatory footer below."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top_y = max(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y1
            for ax in axes_top_row
        )
        bar_y = top_y + 0.010
        text_y = bar_y + 0.008
        for c_start, c_end, label, color in groups:
            x0 = axes_top_row[c_start].get_position().x0 + 0.004
            x1 = axes_top_row[c_end].get_position().x1 - 0.004
            fig.lines.append(
                plt.Line2D(
                    [x0, x1],
                    [bar_y, bar_y],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                )
            )
            fig.text(
                0.5 * (x0 + x1),
                text_y,
                label,
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color=color,
            )

        left_x = axes_top_row[0].get_position().x0
        right_x = axes_top_row[-1].get_position().x1
        fig.text(
            0.5 * (left_x + right_x),
            0.012,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#222222",
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f8fa", "edgecolor": "#d0d7de", "linewidth": 0.8},
        )

    # --- Figure A: Compact 5-column layout directly contrasting DINO vs. Ours ---
    compact_cols = [
        ("(a) Input Events\nON / OFF (48×48)", ev_panels, "#222222", "normal"),
        (
            f"(b) DINOv2 Teacher\nANN · 16×16 tokens\n{fmt_evr('teacher')}" if has_teacher else f"(b) Ours (Baseline)\n12×12 tokens\n{fmt_evr('baseline')}",
            teacher_rgbs if has_teacher else student_rgbs["baseline"],
            "#3b4992",
            "bold",
        ),
        (f"(c) Ours (Baseline SNN)\nw/o dense affinity (12×12)\n{fmt_evr('baseline')}", student_rgbs["baseline"], "#222222", "normal"),
        (f"(d) Ours (Stage 2 SNN)\n+ dense affinity (12×12)\n{fmt_evr('stage2')}", student_rgbs["stage2"], "#222222", "normal"),
        (f"(e) Ours Final (Stage 3 SNN)\n+ HiRes & masked (16×16)\n{fmt_evr('stage3')}", student_rgbs["stage3"], "#0b6e4f", "bold"),
    ]
    fig_c, axes_c = plt.subplots(
        n_rows,
        len(compact_cols),
        figsize=(2.32 * len(compact_cols), 2.16 * n_rows + 0.85),
        dpi=300,
        gridspec_kw={"wspace": 0.065, "hspace": 0.06},
    )
    fig_c.subplots_adjust(top=0.89, bottom=0.075)
    for col_idx, (title, panels, col_color, weight) in enumerate(compact_cols):
        axes_c[0, col_idx].set_title(title, fontsize=8.7, pad=9, linespacing=1.24, color=col_color, fontweight=weight)
        for row_idx in range(n_rows):
            ax = axes_c[row_idx, col_idx]
            ax.imshow(panels[row_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(
                    row_names[row_idx],
                    fontsize=10.0,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )
    if has_teacher:
        add_group_brackets_and_footer(
            fig_c,
            list(axes_c[0]),
            [
                (0, 0, "Event Input", "#444444"),
                (1, 1, "DINO (ANN Teacher)", "#3b4992"),
                (2, 4, "Ours: PureSpikeFormer (Event-Only Binary-Spike SNN)", "#0b6e4f"),
            ],
            "DINO vs. Ours Guide:  (b) DINOv2 Teacher = continuous floating-point DINOv2 (ViT-S/14) foundation model (training target only).\n"
            "(c)–(e) Ours (PureSpikeFormer) = event-only pure binary-spike SNN students: (c) global-only baseline, (d) Stage 2 (+ dense patch affinity), (e) Ours Final Stage 3 (16×16 + masked completion).",
        )
    compact_png = out_dir / "dinov3_fig13_ncaltech101_compact.png"
    compact_pdf = out_dir / "dinov3_fig13_ncaltech101_compact.pdf"
    fig_c.savefig(compact_png, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig_c.savefig(compact_pdf, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig_c)
    saved["compact_png"] = str(compact_png)
    saved["compact_pdf"] = str(compact_pdf)

    # --- Figure B: Full 7-column DINOv3 Fig. 13 comparison with Paired RGB & Teacher reference ---
    if has_teacher:
        full_cols = [
            ("(a) Paired RGB\n(reference only)", [rgb_u8[r] for r in range(n_rows)], "#222222", "normal"),
            ("(b) Input Events\nON / OFF (48×48)", ev_panels, "#222222", "normal"),
            (f"(c) DINOv2 Teacher\nANN · 16×16 tokens\n{fmt_evr('teacher')}", teacher_rgbs, "#3b4992", "bold"),
            (f"(d) Ours (Baseline)\n12×12 · kNN 45.9%\n{fmt_evr('baseline')}", student_rgbs["baseline"], "#222222", "normal"),
            (f"(e) Ours (+ Warm-Start)\n12×12 · kNN 50.7%\n{fmt_evr('warmstart')}", student_rgbs["warmstart"], "#222222", "normal"),
            (f"(f) Ours (Stage 2)\n12×12 · kNN 61.4%\n{fmt_evr('stage2')}", student_rgbs["stage2"], "#222222", "normal"),
            (f"(g) Ours Final (Stage 3)\n16×16 · kNN 63.7%\n{fmt_evr('stage3')}", student_rgbs["stage3"], "#0b6e4f", "bold"),
        ]
        fig_f, axes_f = plt.subplots(
            n_rows,
            len(full_cols),
            figsize=(2.32 * len(full_cols), 2.16 * n_rows + 0.85),
            dpi=300,
            gridspec_kw={"wspace": 0.065, "hspace": 0.06},
        )
        fig_f.subplots_adjust(top=0.89, bottom=0.075)
        for col_idx, (title, panels, col_color, weight) in enumerate(full_cols):
            axes_f[0, col_idx].set_title(
                title,
                fontsize=8.6,
                pad=9,
                linespacing=1.24,
                color=col_color,
                fontweight=weight,
            )
            for row_idx in range(n_rows):
                ax = axes_f[row_idx, col_idx]
                ax.imshow(panels[row_idx])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if col_idx == 0:
                    ax.set_ylabel(
                        row_names[row_idx],
                        fontsize=10.0,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=10,
                    )
        add_group_brackets_and_footer(
            fig_f,
            list(axes_f[0]),
            [
                (0, 1, "Input & Paired Reference", "#444444"),
                (2, 2, "DINO (ANN Teacher)", "#3b4992"),
                (3, 6, "Ours: PureSpikeFormer Students (Event-Only Binary-Spike SNN, Ablation → Final)", "#0b6e4f"),
            ],
            "DINO vs. Ours Guide:  (c) DINOv2 Teacher = continuous floating-point DINOv2 (ViT-S/14) teacher (used only during training; not deployed at inference).\n"
            "(d)–(g) Ours (PureSpikeFormer SNN) = our event-only pure binary-spike models: (d) Baseline, (e) + VP Warm-Start, (f) Stage 2 (+ dense patch affinity), and (g) Ours Final Stage 3 (16×16 + masked completion).",
        )
        full_png = out_dir / "dinov3_fig13_ncaltech101_pca_rgb.png"
        full_pdf = out_dir / "dinov3_fig13_ncaltech101_pca_rgb.pdf"
        fig_f.savefig(full_png, bbox_inches="tight", pad_inches=0.10, facecolor="white")
        fig_f.savefig(full_pdf, bbox_inches="tight", pad_inches=0.10, facecolor="white")
        plt.close(fig_f)
        saved["full_png"] = str(full_png)
        saved["full_pdf"] = str(full_pdf)

    print(json.dumps(saved, indent=2))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DINOv3 Fig. 13 style dense PCA-to-RGB figures.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--from-bundle", type=Path, default=None, help="Render directly from an extracted .npz bundle.")
    parser.add_argument("--bundle-out", type=Path, default=Path("runs/vis/dinov3_ncaltech101_bundle.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("image"))
    parser.add_argument("--render", action="store_true", help="Render PNG/PDF figures after bundle extraction.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-teacher", action="store_true", default=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("data/teacher_cache/ncaltech101/student_split.json"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/ncaltech101_rgb_event/frames_ncaltech101_t16_s48"))
    parser.add_argument("--rgb-root", type=Path, default=Path("data/ncaltech101_rgb_event"))
    parser.add_argument("--paired-manifest", type=Path, default=Path("data/ncaltech101_rgb_event/manifests/paired_index.json"))
    parser.add_argument("--display-ids", nargs="+", default=DEFAULT_DISPLAY_IDS)
    parser.add_argument(
        "--baseline-ckpt",
        type=Path,
        default=Path("iclr_execution_20260907/runs/full_ncaltech101_learned_proxy_s42/best.pt"),
    )
    parser.add_argument(
        "--warmstart-ckpt",
        type=Path,
        default=Path("iclr_execution_20260907/routeA_round4_20261108/runs/stage3_full_20ep/best_purespike_dualsal.pt"),
    )
    parser.add_argument(
        "--stage2-ckpt",
        type=Path,
        default=Path("iclr_execution_20260907/overnight_20261111/runs/stage2_nextgen_dino36911_45ep/best_purespike.pt"),
    )
    parser.add_argument(
        "--stage3-ckpt",
        type=Path,
        default=Path("iclr_execution_20260907/overnight_20261111/runs/stage3_hires256_masked_45ep/best_purespike.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_bundle is not None:
        render_figures_from_bundle(args.from_bundle, args.output_dir)
        return
    bundle_path = extract_bundle_on_gpu(args)
    if args.render:
        render_figures_from_bundle(bundle_path, args.output_dir)


if __name__ == "__main__":
    main()
