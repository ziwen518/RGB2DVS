#!/usr/bin/env python3
"""Generate publication-grade architecture diagram for Next-Gen HiRes PureSpikeFormer.

Outputs crisp vector PDF and high-DPI PNG in `image/`:
  - image/architecture_nextgen_purespikeformer.pdf
  - image/architecture_nextgen_purespikeformer.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


# Publication color palette (ICLR / CVPR modern clean academic palette)
PALETTE = {
    "bg_canvas": "#FFFFFF",
    # Teacher / Training-only zone (Indigo / Slate)
    "teacher_bg": "#F6F8FF",
    "teacher_border": "#4F46E5",
    "teacher_card": "#FFFFFF",
    "teacher_accent": "#EEF2FF",
    "teacher_header": "#3730A3",
    # Student / Deployed SNN zone (Teal / Emerald)
    "student_bg": "#F2FBF9",
    "student_border": "#0F766E",
    "student_card": "#FFFFFF",
    "student_accent": "#CCFBF1",
    "student_header": "#115E59",
    # Spike / Neuromorphic highlights (Amber / Gold)
    "spike_bg": "#FFFBEB",
    "spike_border": "#D97706",
    "spike_fill": "#FEF3C7",
    "spike_text": "#92400E",
    # Losses / Supervision (Rose / Crimson)
    "loss_bg": "#FFF1F2",
    "loss_border": "#E11D48",
    "loss_header": "#9F1239",
    # Detail / Bottom panels (Neutral Cool Slate)
    "detail_bg": "#F8FAFC",
    "detail_border": "#475569",
    "detail_header": "#1E293B",
    # Text & connectors
    "ink": "#0F172A",
    "muted": "#475569",
    "light_border": "#CBD5E1",
    "arrow_main": "#1E293B",
    "arrow_train": "#4F46E5",
    "arrow_loss": "#E11D48",
    "arrow_sal": "#0D9488",
}


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "#FFFFFF",
    ec: str = "#CBD5E1",
    lw: float = 1.2,
    ls: str = "-",
    radius: float = 1.2,
    zorder: int = 2,
    shadow: bool = False,
) -> FancyBboxPatch:
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.30, y - 0.32),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor="#0F172A",
            edgecolor="none",
            alpha=0.055,
            zorder=zorder - 1,
        )
        ax.add_patch(sh)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_pill(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    tc: str,
    fontsize: float = 8.0,
    weight: str = "bold",
    ls: str = "-",
    lw: float = 1.1,
    zorder: int = 6,
) -> None:
    add_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, radius=h * 0.48, zorder=zorder)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=tc,
        zorder=zorder + 1,
    )


def add_arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = "#1E293B",
    lw: float = 1.45,
    ls: str = "-",
    rad: float = 0.0,
    scale: float = 11.0,
    zorder: int = 5,
) -> None:
    arr = FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"-|>,head_length={scale*0.045:.2f},head_width={scale*0.03:.2f}",
        mutation_scale=scale,
        linewidth=lw,
        linestyle=ls,
        edgecolor=color,
        facecolor=color,
        zorder=zorder,
    )
    ax.add_patch(arr)


def add_ortho_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = "#1E293B",
    lw: float = 1.4,
    ls: str = "-",
    scale: float = 11.0,
    zorder: int = 5,
) -> None:
    if len(points) < 2:
        return
    for i in range(len(points) - 2):
        ax.plot(
            [points[i][0], points[i + 1][0]],
            [points[i][1], points[i + 1][1]],
            color=color,
            lw=lw,
            ls=ls,
            solid_capstyle="round",
            zorder=zorder,
        )
    add_arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls, scale=scale, zorder=zorder)


def draw_event_stack_icon(
    ax: plt.Axes, x: float, y: float, w: float, h: float, *, masked: bool = False, zorder: int = 4
) -> None:
    """Draw a crisp 3-slice ON/OFF polarity event tensor icon (optionally block-masked)."""
    n_slices = 3
    dx, dy = 0.85, 0.72
    sw = w - dx * (n_slices - 1)
    sh = h - dy * (n_slices - 1)
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[-1:1:16j, -1:1:16j]
    fuselage = np.exp(-((yy * 3.8) ** 2 + (xx * 1.15) ** 2))
    wings = np.exp(-(((xx + 0.05) * 3.5) ** 2 + (yy * 1.25) ** 2))
    tail = np.exp(-(((xx - 0.55) * 4.5) ** 2 + ((yy - 0.25) * 2.6) ** 2))
    shape = np.clip(fuselage + 0.85 * wings + 0.6 * tail, 0.0, 1.0)
    pol = np.gradient(shape, axis=1) + 0.65 * np.gradient(shape, axis=0)

    for s in range(n_slices):
        sx = x + s * dx
        sy = y + (n_slices - 1 - s) * dy
        add_box(ax, sx, sy, sw, sh, fc="#0F172A", ec="#475569", lw=0.85, radius=0.42, zorder=zorder + s)
        if s == n_slices - 1:
            grid_n = 8
            cw, ch = (sw - 0.56) / grid_n, (sh - 0.56) / grid_n
            sub = pol[::2, ::2]
            mask_cells = {(2, 3), (2, 4), (5, 2), (5, 5), (1, 6)} if masked else set()
            for r in range(grid_n):
                for c in range(grid_n):
                    cx = sx + 0.28 + c * cw
                    cy = sy + 0.28 + (grid_n - 1 - r) * ch
                    if (r, c) in mask_cells:
                        ax.add_patch(
                            Rectangle(
                                (cx, cy), cw * 0.92, ch * 0.92,
                                fc="#334155", ec="#94A3B8", lw=0.35, zorder=zorder + 5,
                            )
                        )
                        continue
                    v = float(sub[r, c]) + float(rng.normal(0, 0.03))
                    if v > 0.045:
                        col = "#FB7185" if v < 0.14 else "#F43F5E"
                    elif v < -0.045:
                        col = "#60A5FA" if v > -0.14 else "#3B82F6"
                    else:
                        continue
                    ax.add_patch(Rectangle((cx, cy), cw * 0.88, ch * 0.88, fc=col, ec="none", zorder=zorder + 5))


def draw_rgb_icon(ax: plt.Axes, x: float, y: float, w: float, h: float, *, zorder: int = 4) -> None:
    """Draw a clean stylized paired RGB image thumbnail."""
    add_box(ax, x, y, w, h, fc="#E0F2FE", ec="#64748B", lw=0.9, radius=0.45, zorder=zorder)
    ax.add_patch(Rectangle((x + 0.25, y + 0.25), w - 0.5, (h - 0.5) * 0.42, fc="#BAE6FD", ec="none", zorder=zorder + 1))
    cx, cy = x + w * 0.5, y + h * 0.55
    body = Polygon(
        [
            (cx - w * 0.34, cy - h * 0.04),
            (cx + w * 0.32, cy + h * 0.04),
            (cx - w * 0.30, cy + h * 0.10),
        ],
        closed=True,
        fc="#F97316",
        ec="#C2410C",
        lw=0.6,
        zorder=zorder + 2,
    )
    wing = Polygon(
        [
            (cx - w * 0.06, cy + h * 0.28),
            (cx + w * 0.10, cy + h * 0.02),
            (cx - w * 0.12, cy - h * 0.25),
        ],
        closed=True,
        fc="#EA580C",
        ec="#9A3412",
        lw=0.6,
        zorder=zorder + 3,
    )
    ax.add_patch(body)
    ax.add_patch(wing)


def draw_dense_pca_grid(
    ax: plt.Axes, x: float, y: float, w: float, h: float, grid: int = 8, *, seed: int = 0, zorder: int = 4
) -> None:
    """Draw a vibrant DINOv3-style dense semantic PCA grid thumbnail."""
    add_box(ax, x, y, w, h, fc="#FFFFFF", ec="#475569", lw=0.85, radius=0.4, zorder=zorder)
    pad = 0.26
    cw = (w - 2 * pad) / grid
    ch = (h - 2 * pad) / grid
    yy, xx = np.mgrid[-1:1:complex(0, grid), -1:1:complex(0, grid)]
    fg = np.exp(-((yy * 1.9) ** 2 + (xx * 1.5) ** 2))
    part = np.exp(-(((xx + 0.15) * 2.6) ** 2 + ((yy - 0.1) * 1.4) ** 2))
    for r in range(grid):
        for c in range(grid):
            f = float(fg[r, c])
            p = float(part[r, c])
            if seed == 0:
                rgb = (
                    np.clip(0.22 + 0.72 * f, 0, 1),
                    np.clip(0.82 - 0.50 * f + 0.25 * p, 0, 1),
                    np.clip(0.35 + 0.58 * p, 0, 1),
                )
            else:
                rgb = (
                    np.clip(0.28 + 0.66 * p, 0, 1),
                    np.clip(0.32 + 0.56 * f, 0, 1),
                    np.clip(0.86 - 0.48 * f, 0, 1),
                )
            ax.add_patch(
                Rectangle(
                    (x + pad + c * cw, y + pad + (grid - 1 - r) * ch),
                    cw * 0.92,
                    ch * 0.92,
                    fc=rgb,
                    ec="none",
                    zorder=zorder + 1,
                )
            )


def draw_binary_spike_strip(
    ax: plt.Axes, x: float, y: float, w: float, h: float, bits: list[str], *, zorder: int = 4
) -> None:
    """Draw a vertical binary {0, 1} spike token column."""
    n = len(bits)
    ch = h / n
    for i, b in enumerate(bits):
        by = y + (n - 1 - i) * ch
        is_one = b == "1"
        fc = "#FEF3C7" if is_one else "#FFFFFF"
        tc = "#B45309" if is_one else "#475569"
        ax.add_patch(Rectangle((x, by), w, ch, fc=fc, ec="#94A3B8", lw=0.8, zorder=zorder))
        ax.text(
            x + w / 2.0,
            by + ch / 2.0,
            b,
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=tc,
            zorder=zorder + 1,
        )


def build_architecture_figure(out_pdf: Path, out_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
        }
    )

    fig, ax = plt.subplots(figsize=(19.2, 11.1), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg_canvas"])
    ax.set_facecolor(PALETTE["bg_canvas"])
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 111)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # 0. TOP TITLE & GLOBAL LEGEND BAR (y = 104.8 .. 109.6)
    # =========================================================================
    ax.text(
        2.4,
        108.0,
        "Next-Gen HiRes PureSpikeFormer Architecture",
        fontsize=14.0,
        fontweight="bold",
        color=PALETTE["ink"],
        va="center",
    )
    ax.text(
        2.4,
        105.4,
        "Label-Free Dense Observable Foundation Distillation, Energy-Compensated Masked Event Completion & Pure-Spike Readout",
        fontsize=8.6,
        color=PALETTE["muted"],
        va="center",
    )

    # Top-right legend pills
    add_pill(
        ax, 108.5, 106.2, 26.2, 2.8,
        "Training-Only (Discarded)",
        fc=PALETTE["teacher_bg"], ec=PALETTE["teacher_border"], tc=PALETTE["teacher_header"],
        ls="--", fontsize=7.5,
    )
    add_pill(
        ax, 135.8, 106.2, 23.6, 2.8,
        "Deployed Event Student",
        fc=PALETTE["student_bg"], ec=PALETTE["student_border"], tc=PALETTE["student_header"],
        fontsize=7.5,
    )
    add_pill(
        ax, 160.5, 106.2, 14.2, 2.8,
        "{0, 1} Spikes",
        fc=PALETTE["spike_fill"], ec=PALETTE["spike_border"], tc=PALETTE["spike_text"],
        fontsize=7.5,
    )
    add_pill(
        ax, 175.8, 106.2, 13.8, 2.8,
        "Distill Losses",
        fc=PALETTE["loss_bg"], ec=PALETTE["loss_border"], tc=PALETTE["loss_header"],
        fontsize=7.5,
    )

    # =========================================================================
    # 1. TIER 1: TRAINING-ONLY DENSE FOUNDATION TEACHER & LOSS HUB (y = 70.0 .. 103.2)
    # =========================================================================
    add_box(
        ax, 2.2, 70.0, 187.4, 33.2,
        fc=PALETTE["teacher_bg"], ec=PALETTE["teacher_border"],
        lw=1.5, ls="--", radius=1.8, zorder=1,
    )
    add_pill(
        ax, 4.8, 99.8, 96.5, 2.7,
        "STAGE A · Training-Only Dense Observable Teacher & Target Bank  (Label-Free U1 · Discarded at Inference)",
        fc="#E0E7FF", ec=PALETTE["teacher_border"], tc=PALETTE["teacher_header"],
        fontsize=7.8, zorder=6,
    )

    # A1. Paired RGB + Event Stream (Train Only) (x = 4.5 .. 28.5)
    add_box(ax, 4.5, 72.2, 24.0, 26.0, fc="#FFFFFF", ec="#C7D2FE", lw=1.1, radius=1.2, shadow=True)
    ax.text(16.5, 96.1, "Paired Train Stream", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["ink"])
    ax.text(16.5, 93.9, "(train_ids only · no labels)", ha="center", va="center", fontsize=7.0, color=PALETTE["muted"])
    draw_rgb_icon(ax, 6.2, 84.8, 8.0, 7.0)
    ax.text(20.4, 89.2, "Paired\nRGB Frame", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#334155")
    ax.text(20.4, 85.8, r"$I_i \in \mathbb{R}^{3\times H\times W}$", ha="center", va="center", fontsize=6.9, color=PALETTE["muted"])
    draw_event_stack_icon(ax, 6.2, 74.8, 8.0, 7.2, masked=False)
    ax.text(20.4, 79.4, "Aligned\nON/OFF Events", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#334155")
    ax.text(20.4, 75.9, r"$X_i \in \mathbb{R}^{16\times 2\times 48^2}$", ha="center", va="center", fontsize=6.9, color=PALETTE["muted"])

    # A2-Left: Pretrained DINOv2 (ViT-S/14) (x = 31.8 .. 53.8, y = 74.4 .. 98.2)
    # Placed side-by-side with Observable Bridge so downward Weight-Init arrow has zero crossings!
    add_box(ax, 31.8, 74.4, 22.0, 23.8, fc="#FFFFFF", ec=PALETTE["teacher_border"], lw=1.2, radius=1.2, shadow=True)
    add_pill(ax, 37.3, 94.8, 11.0, 2.2, "FROZEN", fc="#DBEAFE", ec="#2563EB", tc="#1D4ED8", fontsize=6.8)
    ax.text(42.8, 91.6, "Pretrained\nDINOv2 (ViT-S/14)", ha="center", va="center", fontsize=8.3, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(42.8, 86.2, "Extracts multi-layer\nRGB CLS & dense\n16×16 patch tokens", ha="center", va="center", fontsize=7.0, color=PALETTE["ink"])
    add_box(ax, 33.2, 75.7, 19.2, 6.4, fc="#EEF2FF", ec="#818CF8", lw=0.85, ls="--", radius=0.7)
    ax.text(42.8, 80.1, "Late-Block Weight Init", ha="center", va="center", fontsize=6.9, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(42.8, 77.5, "Blocks [3, 6, 9, 11]\n→ Student [1, 2, 3, 4]", ha="center", va="center", fontsize=6.6, color=PALETTE["ink"])

    # A2-Right: Observable RGB-Event Bridge & EventProxy (x = 56.8 .. 78.2, y = 74.4 .. 98.2)
    add_box(ax, 56.8, 74.4, 21.4, 23.8, fc="#FFFFFF", ec=PALETTE["teacher_border"], lw=1.2, radius=1.2, shadow=True)
    ax.text(67.5, 94.6, "Observable Bridge\n& EventProxy", ha="center", va="center", fontsize=8.3, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(
        67.5, 85.8,
        "Aligns RGB semantics\nwith DVS event stream;\nfilters static unobservable\ncolor/texture & retains\nmotion-supported\nobject structure.",
        ha="center", va="center", fontsize=6.95, color=PALETTE["ink"],
    )
    add_pill(ax, 58.6, 75.8, 17.8, 2.3, "Label-Free Transport", fc="#EEF2FF", ec="#818CF8", tc=PALETTE["teacher_header"], fontsize=6.7)

    # Clean non-crossing arrows A1 -> DINOv2 -> Bridge
    add_arrow(ax, (28.5, 88.8), (31.8, 88.8), color=PALETTE["arrow_train"], lw=1.4)
    add_arrow(ax, (53.8, 88.8), (56.8, 88.8), color=PALETTE["arrow_train"], lw=1.4)
    # Aligned events from A1 routed cleanly beneath DINOv2 into Bridge!
    add_ortho_arrow(
        ax,
        [(28.5, 76.2), (30.2, 76.2), (30.2, 72.4), (67.5, 72.4), (67.5, 74.4)],
        color=PALETTE["arrow_train"],
        lw=1.25,
    )

    # A3. Multi-Granularity Dense Target Cache (x = 81.4 .. 126.4)
    add_box(ax, 81.4, 72.2, 45.0, 26.0, fc="#FFFFFF", ec="#C7D2FE", lw=1.15, radius=1.2, shadow=True)
    ax.text(103.9, 96.1, "Train-Only Dense Target Cache", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["teacher_header"])

    # Sub-card 1: Global target + memory bank
    add_box(ax, 83.2, 87.8, 41.4, 6.2, fc=PALETTE["teacher_accent"], ec="#818CF8", lw=0.95, radius=0.8)
    ax.text(84.7, 92.0, "Global Transported Target & Full Bank", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(84.7, 89.4, r"$z_i^{\mathrm{obs}} \in \mathbb{R}^{384},\;\; B_{\mathrm{tr}} \in \mathbb{R}^{N_{\mathrm{tr}}\times 384},\;\; c_{11} \in \mathbb{R}^{384}$", ha="left", va="center", fontsize=7.1, color=PALETTE["ink"])

    # Sub-card 2: 16x16 HiRes Patch Target
    add_box(ax, 83.2, 80.5, 41.4, 6.2, fc=PALETTE["teacher_accent"], ec="#818CF8", lw=0.95, radius=0.8)
    draw_dense_pca_grid(ax, 118.4, 81.1, 5.0, 5.0, grid=6, seed=1)
    ax.text(84.7, 84.7, "HiRes Dense Patch Target (1-to-1)", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(84.7, 82.1, r"$P_{12} \in \mathbb{R}^{16\times 16\times 384}$  (Block 12 tokens)", ha="left", va="center", fontsize=7.1, color=PALETTE["ink"])

    # Sub-card 3: 12x12 Mid-level Patch Target
    add_box(ax, 83.2, 73.3, 41.4, 6.1, fc=PALETTE["teacher_accent"], ec="#818CF8", lw=0.95, radius=0.8)
    ax.text(84.7, 77.4, "Mid-Layer Structural Patch Target", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["teacher_header"])
    ax.text(84.7, 74.9, r"$P_{10} \in \mathbb{R}^{12\times 12\times 384}$  (Block 10 tokens)", ha="left", va="center", fontsize=7.1, color=PALETTE["ink"])

    add_arrow(ax, (78.2, 86.5), (81.4, 86.5), color=PALETTE["arrow_train"], lw=1.4)

    # A4. Multi-Granularity Label-Free Distillation Losses (x = 130.2 .. 186.2)
    add_box(ax, 130.2, 72.2, 56.0, 26.0, fc="#FFFFFF", ec="#FDA4AF", lw=1.2, radius=1.2, shadow=True)
    ax.text(158.2, 96.1, "Multi-Granularity Label-Free Distillation", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["loss_header"])

    # Loss 1: Global + Neighbor KL + Mid
    add_box(ax, 132.0, 87.8, 52.4, 6.2, fc=PALETTE["loss_bg"], ec=PALETTE["loss_border"], lw=1.0, radius=0.8)
    ax.text(133.5, 92.0, r"$\mathcal{L}_{\mathrm{global}} = 2.2\,\mathcal{L}_{\mathrm{align}} + \mathcal{L}_{\mathrm{geom}} + 0.7\,\mathcal{L}_{\mathrm{nbr}} + 0.3\,\mathcal{L}_{\mathrm{mid}}$", ha="left", va="center", fontsize=7.45, fontweight="bold", color=PALETTE["loss_header"])
    ax.text(133.5, 89.4, "Cosine align + relational Gram + dual-temp bank KL", ha="left", va="center", fontsize=7.0, color=PALETTE["ink"])

    # Loss 2: Saliency-weighted Centered Patch Affinity
    add_box(ax, 132.0, 80.5, 52.4, 6.2, fc=PALETTE["loss_bg"], ec=PALETTE["loss_border"], lw=1.0, radius=0.8)
    ax.text(133.5, 84.7, r"$\mathcal{L}_{\mathrm{patch\text{-}aff}}$: Centered Saliency Patch Affinity", ha="left", va="center", fontsize=7.45, fontweight="bold", color=PALETTE["loss_header"])
    ax.text(133.5, 82.1, r"Spatially centered $16^2$ & $12^2$ Gram weighted by $w_{\mathrm{sal}}w_{\mathrm{sal}}^\top$", ha="left", va="center", fontsize=7.0, color=PALETTE["ink"])

    # Loss 3: Clean-Masked View Invariance
    add_box(ax, 132.0, 73.3, 52.4, 6.1, fc=PALETTE["loss_bg"], ec=PALETTE["loss_border"], lw=1.0, radius=0.8)
    ax.text(133.5, 77.4, r"$\mathcal{L}_{\mathrm{inv}}$: Masked–Clean View Invariance", ha="left", va="center", fontsize=7.45, fontweight="bold", color=PALETTE["loss_header"])
    ax.text(133.5, 74.9, r"$1 - \cos(e_{\mathrm{mask}},\,\mathrm{sg}(e_{\mathrm{clean}}))$ on readout & core spikes", ha="left", va="center", fontsize=7.0, color=PALETTE["ink"])

    # Target Cache -> Losses arrows
    add_arrow(ax, (124.6, 90.9), (132.0, 90.9), color=PALETTE["arrow_loss"], lw=1.35)
    add_arrow(ax, (124.6, 83.6), (132.0, 83.6), color=PALETTE["arrow_loss"], lw=1.35)
    add_ortho_arrow(ax, [(124.6, 76.3), (128.3, 76.3), (128.3, 82.2), (132.0, 82.2)], color=PALETTE["arrow_loss"], lw=1.15, ls="--")

    # =========================================================================
    # 2. TIER 2: DEPLOYED EVENT-ONLY STUDENT — HIRES PURESPIKEFORMER (y = 28.2 .. 65.5)
    # =========================================================================
    add_box(
        ax, 2.2, 28.2, 187.4, 37.3,
        fc=PALETTE["student_bg"], ec=PALETTE["student_border"],
        lw=1.8, ls="-", radius=1.8, zorder=1,
    )
    add_pill(
        ax, 4.8, 62.3, 95.5, 2.7,
        "STAGE B · Deployed Event-Only Student: HiRes PureSpikeFormer  (100% Binary {0, 1} Inter-Block Spikes)",
        fc=PALETTE["student_accent"], ec=PALETTE["student_border"], tc=PALETTE["student_header"],
        fontsize=7.8, zorder=6,
    )

    # B1. Event Stream + Energy-Compensated Spatial Block Masking (x = 4.5 .. 35.8)
    add_box(ax, 4.5, 31.8, 31.3, 29.2, fc="#FFFFFF", ec="#99F6E4", lw=1.15, radius=1.2, shadow=True)
    ax.text(20.15, 58.9, "Event Input & Views", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["student_header"])

    # Clean stream (top half of B1)
    draw_event_stack_icon(ax, 6.0, 48.6, 8.0, 7.4, masked=False)
    ax.text(15.2, 54.7, "Clean Event Stream", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(15.2, 52.2, r"$X \in \mathbb{R}^{T\times 2\times 48\times 48}$", ha="left", va="center", fontsize=7.1, color=PALETTE["ink"])
    ax.text(15.2, 49.8, "T = 16, ON/OFF polarity", ha="left", va="center", fontsize=6.8, color=PALETTE["muted"])

    # Masked augmented view (bottom half of B1, dashed training badge)
    add_box(ax, 5.7, 33.0, 28.9, 14.0, fc="#FFFBEB", ec="#F59E0B", lw=1.0, ls="--", radius=0.9)
    draw_event_stack_icon(ax, 6.6, 34.8, 7.2, 6.8, masked=True)
    ax.text(20.15, 45.1, "Train View 2: Energy-Masked", ha="center", va="center", fontsize=7.2, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(14.6, 41.9, "• Shift ±3 px (p_flip = 0)", ha="left", va="center", fontsize=6.65, color=PALETTE["ink"])
    ax.text(14.6, 39.5, "• 3×3 block drop (18%)", ha="left", va="center", fontsize=6.65, color=PALETTE["ink"])
    ax.text(14.6, 37.1, "• L1 Scale [1.0, 1.35]", ha="left", va="center", fontsize=6.65, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(14.6, 34.7, "  preserves BNTT rate!", ha="left", va="center", fontsize=6.55, color=PALETTE["muted"])

    # B2. Local Spiking Stem (patch_size = 3) (x = 39.0 .. 68.8)
    add_box(ax, 39.0, 33.6, 29.8, 27.4, fc="#FFFFFF", ec=PALETTE["student_border"], lw=1.2, radius=1.2, shadow=True)
    ax.text(53.9, 58.8, "Local Spiking Stem", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["student_header"])
    ax.text(53.9, 56.5, "(patch_size = 3 · per step t)", ha="center", va="center", fontsize=7.0, color=PALETTE["muted"])

    add_box(ax, 40.8, 48.3, 26.2, 6.5, fc="#F8FAFC", ec="#94A3B8", lw=0.95, radius=0.8)
    ax.text(53.9, 52.6, "3×3 Local Conv (2 → 64)", ha="center", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(53.9, 50.0, "BNTT (×16)  →  PLIF {0, 1}", ha="center", va="center", fontsize=7.2, color=PALETTE["spike_text"])

    add_arrow(ax, (53.9, 48.3), (53.9, 45.7), color=PALETTE["arrow_main"], lw=1.3)

    add_box(ax, 40.8, 35.2, 26.2, 10.3, fc="#F8FAFC", ec="#94A3B8", lw=0.95, radius=0.8)
    ax.text(53.9, 43.4, "3×3 Stride-3 Patch Conv", ha="center", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(53.9, 40.8, "BNTT (×16)  +  2D PosBias", ha="center", va="center", fontsize=7.2, color=PALETTE["ink"])
    ax.text(53.9, 37.5, "Patch PLIF  →  Binary Spikes", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["spike_text"])

    # Clean + Masked view merge into Stem
    add_arrow(ax, (35.8, 51.5), (39.0, 51.5), color=PALETTE["arrow_main"], lw=1.5)
    add_ortho_arrow(ax, [(34.6, 40.0), (37.2, 40.0), (37.2, 49.5), (39.0, 49.5)], color=PALETTE["spike_border"], lw=1.15, ls="--", scale=9.5)

    # Binary Spike Token Strip between B2 and B3 (x = 71.8 .. 75.1)
    draw_binary_spike_strip(ax, 71.8, 37.4, 3.3, 17.4, ["1", "0", "1", "...", "0", "1"])
    ax.text(73.45, 58.4, "Spike\nTokens", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(73.45, 34.8, "16×16 = 256\n(D = 384)", ha="center", va="center", fontsize=6.9, color=PALETTE["muted"])

    add_arrow(ax, (68.8, 46.1), (71.8, 46.1), color=PALETTE["arrow_main"], lw=1.45)

    # B3. 4x PureSpikeFormer Backbone (x = 78.2 .. 115.2)
    add_box(ax, 78.2, 33.6, 37.0, 27.4, fc="#FFFFFF", ec=PALETTE["spike_border"], lw=1.4, radius=1.2, shadow=True)
    ax.text(96.7, 58.8, "4 × PureSpikeFormer Blocks", ha="center", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(96.7, 56.5, "(D = 384, Heads = 6, T = 16)", ha="center", va="center", fontsize=7.0, color=PALETTE["muted"])

    blk_labels = [
        ("Block 1", "← DINOv2 Block 3", False),
        ("Block 2", "← DINOv2 Block 6  [Readout]", True),
        ("Block 3", "← DINOv2 Block 9  [Readout]", True),
        ("Block 4", "← DINOv2 Block 11 [Readout]", True),
    ]
    for idx, (bname, bsub, is_ro) in enumerate(blk_labels):
        by = 51.1 - idx * 4.2
        fc_b = "#FEF3C7" if is_ro else "#FFFBEB"
        ec_b = "#D97706" if is_ro else "#F59E0B"
        add_box(ax, 80.2, by, 33.0, 3.4, fc=fc_b, ec=ec_b, lw=1.0, radius=0.6)
        ax.text(81.6, by + 1.7, bname, ha="left", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["spike_text"])
        ax.text(112.0, by + 1.7, bsub, ha="right", va="center", fontsize=6.65, color="#78350F")

    ax.text(
        96.7, 35.6,
        "Global Binary QKV Attn + Local Mixer (d=1,2)\nAll inter-block tensors strictly ∈ {0, 1}",
        ha="center", va="center", fontsize=6.9, color=PALETTE["ink"],
    )

    add_arrow(ax, (75.1, 46.1), (78.2, 46.1), color=PALETTE["arrow_main"], lw=1.45)

    # Zero-crossing Weight Init arrow directly from DINOv2 bottom (42.8, 74.4) -> Corridor (y=67.6) -> Backbone!
    add_ortho_arrow(
        ax,
        [(42.8, 74.4), (42.8, 67.6), (96.7, 67.6), (96.7, 61.0)],
        color=PALETTE["arrow_train"],
        lw=1.35,
        ls="--",
    )
    add_pill(
        ax, 53.5, 66.4, 32.5, 2.4,
        "DINOv2 Blocks [3, 6, 9, 11] Weight Init",
        fc="#EEF2FF", ec=PALETTE["teacher_border"], tc=PALETTE["teacher_header"],
        fontsize=7.0, ls="--", zorder=8,
    )

    # B4. Multi-Depth Signed Spike Readout (x = 118.8 .. 150.8)
    add_box(ax, 118.8, 33.6, 32.0, 27.4, fc="#FFFFFF", ec=PALETTE["student_border"], lw=1.2, radius=1.2, shadow=True)
    ax.text(134.8, 58.8, "Multi-Depth Signed Readout", ha="center", va="center", fontsize=8.4, fontweight="bold", color=PALETTE["student_header"])

    add_box(ax, 120.6, 49.6, 28.4, 6.4, fc="#F0FDFA", ec="#5EEAD4", lw=0.95, radius=0.75)
    ax.text(134.8, 54.0, "Softmax Fusion (Blocks 2, 3, 4)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["student_header"])
    ax.text(134.8, 51.3, r"$\Sigma_{\ell=2}^{4}\,\alpha_\ell W_\ell S_t^{(\ell)} \;\to\; \mathrm{BNTT}_{2D}$", ha="center", va="center", fontsize=7.1, color=PALETTE["ink"])

    add_arrow(ax, (134.8, 49.6), (134.8, 47.4), color=PALETTE["arrow_main"], lw=1.25)

    add_box(ax, 120.6, 41.0, 28.4, 6.4, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=0.95, radius=0.75)
    ax.text(134.8, 45.3, "Dual-Channel Semantic PLIF", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(134.8, 42.7, r"$s_t^{\mathrm{sem}} = s_t^{+} - s_t^{-} \in \{-1, 0, +1\}$", ha="center", va="center", fontsize=7.1, color=PALETTE["ink"])

    add_arrow(ax, (134.8, 41.0), (134.8, 39.0), color=PALETTE["arrow_main"], lw=1.25)

    add_box(ax, 120.6, 34.6, 28.4, 4.4, fc="#F8FAFC", ec="#94A3B8", lw=0.9, radius=0.7)
    ax.text(134.8, 36.8, r"Learned Temporal Pool  $\Sigma_{t=1}^{16}\, w_t s_t^{\mathrm{sem}}$", ha="center", va="center", fontsize=7.1, fontweight="bold", color=PALETTE["ink"])

    # Multi-depth arrows from Blocks 2, 3, 4 -> B4
    for by_tap in (48.6, 44.4, 40.2):
        add_ortho_arrow(
            ax,
            [(113.2, by_tap), (116.8, by_tap), (116.8, 52.8), (120.6, 52.8)],
            color=PALETTE["student_border"],
            lw=1.15,
        )

    # B5. Dense Semantic Patch Map & Saliency-Blended Global Output (x = 154.0 .. 186.2)
    add_box(ax, 154.0, 47.7, 32.2, 13.3, fc="#FFFFFF", ec=PALETTE["student_border"], lw=1.25, radius=1.1, shadow=True)
    draw_dense_pca_grid(ax, 179.0, 49.3, 5.8, 5.8, grid=8, seed=0)
    ax.text(155.6, 58.6, "Dense Semantic Map", ha="left", va="center", fontsize=8.1, fontweight="bold", color=PALETTE["student_header"])
    ax.text(155.6, 55.7, r"$M \in \mathbb{R}^{16\times 16\times 384}$", ha="left", va="center", fontsize=7.5, color=PALETTE["ink"])
    ax.text(155.6, 52.7, "Sharp object & part", ha="left", va="center", fontsize=6.95, color=PALETTE["muted"])
    ax.text(155.6, 50.3, "tokens (DINOv3 Fig.13)", ha="left", va="center", fontsize=6.95, color=PALETTE["muted"])

    add_box(ax, 154.0, 31.8, 32.2, 14.3, fc="#FFFFFF", ec=PALETTE["student_border"], lw=1.25, radius=1.1, shadow=True)
    ax.text(170.1, 44.0, "Global Event Embedding", ha="center", va="center", fontsize=8.1, fontweight="bold", color=PALETTE["student_header"])
    ax.text(170.1, 41.2, r"$e \in \mathbb{R}^{384}$  (Saliency + 2×2 Quad)", ha="center", va="center", fontsize=7.2, color=PALETTE["ink"])
    ax.text(170.1, 38.4, r"$e_{\mathrm{blend}} = \mathrm{Norm}(e_{\mathrm{unif}} + \lambda\, e_{\mathrm{sal}})$", ha="center", va="center", fontsize=7.0, color=PALETTE["muted"])
    add_pill(
        ax, 155.4, 32.9, 29.4, 3.0,
        "N-Caltech101 U1 kNN: 63.7% / 64.9%",
        fc=PALETTE["student_accent"], ec=PALETTE["student_border"], tc=PALETTE["student_header"],
        fontsize=6.9,
    )

    # Arrows B4 -> B5
    add_ortho_arrow(ax, [(149.0, 36.8), (151.8, 36.8), (151.8, 53.8), (154.0, 53.8)], color=PALETTE["arrow_main"], lw=1.35)
    add_ortho_arrow(ax, [(149.0, 36.8), (151.8, 36.8), (151.8, 41.0), (154.0, 41.0)], color=PALETTE["arrow_main"], lw=1.35)

    # Detached Event Saliency routing bar along bottom of Stage B (y = 29.8)
    add_ortho_arrow(
        ax,
        [(20.15, 31.8), (20.15, 29.8), (170.1, 29.8), (170.1, 31.8)],
        color=PALETTE["arrow_sal"],
        lw=1.3,
        ls="-.",
    )
    add_pill(
        ax, 67.5, 28.55, 58.5, 2.5,
        r"Detached Event-Saliency Weights   $w_{\mathrm{sal}} = \mathrm{Pool}_{p\times p}(X_{\mathrm{sum}}) + \epsilon$",
        fc="#F0FDFA", ec=PALETTE["arrow_sal"], tc=PALETTE["student_header"],
        fontsize=6.9, zorder=7,
    )

    # Upward Distillation Supervision Arrows from Student (Stage B) -> Losses (Stage A)
    # 1) Dense Map M & Block 3/4 tokens -> L_patch-aff
    ax.plot([108.0, 108.0, 165.0], [61.0, 67.6, 67.6], color=PALETTE["arrow_loss"], lw=1.1, ls="--", zorder=5)
    add_ortho_arrow(
        ax,
        [(165.0, 61.0), (165.0, 67.6), (187.5, 67.6), (187.5, 83.6), (184.4, 83.6)],
        color=PALETTE["arrow_loss"],
        lw=1.25,
        ls="--",
    )
    add_pill(
        ax, 128.0, 66.4, 34.5, 2.4,
        r"Dense Tokens $(M,\, S^{(3,4)})$ Supervision",
        fc=PALETTE["loss_bg"], ec=PALETTE["loss_border"], tc=PALETTE["loss_header"],
        fontsize=6.9, ls="--", zorder=8,
    )

    # 2) Global clean/masked embeddings -> L_global & L_inv
    add_ortho_arrow(
        ax,
        [(178.0, 46.1), (178.0, 46.9), (188.5, 46.9), (188.5, 90.9), (184.4, 90.9)],
        color=PALETTE["arrow_loss"],
        lw=1.25,
        ls="--",
    )
    add_arrow(ax, (188.5, 76.3), (184.4, 76.3), color=PALETTE["arrow_loss"], lw=1.25, ls="--")

    # =========================================================================
    # 3. TIER 3: BOTTOM ZOOM-IN CIRCUITS & CALIBRATION (y = 2.0 .. 25.0)
    # =========================================================================
    # Clean zoom frustum lines strictly inside the y = 25.0 .. 28.2 gap!
    ax.plot([78.2, 90.5], [28.2, 25.0], color=PALETTE["spike_border"], lw=1.2, ls=":", zorder=2)
    ax.plot([115.2, 123.5], [28.2, 25.0], color=PALETTE["spike_border"], lw=1.2, ls=":", zorder=2)

    # C1. PureSpikeFormer Block Zoom-In (x = 2.2 .. 125.5, y = 2.0 .. 25.0)
    add_box(ax, 2.2, 2.0, 123.3, 23.0, fc=PALETTE["detail_bg"], ec=PALETTE["spike_border"], lw=1.4, radius=1.5, zorder=1)
    add_pill(
        ax, 4.8, 22.4, 83.5, 2.5,
        "ZOOM-IN · Single PureSpikeFormer Block Circuit  (Parallel Global Spike Attention + Local Structure Mixer)",
        fc=PALETTE["spike_fill"], ec=PALETTE["spike_border"], tc=PALETTE["spike_text"],
        fontsize=7.3, zorder=6,
    )

    # Input token box inside C1
    add_box(ax, 4.5, 8.0, 12.5, 11.2, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=1.0, radius=0.8)
    ax.text(10.75, 15.6, "Input\nSpikes", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(10.75, 10.6, r"$S_t^{(\ell-1)}$" + "\n" + r"$\in \{0,1\}^{N\times D}$", ha="center", va="center", fontsize=6.9, color=PALETTE["ink"])

    # Top parallel branch: Normalized Binary Spike Self-Attention
    add_box(ax, 21.5, 13.6, 52.5, 7.8, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, radius=0.9, shadow=True)
    ax.text(47.75, 19.6, "Global Branch: Normalized Binary Spike Self-Attention", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["ink"])
    add_box(ax, 23.0, 14.6, 15.2, 3.5, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=0.85, radius=0.55)
    ax.text(30.6, 16.35, "Linear+BNTT+PLIF", ha="center", va="center", fontsize=6.6, fontweight="bold", color=PALETTE["spike_text"])
    add_arrow(ax, (38.2, 16.35), (41.0, 16.35), color=PALETTE["arrow_main"], lw=1.1, scale=9.5)
    add_box(ax, 41.0, 14.6, 17.5, 3.5, fc="#EEF2FF", ec="#6366F1", lw=0.85, radius=0.55)
    ax.text(49.75, 16.35, r"Binary $(QK^\top / N)\,V$", ha="center", va="center", fontsize=6.9, fontweight="bold", color=PALETTE["teacher_header"])
    add_arrow(ax, (58.5, 16.35), (61.2, 16.35), color=PALETTE["arrow_main"], lw=1.1, scale=9.5)
    add_box(ax, 61.2, 14.6, 11.4, 3.5, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=0.85, radius=0.55)
    ax.text(66.9, 16.35, "Proj + PLIF", ha="center", va="center", fontsize=6.7, fontweight="bold", color=PALETTE["spike_text"])

    # Bottom parallel branch: Event-Observable Local Structure Mixer
    add_box(ax, 21.5, 3.5, 52.5, 8.6, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, radius=0.9, shadow=True)
    ax.text(47.75, 10.3, "Local Branch: Event-Observable Structure Mixer", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["student_header"])
    add_box(ax, 23.0, 4.6, 16.5, 4.0, fc="#F0FDFA", ec="#14B8A6", lw=0.85, radius=0.55)
    ax.text(31.25, 6.6, "3×3 Dilated Convs\n(d = 1, d = 2) + PLIF", ha="center", va="center", fontsize=6.5, fontweight="bold", color=PALETTE["student_header"])
    add_arrow(ax, (39.5, 6.6), (42.3, 6.6), color=PALETTE["arrow_main"], lw=1.1, scale=9.5)
    add_box(ax, 42.3, 4.6, 17.2, 4.0, fc="#F0FDFA", ec="#14B8A6", lw=0.85, radius=0.55)
    ax.text(50.9, 6.6, r"Novelty Route $|\Delta S_t|$" + "\nGate × Softmax Mix", ha="center", va="center", fontsize=6.45, color=PALETTE["ink"])
    add_arrow(ax, (59.5, 6.6), (62.0, 6.6), color=PALETTE["arrow_main"], lw=1.1, scale=9.5)
    add_box(ax, 62.0, 4.6, 10.6, 4.0, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=0.85, radius=0.55)
    ax.text(67.3, 6.6, r"$\sigma(\gamma)\cdot$PLIF", ha="center", va="center", fontsize=6.7, fontweight="bold", color=PALETTE["spike_text"])

    # Split arrows from Input Spikes to Global & Local branches + Residual
    add_ortho_arrow(ax, [(17.0, 15.0), (19.2, 15.0), (19.2, 16.8), (21.5, 16.8)], color=PALETTE["arrow_main"], lw=1.2)
    add_ortho_arrow(ax, [(17.0, 12.0), (19.2, 12.0), (19.2, 7.5), (21.5, 7.5)], color=PALETTE["arrow_main"], lw=1.2)

    # Residual Re-Spike 1
    add_box(ax, 77.2, 9.4, 12.5, 8.2, fc=PALETTE["spike_fill"], ec=PALETTE["spike_border"], lw=1.0, radius=0.8)
    ax.text(83.45, 15.0, "Residual\nRe-Spike", ha="center", va="center", fontsize=7.1, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(83.45, 11.2, "BNTT + PLIF\n→ {0, 1}", ha="center", va="center", fontsize=6.6, color=PALETTE["ink"])

    add_ortho_arrow(ax, [(74.0, 16.8), (75.6, 16.8), (75.6, 14.6), (77.2, 14.6)], color=PALETTE["arrow_main"], lw=1.15)
    add_ortho_arrow(ax, [(74.0, 7.5), (75.6, 7.5), (75.6, 12.4), (77.2, 12.4)], color=PALETTE["arrow_main"], lw=1.15)
    # Skip connection over Global branch (below the ZOOM-IN pill)
    add_ortho_arrow(ax, [(19.2, 16.8), (19.2, 21.8), (83.45, 21.8), (83.45, 17.6)], color="#64748B", lw=0.95, ls="--", scale=9.0)

    # Spiking MLP + Residual Re-Spike 2
    add_box(ax, 92.6, 9.4, 16.5, 8.2, fc="#FFFFFF", ec=PALETTE["spike_border"], lw=1.0, radius=0.8, shadow=True)
    ax.text(100.85, 15.1, "Spiking MLP", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["ink"])
    ax.text(100.85, 11.5, "FC1→BNTT→PLIF\nFC2→BNTT+Res→PLIF", ha="center", va="center", fontsize=6.35, color=PALETTE["muted"])
    add_arrow(ax, (89.7, 13.5), (92.6, 13.5), color=PALETTE["arrow_main"], lw=1.2)

    # Output Spikes inside C1
    add_box(ax, 111.8, 8.0, 11.8, 11.2, fc=PALETTE["spike_bg"], ec=PALETTE["spike_border"], lw=1.0, radius=0.8)
    ax.text(117.7, 15.6, "Output\nSpikes", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(117.7, 10.6, r"$S_t^{(\ell)}$" + "\n" + r"$\in \{0,1\}^{N\times D}$", ha="center", va="center", fontsize=6.9, color=PALETTE["ink"])
    add_arrow(ax, (109.1, 13.5), (111.8, 13.5), color=PALETTE["arrow_main"], lw=1.2)

    # C2. SNN Training Stabilization & Evaluation Protocol (x = 128.0 .. 189.6, y = 2.0 .. 25.0)
    add_box(ax, 128.0, 2.0, 61.6, 23.0, fc=PALETTE["detail_bg"], ec=PALETTE["detail_border"], lw=1.3, radius=1.5, zorder=1)
    add_pill(
        ax, 130.2, 22.4, 57.2, 2.5,
        "KEY MECHANISMS · Energy Masking & Clean-Train BNTT Calibration",
        fc="#E2E8F0", ec=PALETTE["detail_border"], tc=PALETTE["detail_header"],
        fontsize=7.2, zorder=6,
    )

    add_box(ax, 130.0, 13.0, 57.6, 8.2, fc="#FFFFFF", ec="#CBD5E1", lw=0.95, radius=0.8)
    ax.text(131.6, 19.2, "1. Firing-Rate-Preserving Block Masking", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["spike_text"])
    ax.text(
        131.6, 16.5,
        r"$X_{\mathrm{mask}} = (M \odot X)\cdot \mathrm{clamp}\!\left(\|X\|_1 / \|M\odot X\|_1,\, 1.0,\, 1.35\right)$",
        ha="left", va="center", fontsize=7.0, color=PALETTE["ink"],
    )
    ax.text(131.6, 14.2, "Paired [X_clean, X_mask] batch prevents BNTT threshold drift.", ha="left", va="center", fontsize=6.85, color=PALETTE["muted"])

    add_box(ax, 130.0, 3.6, 57.6, 8.2, fc="#FFFFFF", ec="#CBD5E1", lw=0.95, radius=0.8)
    ax.text(131.6, 9.8, "2. Student EMA + Population BNTT Recalibration", ha="left", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["student_header"])
    ax.text(131.6, 7.2, r"EMA ($\mu = 0.995$) + exact clean-train BNTT stats (no_grad).", ha="left", va="center", fontsize=7.0, color=PALETTE["ink"])
    ax.text(131.6, 4.9, "Zero class labels in student training · zero test-split access.", ha="left", va="center", fontsize=6.85, color=PALETTE["muted"])

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Next-Gen PureSpikeFormer architecture diagram.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    img_dir = args.project_root / "image"
    out_pdf = img_dir / "architecture_nextgen_purespikeformer.pdf"
    out_png = img_dir / "architecture_nextgen_purespikeformer.png"
    build_architecture_figure(out_pdf, out_png)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
