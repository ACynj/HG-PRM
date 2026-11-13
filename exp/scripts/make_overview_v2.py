#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_overview_v2.py — 方法总览图 v2 (重画: 修复标题碰撞/文字溢出)
布局: 严格网格对齐, 标题内嵌, 字号>=6.5, 间距充分
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper_figs"
OUT.mkdir(exist_ok=True)

M_PINK, M_PISTACH, M_LEMON = "#F2A7B3", "#A9D6B8", "#F7E8A4"
M_LAVEND, M_BLUE, M_CARAMEL = "#C8B8E8", "#9DB8E8", "#E8C9A0"
D_CARAMEL, D_LAVEND, D_PISTACH = "#A67B4E", "#7A5FA8", "#4E8A6A"
D_PINK, D_BLUE, D_LEMON = "#C05C78", "#3F6AB0", "#B8952E"
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "pdf.fonttype": 42,
})

W, H = 7.16, 3.1          # inch


def rounded(ax, x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.012",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 color=color, lw=lw, linestyle=ls, mutation_scale=13,
                 shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ================= 区域划分 (严格对齐) =================
    # A: [0.005, 0.245]  B: [0.275, 0.475]  C: [0.505, 0.80]  D: [0.825, 0.995]
    top, bot = 0.965, 0.035
    A = (0.005, 0.245); B = (0.275, 0.475); C = (0.505, 0.800); D = (0.825, 0.995)

    def panel(rng, fc, ec, title, tc):
        x0, x1 = rng
        rounded(ax, x0, bot, x1 - x0, top - bot, fc, ec, 1.1)
        ax.text((x0 + x1) / 2, top - 0.045, title, ha="center", va="center",
                fontsize=7.6, weight="bold", color=tc)
        return x0, x1

    # ================= A: 异构图 =================
    x0, x1 = panel(A, "#FDF7F1", D_CARAMEL,
                   "Heterogeneous Graph + Schema", D_CARAMEL)
    cx = (x0 + x1) / 2
    nodes = {
        "Drug":      (0.045, 0.74, M_PINK),
        "Disease":   (0.125, 0.79, M_LAVEND),
        "Symptom":   (0.205, 0.74, M_LEMON),
        "Procedure": (0.045, 0.52, M_PISTACH),
        "Gene":      (0.125, 0.47, M_BLUE),
        "Anatomy":   (0.205, 0.52, M_CARAMEL),
    }
    r = 0.026
    for name, (x, y, c) in nodes.items():
        ax.add_patch(Circle((x, y), r, fc=c, ec="#6A5A45", lw=0.7,
                            transform=ax.transAxes))
        ax.text(x, y - 0.055, name, ha="center", fontsize=6.0, color="#4A3A28")
    edges = [("Drug", "Disease", "treats"),
             ("Disease", "Symptom", "has_symptom"),
             ("Disease", "Gene", "assoc."),
             ("Procedure", "Disease", "analyzes"),
             ("Gene", "Anatomy", "expr._in")]
    for (a, b, lab) in edges:
        (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
        dx, dy = xb - xa, yb - ya
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        arrow(ax, xa + ux * r * 1.1, ya + uy * r * 1.1,
              xb - ux * r * 1.25, yb - uy * r * 1.25, "#B09A78", 1.0)
        ax.text((xa + xb) / 2 + uy * 0.012, (ya + yb) / 2 + ux * 0.012, lab,
                fontsize=5.8, color="#8A744F", style="italic",
                ha="center", va="center", rotation=0)
    ax.text(cx, 0.30, "UMLS: 135 entities · 49 relations · 13 types",
            ha="center", fontsize=6.6, color="#4A3A28")
    rounded(ax, x0 + 0.012, 0.09, x1 - x0 - 0.024, 0.145, "#F6EBDD",
            D_CARAMEL, 0.8)
    ax.text(cx, 0.162, "Schema Σ: 313 legal", ha="center", fontsize=6.2,
            color="#6A5238")
    ax.text(cx, 0.124, "(head-type, relation, tail-type)", ha="center",
            fontsize=6.2, color="#6A5238")

    # ================= B: LLM 轨迹 =================
    x0, x1 = panel(B, "#F6F4FB", D_LAVEND, "LLM Agent Trajectory", D_LAVEND)
    cx = (x0 + x1) / 2
    steps = ["s₁  at e₀ → propose K=4", "s₂  at e₁ → propose K=4",
             "s₃  at e₂ → propose K=4", "s_T  answer entity"]
    bw = (x1 - x0) * 0.84
    for i, s in enumerate(steps):
        yb = 0.735 - i * 0.148
        fc = M_LAVEND if i < 3 else M_PISTACH
        rounded(ax, cx - bw / 2, yb - 0.045, bw, 0.095, fc, D_LAVEND, 0.8)
        ax.text(cx, yb, s, ha="center", va="center", fontsize=6.8,
                color="#3D2E63")
        if i < 3:
            arrow(ax, cx, yb - 0.05, cx, yb - 0.098, D_LAVEND, 1.0)
    ax.text(cx, 0.085, "Qwen2.5-7B (T=0)\n+ same-relation candidates",
            ha="center", va="center", fontsize=6.4, color=GREY)

    # ================= C: HG-PRM 验证器 =================
    x0, x1 = panel(C, "#F1F8F3", D_PISTACH,
                   "HG-PRM: Graph as Process Reward Model", D_PISTACH)
    cx = (x0 + x1) / 2
    # 符号层
    sy, sh = 0.655, 0.205
    rounded(ax, x0 + 0.014, sy, x1 - x0 - 0.028, sh, M_PISTACH, D_PISTACH, 0.9)
    ax.text(cx, sy + sh - 0.038, "Symbolic layer — exact · sparse · unhackable",
            ha="center", fontsize=7.0, weight="bold", color="#2E5E44")
    ax.text(x0 + (x1 - x0) * 0.27, sy + 0.068, "C1  schema", ha="center",
            fontsize=6.4, color="#2E5E44")
    ax.text(x0 + (x1 - x0) * 0.53, sy + 0.068, "C2  rel.-seq. mask",
            ha="center", fontsize=6.4, color="#2E5E44")
    ax.text(x0 + (x1 - x0) * 0.80, sy + 0.068, "C3  type-reach. BFS",
            ha="center", fontsize=6.4, color="#2E5E44")
    ax.text(cx, sy + 0.028, "closed-form on (G, Σ) — no language model",
            ha="center", fontsize=6.2, color="#2E5E44", style="italic")
    # 神经层
    ny, nh = 0.425, 0.17
    rounded(ax, x0 + 0.014, ny, x1 - x0 - 0.028, nh, M_BLUE, D_BLUE, 0.9)
    ax.text(cx, ny + nh - 0.036, "Neural layer — dense · distilled mini-HGT",
            ha="center", fontsize=7.0, weight="bold", color="#2A4A80")
    ax.text(cx, ny + 0.032, "trained on 4,699 symbolic-labeled steps (0 human labels)",
            ha="center", fontsize=6.4, color="#2A4A80")
    # 融合公式
    rounded(ax, x0 + 0.014, 0.075, x1 - x0 - 0.028, 0.115, "#FBF9E8", D_LEMON, 0.8)
    ax.text(cx, 0.1325, r"$f = \alpha\, r_{sym} + \beta\, r_{hgnn}"
                        r" + (1{-}\alpha{-}\beta)\, r_{llm}$",
            ha="center", va="center", fontsize=7.6, color="#1E3E28")

    # ================= D: 部署 =================
    x0, x1 = panel(D, "#FDF1F4", D_PINK, "Deployment", D_PINK)
    cx = (x0 + x1) / 2
    bw = (x1 - x0) * 0.80
    for i, (lab, fc, ec) in enumerate([
            ("Test-time\nbeam search", M_PINK, D_PINK),
            ("GRPO / DPO\nprocess reward", M_PINK, D_PINK),
            ("67.5% vs 28.1%\n(+39.4 pts, p<1e-13)", M_LEMON, D_LEMON)]):
        yb = 0.70 - i * 0.185
        rounded(ax, cx - bw / 2, yb - 0.062, bw, 0.135, fc, ec, 0.8)
        ax.text(cx, yb + 0.005, lab, ha="center", va="center", fontsize=6.6,
                color="#6E3A4A" if i < 2 else "#6E5410",
                weight="normal" if i < 2 else "bold")

    # ================= 大箭头 =================
    arrow(ax, 0.250, 0.5, 0.271, 0.5, GREY, 1.5)
    arrow(ax, 0.480, 0.5, 0.501, 0.5, GREY, 1.5)
    arrow(ax, 0.803, 0.5, 0.823, 0.5, GREY, 1.5)
    # 逐步验证回路 B<->C (虚线)
    arrow(ax, 0.455, 0.30, 0.512, 0.36, D_PISTACH, 1.1, ls="--")
    arrow(ax, 0.512, 0.22, 0.455, 0.16, D_PISTACH, 1.1, ls="--")
    ax.text(0.483, 0.088, "verify\nper step", ha="center", va="center",
            fontsize=5.8, color=D_PISTACH)

    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_overview.pdf (v2)")


if __name__ == "__main__":
    main()
