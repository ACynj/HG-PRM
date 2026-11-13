#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_fig_motivation.py — Fig.0 验证缺口动机图(单栏)
(a) 现状: 结构失明的监督 — outcome奖励(稀疏) + 文本PRM(自己不会算图)
(b) 本文: 图原生验证 — 每步闭式判定 + 掩码示例
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parents[1] / "paper_figs"

M_PINK, M_PISTACH, M_LEMON = "#F2A7B3", "#A9D6B8", "#F7E8A4"
M_LAVEND, M_BLUE, M_CARAMEL = "#C8B8E8", "#9DB8E8", "#E8C9A0"
D_PINK, D_PISTACH, D_LAVEND = "#A84A64", "#3E7A58", "#6B5494"
D_CARAMEL, D_BLUE, D_LEMON = "#8A6238", "#33568C", "#8F7020"
GREY = "#4A4A4A"
RED_X = "#B03A48"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "text.color": "#222222"})


def rbox(ax, x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.010",
                 fc=fc, ec=ec, lw=lw))


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.2, ls="-", both=False):
    st = "<|-|>" if both else "-|>"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=st,
                 color=color, lw=lw, linestyle=ls, mutation_scale=11,
                 shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, fs=6.2, c="#222222", w="normal", ha="center",
        bbox=True):
    kw = dict(ha=ha, va="center", fontsize=fs, color=c, weight=w, zorder=8)
    if bbox:
        kw["bbox"] = dict(fc="white", ec="none", alpha=0.85, pad=0.7)
    ax.text(x, y, s, **kw)


def main():
    fig, ax = plt.subplots(figsize=(3.5, 4.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ================= (a) 现状 =================
    rbox(ax, 0.02, 0.545, 0.96, 0.435, "#FBF3F4", D_PINK, 1.1)
    txt(ax, 0.5, 0.945, "(a) Current: structure-blind supervision",
        fs=7.0, c=D_PINK, w="bold", bbox=False)
    # LLM agent
    rbox(ax, 0.06, 0.72, 0.26, 0.16, M_LAVEND, D_LAVEND)
    txt(ax, 0.19, 0.835, "LLM agent", fs=6.4, c=D_LAVEND, w="bold")
    txt(ax, 0.19, 0.775, "wrong relation\ntype drift…", fs=5.4, c="#4A3E63")
    # 两个失明监督源
    rbox(ax, 0.42, 0.79, 0.52, 0.115, "#F3E9EC", RED_X, 0.8)
    txt(ax, 0.68, 0.885, "Outcome reward  ✗", fs=6.2, c=RED_X, w="bold")
    txt(ax, 0.68, 0.818, "arrives only at the final answer", fs=5.2, c=GREY)
    rbox(ax, 0.42, 0.615, 0.52, 0.115, "#F3E9EC", RED_X, 0.8)
    txt(ax, 0.68, 0.705, "Text-LLM PRM  ✗", fs=6.2, c=RED_X, w="bold")
    txt(ax, 0.68, 0.638, "verifier cannot compute structure", fs=5.2, c=GREY)
    arrow(ax, 0.325, 0.83, 0.415, 0.85, GREY, 1.0, ls="--")
    arrow(ax, 0.325, 0.77, 0.415, 0.67, GREY, 1.0, ls="--")
    txt(ax, 0.5, 0.578, "gap: nobody checks the graph", fs=5.6, c=RED_X,
        w="bold")

    # ================= (b) 本文 =================
    rbox(ax, 0.02, 0.035, 0.96, 0.475, "#EFF7F1", D_PISTACH, 1.1)
    txt(ax, 0.5, 0.478, "(b) HG-PRM: the graph verifies each step",
        fs=7.0, c=D_PISTACH, w="bold", bbox=False)
    # agent
    rbox(ax, 0.055, 0.30, 0.235, 0.13, M_LAVEND, D_LAVEND)
    txt(ax, 0.1725, 0.398, "LLM agent", fs=6.4, c=D_LAVEND, w="bold")
    txt(ax, 0.1725, 0.34, "proposes K=4 + pool", fs=5.2, c="#4A3E63")
    # graph verifier
    rbox(ax, 0.365, 0.255, 0.575, 0.19, M_PISTACH, D_PISTACH)
    txt(ax, 0.6525, 0.412, "Graph-native verifier", fs=6.6, c="#2E5E44",
        w="bold")
    txt(ax, 0.6525, 0.352,
        "C1 schema · C2 program mask\nC3 type-reachability (BFS)",
        fs=5.2, c="#2E5E44")
    txt(ax, 0.6525, 0.293, "closed-form on (G, Σ) — exact, unhackable",
        fs=5.2, c="#2E5E44")
    arrow(ax, 0.295, 0.365, 0.360, 0.355, GREY, 1.2)
    arrow(ax, 0.360, 0.315, 0.295, 0.325, D_PISTACH, 1.2, ls="--",
          both=True)
    txt(ax, 0.3275, 0.442, "mask / score", fs=5.0, c=D_PISTACH)
    # 输出
    arrow(ax, 0.6525, 0.252, 0.6525, 0.205, GREY, 1.2)
    rbox(ax, 0.365, 0.085, 0.575, 0.115, M_LEMON, D_LEMON, 0.9)
    txt(ax, 0.6525, 0.170, "dense process reward f", fs=6.0, c="#6E5410",
        w="bold")
    txt(ax, 0.6525, 0.112, "test-time beam search · GRPO / DPO training",
        fs=5.2, c="#6E5410")
    # 掩码小示例(左下)
    rbox(ax, 0.055, 0.085, 0.235, 0.19, "#F6FBF7", D_PISTACH, 0.7)
    txt(ax, 0.1725, 0.253, "mask in action", fs=5.6, c="#2E5E44", w="bold")
    txt(ax, 0.1725, 0.162,
        "step: issue_in\n(− affects ✗\n(− location_of ✗\n(+ issue_in → Occ. ✓",
        fs=5.0, c="#2E5E44")
    arrow(ax, 0.295, 0.18, 0.360, 0.16, D_PISTACH, 0.9, ls=":")

    fig.savefig(OUT / "fig_motivation.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_motivation.pdf")


if __name__ == "__main__":
    main()
