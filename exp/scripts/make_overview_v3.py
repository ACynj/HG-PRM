#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_overview_v3.py — 方法总览图 v3
原则: 每面板内每行文字独立带白底框, 面板间隔加宽, 标题短, 无跨面板元素
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
D_CARAMEL, D_LAVEND, D_PISTACH = "#8A6238", "#6B5494", "#3E7A58"
D_PINK, D_BLUE, D_LEMON = "#A84A64", "#33568C", "#8F7020"
GREY = "#4A4A4A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0,
                     "pdf.fonttype": 42})

W, H = 7.16, 3.0


def rbox(ax, x, y, w, h, fc, ec, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.008",
                 fc=fc, ec=ec, lw=lw))


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.2, ls="-", both=False):
    st = "<|-|>" if both else "-|>"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=st,
                 color=color, lw=lw, linestyle=ls, mutation_scale=10,
                 shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, fs=6.4, c="#222222", w="normal", ha="center",
        bbox=True):
    kw = dict(ha=ha, va="center", fontsize=fs, color=c, weight=w,
              zorder=8)
    if bbox:
        kw["bbox"] = dict(fc="white", ec="none", alpha=0.85, pad=0.8)
    ax.text(x, y, s, **kw)


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 面板: [x0, x1], 间隔 0.026
    P = {"A": (0.010, 0.225), "B": (0.251, 0.466),
         "C": (0.492, 0.783), "D": (0.809, 0.990)}
    TOP, BOT = 0.955, 0.030
    TITLE_Y = 0.905

    cfg = {"A": ("#FBF4EC", D_CARAMEL, "Graph + Schema", D_CARAMEL),
           "B": ("#F4F1FA", D_LAVEND, "LLM Agent", D_LAVEND),
           "C": ("#EEF7F0", D_PISTACH, "HG-PRM  Verifier", D_PISTACH),
           "D": ("#FAEFF2", D_PINK, "Deployment", D_PINK)}
    for k, (x0, x1) in P.items():
        fc, ec, ti, tc = cfg[k]
        rbox(ax, x0, BOT, x1 - x0, TOP - BOT, fc, ec, 1.1)
        txt(ax, (x0 + x1) / 2, TITLE_Y, ti, fs=7.8, c=tc, w="bold")

    # ---------- A: 4 节点迷你异构图(无边标签, 边类型写在图例行) ----------
    x0, x1 = P["A"]; cx = (x0 + x1) / 2
    nodes = {"Drug": (x0 + 0.034, 0.665, M_PINK),
             "Disease": (cx, 0.780, M_LAVEND),
             "Symptom": (x1 - 0.034, 0.665, M_LEMON),
             "Gene": (cx, 0.560, M_BLUE)}
    lbl_pos = {"Drug": (0.0, -0.056), "Disease": (0.0, 0.048),
               "Symptom": (0.0, -0.056), "Gene": (0.0, -0.056)}
    r = 0.024
    for n, (x, y, c) in nodes.items():
        ax.add_patch(Circle((x, y), r, fc=c, ec="#66543E", lw=0.8, zorder=5))
        ox, oy = lbl_pos[n]
        txt(ax, x + ox, y + oy, n, fs=6.2, c="#4A3826")
    for a, b in [("Drug", "Disease"), ("Disease", "Symptom"),
                 ("Disease", "Gene")]:
        (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
        arrow(ax, xa + (xb - xa) * 0.15, ya + (yb - ya) * 0.15,
              xb - (xb - xa) * 0.17, yb - (yb - ya) * 0.17, "#B09A78", 0.9)
    txt(ax, cx, 0.452, "typed nodes · typed edges", fs=6.0, c=GREY)
    txt(ax, cx, 0.400, "(treats · has_symptom · assoc.)", fs=5.6,
        c="#8A744F")
    rbox(ax, x0 + 0.010, 0.145, x1 - x0 - 0.020, 0.145, M_CARAMEL,
         D_CARAMEL, 0.8)
    txt(ax, cx, 0.253, "Schema Σ", fs=6.6, c="#5C4426", w="bold")
    txt(ax, cx, 0.184, "313 legal triples\n(type, relation, type)",
        fs=5.8, c="#5C4426")
    txt(ax, cx, 0.080, "UMLS: 135 ent · 49 rel · 13 types",
        fs=5.2, c=GREY)

    # ---------- B: 步骤轨迹 ----------
    x0, x1 = P["B"]; cx = (x0 + x1) / 2
    bw = (x1 - x0) * 0.82
    rows = [("s₁  propose K=4", M_LAVEND, 0.770),
            ("s₂  propose K=4", M_LAVEND, 0.585),
            ("s₃  propose K=4", M_LAVEND, 0.400),
            ("s_T  answer", M_PISTACH, 0.215)]
    for s, fc, y in rows:
        rbox(ax, cx - bw / 2, y - 0.052, bw, 0.104, fc, D_LAVEND, 0.8)
        txt(ax, cx, y, s, fs=6.6, c="#3D2E63")
    for y1, y2 in [(0.718, 0.637), (0.533, 0.452), (0.348, 0.267)]:
        arrow(ax, cx, y1, cx, y2, D_LAVEND, 1.0)
    txt(ax, cx, 0.085, "Qwen2.5-7B · T = 0", fs=6.2, c=GREY)

    # ---------- C: 两层验证器 ----------
    x0, x1 = P["C"]; cx = (x0 + x1) / 2
    bw = x1 - x0 - 0.020
    # 符号层
    rbox(ax, x0 + 0.010, 0.660, bw, 0.185, M_PISTACH, D_PISTACH, 0.9)
    txt(ax, cx, 0.800, "Symbolic layer", fs=7.2, c="#2E5E44", w="bold")
    txt(ax, cx, 0.741, "exact · sparse · unhackable", fs=6.0, c="#2E5E44")
    txt(ax, cx, 0.695,
        "C1 schema · C2 mask · C3 type-reach",
        fs=5.8, c="#2E5E44")
    # 神经层
    rbox(ax, x0 + 0.010, 0.395, bw, 0.175, M_BLUE, D_BLUE, 0.9)
    txt(ax, cx, 0.525, "Neural layer", fs=7.2, c="#2A4A80", w="bold")
    txt(ax, cx, 0.468, "dense · mini-HGT", fs=6.0, c="#2A4A80")
    txt(ax, cx, 0.428, "distilled from 4,699 symbolic labels",
        fs=6.0, c="#2A4A80")
    # 融合
    rbox(ax, x0 + 0.006, 0.135, bw + 0.008, 0.185, M_LEMON, D_LEMON, 0.9)
    txt(ax, cx, 0.272, "Fusion", fs=7.2, c="#6E5410", w="bold")
    txt(ax, cx, 0.205,
        r"$f = \alpha\, r_{sym} + \beta\, r_{hgnn}"
        r" + (1{-}\alpha{-}\beta)\, r_{llm}$",
        fs=6.4, c="#4A3A10")

    # ---------- D: 部署 ----------
    x0, x1 = P["D"]; cx = (x0 + x1) / 2
    bw = (x1 - x0) * 0.86
    for lab, fc, ec, y, w, c in [
            ("test-time\nbeam search", M_PINK, D_PINK, 0.760, "normal",
             "#6E3A4A"),
            ("GRPO / DPO\nprocess reward", M_PINK, D_PINK, 0.500,
             "normal", "#6E3A4A"),
            ("67.5% vs 30.0%", M_LEMON, D_LEMON, 0.250, "bold",
             "#6E5410")]:
        rbox(ax, cx - bw / 2, y - 0.085, bw, 0.170, fc, ec, 0.8)
        txt(ax, cx, y, lab, fs=6.4, c=c, w=w)
    txt(ax, cx, 0.085, "(+37.5 pts, p<10⁻¹⁷)", fs=5.8, c=GREY)

    # ---------- 面板间箭头 ----------
    for (a, b) in [("A", "B"), ("B", "C"), ("C", "D")]:
        xa = P[a][1] + 0.002; xb = P[b][0] - 0.002
        arrow(ax, xa, 0.5, xb, 0.5, GREY, 1.4)
    # B<->C 逐步验证回路(短虚线双箭头, 不触面板边框; 语义由图注说明)
    arrow(ax, P["B"][1] + 0.006, 0.32, P["C"][0] - 0.006, 0.32,
          D_PISTACH, 1.0, ls="--", both=True)

    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_overview.pdf (v3)")


if __name__ == "__main__":
    main()
