#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_figures.py — 论文终稿图 (马卡龙配色, 高信息密度)

Fig 1 (fig_overview.pdf, 全宽 figure*): 方法总览
Fig 2 (fig_results.pdf,  全宽 figure*): 四面板结果
  (a) accuracy vs hop   (b) compositional split seen/heldout + Δ
  (c) distraction quartile (d) cost-accuracy frontier + error bars
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figs"
OUT.mkdir(exist_ok=True)

# ---------------- 马卡龙色板 ----------------
M_PINK    = "#F2A7B3"   # rose 玫瑰
M_PISTACH = "#A9D6B8"   # pistachio 开心果
M_LEMON   = "#F7E8A4"   # lemon 柠檬
M_LAVEND  = "#C8B8E8"   # lavender 薰衣草
M_BLUE    = "#9DB8E8"   # blueberry 蓝莓
M_CARAMEL = "#E8C9A0"   # caramel 焦糖
DEEP = {"pink": "#C05C78", "pistach": "#4E8A6A", "lemon": "#B8952E",
        "lavend": "#7A5FA8", "blue": "#3F6AB0", "caramel": "#A67B4E"}
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.linewidth": 0.7, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "pdf.fonttype": 42,
})


def load(tag="main"):
    recs = [json.loads(l) for l in
            (ROOT / "outputs" / f"results_{tag}.jsonl").read_text().splitlines()]
    return [r for r in recs if "error" not in r]


# ===========================================================================
# Fig 1: 方法总览 (全宽 ~ 7.16in x 2.9in)
# ===========================================================================
def fig_overview():
    fig, ax = plt.subplots(figsize=(7.16, 2.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=1.0, r=0.9):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.25,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, mutation_aspect=0.35)
        ax.add_patch(b); return b

    def arrow(x1, y1, x2, y2, color=GREY, lw=1.1, style="-|>", ls="-"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            color=color, lw=lw, linestyle=ls,
                            mutation_scale=11, shrinkA=2, shrinkB=2)
        ax.add_patch(a)

    # ---- (A) 异构图面板 ----
    box(0.5, 8, 26, 30, "#FDF6F0", DEEP["caramel"], 1.2)
    ax.text(13.5, 35.6, "Heterogeneous Graph $\\mathcal{G}$ + Schema $\\Sigma$",
            ha="center", fontsize=8.6, weight="bold", color=DEEP["caramel"])
    # 类型节点
    nodes = {
        "Drug": (5.2, 28.5, M_PINK), "Disease": (13.5, 30.5, M_LAVEND),
        "Symptom": (21.5, 28.5, M_LEMON), "Procedure": (5.2, 17.5, M_PISTACH),
        "Gene": (13.5, 15.0, M_BLUE), "Anatomy": (21.5, 17.5, M_CARAMEL),
    }
    for name, (x, y, c) in nodes.items():
        ax.add_patch(Circle((x, y), 1.55, fc=c, ec="#7A6A55", lw=0.7))
        ax.text(x, y - 3.0, name, ha="center", fontsize=6.4, color="#4A3A28")
    edges = [("Drug", "Disease", "treats", 0.55), ("Disease", "Symptom",
              "has_symptom", 0.55), ("Disease", "Gene", "associated_with", 0.35),
             ("Procedure", "Disease", "analyzes", 0.35),
             ("Gene", "Anatomy", "expressed_in", 0.35)]
    pos = {k: v[:2] for k, v in nodes.items()}
    for (a, b, lab, cu) in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        arrow(x1 + 1.5 * np.sign(x2 - x1) * 0.9, y1 - 1.2 * np.sign(y1 - y2) * 0.5,
              x2 - 1.4 * np.sign(x2 - x1) * 0.9, y2 + 1.2 * np.sign(y1 - y2) * 0.5,
              color="#B09A78", lw=0.9)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.9, lab, fontsize=5.6,
                color="#8A744F", style="italic", ha="center")
    ax.text(13.5, 10.2, "313 schema triples (UMLS)\n135 entities / 49 relations / 13 types",
            ha="center", fontsize=6.2, color=GREY)

    # ---- (B) LLM agent 轨迹面板 ----
    box(29.5, 8, 22, 30, "#F5F3FB", DEEP["lavend"], 1.2)
    ax.text(40.5, 35.6, "LLM Agent Trajectory", ha="center", fontsize=8.6,
            weight="bold", color=DEEP["lavend"])
    steps = ["$s_1$: at $e_0$ · propose $K{=}4$",
             "$s_2$: at $e_1$ · propose $K{=}4$",
             "$s_3$: at $e_2$ · propose $K{=}4$",
             "$s_T$: answer entity"]
    for i, s in enumerate(steps):
        fc = M_LAVEND if i < 3 else M_PISTACH
        box(31.5, 29.5 - i * 6.2, 18, 4.4, fc, DEEP["lavend"], 0.8)
        ax.text(40.5, 31.7 - i * 6.2, s, ha="center", fontsize=6.6)
        if i < 3:
            arrow(40.5, 29.7 - i * 6.2, 40.5, 27.6 - i * 6.2, DEEP["lavend"], 1.0)
    ax.text(40.5, 10.0, "Qwen2.5-7B · T=0 proposals\n+ same-relation neighbor pool",
            ha="center", fontsize=6.2, color=GREY)

    # ---- (C) 两层验证器 ----
    box(54.5, 8, 28, 30, "#F0F8F2", DEEP["pistach"], 1.2)
    ax.text(68.5, 35.6, "HG-PRM: Graph as Process Reward Model",
            ha="center", fontsize=8.6, weight="bold", color=DEEP["pistach"])
    # 符号层
    box(56, 24, 25, 8.2, M_PISTACH, DEEP["pistach"], 0.9)
    ax.text(68.5, 30.4, "Symbolic Layer (exact, sparse, unhackable)",
            ha="center", fontsize=7.0, weight="bold", color="#2E5E44")
    ax.text(68.5, 26.6,
            "C1 schema conformance   C2 relation-sequence mask\nC3 answer-type reachability (BFS)",
            ha="center", fontsize=6.4, color="#2E5E44")
    # 神经层
    box(56, 15, 25, 7.2, M_BLUE, DEEP["blue"], 0.9)
    ax.text(68.5, 20.4, "Neural Layer: distilled mini-HGT (dense)",
            ha="center", fontsize=7.0, weight="bold", color="#2A4A80")
    ax.text(68.5, 17.0, "trained on 4,699 self-labeled steps (0 human labels)",
            ha="center", fontsize=6.4, color="#2A4A80")
    # 融合
    ax.text(68.5, 12.6, "$f = \\alpha\\,r_{sym} + \\beta\\,r_{hgnn} + (1{-}\\alpha{-}\\beta)\\,r_{llm}$",
            ha="center", fontsize=7.0, color="#1E3E28")

    # ---- (D) 输出 ----
    box(85, 8, 14.5, 30, "#FDF0F3", DEEP["pink"], 1.2)
    ax.text(92.2, 35.6, "Deployment", ha="center", fontsize=8.6,
            weight="bold", color=DEEP["pink"])
    box(86.3, 26.5, 12, 5.2, M_PINK, DEEP["pink"], 0.8)
    ax.text(92.3, 29.1, "Test-time\nbeam search", ha="center", fontsize=6.6)
    box(86.3, 19.5, 12, 5.2, M_PINK, DEEP["pink"], 0.8)
    ax.text(92.3, 22.1, "GRPO / DPO\nprocess reward", ha="center", fontsize=6.6)
    box(86.3, 12.5, 12, 5.2, M_LEMON, DEEP["lemon"], 0.8)
    ax.text(92.3, 15.1, "67.5% vs 28.1%\n(+39.4 pts)", ha="center",
            fontsize=6.6, weight="bold", color="#6E5410")

    # 面板间大箭头
    arrow(27.2, 23, 29.0, 23, GREY, 1.4)
    arrow(52.2, 23, 54.0, 23, GREY, 1.4)
    arrow(83.0, 23, 84.8, 23, GREY, 1.4)
    # B→C 逐步验证回路
    arrow(50.5, 14, 56.0, 17.5, DEEP["pistach"], 1.1, ls="--")
    arrow(56.0, 22.5, 50.5, 19.5, DEEP["pistach"], 1.1, ls="--")
    ax.text(53.0, 12.2, "verify\nper step", ha="center", fontsize=5.8,
            color=DEEP["pistach"])

    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_overview.pdf")


# ===========================================================================
# Fig 2: 结果四面板 (全宽 7.16 x 2.7)
# ===========================================================================
NAME = {"greedy": "Greedy", "sc": "Self-Consistency", "tokbon": "Token-Overlap BoN",
        "beam": "HG-PRM (ours)", "beam_symonly": "ours w/o LLM",
        "beam_fused": "ours + HGNN", "beam_nomask": "ours w/o mask",
        "beam_b1": "ours beam=1", "beam_a04": "ours α=0.4", "beam_a08": "ours α=0.8"}
COL = {"greedy": M_CARAMEL, "sc": M_LEMON, "tokbon": M_LAVEND,
       "beam": M_PINK, "beam_symonly": M_PISTACH, "beam_fused": M_BLUE,
       "beam_nomask": "#D9C5A8", "beam_b1": "#E3B9C6", "beam_a04": "#F4D7DE",
       "beam_a08": "#EBBFCB"}
EDG = {"greedy": DEEP["caramel"], "sc": DEEP["lemon"], "tokbon": DEEP["lavend"],
       "beam": DEEP["pink"], "beam_symonly": DEEP["pistach"], "beam_fused": DEEP["blue"],
       "beam_nomask": "#8A7355", "beam_b1": "#A04A62", "beam_a04": "#C05C78",
       "beam_a08": "#C05C78"}


def fig_results(recs):
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.55))

    def acc_of(rs):
        return 100 * sum(r["correct"] for r in rs) / max(1, len(rs))

    # ---- (a) accuracy vs hop ----
    ax = axes[0]
    hops = [2, 3, 4, 5]
    for m in ["greedy", "sc", "tokbon", "beam_symonly", "beam_fused", "beam"]:
        ys = [acc_of([r for r in recs if r["method"] == m and r["hop"] == h])
              for h in hops]
        ax.plot(hops, ys, "-o", color=EDG[m], lw=1.4 if m == "beam" else 0.9,
                ms=3.4, label=NAME[m], zorder=5 if m == "beam" else 3,
                markerfacecolor=COL[m], markeredgecolor=EDG[m])
    ax.set_xlabel("Reasoning hops"); ax.set_ylabel("Answer Hits@1 (%)")
    ax.set_xticks(hops); ax.set_ylim(0, 84)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.legend(fontsize=4.9, frameon=False, ncol=2, loc="lower center",
              handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
    ax.set_title("(a) Accuracy vs. hop length", fontsize=7.8, pad=3)

    # ---- (b) compositional split ----
    ax = axes[1]
    ms = ["greedy", "sc", "tokbon", "beam_symonly", "beam_fused", "beam"]
    x = np.arange(len(ms))
    seen = [acc_of([r for r in recs if r["method"] == m and r["split_comp"] == "seen"])
            for m in ms]
    held = [acc_of([r for r in recs if r["method"] == m and r["split_comp"] == "heldout"])
            for m in ms]
    b1 = ax.bar(x - 0.19, seen, 0.36, color=M_PISTACH, edgecolor=DEEP["pistach"],
                lw=0.6, label="seen type-seq.")
    b2 = ax.bar(x + 0.19, held, 0.36, color=M_PINK, edgecolor=DEEP["pink"],
                lw=0.6, label="held-out type-seq.")
    for xi, (s, h) in enumerate(zip(seen, held)):
        ax.text(xi, max(s, h) + 2.5, f"Δ{h-s:+.1f}", ha="center", fontsize=5.6,
                color=DEEP["pink"] if h - s < -5 else GREY)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m].replace(" ", "\n") for m in ms],
                                          fontsize=5.2)
    ax.set_ylabel("Hits@1 (%)"); ax.set_ylim(0, 92)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left")
    ax.set_title("(b) Compositional split", fontsize=7.8, pad=3)

    # ---- (c) distraction quartile ----
    ax = axes[2]
    qs = sorted(r["distract_avg"] for r in recs)
    n4 = len(qs) // 4
    def quart(v):
        if v <= qs[n4]: return "Q1"
        if v <= qs[2 * n4]: return "Q2"
        if v <= qs[3 * n4]: return "Q3"
        return "Q4"
    labels = ["Q1\n(easy)", "Q2", "Q3", "Q4\n(hard)"]
    ms4 = ["greedy", "sc", "tokbon", "beam"]
    w = 0.2
    for i, m in enumerate(ms4):
        ys = [acc_of([r for r in recs if r["method"] == m and quart(r["distract_avg"]) == q])
              for q in ["Q1", "Q2", "Q3", "Q4"]]
        ax.bar(np.arange(4) + (i - 1.5) * w, ys, w - 0.02, color=COL[m],
               edgecolor=EDG[m], lw=0.5, label=NAME[m])
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("Hits@1 (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.legend(fontsize=5.2, frameon=False, loc="upper right", ncol=1)
    ax.set_title("(c) Difficulty: same-relation distractors", fontsize=7.8, pad=3)

    # ---- (d) cost-accuracy frontier ----
    ax = axes[3]
    for m in ["greedy", "sc", "tokbon", "beam_symonly", "beam_fused", "beam"]:
        rs = [r for r in recs if r["method"] == m]
        acc = acc_of(rs)
        calls = sum(r["n_llm_calls"] for r in rs) / len(rs)
        viol = sum(r["type_violations"] for r in rs) / len(rs)
        dead = sum(r["dead_ends"] for r in rs) / len(rs)
        size = 18 + 26 * (viol + dead)  # 气泡=错误量
        ax.scatter(calls, acc, s=size, color=COL[m], edgecolor=EDG[m],
                   lw=0.9, zorder=5, alpha=0.95)
        dx, dy = (0.35, 2.2)
        ha = "left"
        if m == "beam":
            dx, dy = -0.35, 2.2; ha = "right"
        ax.annotate(NAME[m], (calls, acc), xytext=(calls + dx, acc + dy),
                    fontsize=5.4, ha=ha, color=EDG[m])
    ax.set_xlabel("LLM calls / question")
    ax.set_ylabel("Hits@1 (%)")
    ax.set_xlim(0, 17); ax.set_ylim(0, 84)
    ax.grid(lw=0.35, color="#DDDDDD", zorder=0)
    ax.set_title("(d) Cost–accuracy frontier (bubble = errors/q)", fontsize=7.8, pad=3)

    for ax in axes:
        ax.tick_params(length=2.2, width=0.6, labelsize=6.2)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_results.pdf")


if __name__ == "__main__":
    recs = load("main")
    fig_overview()
    fig_results(recs)
