#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_results_v2.py — 结果四面板图 v2 (按视觉QA修复)
修复: (a)图例入图内左上+差异化marker (b)短名+旋转刻度+Δ对齐 (c)图例右上避柱
      (d)标注手工偏移+气泡尺寸图例+轴标题 (全局)统一标题与间距
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figs"

M_PINK, M_PISTACH, M_LEMON = "#F2A7B3", "#A9D6B8", "#F7E8A4"
M_LAVEND, M_BLUE, M_CARAMEL = "#C8B8E8", "#9DB8E8", "#E8C9A0"
D = {"greedy": "#A67B4E", "sc": "#B8952E", "tokbon": "#7A5FA8",
     "beam": "#C05C78", "beam_symonly": "#4E8A6A", "beam_fused": "#3F6AB0"}
C = {"greedy": M_CARAMEL, "sc": M_LEMON, "tokbon": M_LAVEND,
     "beam": M_PINK, "beam_symonly": M_PISTACH, "beam_fused": M_BLUE}
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.linewidth": 0.7, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "pdf.fonttype": 42,
})

SHORT = {"greedy": "Greedy", "sc": "SC", "tokbon": "TokBoN",
         "beam": "HG-PRM", "beam_symonly": "w/o LLM", "beam_fused": "+HGNN"}
MKR = {"greedy": "o", "sc": "v", "tokbon": "D", "beam": "o",
       "beam_symonly": "s", "beam_fused": "^"}


def load(tag="main"):
    recs = [json.loads(l) for l in
            (ROOT / "outputs" / f"results_{tag}.jsonl").read_text().splitlines()]
    return [r for r in recs if "error" not in r]


def main():
    recs = load()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.6))

    def acc(rs):
        return 100 * sum(r["correct"] for r in rs) / max(1, len(rs))

    # ---------------- (a) hop ----------------
    ax = axes[0]
    hops = [2, 3, 4, 5]
    order = ["greedy", "sc", "tokbon", "beam_fused", "beam_symonly", "beam"]
    for m in order:
        ys = [acc([r for r in recs if r["method"] == m and r["hop"] == h])
              for h in hops]
        ax.plot(hops, ys, marker=MKR[m], color=D[m],
                lw=1.5 if m == "beam" else 0.9,
                ms=3.6 if m == "beam" else 2.8,
                label=SHORT[m], zorder=6 if m == "beam" else 3,
                markerfacecolor=C[m], markeredgecolor=D[m],
                markeredgewidth=0.6)
    ax.set_xlabel("Reasoning hops")
    ax.set_ylabel("Answer Hits@1 (%)")
    ax.set_xticks(hops); ax.set_ylim(0, 88)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    leg = ax.legend(fontsize=5.4, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.5, -0.34), ncol=3, columnspacing=0.9,
                    handlelength=1.3, labelspacing=0.28)
    ax.set_title("(a) Accuracy vs. hops", fontsize=8.0, pad=4)

    # ---------------- (b) compositional ----------------
    ax = axes[1]
    ms = ["greedy", "sc", "tokbon", "beam_symonly", "beam_fused", "beam"]
    x = np.arange(len(ms))
    seen = [acc([r for r in recs if r["method"] == m and r["split_comp"] == "seen"])
            for m in ms]
    held = [acc([r for r in recs if r["method"] == m and r["split_comp"] == "heldout"])
            for m in ms]
    ax.bar(x - 0.19, seen, 0.36, color=M_PISTACH, edgecolor=D["beam_symonly"],
           lw=0.6, label="seen", zorder=3)
    ax.bar(x + 0.19, held, 0.36, color=M_PINK, edgecolor=D["beam"],
           lw=0.6, label="held-out", zorder=3)
    for xi, (s, h) in enumerate(zip(seen, held)):
        dy = 2.6 if xi % 2 == 0 else 11.8  # 交错高度避免相邻Δ碰撞
        ax.text(xi, max(s, h) + dy, f"Δ{h-s:+.1f}", ha="center", fontsize=5.0,
                color=D["beam"] if h - s < -5 else GREY,
                weight="bold" if h - s < -5 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ms], rotation=48, ha="right",
                       fontsize=5.4)
    ax.set_ylabel("Hits@1 (%)"); ax.set_ylim(0, 88)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.legend(fontsize=6.0, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.40), ncol=2, handlelength=1.1,
              columnspacing=0.9)
    ax.set_title("(b) Compositional split", fontsize=8.0, pad=4)

    # ---------------- (c) difficulty ----------------
    ax = axes[2]
    qs = sorted(r["distract_avg"] for r in recs)
    n4 = len(qs) // 4
    def quart(v):
        if v <= qs[n4]: return "Q1"
        if v <= qs[2 * n4]: return "Q2"
        if v <= qs[3 * n4]: return "Q3"
        return "Q4"
    ms4 = ["greedy", "sc", "tokbon", "beam"]
    w = 0.2
    for i, m in enumerate(ms4):
        ys = [acc([r for r in recs if r["method"] == m and quart(r["distract_avg"]) == q])
              for q in ["Q1", "Q2", "Q3", "Q4"]]
        ax.bar(np.arange(4) + (i - 1.5) * w, ys, w - 0.02, color=C[m],
               edgecolor=D[m], lw=0.5, label=SHORT[m], zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\neasy", "Q2", "Q3", "Q4\nhard"], fontsize=6.2)
    ax.set_ylabel("Hits@1 (%)"); ax.set_ylim(0, 108)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right",
              handlelength=1.1, labelspacing=0.3)
    ax.set_title("(c) Distractor difficulty", fontsize=8.0, pad=4)

    # ---------------- (d) cost-accuracy ----------------
    ax = axes[3]
    pts = {}
    for m in ["greedy", "sc", "tokbon", "beam_symonly", "beam_fused", "beam"]:
        rs = [r for r in recs if r["method"] == m]
        calls = sum(r["n_llm_calls"] for r in rs) / len(rs)
        errs = (sum(r["type_violations"] + r["dead_ends"] for r in rs)
                / len(rs))
        pts[m] = (calls, acc(rs), errs)
    # 手工标注偏移: 气泡半径实测约2.4数据单位 → 垂直让位≥4.2 或水平≥2.9
    offs = {"greedy": (0.30, 10.8, "left"), "sc": (-3.4, 11.2, "right"),
            "tokbon": (-3.4, -13.2, "right"),
            "beam_symonly": (2.9, 0.0, "left"),
            "beam_fused": (0.30, -10.8, "left"),
            "beam": (0.38, 9.8, "left")}
    for m, (cx_, cy_, er) in pts.items():
        size = 20 + 30 * er
        ax.scatter(cx_, cy_, s=size, color=C[m], edgecolor=D[m], lw=0.9,
                   zorder=6, alpha=0.95)
        dx, dy, ha = offs[m]
        ax.annotate(SHORT[m], (cx_, cy_), xytext=(cx_ + dx, cy_ + dy),
                    fontsize=5.8, ha=ha, color=D[m], weight="bold")
    # 气泡尺寸图例 (左下空白区, 文字与圆圈充分错开)
    for er, yy in [(0.5, 2.5), (2.0, 9.5)]:
        ax.scatter(17.0, yy, s=20 + 30 * er, facecolors="none",
                   edgecolors=GREY, lw=0.7, zorder=6)
        ax.text(20.6, yy, f"{er:.1f} errors/q", fontsize=5.6, ha="left",
                va="center", color=GREY)
    ax.set_xlabel("LLM calls / question")
    ax.set_ylabel("Hits@1 (%)")
    ax.set_xlim(-1.0, 32.5); ax.set_ylim(0, 88)
    ax.grid(lw=0.35, color="#DDDDDD", zorder=0)
    ax.set_title("(d) Cost–accuracy frontier", fontsize=8.0, pad=4)

    for ax in axes:
        ax.tick_params(length=2.2, width=0.6, labelsize=6.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_results.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_results.pdf (v2)")


if __name__ == "__main__":
    main()
