#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_fig3.py — Fig.3: (a)测试时扩展曲线 (b)DPO训练结果
单栏宽(3.5in)双面板纵向堆叠, 马卡龙配色, 高信息密度。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper_figs"

M_PINK, M_PISTACH, M_LEMON = "#F2A7B3", "#A9D6B8", "#F7E8A4"
M_LAVEND, M_CARAMEL = "#C8B8E8", "#E8C9A0"
D_PINK, D_PISTACH, D_LEMON = "#C05C78", "#4E8A6A", "#B8952E"
D_LAVEND, D_CARAMEL = "#7A5FA8", "#A67B4E"
D_BLUE = "#3F6AB0"
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.6,
    "axes.linewidth": 0.7, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "pdf.fonttype": 42,
})


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4),
                                   gridspec_kw=dict(hspace=0.52))

    # ---------- (a) 测试时扩展曲线 ----------
    Ns = [1, 2, 4, 8]
    sc = [21.0, 21.0, 19.7, 20.4]
    vr = [21.0, 24.2, 26.1, 28.0]
    ah = [21.0, 29.9, 38.2, 51.6]
    ax1.plot(Ns, sc, "-o", color=D_LEMON, lw=1.0, ms=2.8,
             markerfacecolor=M_LEMON, markeredgecolor=D_LEMON,
             label="SC@$n$")
    ax1.plot(Ns, vr, "-s", color=D_LAVEND, lw=1.0, ms=2.8,
             markerfacecolor=M_LAVEND, markeredgecolor=D_LAVEND,
             label="Verif-Rerank@$n$")
    ax1.plot(Ns, ah, "--D", color=D_CARAMEL, lw=1.2, ms=3.0,
             markerfacecolor=M_CARAMEL, markeredgecolor=D_CARAMEL,
             label="Any-hit oracle@$n$")
    ax1.axhline(67.5, color=D_PINK, lw=1.8)
    ax1.text(7.9, 70.4, "HG-PRM beam (ours): 67.5", fontsize=6.6,
             color=D_PINK, weight="bold", ha="right")
    ax1.annotate("oracle ceiling\nat $n{=}8$", xy=(8, 51.6),
                 xytext=(4.4, 45.5), fontsize=5.8, color=D_CARAMEL,
                 arrowprops=dict(arrowstyle="-", color=D_CARAMEL, lw=0.7))
    ax1.set_xticks(Ns)
    ax1.set_xticklabels(["1", "2", "4", "8"])
    ax1.set_xlabel("Sampled trajectories $n$ (T=1.0)")
    ax1.set_ylabel("Hits@1 (%)")
    ax1.set_ylim(10, 100)
    ax1.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax1.legend(fontsize=5.8, frameon=False, loc="upper left",
               handlelength=1.3, labelspacing=0.3, borderpad=0.3)
    ax1.set_title("(a) Post-hoc selection does not scale;\n"
                  "in-loop verification beats the oracle ceiling",
                  fontsize=7.2, pad=4)
    ax1.tick_params(length=2.2, width=0.6, labelsize=6.4)

    # ---------- (b) DPO 训练结果 ----------
    models = ["Base", "SFT", "DPO\noutcome", "DPO\nprocess",
              "DPO\nproc+out"]
    seen = [39.8, 67.8, 84.7, 51.7, 87.3]
    held = [23.8, 54.8, 64.3, 52.4, 73.8]
    viol = [1.32, 0.64, 0.40, 0.49, 0.29]
    x = np.arange(len(models))
    b1 = ax2.bar(x - 0.19, seen, 0.36, color=M_PISTACH,
                 edgecolor=D_PISTACH, lw=0.6, label="seen", zorder=3)
    b2 = ax2.bar(x + 0.19, held, 0.36, color=M_PINK,
                 edgecolor=D_PINK, lw=0.6, label="held-out", zorder=3)
    for xi, (s, h) in enumerate(zip(seen, held)):
        ax2.text(xi, max(s, h) + 2.2, f"Δ{h-s:+.1f}", ha="center",
                 fontsize=5.4, color=GREY)
    # 违规率折线(右轴)
    axv = ax2.twinx()
    axv.plot(x, viol, "-^", color=D_BLUE, lw=1.2, ms=3.4,
             markerfacecolor="#9DB8E8", markeredgecolor=D_BLUE,
             label="viol./q", zorder=5)
    axv.set_ylabel("Type violations / q", fontsize=6.6, color=D_BLUE)
    axv.set_ylim(0, 2.0)
    axv.tick_params(length=2.2, width=0.6, labelsize=6.0, colors=D_BLUE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=6.0)
    ax2.set_ylabel("Hits@1 (%)")
    ax2.set_ylim(0, 120)
    ax2.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = axv.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=5.8, frameon=False,
               loc="upper left", handlelength=1.3, labelspacing=0.3)
    ax2.set_title("(b) Training-time: process pairs close the\n"
                  "compositional gap (DPO, local Qwen2.5-7B)",
                  fontsize=7.2, pad=4)
    ax2.tick_params(length=2.2, width=0.6, labelsize=6.4)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    axv.spines["top"].set_visible(False)
    axv.spines["right"].set_color(D_BLUE)
    axv.spines["right"].set_linewidth(0.7)
    fig.savefig(OUT / "fig_training.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fig_training.pdf")


if __name__ == "__main__":
    main()
