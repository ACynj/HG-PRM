#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_fig_reward.py — 奖励质量图(单栏双面板)
(a) 步级符号分分布: 合法边 vs 幻觉边 直方图 + 均值标注
(b) ROC: 步级(AUC 0.787) 与轨迹级(AUC 0.893) 曲线
数据: 重放 results_main.jsonl 全部轨迹并重算可部署符号分(同 reward_analysis)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper_figs"

M_PINK, M_PISTACH, M_LAVEND, M_CARAMEL = "#F2A7B3", "#A9D6B8", "#C8B8E8", "#E8C9A0"
D_PINK, D_PISTACH, D_LAVEND, D_CARAMEL = "#C05C78", "#4E8A6A", "#7A5FA8", "#A67B4E"
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.4,
    "axes.linewidth": 0.7, "axes.edgecolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "pdf.fonttype": 42,
})


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_steps():
    """重放全部 UMLS 轨迹 → (步级符号分, 步级合法?, 轨迹均分, 轨迹正确?)"""
    from env.kg_env import load_all
    from symbolic.checker import Schema
    from agents.policies import symbolic_score

    data = json.loads((ROOT / "data" / "questions.json").read_text())
    qmap = {q["qid"]: q for q in data["questions"]}
    schema = Schema(triples={tuple(t) for t in data["schema"]})
    ents, cats, adj = load_all()
    ents_rev = {v: k for k, v in ents.items()}

    def ontrack(node, remaining, finals):
        frontier = {node}
        for r in remaining:
            frontier = {t for n in frontier for (rr, t) in adj[n] if rr == r}
            if not frontier:
                return False
        return bool(frontier & finals)

    recs = [json.loads(l) for l in
            (ROOT / "outputs" / "results_main.jsonl").read_text().splitlines()]
    recs = [r for r in recs if "error" not in r and r["method"] in (
        "greedy", "beam", "beam_symonly", "beam_nomask", "beam_b1")]
    step_score, step_pos, step_otrack = [], [], []
    traj_score, traj_correct = [], []
    for r in recs:
        q = qmap[r["qid"]]
        finals = set(q.get("answer_ids", [q["answer_id"]]))
        cur = q["start_id"]
        ts = []
        for i, a in enumerate(r["actions"]):
            rel, nb = a["relation"], a.get("neighbor", "")
            sym = symbolic_score(q, schema, ents, cats, adj, cur, rel, nb,
                                 ents_rev, i)
            nid = ents_rev.get(nb, -1)
            ok = any(rr == rel and t == nid for (rr, t) in adj[cur])
            ot = int(ok and ontrack(nid, q["rel_seq"][i + 1:], finals))
            step_score.append(sym)
            step_pos.append(int(ok))
            step_otrack.append(ot)
            ts.append(sym)
            if ok:
                cur = nid
        traj_score.append(sum(ts) / max(1, len(ts)))
        traj_correct.append(int(r["correct"]))
    return (np.array(step_score), np.array(step_pos),
            np.array(step_otrack), np.array(traj_score),
            np.array(traj_correct))


def roc(scores, labels):
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    tps = np.cumsum(l); fps = np.cumsum(1 - l)
    tpr = np.concatenate([[0], tps / max(1, l.sum())])
    fpr = np.concatenate([[0], fps / max(1, (1 - l).sum())])
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def main():
    ss, sp, so, ts, tc = load_steps()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.45),
                                   gridspec_kw=dict(wspace=0.34))

    # ---- (a) 步级分数分布 ----
    bins = np.linspace(0, 1, 21)
    ax1.hist(ss[sp == 1], bins=bins, color=M_PISTACH, edgecolor=D_PISTACH,
             lw=0.6, label=f"valid edge (n={int(sp.sum())})", zorder=3)
    ax1.hist(ss[sp == 0], bins=bins, color=M_PINK, edgecolor=D_PINK,
             lw=0.6, label=f"hallucinated (n={int((1-sp).sum())})", zorder=3,
             alpha=0.92)
    m1, m0 = ss[sp == 1].mean(), ss[sp == 0].mean()
    ax1.axvline(m1, color=D_PISTACH, lw=1.1, ls="--")
    ax1.axvline(m0, color=D_PINK, lw=1.1, ls="--")
    ax1.annotate(f"{m1:.2f}", xy=(m1, ax1.get_ylim()[1]*0.9),
                 fontsize=6.0, color=D_PISTACH, ha="center", weight="bold")
    ax1.annotate(f"{m0:.2f}", xy=(m0 + 0.30, ax1.get_ylim()[1]*0.45),
                 fontsize=6.0, color=D_PINK, ha="center", weight="bold")
    ax1.set_xlabel("Deployable symbolic score $r_{sym}$")
    ax1.set_ylabel("Reasoning steps")
    ax1.set_xlim(0, 1.02)
    ax1.legend(fontsize=5.8, frameon=False, loc="upper center")
    ax1.set_title("(a) Step scores separate hallucinations", fontsize=7.6,
                  pad=4)
    ax1.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)
    ax1.tick_params(length=2.2, width=0.6, labelsize=6.4)
    despine(ax1)

    # ---- (b) ROC ----
    f0, t0, a0 = roc(ss, sp)          # 合法 vs 幻觉
    f1, t1, a1 = roc(ss, so)          # 在轨判别
    f2, t2, a2 = roc(ts, tc)          # 轨迹正确
    ax2.plot(f0, t0, color=D_PISTACH, lw=1.3,
             label=f"valid vs. hallucinated (AUC {a0:.3f})")
    ax2.plot(f1, t1, color=D_LAVEND, lw=1.3,
             label=f"on-track (AUC {a1:.3f})")
    ax2.plot(f2, t2, color=D_PINK, lw=1.6,
             label=f"trajectory success (AUC {a2:.3f})")
    ax2.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls=":")
    ax2.set_xlabel("False-positive rate")
    ax2.set_ylabel("True-positive rate")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=5.6, frameon=False, loc="lower right")
    ax2.set_title("(b) Reward quality, no answer info used", fontsize=7.6,
                  pad=4)
    ax2.grid(lw=0.35, color="#DDDDDD", zorder=0)
    ax2.tick_params(length=2.2, width=0.6, labelsize=6.4)
    despine(ax2)

    fig.savefig(OUT / "fig_reward.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"fig_reward.pdf  stepAUC={a1:.3f} trajAUC={a2:.3f}")


if __name__ == "__main__":
    main()
