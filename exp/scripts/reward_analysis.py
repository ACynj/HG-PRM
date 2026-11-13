#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reward_analysis.py — 奖励信号质量分析(零API成本)

对 results_main.jsonl 中全部已记录轨迹的每个动作:
  r_sym : 可部署符号分(C1 schema + 局部可执行 + C3 类型可达; 不用黄金答案)
  y     : oracle 在轨标签(沿剩余必需关系序列可到任一答案)
指标:
  AUC(r_sym → y)   符号分对"在轨"的判别力
  分桶均值          类型违规步 sym<=0.4 vs 合法步
  轨迹级相关性      符号均分 vs 轨迹正确 (Spearman) —— "奖励引导方向正确性"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.kg_env import load_all, diagnose_trajectory  # noqa: E402
from symbolic.checker import Schema  # noqa: E402
from agents.policies import symbolic_score  # noqa: E402


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    data = json.loads((ROOT / "data" / "questions.json").read_text())
    schema = Schema(triples={tuple(t) for t in data["schema"]})
    ents, cats, adj = load_all()
    ents_rev = {v: k for k, v in ents.items()}
    qmap = {q["qid"]: q for q in data["questions"]}

    def ontrack(cur, remaining, finals):
        frontier = {cur}
        for r in remaining:
            frontier = {t for n in frontier for (rr, t) in adj[n] if rr == r}
            if not frontier:
                return False
        return bool(frontier & finals)

    recs = [json.loads(l) for l in
            (ROOT / "outputs" / "results_main.jsonl").read_text().splitlines()]
    recs = [r for r in recs if "error" not in r
            and r["method"] in ("greedy", "beam", "beam_symonly",
                                "beam_nomask", "beam_b1")]

    step_scores, step_labels = [], []
    viol_scores, valid_scores = [], []
    traj_sym, traj_correct = [], []
    n_steps = n_viol = 0
    for r in recs:
        q = qmap[r["qid"]]
        finals = set(q.get("answer_ids", [q["answer_id"]]))
        cur = q["start_id"]
        t_sym = []
        y_traj = []
        for i, a in enumerate(r["actions"]):
            rel, nb = a["relation"], a.get("neighbor", "")
            sym = symbolic_score(q, schema, ents, cats, adj, cur, rel, nb,
                                 ents_rev, i)
            step_scores.append(sym)
            # oracle 标签: 动作真实存在 且 动作后仍在轨
            nid = ents_rev.get(nb, -1)
            edge_ok = any(rr == rel and t == nid for (rr, t) in adj[cur])
            y = int(edge_ok and ontrack(nid, q["rel_seq"][i + 1:], finals))
            step_labels.append(y)
            t_sym.append(sym)
            y_traj.append(y)
            if edge_ok:
                valid_scores.append(sym)
                cur = nid
            else:
                viol_scores.append(sym)
            n_steps += 1
            n_viol += int(not edge_ok)
        traj_sym.append(sum(t_sym) / max(1, len(t_sym)))
        traj_correct.append(int(r["correct"]))

    print(f"steps analyzed: {n_steps} (type-violating: {n_viol})")
    print(f"AUC(r_sym -> on-track)        : {auc(step_scores, step_labels):.3f}")
    print(f"mean r_sym | valid edge       : "
          f"{sum(valid_scores)/max(1,len(valid_scores)):.3f} (n={len(valid_scores)})")
    print(f"mean r_sym | hallucinated edge: "
          f"{sum(viol_scores)/max(1,len(viol_scores)):.3f} (n={len(viol_scores)})")

    # 轨迹级 Spearman(简化: 皮尔逊 + 分组正确率)
    import statistics as st
    correct_syms = [s for s, c in zip(traj_sym, traj_correct) if c]
    wrong_syms = [s for s, c in zip(traj_sym, traj_correct) if not c]
    print(f"mean traj r_sym | correct     : "
          f"{st.mean(correct_syms):.3f} (n={len(correct_syms)})")
    print(f"mean traj r_sym | wrong       : "
          f"{st.mean(wrong_syms):.3f} (n={len(wrong_syms)})")
    print(f"AUC(traj r_sym -> correct)    : "
          f"{auc(traj_sym, traj_correct):.3f}")


if __name__ == "__main__":
    main()
