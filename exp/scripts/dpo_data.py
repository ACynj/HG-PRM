#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/dpo_data.py — DPO 训练数据构建(状态对齐的步级偏好对)

数据源: results_main.jsonl 全部方法的 UMLS 轨迹(仅 split_comp=seen 用于训练)
配对原则: 按 (qid, 当前实体, 步序) 分组 — 同一状态下的动作才可比较:
  chosen  = 走后 on-track 的动作(优先) / 高过程分
  rejected= 幻觉动作 或 走后 off-track
三种信号:
  process : chosen/rejected 由 (on-track, 过程分) 决定 —— 本文方法
  outcome : 由"所在轨迹最终是否答对"决定 —— 传统结果奖励
  (SFT 数据: 黄金路径逐步 (obs, gold action))
obs 文本确定性重建(排序邻居, 含黄金邻居), 训练/评测一致。
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.kg_env import load_all  # noqa: E402

MAX_NB = 25
SEED = 5


def rebuild_obs(q, ents, cats, adj, node, step_i, ents_rev):
    """确定性 obs: 排序邻居 + 恒含黄金邻居。与训练/评测共用。"""
    gt_pair = None
    if step_i < len(q["gt_path"]):
        h_name, r, t_name = q["gt_path"][step_i]
        t_id = ents_rev.get(t_name, -1)
        gt_pair = (r, t_id) if t_id >= 0 else None
    pairs = sorted(set(adj[node]))
    if gt_pair:
        # 黄金邻居置于种子随机位置: 防"选第一条边"捷径(位置种子跨重跑一致)
        others = [pp for pp in pairs if pp != gt_pair][: MAX_NB - 1]
        rng = random.Random(hash((q["qid"], step_i, node)) & 0xffffff)
        pos = rng.randrange(len(others) + 1)
        keep = others[:pos] + [gt_pair] + others[pos:]
    else:
        keep = pairs[: MAX_NB]
    lines = [f"- ({r}) -> {ents[t]} [{cats[t]}]" for (r, t) in keep]
    if any(t not in ents for (_, t) in keep):
        keep = [(r, t) for (r, t) in keep if t in ents]
        lines = [f"- ({r}) -> {ents[t]} [{cats[t]}]" for (r, t) in keep]
    return (
        f"Question: Start at entity '{q['start']}'. Follow these relations "
        f"in order: {' -> '.join(q['rel_seq'])}. The final entity should be "
        f"of type '{q['answer_type']}'.\n"
        f"You are currently at: {ents[node]} [{cats[node]}] "
        f"(step {step_i + 1}/{q['hop']})\n"
        f"Available edges:\n" + "\n".join(lines))


def ontrack(adj, node, remaining, finals):
    frontier = {node}
    for r in remaining:
        frontier = {t for n in frontier for (rr, t) in adj[n] if rr == r}
        if not frontier:
            return False
    return bool(frontier & finals)


def main():
    rng = random.Random(SEED)
    data = json.loads((ROOT / "data" / "questions.json").read_text())
    qmap = {q["qid"]: q for q in data["questions"]}
    ents, cats, adj = load_all()
    ents_rev = {v: k for k, v in ents.items()}
    recs = [json.loads(l) for l in
            (ROOT / "outputs" / "results_main.jsonl").read_text().splitlines()]
    recs = [r for r in recs if "error" not in r and r["method"] != "bon_verifier"]

    # (qid, node, step) -> [(action, on_track, edge_ok, traj_correct)]
    groups = defaultdict(list)
    for r in recs:
        q = qmap[r["qid"]]
        finals = set(q.get("answer_ids", [q["answer_id"]]))
        cur = q["start_id"]
        for i, a in enumerate(r["actions"]):
            rel, nb = a["relation"], a.get("neighbor", "")
            nid = ents_rev.get(nb, -1)
            edge_ok = any(rr == rel and t == nid for (rr, t) in adj[cur])
            otk = edge_ok and ontrack(adj, nid, q["rel_seq"][i + 1:], finals)
            groups[(r["qid"], cur, i)].append(
                ({"relation": rel, "neighbor": nb}, otk, edge_ok, r["correct"]))
            if edge_ok:
                cur = nid

    def act_str(a):
        return json.dumps({"relation": a["relation"], "neighbor": a["neighbor"]})

    pairs_process, pairs_outcome, sft_rows = [], [], []
    for (qid, node, step_i), acts in groups.items():
        q = qmap[qid]
        if q["split_comp"] != "seen":      # heldout 不进训练
            continue
        prompt = rebuild_obs(q, ents, cats, adj, node, step_i, ents_rev)
        uniq = {}
        for a, otk, eok, corr in acts:
            key = act_str(a)
            if key not in uniq or (otk, eok) > (uniq[key][1], uniq[key][2]):
                uniq[key] = (a, otk, eok, corr)
        cand = list(uniq.values())
        if len(cand) < 2:
            continue
        # process: on-track 优先
        cand_p = sorted(cand, key=lambda x: (x[1], x[2]), reverse=True)
        if cand_p[0][1] != cand_p[-1][1] or cand_p[0][2] != cand_p[-1][2]:
            pairs_process.append({"prompt": prompt,
                                  "chosen": act_str(cand_p[0][0]),
                                  "rejected": act_str(cand_p[-1][0])})
        # outcome: 所在轨迹对 vs 错
        cand_o = sorted(cand, key=lambda x: x[3], reverse=True)
        if cand_o[0][3] and not cand_o[-1][3]:
            pairs_outcome.append({"prompt": prompt,
                                  "chosen": act_str(cand_o[0][0]),
                                  "rejected": act_str(cand_o[-1][0])})

    # SFT: 黄金路径逐步
    for q in data["questions"]:
        if q["split_comp"] != "seen":
            continue
        cur = q["start_id"]
        for i, (hn, r, tn) in enumerate(q["gt_path"]):
            prompt = rebuild_obs(q, ents, cats, adj, cur, i, ents_rev)
            sft_rows.append({"prompt": prompt,
                             "response": json.dumps({"relation": r,
                                                     "neighbor": tn})})
            cur = ents_rev[tn]

    rng.shuffle(pairs_process)
    rng.shuffle(pairs_outcome)
    out = ROOT / "outputs" / "dpo_data.json"
    out.write_text(json.dumps({
        "pairs_process": pairs_process, "pairs_outcome": pairs_outcome,
        "sft": sft_rows,
        "n_groups_seen": sum(1 for (qid, _, _) in groups
                             if qmap[qid]["split_comp"] == "seen"),
    }, ensure_ascii=False))
    print(f"process pairs: {len(pairs_process)} | outcome pairs: "
          f"{len(pairs_outcome)} | sft rows: {len(sft_rows)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
