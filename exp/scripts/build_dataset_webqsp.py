#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dataset_webqsp.py — WebQSP (RoG 版) 开放问答基准
- 图: 每题 Freebase 子图(实体名+关系名); 类型自动推导: 节点出边关系的 domain 前缀
- schema: 子图内 (head_type, relation, tail_type) 归纳
- 协议: 开放问答(openqa) — 自然语言问题 + 起始实体 + 固定步预算;
  判分: 路径上任一访问节点 ∈ 答案集 (visit-based, WebQSP 标准口径)
- 选题: 答案在子图内且距种子 2-3 跳, 图规模分层抽样
"""
from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "webqsp_src" / "data"
OUT = ROOT / "data" / "questions_webqsp.json"
SEED = 31
N_QUESTIONS = 400
HOPS = 4


def domain_of(rel: str) -> str:
    parts = rel.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def load_test():
    rows = []
    for f in sorted(SRC.glob("test-*.parquet")):
        rows += pq.read_table(f).to_pylist()
    return rows


def main():
    rng = random.Random(SEED)
    rows = load_test()
    print(f"test rows: {len(rows)}")

    questions, schema_all = [], set()
    # 按图规模分层: 小(<300), 中(300-800), 大(>800)
    buckets = defaultdict(list)

    for it in rows:
        q_ent = (it.get("q_entity") or [None])[0]
        answers = set(it.get("a_entity") or [])
        if not q_ent or not answers:
            continue
        g = it["graph"]
        if not g or len(g) > 3000:
            continue
        adj = defaultdict(list)
        for h, r, t in g:
            adj[h].append((r, t))
        if q_ent not in adj:
            continue
        # 答案需在子图内且 2-3 跳可达
        dist = {q_ent: 0}
        dq = deque([q_ent])
        while dq:
            n = dq.popleft()
            if dist[n] >= 3:
                continue
            for _, t in adj[n]:
                if t not in dist:
                    dist[t] = dist[n] + 1
                    dq.append(t)
        reach = {a: d for a, d in dist.items() if a in answers and 2 <= d <= 3}
        if not reach:
            continue
        # 类型推导: 出边 domain
        ntype = {}
        for n, edges in adj.items():
            ntype[n] = domain_of(edges[0][0]) if edges else "fb.leaf"
        for h, r, t in g:
            schema_all.add((ntype.get(h, "fb.leaf"), r, ntype.get(t, "fb.leaf")))
        size = "小" if len(g) < 300 else ("中" if len(g) <= 800 else "大")
        buckets[size].append({
            "qid": it["id"], "hop": HOPS, "protocol": "openqa",
            "question": it["question"],
            "start": q_ent, "start_id": q_ent,
            "answers": sorted(answers), "answer_ids": sorted(answers),
            "n_finals": len(answers),
            "answer_type": ntype.get(sorted(answers)[0], "fb.leaf"),
            "min_dist": min(reach.values()),
            "graph_size": len(g),
            "distract_avg": len(adj[q_ent]), "distract_max": len(adj[q_ent]),
            "_graph": g, "_ntype": ntype,
        })

    per = N_QUESTIONS // 3
    picked = []
    for size in ("小", "中", "大"):
        pool = buckets[size]
        rng.shuffle(pool)
        picked += pool[:per]   # 各桶取满上限 → 用尽可用题
    print({k: len(v) for k, v in buckets.items()}, "-> 抽取", len(picked))

    # 子图合并为一张全局图(实体名全局唯一性近似: 名字相同即同实体)
    adj_all = defaultdict(list)
    for q in picked:
        for h, r, t in q["_graph"]:
            adj_all[h].append((r, t))
            adj_all[t]  # 确保尾节点存在
        q["gt_path"] = []  # 开放问答无唯一黄金路径
        q["rel_seq"] = []
        q["type_seq"] = []
        q["split_comp"] = "seen"
        del q["_graph"], q["_ntype"]

    (ROOT / "data" / "webqsp_graph.json").write_text(json.dumps(
        {"triples": [[h, r, t] for h, edges in adj_all.items()
                      for r, t in edges],
         "ntype": {n: (domain_of(edges[0][0]) if edges else "fb.leaf")
                   for n, edges in adj_all.items()}}, ensure_ascii=False))
    OUT.write_text(json.dumps({
        "source": "WebQSP (RoG-processed, test split), open-QA protocol",
        "schema": [list(x) for x in sorted(schema_all)],
        "n_entities": len(adj_all),
        "protocol": "openqa",
        "questions": picked,
    }, ensure_ascii=False, indent=1))
    import statistics
    print(f"图节点 {len(adj_all)}, schema {len(schema_all)}")
    print(f"min_dist 分布: 2跳 {sum(1 for q in picked if q['min_dist']==2)}, "
          f"3跳 {sum(1 for q in picked if q['min_dist']==3)}")
    print(f"平均图规模 {statistics.mean(q['graph_size'] for q in picked):.0f} -> {OUT}")


if __name__ == "__main__":
    main()
