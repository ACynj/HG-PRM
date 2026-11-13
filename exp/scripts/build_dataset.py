#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dataset.py — 从真实 UMLS KG 构建多跳问答数据集

输入: data/umls_src/*.txt  (dongwookim-ml/kg-data, R-GCN 基准)
输出: data/questions.json
  - 每题: start 实体 / 关系序列 / 期望尾类型 / 黄金答案 / 黄金路径
  - 唯一性保证: 从 start 按该关系序列可达终点唯一(否则丢弃)
  - 切分: hop 桶 {2,3,4,5} × 组合切分(seen/heldout meta-path)
  - 难度特征: 每步同关系邻居数(干扰度)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "umls_src"
OUT = ROOT / "data" / "questions.json"

SEED = 42
PER_HOP = 40          # 每个 hop 桶的问题数
HOPS = [2, 3, 4, 5]
HOLDOUT_FRAC = 0.30   # meta-path 组合切分: 留出 30% 类型序列
MAX_QUESTIONS_PER_SEQ = 3  # 同一类型序列最多生成几题(多样性)


def load():
    ents = {}
    for line in (SRC / "entities.txt").read_text().splitlines():
        i, name = line.split("\t")
        ents[int(i)] = name
    rels = {}
    for line in (SRC / "relations.txt").read_text().splitlines():
        i, name = line.split("\t")
        rels[int(i)] = name
    cats = {}
    for line in (SRC / "entity_category.txt").read_text().splitlines():
        i, c = line.split("\t")
        cats[int(i)] = c
    triples = []
    for line in (SRC / "triples.txt").read_text().splitlines():
        r, h, t = line.split("\t")
        triples.append((int(h), rels[int(r)], int(t)))
    return ents, rels, cats, triples


def build_adj(triples):
    adj = defaultdict(list)  # head -> [(relation, tail)]
    for h, r, t in triples:
        adj[h].append((r, t))
    return adj


def reachable_ends(adj, start, rel_seq):
    """从 start 严格按关系序列游走的全部终点。"""
    frontier = {start: [[]]}  # node -> list of paths
    for r in rel_seq:
        nxt = defaultdict(list)
        for node, paths in frontier.items():
            for (rr, t) in adj[node]:
                if rr == r:
                    for p in paths:
                        nxt[t].append(p + [(node, rr, t)])
        frontier = nxt
        if not frontier:
            return {}
    return frontier


def main():
    random.seed(SEED)
    ents, rels, cats, triples = load()
    adj = build_adj(triples)

    # schema: (head_cat, relation, tail_cat) 集合
    schema_triples = {(cats[h], r, cats[t]) for h, r, t in triples}

    questions = []
    seq_count = defaultdict(int)

    for hop in HOPS:
        attempts = 0
        got = 0
        while got < PER_HOP and attempts < 20000:
            attempts += 1
            start = random.choice(list(ents))
            # 采样关系序列: 每步从 start 出发现有关系的并集中取
            rel_pool = list({r for (_, r) in adj[start]}) or []
            if not rel_pool:
                continue
            # 逐步采样, 保证每一步都走得通
            rel_seq, cur = [], [start]
            nodes = {start}
            ok = True
            for _ in range(hop):
                # 从当前节点集合的所有出边关系中选
                pool = sorted({r for n in nodes for (r, _) in adj[n]})
                if not pool:
                    ok = False
                    break
                r = random.choice(pool)
                nxt = {t for n in nodes for (rr, t) in adj[n] if rr == r}
                if not nxt:
                    ok = False
                    break
                rel_seq.append(r)
                nodes = nxt
            if not ok:
                continue
            ends = reachable_ends(adj, start, rel_seq)
            if len(ends) != 1:
                continue  # 答案必须唯一
            ans, paths = next(iter(ends.items()))
            if ans == start or len(paths[0]) != hop:
                continue
            # 类型序列 key (组合切分用)
            gt = paths[0]
            tkey = tuple((r, cats[t]) for (_, r, t) in gt)
            if seq_count[tkey] >= MAX_QUESTIONS_PER_SEQ:
                continue
            # 干扰度: 每步同关系邻居数的平均
            distract = []
            for (h, r, t) in gt:
                same = sum(1 for (rr, _) in adj[h] if rr == r)
                distract.append(same)
            seq_count[tkey] += 1
            questions.append({
                "qid": f"umls-h{hop}-{got:03d}",
                "hop": hop,
                "start": ents[start],
                "start_id": start,
                "rel_seq": rel_seq,
                "answer": ents[ans],
                "answer_id": ans,
                "answer_type": cats[ans],
                "gt_path": [(ents[h], r, ents[t]) for (h, r, t) in gt],
                "type_seq": list(tkey),
                "distract_avg": sum(distract) / len(distract),
                "distract_max": max(distract),
            })
            got += 1

    # 组合切分: 按类型序列聚类, 30% 序列整体留出
    keys = sorted(seq_count)
    random.shuffle(keys)
    holdout = set(keys[: int(len(keys) * HOLDOUT_FRAC)])
    for q in questions:
        q["split_comp"] = "heldout" if tuple(q["type_seq"]) in holdout else "seen"

    OUT.write_text(json.dumps({
        "source": "UMLS (dongwookim-ml/kg-data, R-GCN benchmark)",
        "schema": [list(x) for x in sorted(schema_triples)],
        "n_entities": len(ents),
        "n_relations": len(rels),
        "n_triples": len(triples),
        "questions": questions,
    }, ensure_ascii=False, indent=1))

    # 统计
    print(f"schema triples: {len(schema_triples)}")
    for hop in HOPS:
        qs = [q for q in questions if q["hop"] == hop]
        ho = [q for q in qs if q["split_comp"] == "heldout"]
        print(f"hop={hop}: {len(qs)} questions "
              f"(heldout {len(ho)}), avg-distract "
              f"{sum(q['distract_avg'] for q in qs)/max(1,len(qs)):.1f}")
    print(f"unique type-seqs: {len(keys)}, holdout: {len(holdout)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
