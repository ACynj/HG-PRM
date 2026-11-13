#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dataset_aminer.py — Aminer 开放协议多跳数据集(大规模第三域)

真实姓名实体(16.4万作者/12.7万论文/101会议/392k term边, 约29万节点)
开放协议: 问题只给 (起始实体, 期望答案类型, 步数预算), 不声明关系序列
  → 动作空间 = 全部 schema 合法边, C2' 掩码泛化为 C1 schema 合法性
黄金路径仅用于: 答案集合生成(any-hit<=3) + tokbon 的 oracle 词面奖励
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "aminer_src"
OUT = ROOT / "data" / "questions_aminer.json"

SEED = 23
PER_HOP = 30
HOPS = [2, 3, 4, 5]
MAX_FINALS = 3
MAX_FRONTIER = 800


def load_aminer():
    authors = {}          # idx -> name  (文件格式: name\tid)
    for line in (SRC / "author2id.txt").read_text(encoding="utf-8",
                                                  errors="ignore").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2 and parts[1].strip().isdigit() and parts[0].strip():
            authors[int(parts[1])] = parts[0][:40]
    confs = {}
    for line in (SRC / "conf2id.txt").read_text(errors="ignore").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2 and parts[1].strip().isdigit() and parts[0].strip():
            confs[int(parts[1])] = parts[0][:50]
    pa, pc, pt = [], [], []

    def pairs(fname):
        for line in (SRC / fname).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                yield int(parts[0]), int(parts[1])

    pa = list(pairs("paper_author.txt"))
    pc = list(pairs("paper_conference.txt"))
    pt = list(pairs("paper_type.txt"))

    ents, cats = {}, {}
    A0, C0, T0 = 10 ** 6, 2 * 10 ** 6, 3 * 10 ** 6
    for p in {p for p, _ in pa} | {p for p, _ in pc} | {p for p, _ in pt}:
        ents[p] = f"Paper#{p}"
        cats[p] = "Paper"
    for a in {a for _, a in pa}:
        ents[A0 + a] = authors.get(a, f"Author#{a}")
        cats[A0 + a] = "Author"
    for c in {c for _, c in pc}:
        ents[C0 + c] = confs.get(c, f"Conference#{c}")
        cats[C0 + c] = "Conference"
    for t in {t for _, t in pt}:
        ents[T0 + t] = f"Term#{t}"
        cats[T0 + t] = "Term"

    adj = defaultdict(list)
    for p, a in pa:
        adj[p].append(("written_by", A0 + a))
        adj[A0 + a].append(("writes", p))
    for p, c in pc:
        adj[p].append(("published_in", C0 + c))
        adj[C0 + c].append(("publishes", p))
    for p, t in pt:
        adj[p].append(("has_term", T0 + t))
        adj[T0 + t].append(("term_of", p))
    schema = {(cats[h], r, cats[t]) for h in adj for (r, t) in adj[h]}
    return ents, cats, adj, schema


def reachable_sets(adj, start, rel_seq):
    frontiers = [{start}]
    for r in rel_seq:
        nxt = {t for n in frontiers[-1] for (rr, t) in adj[n] if rr == r}
        if not nxt or len(nxt) > MAX_FRONTIER:
            return None
        frontiers.append(nxt)
    return frontiers


def reconstruct_path(adj, ents, start, rel_seq, answer):
    path, cur = [], start
    for i, r in enumerate(rel_seq):
        rem = rel_seq[i + 1:]
        for (rr, t) in adj[cur]:
            if rr != r:
                continue
            if not rem:
                if t == answer:
                    path.append((ents[cur], r, ents[t]))
                    return path
            else:
                fs = reachable_sets(adj, t, rem)
                if fs and answer in fs[-1]:
                    path.append((ents[cur], r, ents[t]))
                    cur = t
                    break
        else:
            return None
    return path


def main():
    random.seed(SEED)
    ents, cats, adj, schema = load_aminer()
    ents_rev = {v: k for k, v in ents.items()}
    print(f"nodes {len(ents)}, edges {sum(len(v) for v in adj.values())}, "
          f"schema {len(schema)}")

    paper_authors = defaultdict(set)
    author_papers = defaultdict(set)
    for h in ents:
        for (r, t) in adj[h]:
            if r == "written_by":
                paper_authors[h].add(t)
                author_papers[t].add(h)
    small_authors = [a for a, ps in author_papers.items() if len(ps) <= 3]
    single_authors = [a for a in author_papers
                      if 1 <= len(author_papers[a]) <= 3
                      and all(paper_authors[p] == {a}
                              for p in author_papers[a])]
    papers_1a = [p for p, aset in paper_authors.items()
                 if 1 <= len(aset) <= 3
                 and any(len(author_papers[x]) <= 3 for x in aset)]
    print(f"small_authors {len(small_authors)}, single {len(single_authors)}, "
          f"papers_1a {len(papers_1a)}")

    questions, seq_count = [], defaultdict(int)

    def add_q(hop, start, rel_seq, got):
        fs = reachable_sets(adj, start, rel_seq)
        if not fs or not (1 <= len(fs[-1]) <= MAX_FINALS):
            return None
        finals = sorted(fs[-1])
        answer = finals[0]
        if answer == start:
            return None
        path = reconstruct_path(adj, ents, start, rel_seq, answer)
        if not path or len(path) != hop:
            return None
        tkey = tuple((r, cats[ents_rev[path[i][2]]])
                     for i, (_, r, _) in enumerate(path))
        if seq_count[tkey] >= PER_HOP:
            return None
        distract, cur = [], start
        for (hn, r, tn) in path:
            distract.append(sum(1 for (rr, _) in adj[cur] if rr == r))
            cur = ents_rev[tn]
        seq_count[tkey] += 1
        questions.append({
            "qid": f"aminer-h{hop}-{got:03d}",
            "hop": hop, "start": ents[start], "start_id": start,
            "rel_seq": rel_seq,          # 黄金路径溯源, 不展示给 agent
            "protocol": "open",
            "answer": ents[answer], "answer_id": answer,
            "answer_ids": finals, "n_finals": len(finals),
            "answer_type": cats[answer],
            "gt_path": path, "type_seq": list(tkey),
            "distract_avg": sum(distract) / len(distract),
            "distract_max": max(distract),
        })
        return True

    for hop in HOPS:
        got, attempts = 0, 0
        while got < PER_HOP and attempts < 8000:
            attempts += 1
            if hop == 2:
                a = random.choice(small_authors)
                ok = add_q(hop, a, ["writes", "published_in"], got)
            elif hop == 3:
                p = random.choice(papers_1a)
                ok = add_q(hop, p, ["written_by", "writes", "published_in"],
                           got)
            elif hop == 4:
                a0 = random.choice(single_authors if random.random() < 0.7
                                   else small_authors)
                ok = add_q(hop, a0,
                           ["writes", "written_by", "writes", "published_in"],
                           got)
            else:
                a0 = random.choice(single_authors if random.random() < 0.7
                                   else small_authors)
                p0 = random.choice(sorted(author_papers[a0]))
                ok = add_q(hop, p0,
                           ["written_by", "writes", "written_by", "writes",
                            "published_in"], got)
            if ok:
                got += 1

    keys = sorted(seq_count)
    random.shuffle(keys)
    holdout = set(keys[: int(len(keys) * 0.30)])
    for q in questions:
        q["split_comp"] = "heldout" if tuple(q["type_seq"]) in holdout else "seen"

    OUT.write_text(json.dumps({
        "source": "Aminer (Jhy1993/Datasets-for-Heterogeneous-Graph), "
                  "open protocol",
        "schema": [list(x) for x in sorted(schema)],
        "n_entities": len(ents), "protocol": "open",
        "questions": questions,
    }, ensure_ascii=False, indent=1))
    for hop in HOPS:
        qs = [q for q in questions if q["hop"] == hop]
        ho = [q for q in qs if q["split_comp"] == "heldout"]
        print(f"hop={hop}: {len(qs)} (heldout {len(ho)}), distract "
              f"{sum(q['distract_avg'] for q in qs)/max(1,len(qs)):.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
