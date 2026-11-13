#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dataset_dblp.py — DBLP 第二域多跳问答数据集(跨域实验)

数据源: Jhy1993/Datasets-for-Heterogeneous-Graph DBLP_1 (HAN 系基准)
  节点类型: Paper(14376) / Author(14475) / Conference(20) / Term(8920)
  关系(含逆): writes, written_by, published_in, publishes, has_term, term_of

设计:
  - 实体名用 ID (Author #123): 剥离语义捷径, 纯结构推理 → 对 token-overlap
    基线是最有利设置(词面=ID), 对我们的符号层无影响
  - hop {2,3,4,5}; 集合前向可达判唯一(不自爆路径枚举)
  - 黄金路径: 反向可达集贪心重建
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "dblp_src"
OUT = ROOT / "data" / "questions_dblp.json"

SEED = 11
PER_HOP = 30
HOPS = [2, 3, 4, 5]
HOLDOUT_FRAC = 0.30
MAX_PER_SEQ = 2


def load_dblp():
    pa = []   # (paper, author)
    for line in (SRC / "paper_author.dat").read_text().splitlines():
        p, a, _ = line.split("\t")
        pa.append((int(p), int(a)))
    pc = []
    for line in (SRC / "paper_conference.dat").read_text().splitlines():
        p, c, _ = line.split("\t")
        pc.append((int(p), int(c)))
    pt = []
    for line in (SRC / "paper_type.dat").read_text().splitlines():
        p, t, _ = line.split("\t")
        pt.append((int(p), int(t)))

    ents, cats = {}, {}
    # paper: 1..max_paper
    max_p = max(max(p for p, _ in pa), max(p for p, _ in pc),
                max(p for p, _ in pt))
    for p in range(1, max_p + 1):
        ents[p] = f"Paper#{p}"
        cats[p] = "Paper"
    max_a = max(a for _, a in pa)
    for a in range(1, max_a + 1):
        ents[10 ** 6 + a] = f"Author#{a}"
        cats[10 ** 6 + a] = "Author"
    max_c = max(c for _, c in pc)
    for c in range(1, max_c + 1):
        ents[2 * 10 ** 6 + c] = f"Conf#{c}"
        cats[2 * 10 ** 6 + c] = "Conference"
    max_t = max(t for _, t in pt)
    for t in range(1, max_t + 1):
        ents[3 * 10 ** 6 + t] = f"Term#{t}"
        cats[3 * 10 ** 6 + t] = "Term"

    adj = defaultdict(list)
    for p, a in pa:
        adj[p].append(("written_by", 10 ** 6 + a))     # paper -> author
        adj[10 ** 6 + a].append(("writes", p))         # author -> paper
    for p, c in pc:
        adj[p].append(("published_in", 2 * 10 ** 6 + c))
        adj[2 * 10 ** 6 + c].append(("publishes", p))
    for p, t in pt:
        adj[p].append(("has_term", 3 * 10 ** 6 + t))
        adj[3 * 10 ** 6 + t].append(("term_of", p))

    # schema 归纳(粗类型级)
    schema = {(cats[h], r, cats[t]) for h in adj for (r, t) in adj[h]}
    return ents, cats, adj, schema


MAX_FRONTIER = 1500  # 超过即放弃(防扇出爆炸; 此类walk几乎不可能终点唯一)

def reachable_sets(adj, start, rel_seq):
    """逐步前向可达节点集合列表 (集合制, 不枚举路径)。"""
    frontiers = [{start}]
    for r in rel_seq:
        nxt = {t for n in frontiers[-1] for (rr, t) in adj[n] if rr == r}
        if not nxt or len(nxt) > MAX_FRONTIER:
            return None
        frontiers.append(nxt)
    return frontiers


def reconstruct_path(adj, ents, cats, start, rel_seq, answer, ents_rev):
    """黄金路径贪心重建: 每步选一个仍在'可到答案'集合中的邻居。"""
    path, cur = [], start
    for i, r in enumerate(rel_seq):
        rem = rel_seq[i + 1:]
        for (rr, t) in adj[cur]:
            if rr != r:
                continue
            if not rem:
                if t == answer:
                    path.append((ents[cur], r, ents[t]))
                    return path, t
            else:
                fs = reachable_sets(adj, t, rem)
                if fs and answer in fs[-1]:
                    path.append((ents[cur], r, ents[t]))
                    cur = t
                    break
        else:
            return None, cur
    return path, cur


def main():
    random.seed(SEED)
    ents, cats, adj, schema = load_dblp()
    ents_rev = {v: k for k, v in ents.items()}
    print(f"nodes {len(ents)}, edges {sum(len(v) for v in adj.values())}, "
          f"schema {len(schema)}")

    questions, seq_count = [], defaultdict(int)
    # 预计算类型转移表: type -> {rel}, (type, rel) -> {target_types}
    ok_next = defaultdict(set)
    rel_targets = defaultdict(set)
    for h in adj:
        for (r, t) in adj[h]:
            ok_next[cats[h]].add(r)
            rel_targets[(cats[h], r)].add(cats[t])

    # ============ 反向构造: 4 种模式, 天然满足 any-hit (finals<=3) ============
    # hop2: A --writes--> P --published_in--> C(finals=confs(A.papers))
    # hop3: P --written_by--> A --writes--> P --published_in--> C   (P 单作者)
    # hop4: A --writes--> P --written_by--> A2 (finals=papers(A2))
    # hop5: A --writes--> P --written_by--> A2 --writes--> P2 --published_in--> C
    MAX_FINALS = 3
    paper_authors = defaultdict(set)
    author_papers = defaultdict(set)
    for p, a in [(int(l.split("\t")[0]), int(l.split("\t")[1]))
                 for l in (SRC / "paper_author.dat").read_text().splitlines()]:
        paper_authors[p].add(10 ** 6 + a)
        author_papers[10 ** 6 + a].add(p)
    small_authors = [a for a, ps in author_papers.items() if len(ps) <= 3]
    papers_1a_3p = []  # 论文: 作者数<=3 且 至少一位作者 papers<=3
    for p, aset in paper_authors.items():
        if 1 <= len(aset) <= 3 and any(len(author_papers[x]) <= 3
                                       for x in aset):
            papers_1a_3p.append(p)

    def finals_of(start, rel_seq):
        fs = reachable_sets(adj, start, rel_seq)
        return fs

    def add_q(hop, start, rel_seq, got):
        fs = finals_of(start, rel_seq)
        if not fs or not (1 <= len(fs[-1]) <= MAX_FINALS):
            return None
        finals = sorted(fs[-1])
        answer = finals[0]
        if answer == start:
            return None
        path, _ = reconstruct_path(adj, ents, cats, start, rel_seq, answer,
                                   ents_rev)
        if not path or len(path) != hop:
            return None
        tkey = tuple((r, cats[ents_rev[path[i][2]]])
                     for i, (_, r, _) in enumerate(path))
        # DBLP 类型序列本身少, 上限放宽到 PER_HOP
        if seq_count[tkey] >= PER_HOP:
            return None
        distract, cur = [], start
        for (hn, r, tn) in path:
            distract.append(sum(1 for (rr, _) in adj[cur] if rr == r))
            cur = ents_rev[tn]
        seq_count[tkey] += 1
        questions.append({
            "qid": f"dblp-h{hop}-{got:03d}",
            "hop": hop, "start": ents[start], "start_id": start,
            "rel_seq": rel_seq, "answer": ents[answer],
            "answer_id": answer, "answer_ids": finals,
            "n_finals": len(finals), "answer_type": cats[answer],
            "gt_path": path, "type_seq": list(tkey),
            "distract_avg": sum(distract) / len(distract),
            "distract_max": max(distract),
        })
        return True

    # 单一作者论文的作者(hop4/5 保证链成立); 混合随机链靠 <=3 过滤
    single_authors = [a for a in small_authors
                      if all(paper_authors[p] == {a} for p in author_papers[a])]

    for hop in HOPS:
        got, attempts = 0, 0
        while got < PER_HOP and attempts < 6000:
            attempts += 1
            if hop == 2:
                a = random.choice(small_authors)
                ok = add_q(hop, a, ["writes", "published_in"], got)
            elif hop == 3:
                p = random.choice(papers_1a_3p)
                ok = add_q(hop, p, ["written_by", "writes", "published_in"],
                           got)
            elif hop == 4:
                a0 = random.choice(single_authors if random.random() < 0.7
                                   else small_authors)
                ok = add_q(hop, a0,
                           ["writes", "written_by", "writes", "published_in"],
                           got)
            else:  # hop 5: P0 -> A0 -> P1 -> A2 -> P2 -> C
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
    holdout = set(keys[: int(len(keys) * HOLDOUT_FRAC)])
    for q in questions:
        q["split_comp"] = "heldout" if tuple(q["type_seq"]) in holdout else "seen"

    OUT.write_text(json.dumps({
        "source": "DBLP-1 (Jhy1993/Datasets-for-Heterogeneous-Graph, HAN line)",
        "schema": [list(x) for x in sorted(schema)],
        "n_entities": len(ents),
        "questions": questions,
    }, ensure_ascii=False, indent=1))
    for hop in HOPS:
        qs = [q for q in questions if q["hop"] == hop]
        ho = [q for q in qs if q["split_comp"] == "heldout"]
        print(f"hop={hop}: {len(qs)} (heldout {len(ho)}), "
              f"avg-distract {sum(q['distract_avg'] for q in qs)/max(1,len(qs)):.1f}")
    print(f"unique type-seqs {len(keys)}, holdout {len(holdout)} -> {OUT}")


if __name__ == "__main__":
    main()
