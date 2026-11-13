#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env/kg_env.py — UMLS 多跳遍历推理环境

任务形式(受控遍历, 对齐 GraphPRM/NLGraph 设定):
  问题 = (起始实体, 有序关系序列, 期望答案类型)
  动作 = 从当前实体选择一条出边 (relation, neighbor)
  执行 hop 步后所在实体即答案; 落在 GT 实体 = 正确

环境特性:
  - 观测只暴露当前实体的邻居(含类型), 不给全图 → 每步是真实决策
  - 邻居上限 40, 恒包含黄金邻居(信息可得, 选择是难点)
  - 无效动作(不存在的边)记违规并停留原地(消耗步数), 对齐 KG-R1 处理
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_NEIGHBORS = 40


def load_all():
    src = ROOT / "data" / "umls_src"
    ents, rels, cats = {}, {}, {}
    for line in (src / "entities.txt").read_text().splitlines():
        i, n = line.split("\t"); ents[int(i)] = n
    for line in (src / "relations.txt").read_text().splitlines():
        i, n = line.split("\t"); rels[int(i)] = n
    for line in (src / "entity_category.txt").read_text().splitlines():
        i, c = line.split("\t"); cats[int(i)] = c
    adj = defaultdict(list)
    for line in (src / "triples.txt").read_text().splitlines():
        r, h, t = line.split("\t")
        adj[int(h)].append((rels[int(r)], int(t)))
    return ents, cats, adj


class KGEnv:
    def __init__(self, question: dict, ents: dict, cats: dict, adj: dict,
                 seed: int = 0):
        self.q = question
        self.ents, self.cats, self.adj = ents, cats, adj
        self.rng = random.Random(seed)
        self.cur = question["start_id"]
        self.step_i = 0
        self.hop = question["hop"]
        self.done = False
        self.visited = {question["start_id"]}

    # ---- 观测 --------------------------------------------------------
    def visible_neighbors(self):
        """当前实体的邻居列表(截断), 保证黄金邻居在内。"""
        gt_next = None
        if self.q.get("gt_path") and self.step_i < len(self.q["gt_path"]):
            gt_next = self.q["gt_path"][self.step_i]
        gt_pair = (gt_next[1], gt_next[2]) if gt_next else None
        pairs = list(dict.fromkeys(self.adj[self.cur]))  # 去重保序
        if gt_pair and gt_pair in pairs:
            pairs.remove(gt_pair)
            others = [p for p in pairs if p != gt_pair]
            self.rng.shuffle(others)
            keep = [gt_pair] + others[: MAX_NEIGHBORS - 1]
        else:
            self.rng.shuffle(pairs)
            keep = pairs[:MAX_NEIGHBORS]
        return keep

    def observation(self) -> str:
        q = self.q
        neigh = self.visible_neighbors()
        lines = [f"- ({r}) -> {self.ents[t]} [{self.cats[t]}]"
                 for (r, t) in neigh]
        if q.get("protocol") == "openqa":
            head = (f"Question: {q['question']}\nAnswer it by navigating "
                    f"the knowledge graph: starting from '{q['start']}', "
                    f"follow one edge per step (you have {q['hop']} steps) "
                    f"and finish on the entity that answers the question.\n")
        elif q.get("protocol") == "open":
            head = (f"Question: Starting from '{q['start']}' "
                    f"({self.cats[q['start_id']]}), reach any entity of type "
                    f"'{q['answer_type']}' in exactly {q['hop']} steps by "
                    f"following edges. Choose the best edge at each step.\n")
        else:
            head = (f"Question: Start at entity '{q['start']}'. Follow these "
                    f"relations in order: {' -> '.join(q['rel_seq'])}. "
                    f"The final entity should be of type "
                    f"'{q['answer_type']}'.\n")
        return (head +
                f"You are currently at: {self.ents[self.cur]} "
                f"[{self.cats[self.cur]}] (step {self.step_i + 1}/{q['hop']})\n"
                f"Available edges:\n" + "\n".join(lines))

    # ---- 动作 --------------------------------------------------------
    def act(self, relation: str, neighbor_name: str) -> dict:
        """执行一步。返回 {valid, moved, violation, prev, cur}。"""
        nid = self.ename2id.get(neighbor_name, -1)
        prev = self.cur
        pairs = {(r, t) for (r, t) in self.adj[self.cur]}
        if isinstance(nid, int) and nid < 0:
            valid = False            # 整型ID域: 查无此实体
        else:
            valid = (relation, nid) in pairs   # 字符串名域(如WebQSP)
        if valid:
            self.cur = nid
            self.visited.add(nid)
        self.step_i += 1
        if self.step_i >= self.hop:
            self.done = True
        return {"valid": valid, "violation": not valid, "prev": prev,
                "cur": self.cur, "relation": relation,
                "neighbor": neighbor_name}

    # ---- 名字 -> id ---------------------------------------------------
    @property
    def ename2id(self):
        if not hasattr(self, "_e2i"):
            self._e2i = {v: k for k, v in self.ents.items()}
        return self._e2i

    @property
    def answer_correct(self) -> bool:
        finals = self.q.get("answer_ids") or [self.q.get("answer_id")]
        if self.q.get("protocol") == "openqa":
            return bool(self.visited & set(finals))   # visit-based
        return self.cur in finals


# ---------------------------------------------------------------------------
# 事后诊断指标(用黄金信息, 仅用于评估不用于决策)
# ---------------------------------------------------------------------------
def diagnose_trajectory(question, adj, ents, cats, actions):
    """逐动作的 oracle 诊断: 类型合法? 在轨? (供 dead-end/type-validity 指标)
    开放协议下 dead-end 无 declared 程序可判, 仅计违规。"""
    n_viol = n_dead = 0
    cur = question["start_id"]
    open_mode = question.get("protocol") in ("open", "openqa")
    for i, a in enumerate(actions):
        valid_edge = False
        for (r, t) in adj[cur]:
            if r == a["relation"] and ents.get(t) == a["neighbor"]:
                valid_edge = True; cur = t; break
        if not valid_edge:
            n_viol += 1
            continue
        if open_mode:
            continue
        remaining = question["rel_seq"][i + 1:]
        frontier = {cur}
        for r in remaining:
            frontier = {t for n in frontier for (rr, t) in adj[n] if rr == r}
            if not frontier:
                break
        finals = set(question.get("answer_ids") or [question.get("answer_id")])
        if not (finals & frontier):
            n_dead += 1
    return {"type_violations": n_viol, "dead_ends": n_dead,
            "n_steps": len(actions)}
