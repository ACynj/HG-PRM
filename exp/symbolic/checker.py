#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbolic/checker.py — HG-PRM 符号约束层（论文 Sec. IV-A 的 C1/C2/C3）

三个在异构图上闭式可判定的检查：
  C1 schema conformance   动作调用的关系类型是否与当前实体类型匹配
  C2 meta-path validity   路径类型序列是否构成合法 meta-path（schema 上可组合）
  C3 reachability         扩展后的实体是否仍在目标可达前沿内

设计约束（与论文定理对应）：
  - 全部检查只用 (G, Sigma) 的图结构，不做任何语言判断  → 不可 hack（Thm. 2）
  - 返回值是精确的 {0, 1} / 可选软进度分，不掺语义        → 稀疏但精确
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Schema:
    """异构图 schema：合法的 (head_type, relation, tail_type) 三元组集合。"""
    triples: set[tuple[str, str, str]]
    # 合法类型转移表 (head_type, relation) -> {tail_type,...}，由 triples 构建
    _trans: dict[tuple[str, str], set[str]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        for h, r, t in self.triples:
            self._trans.setdefault((h, r), set()).add(t)

    def conforms(self, head_type: str, relation: str, tail_type: str) -> bool:
        """C1: 三元组是否在 schema 内。O(1)。"""
        return tail_type in self._trans.get((head_type, relation), set())

    def admissible_relations(self, head_type: str) -> list[str]:
        """当前实体类型下可用的关系（动作空间裁剪，供 agent prompt 用）。"""
        return sorted({r for (h, r) in self._trans if h == head_type})


@dataclass
class ReasoningState:
    """agent 的推理状态 s_t = (question, current_entity, path)。"""
    question: str
    current_entity: str
    path: list[tuple[str, str, str]] = field(default_factory=list)  # (h, r, t) 序列

    def type_sequence(self, G: nx.MultiDiGraph) -> list[str]:
        """路径的类型序列 Ψ(P)：[t0, r1, t1, r2, t2, ...]，用于 C2。"""
        seq = []
        for h, r, t in self.path:
            seq.append(G.nodes[h]["type"])
            seq.append(r)
        seq.append(G.nodes[self.path[-1][2]]["type"]) if self.path else None
        return seq


# ---------------------------------------------------------------------------
# 符号检查器
# ---------------------------------------------------------------------------
class SymbolicChecker:
    def __init__(self, G: nx.MultiDiGraph, schema: Schema,
                 meta_path_bank: Optional[set[tuple[str, ...]]] = None):
        self.G = G
        self.schema = schema
        # meta-path 先验库：类型序列的元组集合；为空则退化为“schema 可组合即合法”
        self.meta_path_bank = meta_path_bank or None

    # ---- C1 -------------------------------------------------------------
    def c1_schema_conformance(self, state: ReasoningState, relation: str,
                              next_entity: str) -> bool:
        h_type = self.G.nodes[state.current_entity]["type"]
        t_type = self.G.nodes[next_entity]["type"]
        return self.schema.conforms(h_type, relation, t_type)

    # ---- C2 -------------------------------------------------------------
    def c2_metapath_validity(self, state: ReasoningState, relation: str,
                             next_entity: str) -> bool:
        """检查 (历史类型序列 + 新增 (r, tail_type)) 是否可组合成合法 meta-path。"""
        h_type = self.G.nodes[state.current_entity]["type"]
        t_type = self.G.nodes[next_entity]["type"]
        if not self.schema.conforms(h_type, relation, t_type):
            return False
        if self.meta_path_bank is None:
            return True  # 无先验库：schema 可组合即合法
        # 有先验库：完整序列（到达答案时）须命中库；中间步骤只查前缀可行性
        new_seq = tuple(
            state.type_sequence(self.G) + [relation, t_type]
        ) if state.path else (h_type, relation, t_type)
        for mp in self.meta_path_bank:
            if len(mp) >= len(new_seq) and mp[:len(new_seq)] == new_seq:
                return True
        return False

    # ---- C3 -------------------------------------------------------------
    def c3_reachability(self, state: ReasoningState, next_entity: str,
                        answer_frontier: set[str], hop_budget: int) -> bool:
        """next_entity 是否仍在“可到达答案候选”的前沿内。

        answer_frontier: 问题 q 的答案候选集合 F(q)（离线预计算或按需 BFS）。
        简化实现：next_entity 能在剩余 hop 预算内触达 F(q) 中任一候选。
        """
        if next_entity in answer_frontier:
            return True
        remaining = hop_budget - len(state.path) - 1
        if remaining <= 0:
            return False
        for target in answer_frontier:
            if nx.has_path(self.G, next_entity, target):
                if nx.shortest_path_length(self.G, next_entity, target) <= remaining:
                    return True
        return False

    # ---- 融合：步骤符号奖励 r_sym --------------------------------------
    def step_reward(self, state: ReasoningState, relation: str,
                    next_entity: str, answer_frontier: set[str],
                    hop_budget: int, w=(0.4, 0.3, 0.3)) -> float:
        c1 = self.c1_schema_conformance(state, relation, next_entity)
        c2 = self.c2_metapath_validity(state, relation, next_entity)
        c3 = self.c3_reachability(state, next_entity, answer_frontier, hop_budget)
        return float(w[0] * c1 + w[1] * c2 + w[2] * c3)


# ---------------------------------------------------------------------------
# 冒烟测试：一个 UMLS 风格的迷你 schema
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    schema = Schema(triples={
        ("Disease", "may_treat", "Drug"),          # 注意方向故意造一个反例
        ("Drug", "treats", "Disease"),
        ("Disease", "has_symptom", "Symptom"),
        ("Symptom", "symptom_of", "Disease"),
        ("Drug", "causes", "Symptom"),
    })
    G = nx.MultiDiGraph()
    G.add_node("flu", type="Disease")
    G.add_node("aspirin", type="Drug")
    G.add_node("fever", type="Symptom")
    G.add_edge("flu", "fever", key="has_symptom", relation="has_symptom")
    G.add_edge("fever", "flu", key="symptom_of", relation="symptom_of")
    G.add_edge("aspirin", "flu", key="treats", relation="treats")

    checker = SymbolicChecker(G, schema)
    state = ReasoningState(question="What treats flu?", current_entity="flu")

    # 合法动作：flu --symptom_of? 不，schema 里 Disease 的关系是 may_treat/has_symptom
    print("C1 flu-[*]->aspirin (Disease may_treat Drug):",
          checker.c1_schema_conformance(state, "may_treat", "aspirin"))   # True
    print("C1 flu-[treats]->aspirin (非法：treats 头类型应为 Drug):",
          checker.c1_schema_conformance(state, "treats", "aspirin"))      # False
    print("C3 aspirin 可达答案 {aspirin}:",
          checker.c3_reachability(state, "aspirin", {"aspirin"}, 3))      # True
    print("step_reward(flu,has_symptom,fever):",
          checker.step_reward(state, "has_symptom", "fever", {"aspirin"}, 3))
    print("admissible relations at Disease:", schema.admissible_relations("Disease"))
