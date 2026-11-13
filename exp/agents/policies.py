#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agents/policies.py — 四种方法策略 + 带磁盘缓存的 LLM 调用

方法:
  greedy    逐步贪心: LLM 每步选一条最优边 (outcome 无过程监督的缩影)
  sc        Self-Consistency: 采样 N 条完整轨迹, 终点多数投票
  tokbon    Token-overlap BoN (Kansal&Jha 式词面奖励, 注意: 用到黄金路径=oracle 参照)
  beam      本文方法: LLM 提议 K 候选 + 符号检查器(C1/C2/C3)融合打分, 保 top-b

公平性: beam 的 C1/C2/C3 全部只用 (问题文本, schema, 期望答案类型), 不用黄金答案。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.kg_env import KGEnv, MAX_NEIGHBORS  # noqa: E402
from symbolic.checker import Schema  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 带缓存的 LLM 调用
# ---------------------------------------------------------------------------
_client = None


def get_llm():
    global _client
    if _client is None:
        import os
        provider = os.environ.get("HGPRM_PROVIDER", "siliconflow")
        model = os.environ.get("HGPRM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        from agents.api_llm import get_client
        _client = get_client(provider, model=model)
    return _client


def cached_chat(messages, temperature, max_tokens=512, tag=""):
    import os
    model = os.environ.get("HGPRM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    payload = {"m": messages, "t": temperature, "k": max_tokens, "g": tag}
    if model != "Qwen/Qwen2.5-7B-Instruct":
        payload["mod"] = model   # 仅非默认模型进键 → 与7B缓存完全隔离
    key = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    with _cache_lock:
        if path.exists():
            return json.loads(path.read_text())["out"]
    out = get_llm().chat(messages, temperature=temperature, max_tokens=max_tokens)
    with _cache_lock:
        path.write_text(json.dumps({"out": out}, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------
def parse_json_blocks(text: str):
    """尽力从 LLM 输出抽取 JSON 对象列表。"""
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return objs


SYSTEM_STEP = (
    "You are a precise graph reasoner executing a fixed relation sequence. "
    "Given the current entity and its available edges, choose the single best "
    "edge to follow. Reply with ONLY one JSON object: "
    '{"relation": "<relation>", "neighbor": "<entity name>"} '
    "Copy names EXACTLY as given."
)

SYSTEM_PROPOSE = (
    "You are a precise graph reasoner. Given the current entity and its "
    "available edges, propose the top candidate edges for the next step, best "
    "first. Reply with ONLY a JSON array of objects: "
    '[{"relation": "...", "neighbor": "...", "score": 0.0-1.0}, ...] '
    "up to 4 items. Copy names EXACTLY as given."
)


def _msg(obs, system):
    return [{"role": "system", "content": system},
            {"role": "user", "content": obs}]


# ---------------------------------------------------------------------------
# 逐步贪心 / 采样轨迹
# ---------------------------------------------------------------------------
def run_trajectory(question, ents, cats, adj, schema: Schema, temperature,
                   nonce=""):
    """跑一条轨迹, 返回 (final_correct, actions, env)。
    nonce 写入 prompt 后缀: 后端对相同 prompt 确定性输出, 必须扰动才能采样。"""
    env = KGEnv(question, ents, cats, adj, seed=hash(question["qid"]) & 0xffff)
    actions = []
    while not env.done:
        obs = env.observation()
        if nonce:
            obs = obs + f"\n[sampling run {nonce}]"
        out = cached_chat(_msg(obs, SYSTEM_STEP), temperature)
        acts = parse_json_blocks(out)
        a = acts[0] if acts else {}
        rec = env.act(a.get("relation", ""), a.get("neighbor", ""))
        actions.append(rec)
    return env.answer_correct, actions, env


# ---------------------------------------------------------------------------
# 方法 1: greedy
# ---------------------------------------------------------------------------
def method_greedy(question, ents, cats, adj, schema, cfg):
    correct, actions, env = run_trajectory(question, ents, cats, adj, schema,
                                           temperature=0.0)
    return {"correct": correct, "actions": actions,
            "final": env.ents[env.cur], "n_llm_calls": question["hop"]}


# ---------------------------------------------------------------------------
# 方法 2/3: 采样类 (SC / token-overlap BoN)
# ---------------------------------------------------------------------------
def method_sampling(question, ents, cats, adj, schema, cfg, mode):
    n = cfg.get("n_sample", 8)
    results = []
    with ThreadPoolExecutor(max_workers=min(n, 4)) as ex:
        futs = [ex.submit(run_trajectory, question, ents, cats, adj, schema,
                          1.0, nonce=f"s{i}") for i in range(n)]
        results = [f.result() for f in futs]

    finals = [env.ents[env.cur] for (_, _, env) in results]
    all_actions = [a for (_, a, _) in results]
    n_calls = sum(len(a) for a in all_actions)
    ents_rev = {v: kk for kk, v in ents.items()}

    if mode == "sc":
        votes = defaultdict(int)
        for f in finals:
            votes[f] += 1
        best = max(votes, key=votes.get)
        correct = best == question["answer"]
        best_i = 0
    elif mode == "tokbon":
        # 词面覆盖奖励(oracle 参照, 对齐 Kansal&Jha 的 token overlap)
        gt_tokens = {e.lower() for (e, _, _) in question["gt_path"]} | {
            question["answer"].lower()}
        best_i, best_cov = 0, -1.0
        for i, (_, actions, _) in enumerate(results):
            traj_tokens = {a["neighbor"].lower() for a in actions
                           if a.get("neighbor")}
            cov = (len(traj_tokens & gt_tokens) / max(1, len(gt_tokens))
                   if gt_tokens else 0.0)
            bonus = 0.5 if results[i][2].answer_correct else 0.0
            if cov + bonus > best_cov:
                best_cov, best_i = cov + bonus, i
        correct = results[best_i][0]
    else:  # bon_verifier: 可部署符号分重排(不用黄金答案/词面)
        def cur_after(actions_prefix):
            ents_rev2 = {v: k for k, v in ents.items()}
            cur = question["start_id"]
            for a in actions_prefix:
                nid = ents_rev2.get(a.get("neighbor", ""), -1)
                if any(r == a["relation"] and t == nid for (r, t) in adj[cur]):
                    cur = nid
            return cur

        def traj_score(i):
            (_, actions, env) = results[i]
            s = 0.0
            for j, a in enumerate(actions):
                s += symbolic_score(question, schema, ents, cats, adj,
                                    cur_after(actions[:j]),
                                    a["relation"], a.get("neighbor", ""),
                                    ents_rev, j)
            type_ok = 1.0 if cats.get(env.cur) == question["answer_type"] else 0.0
            return s / max(1, len(actions)) + type_ok
        best_i = max(range(len(results)), key=traj_score)
        correct = results[best_i][0]
    return {"correct": correct, "actions": all_actions[best_i],
            "all_finals": finals, "best_i": best_i,
            "final": finals[best_i],
            "n_llm_calls": n_calls}


# ---------------------------------------------------------------------------
# 方法 4 (本文): 符号引导 beam search
# ---------------------------------------------------------------------------
def symbolic_score(question, schema: Schema, ents, cats, adj,
                   cur_node, relation, neighbor_name, ents_rev, step_i):
    """C1+C2+C3 全部可部署(不用黄金答案): C3 用'类型可达'而非答案可达。

    step_i: 当前是第几步(0-based), 执行本动作后剩余步数 = hop - step_i - 1。
    """
    nid = ents_rev.get(neighbor_name)
    known = nid is not None            # 兼容整型ID域与字符串名域
    # C1 schema 合法
    h_type, t_type = cats[cur_node], (cats.get(nid, "UNKNOWN"))
    c1 = schema.conforms(h_type, relation, t_type)
    # C2 关系在当前节点的出边中存在(局部可执行)
    rel_ok = known and any(r == relation for (r, _) in adj[cur_node])
    # C3 从候选邻居出发, 剩余步数内(含自身)能否触达期望答案类型(BFS, 不用答案)
    remaining = max(0, question["hop"] - step_i - 1)
    frontier = {nid} if known else set()
    c3 = known and cats.get(nid) == question["answer_type"]
    for _ in range(remaining):
        if not frontier or c3:
            break
        frontier = {t for n2 in frontier for (_, t) in adj[n2]}
        if any(cats.get(x) == question["answer_type"] for x in frontier):
            c3 = True
    return 0.4 * c1 + 0.3 * rel_ok + 0.3 * c3


def method_beam(question, ents, cats, adj, schema, cfg):
    """LLM 提议 ∪ 必需关系邻居 → C2 硬掩码 → C3+LLM 融合打分 → top-b。

    候选池构造是关键: 问题声明了每步必须走的关系 r_i (meta-path 约束),
    符号层据此做动作掩码(C2'), LLM 只需在同关系邻居中排序 → 职责正确分工。
    """
    import copy
    k = cfg.get("k_propose", 4)
    b = cfg.get("beam_width", 2)
    alpha = cfg.get("alpha", 0.6)
    ents_rev = {v: kk for kk, v in ents.items()}

    env0 = KGEnv(question, ents, cats, adj, seed=hash(question["qid"]) & 0xffff)
    beams = [(0.0, env0, [])]
    n_calls = 0
    while beams and not all(e.done for (_, e, _) in beams):
        new_beams = []
        for (score, env, actions) in beams:
            if env.done:
                new_beams.append((score, env, actions))
                continue
            required_rel = (None if question.get("protocol") in ("open", "openqa")
                            else question["rel_seq"][env.step_i])
            # 候选池: (开放协议) 全部邻居 ∪ LLM 提议; (受控) 必需关系邻居 ∪ 提议
            if required_rel is None:
                same_rel = [(r, ents[t]) for (r, t) in
                            env.visible_neighbors()][:8]
            else:
                same_rel = [(required_rel, ents[t])
                            for (r, t) in env.visible_neighbors()
                            if r == required_rel][:8]
            out = cached_chat(_msg(env.observation(), SYSTEM_PROPOSE), 0.0)
            n_calls += 1
            llm_cands = parse_json_blocks(out)[:k]
            llm_scores = {}
            for c in llm_cands:
                key = (c.get("relation", ""), c.get("neighbor", ""))
                if key[0] and key[1]:
                    try:
                        llm_scores[key] = min(max(float(c.get("score", 0.5)), 0.0), 1.0)
                    except (TypeError, ValueError):
                        llm_scores[key] = 0.5
            pool = same_rel + [kk for kk in llm_scores if kk not in set(same_rel)]
            # --- C2' 硬掩码: 受控协议=必须遵守声明关系序列; 开放协议=C1 由符号分软引导 ---
            if required_rel is None or cfg.get("nomask"):
                masked = pool if pool else [("", env.ents[env.cur])]
            else:
                masked = [(rel, nb) for (rel, nb) in pool if rel == required_rel]
                if not masked:
                    masked = [(required_rel, env.ents[env.cur])]
            for (rel, nb) in masked:
                sym = symbolic_score(question, schema, ents, cats, adj,
                                     env.cur, rel, nb, ents_rev, env.step_i)
                llm_s = llm_scores.get((rel, nb), 0.5)
                fused = alpha * sym + (1 - alpha) * llm_s
                env2 = copy.deepcopy(env)
                if hasattr(env2, "_e2i"):
                    delattr(env2, "_e2i")
                env2.act(rel, nb)
                new_beams.append((score + fused, env2, actions + [{
                    "relation": rel, "neighbor": nb, "sym": sym,
                    "llm": llm_s, "fused": fused}]))
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:b]
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:b]
    # 终局选择: 先按"终点类型=问题声明类型"(可部署), 再按累计融合分
    def _rank(item):
        score, env, _ = item
        return (1 if cats.get(env.cur) == question["answer_type"] else 0, score)
    best_score, best_env, best_actions = max(beams, key=_rank)
    return {"correct": best_env.answer_correct, "actions": best_actions,
            "final": ents[best_env.cur], "score": best_score,
            "n_llm_calls": n_calls}


# ---------------------------------------------------------------------------
# HGNN 打分器(P2 训练后加载)
# ---------------------------------------------------------------------------
_hgnn = {"model": None, "meta": None, "adj_bin": None}


def hgnn_score(question, cur, relation, neighbor_name, ents, cats, adj,
               ents_rev, step_i):
    """加载训练好的 mini-HGT, 输出候选可部署稠密分(0-1)。"""
    if _hgnn["model"] is None:
        import torch
        sys.path.insert(0, str(ROOT / "scripts"))
        import train_hgnn as TH
        from train_hgnn import MiniHGTScorer, DEVICE
        meta = json.loads((ROOT / "outputs" / "hgnn_meta.json").read_text())
        # 回填类型查找表(训练时全局, 推理进程需重建)
        TH._typ_of.update({int(k): v for k, v in meta["typ_of"].items()})
        sd = torch.load(ROOT / "outputs" / "hgnn_scorer.pt",
                        map_location=DEVICE, weights_only=True)
        rel_list = sorted(meta["rel_idx"], key=lambda k: meta["rel_idx"][k])
        model = MiniHGTScorer(len(ents), len(meta["typ_idx"]), len(rel_list))
        model.load_state_dict(sd); model.eval()
        # 类型直方图矩阵
        tidx = meta["typ_idx"]
        ab = __import__("torch").zeros(len(ents), len(tidx))
        for h in adj:
            for (_, t) in adj[h]:
                ab[h][tidx[cats[t]]] += 1
        _hgnn.update(model=model, meta=meta, adj_bin=ab)
    import torch
    m, meta = _hgnn["model"], _hgnn["meta"]
    nid = ents_rev.get(neighbor_name, 0)
    with torch.no_grad():
        s = m(
            torch.tensor([cur]),
            torch.tensor([nid]),
            torch.tensor([meta["rel_idx"].get(relation, 0)]),
            torch.tensor([meta["typ_idx"][question["answer_type"]]]),
            torch.tensor([float(question["hop"] - step_i - 1)]),
            _hgnn["adj_bin"])
    return float(torch.sigmoid(s)[0])


def method_beam_variant(question, ents, cats, adj, schema, cfg, mode):
    """beam 变体: mode='symonly'(纯符号) / 'fused'(符号+HGNN+LLM)。"""
    res = None
    if mode == "symonly":
        cfg2 = dict(cfg); cfg2["alpha"] = 1.0
        res = method_beam(question, ents, cats, adj, schema, cfg2)
    elif mode == "fused":
        # 复用 method_beam 主流程但注入 HGNN 分: 为简洁重写循环
        import copy
        b = cfg.get("beam_width", 2)
        k = cfg.get("k_propose", 4)
        ents_rev = {v: kk for kk, v in ents.items()}
        env0 = KGEnv(question, ents, cats, adj, seed=hash(question["qid"]) & 0xffff)
        beams = [(0.0, env0, [])]
        n_calls = 0
        while beams and not all(e.done for (_, e, _) in beams):
            new_beams = []
            for (score, env, actions) in beams:
                if env.done:
                    new_beams.append((score, env, actions)); continue
                required_rel = question["rel_seq"][env.step_i]
                same_rel = [(required_rel, ents[t])
                            for (r, t) in env.visible_neighbors()
                            if r == required_rel][:8]
                out = cached_chat(_msg(env.observation(), SYSTEM_PROPOSE), 0.0)
                n_calls += 1
                llm_scores = {}
                for c in parse_json_blocks(out)[:k]:
                    key = (c.get("relation", ""), c.get("neighbor", ""))
                    if key[0] and key[1]:
                        try:
                            llm_scores[key] = min(max(float(c.get("score", .5)), 0.), 1.)
                        except (TypeError, ValueError):
                            llm_scores[key] = 0.5
                pool = same_rel + [kk for kk in llm_scores if kk not in set(same_rel)]
                masked = [(rel, nb) for (rel, nb) in pool if rel == required_rel]
                if not masked:
                    masked = [(required_rel, env.ents[env.cur])]
                for (rel, nb) in masked:
                    sym = symbolic_score(question, schema, ents, cats, adj,
                                         env.cur, rel, nb, ents_rev, env.step_i)
                    hn = hgnn_score(question, env.cur, rel, nb, ents, cats,
                                    adj, ents_rev, env.step_i)
                    llm_s = llm_scores.get((rel, nb), 0.5)
                    fused = 0.3 * sym + 0.4 * hn + 0.3 * llm_s
                    env2 = copy.deepcopy(env)
                    if hasattr(env2, "_e2i"):
                        delattr(env2, "_e2i")
                    env2.act(rel, nb)
                    new_beams.append((score + fused, env2, actions + [{
                        "relation": rel, "neighbor": nb, "sym": sym,
                        "hgnn": hn, "llm": llm_s, "fused": fused}]))
            new_beams.sort(key=lambda x: -x[0])
            beams = new_beams[:b]

        def _rank(item):
            score, env, _ = item
            return (1 if cats.get(env.cur) == question["answer_type"] else 0, score)
        best_score, best_env, best_actions = max(beams, key=_rank)
        res = {"correct": best_env.answer_correct, "actions": best_actions,
               "final": ents[best_env.cur], "score": best_score,
               "n_llm_calls": n_calls}
    return res


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------
def run_method(method, question, ents, cats, adj, schema, cfg=None):
    cfg = cfg or {}
    if method == "greedy":
        return method_greedy(question, ents, cats, adj, schema, cfg)
    if method == "sc":
        return method_sampling(question, ents, cats, adj, schema, cfg, "sc")
    if method == "tokbon":
        return method_sampling(question, ents, cats, adj, schema, cfg, "tokbon")
    if method == "bon_verifier":
        return method_sampling(question, ents, cats, adj, schema, cfg,
                               "bon_verifier")
    if method == "beam":
        return method_beam(question, ents, cats, adj, schema, cfg)
    if method == "beam_symonly":
        return method_beam_variant(question, ents, cats, adj, schema, cfg,
                                   "symonly")
    if method == "beam_fused":
        return method_beam_variant(question, ents, cats, adj, schema, cfg,
                                   "fused")
    if method == "beam_nomask":
        cfg2 = dict(cfg); cfg2["nomask"] = True
        return method_beam(question, ents, cats, adj, schema, cfg2)
    if method == "beam_b1":
        cfg2 = dict(cfg); cfg2["beam_width"] = 1
        return method_beam(question, ents, cats, adj, schema, cfg2)
    if method == "beam_a04":
        cfg2 = dict(cfg); cfg2["alpha"] = 0.4
        return method_beam(question, ents, cats, adj, schema, cfg2)
    if method == "beam_a08":
        cfg2 = dict(cfg); cfg2["alpha"] = 0.8
        return method_beam(question, ents, cats, adj, schema, cfg2)
    raise ValueError(method)
