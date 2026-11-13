#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_hgnn.py — P2: 符号信号 → 轻量异构GNN奖励模型的蒸馏

数据: results_main.jsonl 全部方法的轨迹状态-动作对 + 负采样
标签(oracle, 训练用):
  y = 该动作后, 沿剩余必需关系序列仍能到达黄金答案 (on-track)
特征(可部署, 不用黄金答案):
  实体嵌入(135) + 类型嵌入 + 关系嵌入 + 剩余步数 + 候选1-hop类型直方图
  → 一层类型感知聚合(mini-HGT 原型, 纯 torch 实现)

关键评测: 只在 seen 组合问题上训练, 在 heldout 组合上测奖励模型本身
的组合泛化 — 对应论文 Sec. V 的 compositional protocol。
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.kg_env import load_all  # noqa: E402

DEVICE = "cpu"
SEED = 7
EPOCHS = 60
LR = 3e-3
DIM = 64


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def build_examples():
    data = json.loads((ROOT / "data" / "questions.json").read_text())
    qmap = {q["qid"]: q for q in data["questions"]}
    ents, cats, adj = load_all()
    ents_rev = {v: k for k, v in ents.items()}

    # 精确 on-track 判定: 从 node 沿剩余序列能否到答案(带记忆化不必, 图小)
    def ontrack(node, remaining, ans):
        frontier = {node}
        for r in remaining:
            frontier = {t for n in frontier for (rr, t) in adj[n] if rr == r}
            if not frontier:
                return False
        return ans in frontier

    rng = random.Random(SEED)
    examples = []  # (qid, cur_id, cand_id, req_rel, remaining_hops, ans_type, y)
    seen_states = set()

    def add_example(q, cur, cand_name, remaining_idx, y):
        cid = ents_rev.get(cand_name, -1)
        if cid < 0:
            return
        key = (q["qid"], cur, cid, remaining_idx)
        if key in seen_states:
            return
        seen_states.add(key)
        examples.append({
            "qid": q["qid"], "cur": cur, "cand": cid,
            "req_rel": q["rel_seq"][remaining_idx - 1] if remaining_idx > 0 else "",
            "remaining_hops": q["hop"] - remaining_idx,
            "ans_type": q["answer_type"], "y": float(y),
            "split": q["split_comp"],
        })

    recs_path = ROOT / "outputs" / "results_main.jsonl"
    for line in recs_path.read_text().splitlines():
        r = json.loads(line)
        q = qmap[r["qid"]]
        # 重放轨迹得到每步当前实体
        cur = q["start_id"]
        for i, a in enumerate(r.get("actions", [])):
            # 正例候选 = 实际动作(标签由 oracle 决定, 而非方法对错)
            moved = False
            for (rr, t) in adj[cur]:
                if rr == a["relation"] and ents.get(t) == a.get("neighbor"):
                    add_example(q, cur, a["neighbor"], i, 1.0)
                    cur = t
                    moved = True
                    break
            if not moved:
                add_example(q, cur, a["neighbor"], i, 0.0)  # 幻觉动作=负例
            # 负采样: 同状态随机邻居
            for (rr, t) in rng.sample(adj[cur], min(3, len(adj[cur]))):
                remaining = q["rel_seq"][i:]
                y = 1.0 if (rr == q["rel_seq"][i] if i < q["hop"] else False) and \
                    ontrack(t, q["rel_seq"][i + 1:], q["answer_id"]) else 0.0
                add_example(q, cur, ents[t], i, y)
    return examples, ents, cats, adj, qmap


# ---------------------------------------------------------------------------
# mini 异构GNN (类型感知聚合 + 打分头)
# ---------------------------------------------------------------------------
class MiniHGTScorer(nn.Module):
    def __init__(self, n_ent, n_type, n_rel, dim=DIM):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.typ = nn.Embedding(n_type, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.agg = nn.Linear(dim, dim)           # 1-hop 类型感知聚合
        self.state = nn.Linear(dim * 4, dim)     # cur|type|reqrel|anstype
        self.cand = nn.Linear(dim * 3, dim)      # ent|type|agg
        self.head = nn.Sequential(
            nn.Linear(dim * 3 + 16, 128), nn.ReLU(),
            nn.Linear(128, 1))
        self.hop_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU())

    def forward(self, cur, cand, req_rel_idx, ans_type_idx, hop_left,
                adj_bin):
        h_cur_e = self.ent(cur)
        h_cand_e = self.ent(cand)
        # 1-hop 类型感知聚合: 候选邻居的类型构成(小图用稠密矩阵)
        agg = torch.clamp(adj_bin[cand], 0.0, 5.0) @ self.typ.weight  # (B, dim)
        h_state = torch.relu(self.state(torch.cat([
            h_cur_e, self.typ(cur_typ_of(cur)), self.rel(req_rel_idx),
            self.typ(ans_type_idx)], -1)))
        h_cand = torch.relu(self.cand(torch.cat(
            [h_cand_e, self.typ(cand_typ_of(cand)), self.agg(agg)], -1)))
        hopf = self.hop_mlp(hop_left.unsqueeze(-1))
        h = torch.cat([h_state, h_cand, h_state * h_cand, hopf], -1)
        return self.head(h).squeeze(-1)


# 类型索引映射(运行时构建)
_typ_of = {}


def cur_typ_of(idx_tensor):
    return torch.tensor([_typ_of[int(i)] for i in idx_tensor])


def cand_typ_of(idx_tensor):
    return torch.tensor([_typ_of[int(i)] for i in idx_tensor])


def main():
    torch.manual_seed(SEED)
    examples, ents, cats, adj, qmap = build_examples()
    print(f"examples: {len(examples)} "
          f"(heldout {sum(1 for e in examples if e['split']=='heldout')})")

    # 索引表
    ents_rev = {v: k for k, v in ents.items()}
    rel_list = sorted({r for n in adj for (r, _) in adj[n]})
    rel_idx = {r: i for i, r in enumerate(rel_list)}
    typ_list = sorted(set(cats.values()))
    typ_idx = {t: i for i, t in enumerate(typ_list)}
    global _typ_of
    _typ_of = {eid: typ_idx[cats[eid]] for eid in ents}

    # 稠密邻接×类型矩阵 (135×13 → 类型直方图), 再对类型嵌入聚合
    n_ent = len(ents)
    adj_bin = torch.zeros(n_ent, len(typ_list))
    for h in adj:
        for (r, t) in adj[h]:
            adj_bin[h][typ_idx[cats[t]]] += 1

    train = [e for e in examples if e["split"] == "seen"]
    test = [e for e in examples if e["split"] == "heldout"]
    print(f"train {len(train)} / test {len(test)}")

    model = MiniHGTScorer(n_ent, len(typ_list), len(rel_list))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    def batchify(es):
        cur = torch.tensor([e["cur"] for e in es])
        cand = torch.tensor([e["cand"] for e in es])
        rr = torch.tensor([rel_idx.get(e["req_rel"], 0) for e in es])
        at = torch.tensor([typ_idx[e["ans_type"]] for e in es])
        hl = torch.tensor([float(e["remaining_hops"]) for e in es])
        y = torch.tensor([e["y"] for e in es])
        return cur, cand, rr, at, hl, y

    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), 256):
            b = train[i:i + 256]
            cur, cand, rr, at, hl, y = batchify(b)
            logits = model(cur, cand, rr, at, hl, adj_bin)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                cur, cand, rr, at, hl, y = batchify(test)
                logits = torch.sigmoid(model(cur, cand, rr, at, hl, adj_bin))
                pred = (logits > 0.5).float()
                acc = (pred == y).float().mean().item()
                pos = y == 1
                rec = (pred[pos] == 1).float().mean().item() if pos.any() else 0
                print(f"ep{ep+1} loss {tot/len(train):.4f} | heldout acc "
                      f"{acc:.3f} recall {rec:.3f}")

    torch.save(model.state_dict(), ROOT / "outputs" / "hgnn_scorer.pt")
    # 保存索引表供推理
    json.dump({"rel_idx": rel_idx, "typ_idx": typ_idx, "typ_of": _typ_of},
              open(ROOT / "outputs" / "hgnn_meta.json", "w"))
    print("saved -> outputs/hgnn_scorer.pt")


if __name__ == "__main__":
    main()
