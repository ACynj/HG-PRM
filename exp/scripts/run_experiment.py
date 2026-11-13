#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_experiment.py — 运行全部方法 × 全部问题, 落盘逐题记录

用法:
  python3 scripts/run_experiment.py --methods greedy,sc,tokbon,beam \
      --limit 4 --threads 4        # 冒烟
  python3 scripts/run_experiment.py --methods greedy,sc,tokbon,beam \
      --threads 8                  # 全量
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.kg_env import load_all, diagnose_trajectory  # noqa: E402
from symbolic.checker import Schema  # noqa: E402
from agents.policies import run_method  # noqa: E402

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_domain(domain):
    """返回 (questions_data, ents, cats, adj)。"""
    if domain == "umls":
        data = json.loads((ROOT / "data" / "questions.json").read_text())
        ents, cats, adj = load_all()
        return data, ents, cats, adj
    if domain == "dblp":
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from build_dataset_dblp import load_dblp
        data = json.loads((ROOT / "data" / "questions_dblp.json").read_text())
        ents, cats, adj, _schema = load_dblp()
        return data, ents, cats, adj
    if domain == "webqsp":
        data = json.loads((ROOT / "data" / "questions_webqsp.json").read_text())
        g = json.loads((ROOT / "data" / "webqsp_graph.json").read_text())
        ents = {n: n for n in g["ntype"]}          # 实体名即ID
        cats = {n: t for n, t in g["ntype"].items()}
        adj = defaultdict(list)
        for h, r, t in g["triples"]:
            adj[h].append((r, t))
            adj[t]  # 物化尾节点 → 运行期零插入, 多线程只读安全
        return data, ents, cats, adj
    if domain == "aminer":
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from build_dataset_aminer import load_aminer
        data = json.loads((ROOT / "data" / "questions_aminer.json").read_text())
        ents, cats, adj, _schema = load_aminer()
        return data, ents, cats, adj
    raise ValueError(domain)

CFG = {"n_sample": 8, "k_propose": 4, "beam_width": 2, "alpha": 0.6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="greedy,sc,tokbon,beam")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--domain", default="umls")
    args = ap.parse_args()
    methods = args.methods.split(",")

    data, ents, cats, adj = load_domain(args.domain)
    schema = Schema(triples={tuple(t) for t in data["schema"]})
    questions = data["questions"]
    if args.limit:
        # 每 hop 桶取前 limit/4, 保证冒烟覆盖各难度
        per = max(1, args.limit // 4)
        questions = ([q for q in questions if q["hop"] == 2][:per] +
                     [q for q in questions if q["hop"] == 3][:per] +
                     [q for q in questions if q["hop"] == 4][:per] +
                     [q for q in questions if q["hop"] == 5][:per])

    out_path = OUT_DIR / f"results_{args.tag}.jsonl"
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add((r["method"], r["qid"]))
            except json.JSONDecodeError:
                pass
    fout = open(out_path, "a", encoding="utf-8")

    jobs = [(m, q) for m in methods for q in questions
            if (m, q["qid"]) not in done_keys]
    print(f"{len(jobs)} jobs (cache hit skipped: {len(done_keys)})")

    def work(m, q):
        t0 = time.time()
        try:
            res = run_method(m, q, ents, cats, adj, schema, CFG)
        except Exception as e:  # noqa: BLE001
            return {"method": m, "qid": q["qid"], "error": str(e)[:300]}
        diag = diagnose_trajectory(q, adj, ents, cats, res["actions"])
        rec = {"method": m, "qid": q["qid"], "hop": q["hop"],
               "split_comp": q["split_comp"], "distract_avg": q["distract_avg"],
               "answer": q.get("answer") or q["answers"][0], "final": res.get("final"),
               "correct": bool(res["correct"]),
               "type_violations": diag["type_violations"],
               "dead_ends": diag["dead_ends"], "n_steps": diag["n_steps"],
               "n_llm_calls": res.get("n_llm_calls", 0),
               "actions": res["actions"], "secs": round(time.time() - t0, 1)}
        if "all_finals" in res:
            rec["all_finals"] = res["all_finals"]
        return rec

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(work, m, q): (m, q) for (m, q) in jobs}
        for f in as_completed(futs):
            rec = f.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done}/{len(jobs)} done")
    fout.close()
    print(f"all done -> {out_path}")


if __name__ == "__main__":
    main()
