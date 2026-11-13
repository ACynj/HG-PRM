#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/significance.py — McNemar 精确检验 (成对二元正确性) + Bootstrap CI
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tag="main"):
    recs = [json.loads(l) for l in
            (ROOT / "outputs" / f"results_{tag}.jsonl").read_text().splitlines()]
    return [r for r in recs if "error" not in r]


def correct_map(recs, method):
    return {r["qid"]: r["correct"] for r in recs if r["method"] == method}


def mcnemar_exact(a: dict, b: dict):
    """返回 (b01, b10, p_two_sided)。b01: a对b错; b10: a错b对。"""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for k in keys if a[k] and not b[k])
    b10 = sum(1 for k in keys if not a[k] and b[k])
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # 双侧精确二项 p = 2 * P(X <= k), X~Bin(n, 0.5), 截断到 1
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b01, b10, min(1.0, p)


def bootstrap_ci(vals, n_boot=10000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(vals)
    stats = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        stats.append(s)
    stats.sort()
    return stats[int(0.025 * n_boot)] * 100, stats[int(0.975 * n_boot)] * 100


def main():
    recs = load("main")
    methods = sorted({r["method"] for r in recs})
    print("=== McNemar exact tests vs beam ===")
    beam = correct_map(recs, "beam")
    for m in methods:
        if m == "beam":
            continue
        other = correct_map(recs, m)
        if len(set(beam) & set(other)) < 10:
            continue
        b01, b10, p = mcnemar_exact(beam, other)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"beam vs {m:14s}: +{b01:3d}/-{b10:3d}  p={p:.2e}  {sig}")
    print("\n=== 95% Bootstrap CI (accuracy %) ===")
    for m in methods:
        cm = correct_map(recs, m)
        vals = list(cm.values())
        if len(vals) >= 10:
            lo, hi = bootstrap_ci(vals)
            print(f"{m:14s}: {100*sum(vals)/len(vals):5.1f}%  [{lo:.1f}, {hi:.1f}]  n={len(vals)}")


if __name__ == "__main__":
    main()
