#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_tables.py — 汇总 results_*.jsonl → 主表/切分表 + LaTeX
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tag="main"):
    recs = [json.loads(l) for l in
            (ROOT / "outputs" / f"results_{tag}.jsonl").read_text().splitlines()]
    return [r for r in recs if "error" not in r]


def pct(x, n):
    return 100.0 * x / max(1, n)


def agg(recs, keyfn):
    g = defaultdict(lambda: {"n": 0, "acc": 0, "viol": 0, "dead": 0, "calls": 0})
    for r in recs:
        k = keyfn(r)
        d = g[k]
        d["n"] += 1
        d["acc"] += int(r["correct"])
        d["viol"] += r["type_violations"]
        d["dead"] += r["dead_ends"]
        d["calls"] += r.get("n_llm_calls", 0)
    return g


def main(tag="main"):
    recs = load(tag)
    methods = ["greedy", "sc", "bon_verifier", "tokbon", "beam",
               "beam_symonly", "beam_fused"]
    name = {"greedy": "Greedy LLM", "sc": "Self-Consistency ($n{=}8$)",
            "bon_verifier": "Verifier-Rerank BoN ($n{=}8$)",
            "tokbon": "Token-Overlap BoN$^\\dagger$ ($n{=}8$)",
            "beam": "HG-PRM Beam (ours)",
            "beam_symonly": "\\;\\;w/o LLM score (sym-only)",
            "beam_fused": "\\;\\;+ HGNN module (optional)"}

    # ---- 主表 ----
    print("=" * 78)
    print(f"{'method':22s} {'Acc%':>6s} {'viol/q':>7s} {'dead/q':>7s} {'calls/q':>8s}")
    main_g = agg(recs, lambda r: r["method"])
    for m in methods:
        d = main_g.get(m)
        if not d:
            continue
        print(f"{name[m]:22s} {pct(d['acc'],d['n']):6.1f} "
              f"{d['viol']/d['n']:7.2f} {d['dead']/d['n']:7.2f} "
              f"{d['calls']/d['n']:8.1f}")

    # ---- 按 hop ----
    print("\n--- Accuracy by hop ---")
    hop_g = agg(recs, lambda r: (r["method"], r["hop"]))
    hops = sorted({r["hop"] for r in recs})
    hdr = f"{'method':22s}" + "".join(f" h{h:>d}  " for h in hops)
    print(hdr)
    for m in methods:
        row = f"{name[m]:22s}"
        for h in hops:
            d = hop_g.get((m, h))
            row += f" {pct(d['acc'],d['n']):5.1f}" if d else "   - "
        print(row)

    # ---- 按 compositional split ----
    print("\n--- Accuracy by compositional split (seen / heldout type-seq) ---")
    comp_g = agg(recs, lambda r: (r["method"], r["split_comp"]))
    for m in methods:
        s, ho = comp_g.get((m, "seen")), comp_g.get((m, "heldout"))
        if s and ho:
            print(f"{name[m]:22s} seen {pct(s['acc'],s['n']):5.1f} "
                  f"(n={s['n']})  heldout {pct(ho['acc'],ho['n']):5.1f} "
                  f"(n={ho['n']})  Δ={pct(ho['acc'],ho['n'])-pct(s['acc'],s['n']):+.1f}")

    # ---- 难度(干扰度)四分位 ----
    print("\n--- Accuracy by distraction quartile (avg same-rel neighbors) ---")
    qs = sorted(r["distract_avg"] for r in recs)
    def quart(r):
        v = r["distract_avg"]
        if v <= qs[len(qs)//4]: return "Q1(easy)"
        if v <= qs[len(qs)//2]: return "Q2"
        if v <= qs[3*len(qs)//4]: return "Q3"
        return "Q4(hard)"
    d_g = agg(recs, lambda r: (r["method"], quart(r)))
    for m in methods:
        row = f"{name[m]:22s}"
        for qk in ["Q1(easy)", "Q2", "Q3", "Q4(hard)"]:
            d = d_g.get((m, qk))
            row += f" {pct(d['acc'],d['n']):5.1f}" if d else "   - "
        print(row)

    # LaTeX 主表
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Best Hits@1 and fewest violations in bold. Main results "
        r"on the UMLS multi-hop traversal benchmark (160 questions; answer "
        r"Hits@1, per-question type violations, dead-ends, LLM calls). "
        r"$^\dagger$uses ground-truth path tokens (oracle-lexical reward).}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccc}\toprule",
        r"Method & Hits@1 & Viol./q & Dead/q & Calls/q \\ \midrule",
    ]
    for m in methods:
        d = main_g.get(m)
        if not d:
            continue
        mname = name[m]
        a_v = f"{pct(d['acc'],d['n']):.1f}"
        v_v = f"{d['viol']/d['n']:.2f}"
        if m == "beam":      # 本文方法: 行名+最佳Hits@1+最少违规加粗
            mname = r"\textbf{" + name[m] + "}"
            a_v, v_v = r"\textbf{" + a_v + "}", r"\textbf{" + v_v + "}"
        lines.append(f"{mname} & {a_v} & {v_v} & "
                     f"{d['dead']/d['n']:.2f} & "
                     f"{d['calls']/d['n']:.1f} \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (ROOT / "outputs" / "table_main.tex").write_text("\n".join(lines))
    print("\nsaved outputs/table_main.tex")

    # LaTeX 组合切分表
    c2 = [
        r"\begin{table}[t]\centering",
        r"\caption{Best per column in bold. Compositional split: accuracy "
        r"on seen vs.\ held-out meta-path type sequences (the HGNN layer is "
        r"trained only on \emph{seen} trajectories). "
        r"$^\dagger$uses ground-truth path tokens (oracle-lexical reward).}",
        r"\label{tab:comp}",
        r"\begin{tabular}{lccc}\toprule",
        r"Method & Seen & Held-out & $\Delta$ \\ \midrule",
    ]
    for m in methods:
        s, ho = comp_g.get((m, "seen")), comp_g.get((m, "heldout"))
        if s and ho:
            a1, a2 = pct(s['acc'], s['n']), pct(ho['acc'], ho['n'])
            nm, f1, f2 = name[m], f"{a1:.1f}", f"{a2:.1f}"
            fd = f"{a2-a1:+.1f}"
            if m == "beam":
                nm, f2 = r"\textbf{" + name[m] + "}", r"\textbf{" + f2 + "}"
            c2.append(f"{nm} & {f1} & {f2} & {fd} \\\\")
    c2 += [r"\bottomrule\end{tabular}\end{table}"]
    (ROOT / "outputs" / "table_comp.tex").write_text("\n".join(c2))
    print("saved outputs/table_comp.tex")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
