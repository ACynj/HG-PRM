#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_dpo.py — 本地模型步级贪心 rollout 评测(批量生成)
对比: base / SFT / DPO-outcome / DPO-process × UMLS 160 题(seen/heldout)
obs 与训练一致(dpo_data.rebuild_obs)。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dpo_data import rebuild_obs  # noqa: E402
from env.kg_env import load_all, diagnose_trajectory  # noqa: E402

MODEL = "/mnt/ssd/ynj/pro3/models/qwen2.5-7b"
OUT = ROOT / "outputs"

SYSTEM = ("You are a precise graph reasoner executing a fixed relation "
          "sequence. Given the current entity and its available edges, "
          "choose the single best edge. Reply with ONLY one JSON object: "
          '{"relation": "<relation>", "neighbor": "<entity name>"} '
          "Copy names EXACTLY as given.")


def parse_json_first(text):
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@torch.no_grad()
def rollout(model, tok, questions, ents, cats, adj, ents_rev, batch=32):
    """批量步级贪心: 每步把所有活跃问题的 obs 一次批量生成。"""
    state = {q["qid"]: {"q": q, "cur": q["start_id"], "step": 0,
                        "actions": [], "done": False} for q in questions}
    pad = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    while True:
        active = [s for s in state.values() if not s["done"]]
        if not active:
            break
        for i in range(0, len(active), batch):
            chunk = active[i:i + batch]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": rebuild_obs(
                         s["q"], ents, cats, adj, s["cur"], s["step"],
                         ents_rev)}],
                    tokenize=False, add_generation_prompt=True)
                for s in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1792).to("cuda")
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                     skip_special_tokens=True)
            for s, t in zip(chunk, texts):
                a = parse_json_first(t)
                rec = {"relation": a.get("relation", ""),
                       "neighbor": a.get("neighbor", "")}
                s["actions"].append(rec)
                nid = ents_rev.get(rec["neighbor"], -1)
                ok = any(rr == rec["relation"] and t2 == nid
                         for (rr, t2) in adj[s["cur"]])
                if ok:
                    s["cur"] = nid
                s["step"] += 1
                if s["step"] >= s["q"]["hop"]:
                    s["done"] = True
    return state


def evaluate(name, model, tok, questions, ents, cats, adj, ents_rev):
    state = rollout(model, tok, questions, ents, cats, adj, ents_rev)
    rows = []
    for s in state.values():
        q = s["q"]
        finals = set(q.get("answer_ids", [q["answer_id"]]))
        correct = s["cur"] in finals
        diag = diagnose_trajectory(q, adj, ents, cats, s["actions"])
        rows.append({"model": name, "qid": q["qid"], "hop": q["hop"],
                     "split_comp": q["split_comp"], "correct": correct,
                     "type_violations": diag["type_violations"],
                     "final": ents[s["cur"]]})
    return rows


def main():
    data = json.loads((ROOT / "data" / "questions.json").read_text())
    questions = data["questions"]
    ents, cats, adj = load_all()
    ents_rev = {v: k for k, v in ents.items()}

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=None).to("cuda")

    all_rows = evaluate("base", base, tok, questions, ents, cats, adj,
                        ents_rev)
    del base; torch.cuda.empty_cache()

    for name in ["sft", "dpo_outcome", "dpo_process", "dpo_combined"]:
        ckpt = OUT / f"ckpt_{name}"
        if not ckpt.exists():
            print(f"skip {name} (no ckpt)")
            continue
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map=None).to("cuda")
        model = PeftModel.from_pretrained(model, str(ckpt))
        model.eval()
        all_rows += evaluate(name, model, tok, questions, ents, cats, adj,
                             ents_rev)
        del model; torch.cuda.empty_cache()

    (OUT / "results_dpo.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows))
    # 汇总
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0.0, 0])
    for r in all_rows:
        a = agg[r["model"]]
        a[0] += r["correct"]; a[1] += 1; a[2] += r["type_violations"]; a[3] += 1
    print(f"{'model':14s} {'Acc':>6s} {'viol':>5s}")
    for m, (c, n, v, _) in agg.items():
        print(f"{m:14s} {100*c/n:6.1f} {v/n:5.2f}")
    # seen/heldout
    for m in ["base", "sft", "dpo_outcome", "dpo_process", "dpo_combined"]:
        rs = [r for r in all_rows if r["model"] == m]
        seen = [r for r in rs if r["split_comp"] == "seen"]
        held = [r for r in rs if r["split_comp"] == "heldout"]
        if seen and held:
            print(f"{m:14s} seen {100*sum(r['correct'] for r in seen)/len(seen):5.1f}"
                  f"  heldout {100*sum(r['correct'] for r in held)/len(held):5.1f}")
    print("-> outputs/results_dpo.jsonl")


if __name__ == "__main__":
    main()
