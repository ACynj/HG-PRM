# HG-PRM: Graphs as Process Reward Models

**Type-Constrained Verification for LLM Reasoning on Heterogeneous Graphs**

Official code for the TKDE submission *"Graphs as Process Reward Models:
Type-Constrained Verification for LLM Reasoning on Heterogeneous Graphs"*
(Zhu, You, Yang, Liu).

## Overview

LLM agents reasoning over heterogeneous graphs and knowledge graphs receive
step-level supervision that is **structure-blind**: RL pipelines give only
terminal outcome rewards, and process reward models (PRMs) are fine-tuned
text LLMs that cannot reliably perform the structural computation they
judge. **HG-PRM** closes this *verification gap* by making the graph itself
the process reward model:

- **Symbolic layer** — exactly decidable type-consistency checks
  (C1 schema conformance, C2 relation-program conformance / hard masking,
  C3 answer-type reachability), immune to reward hacking;
- **Neural layer** — a <1M-parameter mini-HGT scorer distilled purely from
  symbolic labels (zero human annotation).

It plugs into **test-time beam search** (67.5% Hits@1 on UMLS, +37.5 over
the strongest deployable baseline, beyond even the 8-sample oracle post-hoc
selection ceiling) and **training-time DPO/GRPO** (process pairs lift
held-out composition accuracy 64.3%→73.8% on top of outcome pairs).

## Repository Structure

```
exp/
├── symbolic/checker.py        # C1/C2/C3 symbolic verification
├── env/kg_env.py              # stepwise multi-hop reasoning environment
├── agents/
│   ├── api_llm.py             # OpenAI-compatible client (SiliconFlow etc.)
│   └── policies.py            # greedy / SC / BoN / verifier-rerank / HG-PRM beam
├── scripts/
│   ├── build_dataset*.py      # UMLS / DBLP / Aminer / WebQSP benchmark construction
│   ├── run_experiment.py      # main experiment runner (all methods × domains)
│   ├── significance.py        # McNemar exact tests + bootstrap CIs
│   ├── reward_analysis.py     # step/trajectory reward-quality analysis
│   ├── train_dpo.py           # LoRA SFT + DPO (process / outcome / combined)
│   ├── eval_dpo.py            # batched local-model rollout evaluation
│   └── make_*.py              # every table & figure regeneration script
├── data/
│   ├── umls_src/ dblp_src/ aminer_src/   # raw public KG sources
│   └── questions*.json        # built benchmarks (with compositional splits)
└── outputs/
    ├── results_*.jsonl        # per-question records of every experiment
    ├── hgnn_scorer.pt         # trained mini-HGT reward model (<1MB)
    ├── table_*.tex            # generated LaTeX tables
    └── llm_cache.zip          # all cached LLM calls → full reproducibility
docs/                          # detailed results report & research-gap analysis
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) (Optional) rebuild benchmarks from raw sources
python exp/scripts/build_dataset.py
python exp/scripts/build_dataset_dblp.py
python exp/scripts/build_dataset_aminer.py

# 2) Run the main comparison (needs an OpenAI-compatible API key)
export SILICONFLOW_API_KEY=sk-...
python exp/scripts/run_experiment.py --methods greedy,sc,tokbon,beam --tag main

# 3) Statistics & tables
python exp/scripts/significance.py
python exp/scripts/make_tables.py main

# 4) Training-time validation (local Qwen2.5-7B, single GPU)
python exp/scripts/dpo_data.py          # build state-aligned pairs
python exp/scripts/train_dpo.py all     # SFT + DPO variants
python exp/scripts/eval_dpo.py          # evaluate all checkpoints
```

All reported numbers regenerate exactly from `outputs/llm_cache.zip`
(unzip into `exp/outputs/llm_cache/` first) — every LLM call is
content-hashed and replayed from cache.

## Base Model & Checkpoints

- Policy model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Trained checkpoints (SFT / DPO-outcome / DPO-process / DPO-combined LoRA
  adapters, ~323MB each) are released on Hugging Face:
  `https://huggingface.co/ACynj/HG-PRM-checkpoints` *(upload in progress)*

## Key Results

| Setting | Result |
|---|---|
| UMLS test-time (controlled protocol) | **67.5%** vs 30.0 best deployable (+37.5, p<1e-17); beyond 8-sample oracle ceiling (51.6%) |
| UMLS training-time (DPO, held-out compositions) | **73.8%** vs 64.3 outcome-only; violations −78% |
| Aminer open protocol (439k nodes) | 92.2% with **zero** type violations (only such method) |
| Policy-scale (~72B qwen-plus) | margin narrows +37.5→+14.4 but stays significant (p=0.002) |
| HGNN module (optional) | compositional-generalization add-on: halves held-out drop (−7.6→−4.6) at −9.4 in-domain |
| WebQSP open-domain QA (Freebase, n=157) | 68.8% vs 61.1% greedy (+7.6, p=0.11); zero violations vs 0.30/q |
| Reward quality | trajectory-level AUC 0.898 without answer information |

## Citation

```bibtex
@article{zhu2026hgprm,
  title   = {Graphs as Process Reward Models: Type-Constrained Verification
             for LLM Reasoning on Heterogeneous Graphs},
  author  = {Zhu, Ling and You, Nengjie and Yang, Ze and Liu, Guangyu},
  journal = {IEEE Transactions on Knowledge and Data Engineering (submission)},
  year    = {2026}
}
```

## License

MIT License. Benchmark data derives from public datasets
(UMLS via [kg-data](https://github.com/dongwookim-ml/kg-data);
DBLP/Aminer via [Datasets-for-Heterogeneous-Graph](https://github.com/Jhy1993/Datasets-for-Heterogeneous-Graph)).

## TOIS Version

A parallel submission version for ACM TOIS is in `paper_tois/` (acmsmall format, 20 pages).
