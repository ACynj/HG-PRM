#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_dpo.py — 本地 Qwen2.5-7B-Instruct 的 SFT / DPO-outcome / DPO-process
LoRA(r=32), bf16, 单卡; 三种配置顺序训练并各自保存。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/mnt/ssd/ynj/pro3/models/qwen2.5-7b"
DATA = json.loads((ROOT / "outputs" / "dpo_data.json").read_text())
OUT = ROOT / "outputs"

SYSTEM = ("You are a precise graph reasoner executing a fixed relation "
          "sequence. Given the current entity and its available edges, "
          "choose the single best edge. Reply with ONLY one JSON object: "
          '{"relation": "<relation>", "neighbor": "<entity name>"} '
          "Copy names EXACTLY as given.")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=None).to("cuda")
    model.config.use_cache = False
    return tok, model


def dpo_run(name, pairs, tok, model):
    from datasets import Dataset
    ds = Dataset.from_list([
        {"prompt": [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": p["prompt"]}],
         "chosen": [{"role": "assistant", "content": p["chosen"]}],
         "rejected": [{"role": "assistant", "content": p["rejected"]}]}
        for p in pairs])
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = DPOConfig(
        output_dir=str(OUT / f"ckpt_{name}"), per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=5e-5,
        lr_scheduler_type="cosine", warmup_steps=10, bf16=True,
        logging_steps=10, save_strategy="no", report_to=[], max_length=2048,
        beta=0.1, seed=5, gradient_checkpointing=True)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(OUT / f"ckpt_{name}"))
    del trainer
    torch.cuda.empty_cache()
    print(f"[done] {name}")


def sft_run(tok, model):
    from datasets import Dataset
    ds = Dataset.from_list([
        {"messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": r["prompt"]},
                      {"role": "assistant", "content": r["response"]}]}
        for r in DATA["sft"]])
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=str(OUT / "ckpt_sft"), per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=10, bf16=True,
        logging_steps=20, save_strategy="no", report_to=[], max_length=2048,
        seed=5, gradient_checkpointing=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(OUT / "ckpt_sft"))
    del trainer
    torch.cuda.empty_cache()
    print("[done] sft")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tok, model = load_model()
    if which in ("all", "sft"):
        sft_run(tok, model)
    if which in ("all", "dpo_outcome"):
        dpo_run("dpo_outcome", DATA["pairs_outcome"], tok, model)
    if which in ("all", "dpo_process"):
        dpo_run("dpo_process", DATA["pairs_process"], tok, model)
    print("ALL TRAINING DONE")
