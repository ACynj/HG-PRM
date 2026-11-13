#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agents/api_llm.py — 统一的 API LLM 调用封装（dmxapi / 硅基流动，均为
OpenAI 兼容接口）。密钥从 exp/.env 读取，不写入代码。

用法:
    from agents.api_llm import get_client
    llm = get_client("siliconflow", model="Qwen/Qwen2.5-7B-Instruct")
    out = llm.chat([{"role": "user", "content": "..."}], temperature=0.2)
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

PROVIDERS = {
    # 中转站，支持 GPT/Claude/Qwen 等多家模型
    "dmxapi": {
        "base_url": os.environ.get("DMXAPI_BASE_URL", "https://www.dmxapi.cn/v1"),
        "api_key_env": "DMXAPI_API_KEY",
    },
    # 硅基流动官方
    "siliconflow": {
        "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "api_key_env": "SILICONFLOW_API_KEY",
    },
}


class LLMClient:
    def __init__(self, provider: str, model: str):
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise RuntimeError(f"{cfg['api_key_env']} 未设置（检查 exp/.env）")
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(base_url=cfg["base_url"], api_key=key)
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 1024, retries: int = 5) -> str:
        for i in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                if i == retries - 1:
                    raise
                time.sleep(min(60, 3 * (2 ** i)) + 1)
        return ""


def get_client(provider: str = "siliconflow",
               model: str = "Qwen/Qwen2.5-7B-Instruct") -> LLMClient:
    return LLMClient(provider, model)
