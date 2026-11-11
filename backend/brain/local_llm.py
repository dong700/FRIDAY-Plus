"""本地小模型（Ollama）客户端

用途：给「快速决策 / 记忆提取 / TFGR 节点整理」这类高频、小输入的任务用，
省掉云端模型的 token 和往返延迟。连不上或超时就抛异常，调用方自己回退云端。

配置（环境变量）：
    FRIDAY_OLLAMA_URL     默认 http://127.0.0.1:11434
    FRIDAY_LOCAL_MODEL    默认 qwen2.5:1.5b
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from core.logger import get_logger

logger = get_logger("brain.local_llm")

_BASE = os.getenv("FRIDAY_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("FRIDAY_LOCAL_MODEL", "qwen2.5:1.5b")

# 可用性缓存：别每次调用都去探活
_probe_at = 0.0
_probe_ok = False
_PROBE_TTL = 20.0


async def available(force: bool = False) -> bool:
    """Ollama 是否在跑、目标模型是否已拉下来。结果缓存 20 秒。"""
    global _probe_at, _probe_ok
    now = time.time()
    if not force and now - _probe_at < _PROBE_TTL:
        return _probe_ok
    _probe_at = now
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(f"{_BASE}/api/tags")
            if r.status_code != 200:
                _probe_ok = False
                return False
            names = [m.get("name", "") for m in (r.json().get("models") or [])]
            # 允许 qwen2.5:1.5b / qwen2.5:1.5b-instruct 这类前缀匹配
            _probe_ok = any(n == MODEL or n.startswith(MODEL) for n in names)
    except Exception:
        _probe_ok = False
    return _probe_ok


async def chat(
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> str:
    """调用本地模型，返回文本。失败抛异常（由调用方回退云端）。"""
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{_BASE}/api/chat", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"ollama {r.status_code}: {r.text[:120]}")
        data = r.json()
    return str((data.get("message") or {}).get("content") or "")


async def chat_json(
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """要 JSON 的快捷方式；解析失败返回 None。"""
    text = await chat(messages, json_mode=True, max_tokens=max_tokens, timeout=timeout)
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
