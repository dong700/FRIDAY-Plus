"""OpenViking 本地上下文库接入（2026-11-10）

分工：
    - embedding 走本地 Ollama（nomic-embed-text / 768 维），不花云端 token
    - semantic / VLM 仍走云端（ark），本地模型不参与任何「提前判断」
    - 写入是异步索引（wait=false），不挂在对话主链路上

对外：
    find(query, limit)     语义检索，返回 [{"uri","score","text"}]
    write(uri, content)    覆盖写一篇文档
    sync_tfgr()            把最近的 TFGR 推断按天沉淀成文档，内容没变就跳过
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from core.logger import get_logger

logger = get_logger("brain.ov_memory")

BASE_URL = os.getenv("FRIDAY_OV_URL", "http://127.0.0.1:18095").rstrip("/")
TIMEOUT_S = float(os.getenv("FRIDAY_OV_TIMEOUT", "6"))
# 检索结果最多带多少字进上下文：宁可少，别把 prompt 撑爆
SNIPPET_CHARS = int(os.getenv("FRIDAY_OV_SNIPPET", "240"))

# 服务每次重启 account/user 都是默认值，dev 模式下不校验
_HEADERS = {
    "X-OpenViking-Account": "default",
    "X-OpenViking-User": "default",
}


def enabled() -> bool:
    return os.getenv("FRIDAY_OV_ENABLED", "1") == "1"


def _data_dir() -> Path:
    try:
        from core.config import get_data_dir

        d = Path(get_data_dir())
    except Exception:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _post(path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    to = aiohttp.ClientTimeout(total=timeout or TIMEOUT_S)
    try:
        async with aiohttp.ClientSession(timeout=to, headers=_HEADERS) as sess:
            async with sess.post(url, json=payload) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400 or body.get("status") == "error":
                    err = (body.get("error") or {}).get("message", "")
                    logger.warning(f"[OV] {path} -> {resp.status} {err[:200]}")
                    return None
                return body
    except Exception as e:
        logger.debug(f"[OV] {path} 不可用: {e}")
        return None


async def available() -> bool:
    """探活（不缓存太久：服务可能被用户手动关掉）。"""
    url = f"{BASE_URL}/health"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as sess:
            async with sess.get(url) as resp:
                return resp.status == 200
    except Exception:
        return False


def _overlap(query: str, text: str) -> float:
    """query 里的实词有多少出现在文档里（中文按字，英文按词）。

    本地 embedding 在中文短句上经常把「意图类」文档排得过高，
    用一次廉价的字面重合度做重排，把真正提到这件事的记录顶上来。
    """
    q = {c for c in query if "\u4e00" <= c <= "\u9fff"}
    q |= set(re.findall(r"[a-zA-Z0-9_]{2,}", query.lower()))
    if not q:
        return 0.0
    t = (text or "").lower()
    return sum(1 for x in q if x in t) / len(q)


async def find(query: str, limit: int = 6, min_score: float = 0.5) -> List[Dict[str, Any]]:
    """语义检索历史上下文。服务没起来/超时都返回空列表，不抛异常。"""
    if not enabled() or not (query or "").strip():
        return []
    body = await _post(
        "/api/v1/search/find",
        {"query": query.strip()[:300], "limit": int(limit), "read_content": True},
    )
    if not body:
        return []
    result = body.get("result") or {}
    out: List[Dict[str, Any]] = []
    for ctx_type, items in (("memory", result.get("memories") or []),
                            ("resource", result.get("resources") or [])):
        for it in items or []:
            text = (it.get("abstract") or it.get("content") or "").strip()
            if not text:
                continue
            uri = it.get("uri", "")
            # 目录 overview 只是索引壳子，带进上下文没信息量
            if uri.endswith("/.overview.md"):
                continue
            out.append({
                "uri": uri,
                "score": round(float(it.get("score") or 0.0), 3),
                "text": text[:SNIPPET_CHARS].replace("\n", " "),
                "type": ctx_type,
            })
    out = [x for x in out if x["score"] >= float(min_score)]
    # 语义分 + 字面重合度重排（0.8/0.2），避免中文短句被泛化的「意图」文档挤掉
    for x in out:
        x["rank"] = round(0.8 * x["score"] + 0.2 * _overlap(query, x["text"]), 3)
    out.sort(key=lambda x: -x["rank"])
    return out[: int(limit)]


async def write(uri: str, content: str) -> bool:
    if not enabled():
        return False
    body = await _post(
        "/api/v1/content/write",
        {
            "uri": uri,
            "content": content,
            "mode": "replace",
            "wait": False,
            "edit_source": "agent",
        },
        timeout=30,
    )
    return bool(body)


def _sync_state_path() -> Path:
    return _data_dir() / "ov_sync.json"


def _load_state() -> Dict[str, str]:
    p = _sync_state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: Dict[str, str]) -> None:
    try:
        _sync_state_path().write_text(
            json.dumps(state, ensure_ascii=False, indent=0), encoding="utf-8"
        )
    except Exception:
        logger.debug("[OV] 同步状态写入失败", exc_info=True)


async def sync_tfgr(days: int = 7) -> Dict[str, Any]:
    """把 TFGR 推断沉淀成 OpenViking 文档。

    两级：
    1. 重要的单条（钉住的 / 有到点时间的 / 重要度高的）单独成文，检索时能精确命中；
    2. 其余按天汇总成一篇，兜底「很久以前大概在想什么」。
    事实类事件仍在本地 RAG 里，不进 OpenViking，避免两边重复。
    """
    if not enabled():
        return {"skipped": True}
    try:
        from brain.tfgr import MENTAL_KINDS, USER_KINDS, get_tfgr

        g = get_tfgr()
        events = g.recent(n=2000, window_s=days * 86400)
    except Exception:
        logger.debug("[OV] 读取 TFGR 失败", exc_info=True)
        return {"skipped": True}

    now = time.time()
    state = _load_state()
    written, skipped, nodes = 0, 0, 0

    # ---- 1. 重要节点单独成文（最多 40 条/轮，避免一次写爆） ----
    important = []
    for e in events:
        if e.meta.get("deleted"):
            continue
        if e.meta.get("pin") or e.meta.get("due_at") or g.importance(e, now) >= 0.85:
            important.append(e)
    important.sort(key=lambda e: -g.importance(e, now))

    for e in important[:40]:
        head = [f"# {e.text[:80]}", ""]
        meta_bits = [
            f"记录于 {datetime.fromtimestamp(e.ts).strftime('%Y-%m-%d %H:%M')}",
            f"类型 {e.kind}",
            f"置信度 {float(e.confidence):.2f}",
        ]
        due_at = e.meta.get("due_at")
        if due_at:
            due_dt = datetime.fromtimestamp(float(due_at))
            meta_bits.append(f"到点时间 {due_dt.strftime('%Y-%m-%d %H:%M')}")
            if e.meta.get("done"):
                meta_bits.append("状态 已提醒")
        if e.entities:
            meta_bits.append("相关 " + "、".join(e.entities[:5]))
        body = "\n".join(head) + "\n" + "；".join(meta_bits) + "\n\n" + e.text + "\n"
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
        uri = f"viking://resources/friday/mem/{e.id}.md"
        nodes += 1
        if state.get(uri) == digest:
            skipped += 1
            continue
        if await write(uri, body):
            state[uri] = digest
            written += 1

    # ---- 2. 按天汇总 ----
    by_day: Dict[str, List[Any]] = {}
    for e in events:
        if e.meta.get("deleted"):
            continue
        # 推断（belief/story/intent/pattern）+ 他让我记的事（memory）都沉淀；
        # 纯机械事件留在本地 RAG，不进 OpenViking
        if e.kind not in MENTAL_KINDS and e.kind not in USER_KINDS:
            continue
        day = datetime.fromtimestamp(e.ts).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(e)

    for day, items in sorted(by_day.items()):
        items.sort(key=lambda x: x.ts)
        lines = [f"# FRIDAY 推断记录 {day}", ""]
        for e in items:
            hhmm = datetime.fromtimestamp(e.ts).strftime("%H:%M")
            ents = f"（{', '.join(e.entities[:4])}）" if e.entities else ""
            due_txt = ""
            if e.meta.get("due_at"):
                due_dt = datetime.fromtimestamp(float(e.meta["due_at"]))
                due_txt = f" [到点 {due_dt.strftime('%m-%d %H:%M')}]"
            lines.append(f"- {hhmm} [{e.kind} {float(e.confidence):.2f}]{due_txt} {e.text}{ents}")
        content = "\n".join(lines) + "\n"
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
        uri = f"viking://resources/friday/tfgr/{day}.md"
        if state.get(uri) == digest:
            skipped += 1
            continue
        if await write(uri, content):
            state[uri] = digest
            written += 1

    if written:
        _save_state(state)
    return {"days": len(by_day), "nodes": nodes, "written": written, "skipped": skipped}
