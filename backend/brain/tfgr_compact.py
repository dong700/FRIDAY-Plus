"""TFGR 节点语义整理（本地小模型）

机械去重（tfgr.prune）只能处理「字面完全一样」的节点；桌面事件里更常见的是
「同一个意思换了说法」——比如「切到 VS Code 写代码」「在 VS Code 里改脚本」。
这里用本地模型把这类近重复节点合并成一条，减少节点数，也方便以后检索。

约束：
- 只动非推断（非 belief/story/intent/pattern）事件；
- 模型必须返回现有 id，不许编造；拿不准就不合并；
- 单次最多处理几组、每组送 40 条，CPU 推理也不会拖太久。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from core.logger import get_logger

logger = get_logger("brain.tfgr_compact")

# 不参与语义合并的类型：推断(留给模型自己改) / 工具产物 / 人物档案
SKIP_KINDS = {"belief", "story", "intent", "pattern", "video.edit",
              "person.note", "person.name"}

_PROMPT = """你在整理一张「桌面事件」时间线的重复节点。
下面是一组同类事件（id + 内容 + 多少秒前）。请把**意思重复**的节点合并：

规则：
1. 只能使用下面出现过的 id，不许编造；
2. 只有确实在说同一件事才合并；拿不准就保持原样（不要放进任何一组）；
3. 合并后的 text 用一句话说清这件事，保留具体软件名/文件名，不要添加新信息；
4. 最多合并 6 组；没有重复就返回空数组。

事件：
{items}

只输出 JSON：
{{"merge": [{{"keep": "保留的id", "text": "合并后的一句话", "drop": ["被合并掉的id", "..."]}}]}}"""


async def compact_tfgr(window_hours: float = 24.0, max_groups: int = 4) -> Dict[str, int]:
    """用本地模型合并近重复节点。返回 {groups, merged, dropped}。

    默认关闭：本地模型不参与任何判断（用户要求）。需要时设 FRIDAY_TFGR_LLM=1 显式开启。
    """
    import os
    if os.getenv("FRIDAY_TFGR_LLM", "0") != "1":
        return {"groups": 0, "merged": 0, "dropped": 0, "skipped": 1}

    from brain.local_llm import available, chat_json

    if not await available():
        return {"groups": 0, "merged": 0, "dropped": 0, "skipped": 1}

    from brain.tfgr import get_tfgr
    tfgr = get_tfgr()
    now = time.time()
    events = [e for e in tfgr.recent(n=300, window_s=window_hours * 3600)
              if e.kind not in SKIP_KINDS]

    groups: Dict[str, List[Any]] = {}
    for e in events:
        groups.setdefault(e.kind, []).append(e)

    used_groups = merged = dropped = 0
    for kind, items in groups.items():
        if used_groups >= max_groups:
            break
        if len(items) < 3:
            continue
        batch = items[-40:]
        payload = [{"id": e.id, "t": e.text[:80], "ago_s": int(now - e.ts)} for e in batch]
        try:
            obj = await chat_json([
                {"role": "user", "content": _PROMPT.format(
                    items=json.dumps(payload, ensure_ascii=False))},
            ], max_tokens=600, timeout=45)
        except Exception as e:
            logger.warning(f"[TFGR-LLM] 本地模型整理失败: {e}")
            break
        if not obj:
            continue
        used_groups += 1
        valid_ids = {e.id for e in batch}
        for m in (obj.get("merge") or [])[:6]:
            if not isinstance(m, dict):
                continue
            keep = str(m.get("keep") or "")
            text = str(m.get("text") or "").strip()
            drops = [str(x) for x in (m.get("drop") or [])]
            if keep not in valid_ids or not text or not drops:
                continue
            tfgr.update(keep, text=text, note="本地模型语义合并")
            merged += 1
            for d in drops:
                if d in valid_ids and d != keep:
                    tfgr.delete(d, reason="本地模型语义合并")
                    dropped += 1
    if merged or dropped:
        logger.info(f"[TFGR-LLM] 语义合并：{merged} 组合并，删除 {dropped} 条重复节点")
    return {"groups": used_groups, "merged": merged, "dropped": dropped}
