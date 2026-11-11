"""ATG 门面 (2026-11-03)

把 Lucy 的活动时序引擎接到 FRIDAY，并明确它的输入来源：
    1. brain —— 主脑思考/发言时顺带产出的"用户在做什么"（复用已有调用，不额外花 token）
    2. vision —— 桌面拼图经过主脑理解后的活动描述
    3. user —— 用户自己说的话

设计要点：ATG 自己不调用模型。谁调用了主脑，谁就顺手把结论写进来。
这样"要不要主动说话"的思考结果也变成了对用户建模的输入，一份 token 用两次。
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from brain.activity_timeline import ActivityTimeline

_tl = ActivityTimeline()
_lock = threading.RLock()

# ---------------- 模型自写的活动叙事（ATG 的正业） ----------------
# 设计：ATG 不再由视觉信号直接写，改成「模型看见画面 → 自己决定写一条叙事」。
# 覆盖式：同一活动内重复写只刷新最后一条，不追加日志；总长 TTL 滚动 30 条。
NARRATIVE_MAX = int(os.getenv("FRIDAY_ATG_NARR_MAX", "30"))

_narrative: Deque[Dict] = deque(maxlen=NARRATIVE_MAX)
_narr_lock = threading.RLock()
_narr_loaded = False


def _narr_path() -> Path:
    try:
        from core.config import get_data_dir
        d = Path(get_data_dir())
    except Exception:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return Path(os.getenv("FRIDAY_ATG_PATH") or (d / "atg_narrative.jsonl"))


def _load_narrative() -> None:
    global _narr_loaded
    with _narr_lock:
        if _narr_loaded:
            return
        _narr_loaded = True
        try:
            p = _narr_path()
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines()[-NARRATIVE_MAX:]:
                    if line.strip():
                        _narrative.append(json.loads(line))
        except Exception:
            pass


def write_narrative(
    text: str,
    basis: Optional[List[str]] = None,
    source: str = "mind",
    now: Optional[float] = None,
    replace_current: bool = True,
) -> Dict[str, object]:
    """模型自己写一条「这段时间在干什么」。返回 {"success", "replaced", "count"}。

    同一活动（activity_id 相同）内再次写入 → 覆盖上一条，不堆日志。
    活动变了 → 追加新条目，旧条目留在环形缓冲里当历史。
    """
    _load_narrative()
    body = (text or "").strip().replace("\n", " ")
    if not body:
        return {"success": False, "error": "text 为空"}
    if len(body) > 120:
        body = body[:120]
    now_ts = now or time.time()

    # ATG 图上显示的节点只由模型自己写：这里同时喂给活动时间线，
    # 否则页面上的标签其实是视觉模块猜的，和模型说过的话对不上。
    try:
        activity_id = observe(body[:80], source=source if source != "chat" else "mind", now=now_ts)
    except Exception:
        activity_id = None
    if not activity_id:
        with _lock:
            activity_id = (_last or {}).get("id") if _last else None

    with _narr_lock:
        entry = {
            "ts": now_ts,
            "text": body,
            "basis": [str(b)[:40] for b in (basis or [])][:4],
            "source": source,
            "activity_id": activity_id,
            "end_ts": None,
        }
        replaced = False
        if replace_current and _narrative:
            last = _narrative[-1]
            if last.get("activity_id") == activity_id:
                last["ts"] = now_ts
                last["text"] = body
                last["basis"] = entry["basis"]
                last["source"] = source
                replaced = True
        if not replaced:
            _narrative.append(entry)
        try:
            with open(_narr_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(_narrative[-1] if replaced else entry,
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass
        return {"success": True, "replaced": replaced, "count": len(_narrative)}


def narrative_recent(n: int = 6) -> List[Dict]:
    """最近 n 条活动叙事（旧→新）。"""
    _load_narrative()
    with _narr_lock:
        return list(_narrative)[-n:]


def narrative_view(max_items: int = 6) -> str:
    """给模型看的「你自己写过的活动史」，带序号方便 basis 引用。"""
    _load_narrative()
    items = narrative_recent(max_items)
    if not items:
        return "（你还没有记录活动）"
    now = time.time()
    lines = []
    with _narr_lock:
        base = len(_narrative) - len(items)
    for i, e in enumerate(items):
        dt = now - float(e.get("ts", now))
        stamp = f"-{int(dt)}s" if dt < 120 else f"-{int(dt / 60)}m"
        lines.append(f"#{base + i} {stamp} {e.get('text', '')}")
    # 明确顺序，别让模型把「先后」猜错（实测它会因为没写顺序而回答「不记得」）
    return "（按时间先后，越靠下越新）\n" + "\n".join(lines)


def narrative_stats() -> Dict[str, object]:
    _load_narrative()
    with _narr_lock:
        return {"count": len(_narrative), "max": NARRATIVE_MAX}

# ATG(区间) → TFGR(点/区间) 的沉淀状态：活动 id 变化时才落事件，避免高频刷屏
_last: Optional[dict] = None
_MIN_SPAN_S = 60.0     # 短于 1 分钟的活动不值得单独成区间（例如一次切窗口抖动）


def _sync_tfgr(activity_id: str, desc: str, now_ts: float, source: str) -> None:
    """活动切换时：旧活动沉淀成 TFGR 区间事件，新切换记一个点事件。"""
    global _last
    prev = _last
    if prev and prev["id"] == activity_id:
        return
    try:
        from brain.tfgr import get_tfgr
    except Exception:
        _last = {"id": activity_id, "desc": desc, "ts": now_ts, "source": source}
        return
    tfgr = get_tfgr()
    if prev:
        span = max(0.0, now_ts - prev["ts"])
        if span >= _MIN_SPAN_S:
            tfgr.record_interval("activity", prev["desc"], prev["ts"], now_ts,
                                 source=prev["source"])
        tfgr.record("activity.switch", f"{prev['desc']} → {desc}", ts=now_ts,
                    source=source, meta={"from": prev["desc"], "to": desc})
    else:
        tfgr.record("activity.start", desc, ts=now_ts, source=source)
    _last = {"id": activity_id, "desc": desc, "ts": now_ts, "source": source}


def observe(description: str, source: str = "vision",
            now: Optional[float] = None, task_context: str = "") -> Optional[str]:
    """写入一条观察。description 应尽量短（一句话），长文本会被截断。"""
    desc = (description or "").strip().replace("\n", " ")
    if not desc:
        return None
    if len(desc) > 80:
        desc = desc[:80]
    now_ts = now or time.time()
    with _lock:
        try:
            activity_id = _tl.observe_activity(
                description=desc,
                task_context=task_context,
                now=now_ts,
                source=source,
                raw_signal=description or "",
            )
        except Exception:
            return None
    # 落盘 IO 放到锁外，别让 ATG 观察路径被写文件拖慢
    try:
        _sync_tfgr(activity_id, desc, now_ts, source)
    except Exception:
        pass
    return activity_id


def observe_brain(text: str, now: Optional[float] = None) -> Optional[str]:
    """把主脑这次思考/主动发言的结论写进 ATG（source=brain）。

    注意：写的是"这段时间主脑观察到的事"，不是把整段话当活动。
    调用方应尽量传入简短的观察结论；传整段对话也行，会被截断。
    """
    return observe(text, source="brain", now=now)


def snapshot(max_items: int = 12) -> str:
    """给主脑看的活动时序文本（现在在干什么 / 刚才干了多久）。

    重启后内存里的时间线是空的，这时退回你自己写过的叙事，保证模型上下文
    和 ATG 页面上看到的是同一份东西。
    """
    text = ""
    with _lock:
        try:
            text = _tl.serialize_timeline_text(time.time(), max_items=max_items)
        except TypeError:
            try:
                text = _tl.serialize_timeline_text(time.time())
            except Exception:
                text = ""
        except Exception:
            text = ""
    if text and text.strip():
        return text
    return narrative_view(max_items)


def graph_snapshot(limit: int = 40, now: Optional[float] = None) -> Dict[str, object]:
    """结构化快照：节点 + 边 + 当前活动。

    给调试面板用（/api/atg）。注意 snapshot() 返回的是给模型看的文本，
    这里是图本身，前端要画时间轴/关系网得用它。
    """
    now_ts = now or time.time()
    with _lock:
        G = _tl.G
        nodes: List[Dict[str, object]] = []
        for nid, attrs in list(G.nodes(data=True)):
            node = attrs.get("data")
            if node is None:
                continue
            try:
                status = node.status.value if hasattr(node.status, "value") else str(node.status)
            except Exception:
                status = "unknown"
            nodes.append({
                "id": node.id,
                "desc": node.description,
                "task": node.task_context,
                "owner": node.owner,
                "start": node.t_start,
                "end": node.t_end,
                "status": status,
                "duration": round(node.duration_until(now_ts), 1),
                "confidence": round(float(node.confidence), 3),
                "source": node.source,
                "entities": list(node.entities or []),
                "interrupt_by": node.interrupt_by,
                "resume_of": node.resume_of,
                "snapshot": node.context_snapshot,
                "ongoing": node.is_ongoing,
            })
        nodes.sort(key=lambda n: float(n["start"] or 0), reverse=True)

        # 进程重启后时间线是空的（它不落盘），但你自己写过的叙事还在文件里。
        # 用叙事补一个只读视图，避免 ATG 页面看起来像“什么都没记”。
        _from_narrative = False
        if not nodes:
            items = narrative_recent(max(1, min(limit, NARRATIVE_MAX)))
            if items:
                for i, e in enumerate(items):
                    start = float(e.get("ts") or now_ts)
                    nxt = float(items[i + 1].get("ts") or now_ts) if i + 1 < len(items) else now_ts
                    end = max(start + 1.0, min(nxt, now_ts))
                    nodes.append({
                        "id": f"narr-{i}",
                        "desc": str(e.get("text") or ""),
                        "task": "",
                        "owner": "friday",
                        "start": start,
                        "end": end,
                        "status": "live" if i + 1 >= len(items) else "done",
                        "duration": round(max(0.0, end - start), 1),
                        "confidence": 1.0,
                        "source": str(e.get("source") or "mind"),
                        "entities": [],
                        "interrupt_by": None,
                        "resume_of": None,
                        "snapshot": "",
                        "ongoing": i + 1 >= len(items),
                    })
                nodes.sort(key=lambda n: float(n["start"] or 0), reverse=True)
                _from_narrative = True

        edges: List[Dict[str, object]] = []
        try:
            for u, v, key, attrs in list(G.edges(keys=True, data=True)):
                data = attrs.get("data")
                if data is None:
                    continue
                etype = getattr(data.edge_type, "value", str(data.edge_type))
                edges.append({"from": u, "to": v, "type": etype,
                              "strength": float(getattr(data, "strength", 1.0)),
                              "note": getattr(data, "note", "")})
        except Exception:
            pass

        if _from_narrative and not edges:
            edges = [
                {"from": nodes[i + 1]["id"], "to": nodes[i]["id"], "type": "BEFORE",
                 "strength": 1.0, "note": "模型叙事"}
                for i in range(len(nodes) - 1)
            ]

        cur = None
        try:
            node = _tl.get_current_activity(now_ts)
            if node is not None:
                cur = {
                    "id": node.id,
                    "desc": node.description,
                    "task": node.task_context,
                    "duration": round(node.duration_until(now_ts), 1),
                    "confidence": round(float(node.confidence), 3),
                    "source": node.source,
                    "entities": list(node.entities or []),
                }
        except Exception:
            pass

        return {
            "now": now_ts,
            "current": cur,
            "nodes": nodes[:limit],
            "edges": edges[-limit * 2:],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }


def current() -> str:
    with _lock:
        try:
            node = _tl.get_current_activity(time.time())
            return node.description if node else ""
        except Exception:
            return ""


def recent(n: int = 5) -> List[str]:
    """最近 n 条活动描述（供调试/面板）。"""
    out: List[str] = []
    with _lock:
        try:
            for a in _tl._find_recent_activities(time.time(), 3600):  # noqa: SLF001
                out.append(a.description)
        except Exception:
            pass
    return out[-n:]
