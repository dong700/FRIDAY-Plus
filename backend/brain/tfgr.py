"""TFGR — Time-Flow Graph（2026-11-04）

定位（和 ATG 互补）：
    ATG  = 连续活动「区间」：在做什么、持续多久（写代码 32 分钟 / 看视频 18 分钟）
    TFGR = 离散事件「点」+ 因果：什么时候发生了什么（切到 B 站、开始看第 3 集、主动说了句什么）

    ATG 负责「现在的状态」，TFGR 负责「事件链」。两者都从同一条 Signal 流里长出来：
        Signal → ATG 聚合 → 活动结束 → 沉淀成 TFGR 区间事件
        Signal → TFGR 直接记点（用户说话 / 应用切换 / 视频场景 / 工具调用 / 钩子）

存储：
    - 内存有界（默认 2000 条），JSONL 落盘（超过 2MB 自动轮转），重启可恢复最近事件
    - 每条事件可以带 entities / refs（文件、URL、视频片段、图片），供检索和因果链接

对外：
    record()            记一个点事件
    record_interval()   记一个区间事件（ATG 活动结束时用）
    link()              加一条因果/时序边
    recent()            最近 n 条（给上下文）
    around()            某个时刻前后的事件
    search()            关键词/实体检索
    summarize()         压缩成几行文本给模型看
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from core.logger import get_logger

logger = get_logger("brain.tfgr")

MAX_EVENTS = int(os.getenv("FRIDAY_TFGR_MAX", "2000"))
MAX_FILE_MB = 2.0

# 桌面感知类节点的清理策略（2026-11-09）：
# 窗口切换/静默诊断这类点事件一天能写几千条，不清理会把节点池撑满、拖慢检索。
PRUNE_TTL_DAYS = float(os.getenv("FRIDAY_TFGR_TTL_DAYS", "3"))
PRUNE_MAX_ALIVE = int(os.getenv("FRIDAY_TFGR_MAX_ALIVE", "800"))
# 纯噪声类型：优先合并/删除（诊断记录、窗口轮询）
NOISE_KINDS = {"mind.silent", "activity.switch", "video.window", "activity.start"}
# 近重复判定的时间窗：同 kind 同文本在这个窗口内只留最后一条
DUP_WINDOW_S = float(os.getenv("FRIDAY_TFGR_DUP_WINDOW", "1800"))

# TFGR 只收「模型自己的推断」，不收事实（用户原话/字幕/帧描述属于 RAG）
MENTAL_KINDS = ("belief", "story", "intent", "pattern")
# 用户明确让记的事（偏好/事实/计划）。和模型的推断分开存：
# 注入时必须标明来源，否则模型会把自己猜的当成「用户亲口说的」用。
USER_KINDS = ("memory",)
THOUGHT_WINDOW_S = float(os.getenv("FRIDAY_THOUGHT_WINDOW", "300"))
THOUGHT_MAX_PER_WINDOW = int(os.getenv("FRIDAY_THOUGHT_QUOTA", "3"))
THOUGHT_MIN_CONF = float(os.getenv("FRIDAY_THOUGHT_MIN_CONF", "0.2"))

# ---------- 重要度（2026-11-10） ----------
# 清理不再只看「多老」，还要看「多重要」：一条 10 天后要提醒的事，
# 比今天下午的窗口切换重要得多，不能因为超 TTL 就被丢掉。
KIND_WEIGHT = {
    "memory": 1.00,
    "intent": 1.00,
    "belief": 0.85,
    "pattern": 0.70,
    "story": 0.55,
    "video.edit": 0.80,
    "task.done": 0.60,
    "chat.memory": 0.85,
}
# 半个月没被用到，权重衰减一半（靠 last_hit 更新，越常用的越不容易被清）
MENTAL_HALF_LIFE_DAYS = float(os.getenv("FRIDAY_MEM_HALF_LIFE", "14"))
PIN_BOOST = 1.00      # 明确要求「记住这个」
DUE_BOOST = 1.50      # 有明确时间点的事（到点提醒）
NEGATE_FACTOR = 0.55  # 被用户否定过的推断，重要度直接打折
# 注入稳定性：刚注入过的推断给一点「粘性」，避免每轮排序抖动、prompt 来回换
INJECT_STICKY_S = float(os.getenv("FRIDAY_INJECT_STICKY", "600"))
STICKY_BOOST = 0.08
# 提醒提前多久就算「要提」（默认 6 小时，够模型提前打个招呼）
DUE_LEAD_S = float(os.getenv("FRIDAY_DUE_LEAD", "21600"))
# 用户否定「你刚才说的」的常见说法：命中就把最近注入过的推断降权
NEGATION_RE = re.compile(
    r"你记错|记错了|我没说过|没说过|不是这样|别乱说|胡说|搞错|不对吧|没有这回事")



def _data_dir() -> Path:
    try:
        from core.config import get_data_dir
        d = get_data_dir()
    except Exception:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _norm(text: str) -> str:
    """用于近重复判定的归一化（只留字母数字，中文按字保留）。"""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())[:120]


def _query_overlap(query: str, text: str) -> float:
    """当前问题里的实词有多少出现在这条记忆里（0~1）。

    中文短句上 embedding 排序不稳，用一次廉价的字面重合度，
    把「正好在问的那件事」顶到前面，避免每次都注入同一批钉子。
    """
    q = {c for c in (query or "") if "\u4e00" <= c <= "\u9fff"}
    q |= set(re.findall(r"[a-zA-Z0-9_]{2,}", (query or "").lower()))
    if not q:
        return 0.0
    t = (text or "").lower()
    return sum(1 for x in q if x in t) / len(q)


def _near_dup(body_norm: str, others: List[str], ratio: float = 0.75) -> bool:
    """和已有文本是否算同一条（包含关系或高度相似）。"""
    for o in others:
        if not o:
            continue
        if body_norm in o or o in body_norm:
            return True
        if difflib.SequenceMatcher(None, body_norm, o).ratio() >= ratio:
            return True
    return False


@dataclass
class TFEvent:
    id: str
    ts: float
    kind: str
    text: str
    source: str = "system"
    end_ts: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    basis: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TFEvent":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class TimeFlowGraph:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._events: Deque[TFEvent] = deque(maxlen=MAX_EVENTS)
        self._edges: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTS * 2)
        self._lock = threading.RLock()
        self._path = Path(path) if path else (
            Path(os.getenv("FRIDAY_TFGR_PATH")) if os.getenv("FRIDAY_TFGR_PATH")
            else _data_dir() / "tfgr.jsonl"
        )
        self._load()

    # ---------- 写 ----------

    def record(
        self,
        kind: str,
        text: str,
        ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        source: str = "system",
        entities: Optional[List[str]] = None,
        refs: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        basis: Optional[List[str]] = None,
    ) -> str:
        ev = TFEvent(
            id=uuid.uuid4().hex[:10],
            ts=ts or time.time(),
            kind=kind,
            text=(text or "").strip()[:300],
            source=source,
            end_ts=end_ts,
            entities=entities or [],
            refs=refs or [],
            meta=meta or {},
            confidence=float(confidence),
            basis=list(basis or []),
        )
        with self._lock:
            self._events.append(ev)
            self._auto_link(ev)
            self._append(ev)
        return ev.id

    def record_interval(
        self,
        kind: str,
        text: str,
        start_ts: float,
        end_ts: float,
        source: str = "system",
        entities: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """区间事件（活动开始时 ts=start，结束时补记 end）。时间戳可能早于已有事件，
        写入后按 ts 重排，保证 recent()/around() 的时序语义。"""
        ev_id = self.record(kind, text, ts=start_ts, end_ts=end_ts, source=source,
                            entities=entities, meta=meta)
        with self._lock:
            self._events = deque(sorted(self._events, key=lambda e: e.ts),
                                  maxlen=MAX_EVENTS)
        return ev_id

    # ---------- 思考写入（TFGR 的正业） ----------

    def thought_quota_left(self, now: Optional[float] = None) -> int:
        """当前窗口还剩几条写入额度（硬上限，防刷屏）。"""
        now_ts = now or time.time()
        with self._lock:
            used = sum(
                1 for e in self._events
                if e.kind in MENTAL_KINDS and now_ts - e.ts < THOUGHT_WINDOW_S
            )
        return max(0, THOUGHT_MAX_PER_WINDOW - used)

    def record_thought(
        self,
        kind: str,
        text: str,
        confidence: float = 0.6,
        basis: Optional[List[str]] = None,
        source: str = "mind",
        entities: Optional[List[str]] = None,
        now: Optional[float] = None,
        allow_conflict: bool = False,
    ) -> Dict[str, Any]:
        """模型自己决定写一条推断。返回 {"success": bool, ...}，失败带原因。

        只允许 belief/story/intent/pattern；置信度低于 THOUGHT_MIN_CONF 直接拒，
        basis 必填（引用 ATG 序号或 rag id），用于事后追溯，防幻觉。
        """
        kind = (kind or "").strip().lower()
        if kind not in MENTAL_KINDS:
            return {"success": False, "error": f"kind 必须是 {list(MENTAL_KINDS)}"}
        body = (text or "").strip()
        if not body:
            return {"success": False, "error": "text 为空"}
        if confidence < THOUGHT_MIN_CONF:
            return {"success": False, "error": f"confidence<{THOUGHT_MIN_CONF} 不值得记"}
        if not basis:
            return {"success": False, "error": "basis 必填（引用 atg 序号或 rag_id）"}
        left = self.thought_quota_left(now=now)
        if left <= 0:
            return {
                "success": False,
                "error": f"配额已满：每 {int(THOUGHT_WINDOW_S)}s 最多 "
                         f"{THOUGHT_MAX_PER_WINDOW} 条，等下一窗口再写",
            }
        # 写入前先跟历史对一遍：像「他现在在写代码」和「他已经出门了」这种自相矛盾，
        # 不能直接追加成两条，先交回给模型决定是改、是删，还是留个钩子去确认。
        conflicts = self.find_conflicts(kind, body, entities, now=now)
        if conflicts and not allow_conflict:
            return {
                "success": False,
                "conflict": True,
                "message": "这条和已有记录可能矛盾，先处理再写：确认是补充细节就用 force=True 重写；"
                           "是之前判断错了就用 tfgr_update 改；确认过时/不成立就用 tfgr_delete 删；"
                           "拿不准就先留钩子去确认。",
                "candidates": conflicts,
            }
        ev_id = self.record(kind, body, ts=now, source=source, entities=entities,
                            confidence=min(0.99, max(0.0, confidence)),
                            basis=[str(b)[:40] for b in basis][:4])
        return {"success": True, "id": ev_id, "quota_left": left - 1}

    # ---------- 查询 / 修订 / 删除（给模型纠错用的 CRUD） ----------

    def _alive(self) -> List["TFEvent"]:
        return [e for e in self._events if not e.meta.get("deleted")]

    def get(self, event_id: str) -> Optional[TFEvent]:
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    return e
        return None

    def find_conflicts(
        self,
        kind: str,
        text: str,
        entities: Optional[List[str]] = None,
        window_s: float = 48 * 3600,
        limit: int = 4,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """找可能和这条新推断矛盾的旧记录（同实体 / 文字高度相似）。"""
        now_ts = now or time.time()
        ents = {_norm(x) for x in (entities or []) if x}
        body = _norm(text)
        if not body:
            return []
        with self._lock:
            items = [e for e in self._events
                     if not e.meta.get("deleted") and now_ts - e.ts <= window_s
                     and e.kind in MENTAL_KINDS]
        out: List[Dict[str, Any]] = []
        for e in items:
            sim = difflib.SequenceMatcher(None, body, _norm(e.text)).ratio()
            shared = ents & {_norm(x) for x in e.entities}
            if e.kind == kind:
                sim += 0.06
            if shared:
                sim += 0.08
            if sim >= 0.62:
                out.append({
                    "id": e.id,
                    "kind": e.kind,
                    "text": e.text[:90],
                    "confidence": round(float(e.confidence), 2),
                    "ago_s": int(now_ts - e.ts),
                    "similarity": round(min(1.0, sim), 2),
                    "entities": list(e.entities[:4]),
                })
        out.sort(key=lambda x: -x["similarity"])
        return out[:limit]

    def update(
        self,
        event_id: str,
        text: Optional[str] = None,
        kind: Optional[str] = None,
        confidence: Optional[float] = None,
        entities: Optional[List[str]] = None,
        basis: Optional[List[str]] = None,
        note: str = "",
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """改一条已有推断。保留 before 快照，重启后还能看出改过什么。"""
        with self._lock:
            for e in self._events:
                if e.id != event_id:
                    continue
                if e.meta.get("deleted"):
                    return {"success": False, "error": "该事件已删除，改不了"}
                before = {"text": e.text, "kind": e.kind,
                          "confidence": round(float(e.confidence), 2)}
                if text:
                    e.text = text.strip()[:300]
                if kind and kind in MENTAL_KINDS:
                    e.kind = kind
                if confidence is not None:
                    e.confidence = min(0.99, max(0.0, float(confidence)))
                if entities is not None:
                    e.entities = list(entities)[:8]
                if basis:
                    e.basis = [str(b)[:40] for b in basis][:4]
                rev = int(e.meta.get("rev") or 0) + 1
                e.meta["rev"] = rev
                e.meta["updated_at"] = now or time.time()
                hist = e.meta.setdefault("history", [])
                hist.append({"at": now or time.time(), "before": before, "note": (note or "")[:80]})
                e.meta["history"] = hist[-6:]
                self._persist()
                return {"success": True, "id": event_id, "rev": rev, "before": before}

        return {"success": False, "error": "找不到这条事件（id 可能已经轮换出窗口）"}

    def delete(self, event_id: str, reason: str = "", now: Optional[float] = None) -> Dict[str, Any]:
        """软删除：从检索/摘要里消失，但文件和审计记录保留。"""
        with self._lock:
            for e in self._events:
                if e.id != event_id:
                    continue
                e.meta["deleted"] = True
                e.meta["deleted_at"] = now or time.time()
                e.meta["deleted_reason"] = (reason or "")[:120]
                self._persist()
                return {"success": True, "id": event_id, "deleted": True}
        return {"success": False, "error": "找不到这条事件"}

    def _persist(self) -> None:
        """整份重写当前事件文件（append-only 模式下 update/delete 才能生效）。"""
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text("".join(e.to_json() + "\n" for e in self._events),
                           encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.warning("[TFGR] 修订落盘失败", exc_info=True)

    def mental_recent(self, n: int = 10, window_s: Optional[float] = None) -> List[TFEvent]:
        now = time.time()
        with self._lock:
            items = [e for e in self._events
                     if e.kind in MENTAL_KINDS and not e.meta.get("deleted")]
        if window_s is not None:
            items = [e for e in items if now - e.ts <= window_s]
        return items[-n:]

    def mental_summary(self, window_s: float = 86400, max_items: int = 8) -> str:
        """给模型看的「你自己想过什么」，带置信度与依据。"""
        items = self.mental_recent(max_items, window_s=window_s)
        if not items:
            return "（你还没有写过判断）"
        now = time.time()
        lines = []
        for e in items:
            dt = now - e.ts
            stamp = f"-{int(dt)}s" if dt < 120 else f"-{int(dt / 60)}m"
            basis = f" ←{'/'.join(e.basis[:3])}" if e.basis else ""
            lines.append(f"{stamp} {e.kind}({e.confidence:.2f}): {e.text[:80]}{basis}")
        return "\n".join(lines)

    # ---------- 重要度 / 提醒 / 主动注入（2026-11-10） ----------

    def importance(self, e: TFEvent, now: Optional[float] = None) -> float:
        """这条记录有多值得再被想起来（越大越重要，不做归一）。

        计分：kind 基础权重 + 置信度 + 实体数 + 手动 pin + 有到点时间，
        再按「多久没被用到」（meta.last_hit）做半衰期衰减，防止旧判断永远霸屏。
        """
        now_ts = now or time.time()
        score = KIND_WEIGHT.get(e.kind, 0.35)
        score += 0.25 * min(1.0, max(0.0, float(e.confidence)))
        if e.entities:
            score += min(0.15, 0.05 * len(e.entities))
        if e.meta.get("pin"):
            score += PIN_BOOST
        if e.meta.get("due_at"):
            score += DUE_BOOST
        last = float(e.meta.get("last_hit") or e.ts)
        age_days = max(0.0, (now_ts - last) / 86400.0)
        score *= 0.5 ** (age_days / max(1.0, MENTAL_HALF_LIFE_DAYS))
        # 用户否定过的推断：降权，别再当事实反复说
        if e.meta.get("negated"):
            score *= NEGATE_FACTOR
        return round(score, 3)

    def touch(self, event_id: str, now: Optional[float] = None) -> None:
        """记录「这条被用到了」：常用节点的衰减慢，长期没人用的才该被清。"""
        now_ts = now or time.time()
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    e.meta["last_hit"] = now_ts
                    return

    def get(self, event_id: str) -> Optional[TFEvent]:
        """按 id 取一条事件（OpenViking 命中回标日期/来源时用）。"""
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    return e
        return None

    def penalize_recent(self, user_text: str, window_s: float = 1800,
                        max_items: int = 3, factor: float = 0.6,
                        now: Optional[float] = None) -> List[str]:
        """负反馈：用户否定了刚才的说法 → 最近注入过的推断降权。

        - 用户这句话和某条推断有字面重合：只罚那几条（精准）；
        - 只说了「你记错了 / 我没说过」这类：罚最近注入的前 2 条；
        - 只说「不对」这种泛否定：不动（避免误伤）。
        降权是永久的，直到同一件事被重新确认。
        """
        text = (user_text or "").strip()
        if not text:
            return []
        explicit = bool(NEGATION_RE.search(text))
        now_ts = now or time.time()
        changed: List[str] = []
        with self._lock:
            cands = [
                e for e in self._events
                if e.kind in MENTAL_KINDS and not e.meta.get("deleted")
                and now_ts - float(e.meta.get("last_inject") or 0) <= window_s
            ]
            if not cands:
                return []
            cands.sort(key=lambda e: -float(e.meta.get("last_inject") or 0))
            hit = [e for e in cands if _query_overlap(text, e.text) >= 0.3]
            target = hit[:max_items] if hit else (cands[:2] if explicit else [])
            for e in target:
                e.confidence = max(0.1, round(float(e.confidence) * factor, 2))
                e.meta["negated"] = now_ts
                changed.append(e.id)
            if changed:
                self._persist()
        return changed

    def remember(
        self,
        text: str,
        due_in_days: Optional[float] = None,
        due_at: Optional[float] = None,
        entities: Optional[List[str]] = None,
        source: str = "user",
        pin: bool = True,
    ) -> Dict[str, Any]:
        """记一件「以后还要用」的事，可以带明确时间点。

        和 record_thought 的区别：走用户/系统语义，不占用模型推断配额，
        默认 pin + 重要，清理时受保护。
        例：用户说「10 天后提醒我交报告」→ due_in_days=10。
        """
        body = (text or "").strip()
        if not body:
            return {"success": False, "error": "text 为空"}
        ts_now = time.time()
        due_ts: Optional[float] = None
        if due_at is not None:
            due_ts = float(due_at)
        elif due_in_days is not None:
            due_ts = ts_now + float(due_in_days) * 86400
        meta: Dict[str, Any] = {}
        if pin or due_ts:
            meta["pin"] = True
        if due_ts:
            meta["due_at"] = due_ts

        # 同一件事说两遍只留一条（重复的提醒会让「到点提醒」刷屏）
        body_norm = _norm(body)
        with self._lock:
            for e in self._events:
                if e.meta.get("deleted") or e.meta.get("done"):
                    continue
                if bool(e.meta.get("due_at")) != bool(due_ts):
                    continue
                if (body_norm == _norm(e.text)
                        or difflib.SequenceMatcher(None, body_norm, _norm(e.text)).ratio() > 0.82):
                    e.meta["pin"] = True
                    e.meta["last_hit"] = ts_now
                    # 重新确认了同一件事 → 撤销之前的否定降权
                    e.meta.pop("negated", None)
                    if due_ts:
                        e.meta["due_at"] = min(float(e.meta.get("due_at") or due_ts), due_ts)
                    self._persist()
                    return {"success": True, "id": e.id, "merged": True,
                            "due_at": e.meta.get("due_at")}

        ev_id = self.record("memory", body, ts=ts_now, source=source,
                            entities=entities or [], meta=meta,
                            confidence=0.9, basis=["user.remember"])
        out: Dict[str, Any] = {"success": True, "id": ev_id}
        if due_ts:
            out["due_at"] = due_ts
            out["due_in_days"] = round((due_ts - ts_now) / 86400.0, 1)
        return out

    def mark_done(self, event_id: str, note: str = "", now: Optional[float] = None) -> Dict[str, Any]:
        """提醒已处理：不再进「到点提醒」，但记录本身留着（以后还能查到）。"""
        now_ts = now or time.time()
        with self._lock:
            for e in self._events:
                if e.id == event_id:
                    e.meta["done"] = True
                    e.meta["done_at"] = now_ts
                    if note:
                        e.meta["done_note"] = note[:200]
                    self._persist()
                    return {"success": True, "id": event_id}
        return {"success": False, "error": "没有这条记录"}

    def due(self, window_s: float = DUE_LEAD_S, now: Optional[float] = None,
            include_done: bool = False) -> List[TFEvent]:
        """到点（或即将到点）的提醒，按时间先后排。"""
        now_ts = now or time.time()
        with self._lock:
            items = [
                e for e in self._events
                if e.meta.get("due_at") and not e.meta.get("deleted")
                and (include_done or not e.meta.get("done"))
                and float(e.meta["due_at"]) <= now_ts + float(window_s)
            ]
        items.sort(key=lambda e: float(e.meta["due_at"]))
        return items

    def due_next_ts(self, now: Optional[float] = None) -> Optional[float]:
        """下一条提醒的时间点（给心智循环算下次醒来用）。"""
        items = self.due(window_s=365 * 86400, now=now)
        return float(items[0].meta["due_at"]) if items else None

    def digest(self, max_items: int = 8, max_chars: int = 1200,
               now: Optional[float] = None, query: str = "",
               exclude_texts: Optional[List[str]] = None) -> str:
        """给模型注入用的「长期记忆摘要」，按重要度 + 来源分层。

        顺序：到点提醒 > 带时间的计划 > 用户让记的事实 > 钉住的推断 > 高重要度推断。
        每条都标明来源（「他让记」/「我推断」），避免模型把自己猜的当用户说的用。

        query: 当前用户这句话。给了就把「和这句话相关」的权重加上，
               免得每轮都注入同一批钉子、真正在问的事反而没带上。
        exclude_texts: 别的来源（比如 OpenViking 检索）已经展示过的文本，
               推断类不再重复注入；到点提醒 / 他让记的事不受影响（宁可重复也不能漏）。
        有字符预算，宁少不撑爆 prompt。
        """
        now_ts = now or time.time()
        query = (query or "").strip()
        excluded = [_norm(x) for x in (exclude_texts or []) if (x or "").strip()]
        with self._lock:
            items = [e for e in self._events if not e.meta.get("deleted")]
        if not items:
            return ""

        picked: List[TFEvent] = []
        seen = set()
        # 已注入文本的归一化，用于在推断层做跨来源去重
        shown: List[str] = list(excluded)

        def take(ev: TFEvent, dedup: bool = False) -> None:
            if ev.id in seen or len(picked) >= max_items:
                return
            body = _norm(ev.text)
            if dedup and _near_dup(body, shown):
                return
            seen.add(ev.id)
            picked.append(ev)
            shown.append(body)

        def _score(ev: TFEvent) -> float:
            score = self.importance(ev, now_ts)
            if query:
                score += 0.6 * _query_overlap(query, ev.text)
            # 刚注入过的：给一点粘性，避免两个相近的推断每轮互换位置
            last_inj = float(ev.meta.get("last_inject") or 0)
            if now_ts - last_inj <= INJECT_STICKY_S:
                score += STICKY_BOOST
            return score

        def _rank(ev: TFEvent):
            return (-_score(ev), -ev.ts, ev.id)

        # 1) 到点/快到的提醒（最重要，不参与去重）
        for e in self.due(window_s=DUE_LEAD_S, now=now_ts):
            take(e)
        # 2) 带明确时间点的计划（还没到窗口的也提示一下）
        for e in sorted([x for x in items if x.meta.get("due_at") and not x.meta.get("done")],
                        key=lambda x: float(x.meta["due_at"])):
            take(e)
        # 3) 用户明确让记的事（事实/偏好），优先级高于模型自己的推断
        for e in sorted([x for x in items if x.kind in USER_KINDS], key=_rank):
            take(e)
        # 4) 钉住的推断（和 OpenViking 已展示过的去重）
        for e in sorted([x for x in items if x.meta.get("pin") and x.kind not in USER_KINDS],
                        key=_rank):
            take(e, dedup=True)
        # 5) 余下的推断，按「重要度 + 与当前问题的相关性」
        for e in sorted([x for x in items if x.kind in MENTAL_KINDS], key=_rank):
            take(e, dedup=True)

        lines: List[str] = []
        used = 0
        for e in picked:
            self.touch(e.id, now_ts)
            if e.kind in MENTAL_KINDS:
                # 记下「这条刚被注入过」，供下一轮的粘性排序和负反馈定位
                e.meta["last_inject"] = now_ts
            age = now_ts - e.ts
            when = f"{int(age / 86400)}天前" if age >= 86400 else f"{max(1, int(age / 60))}分钟前"
            due_txt = ""
            if e.meta.get("due_at") and not e.meta.get("done"):
                left = float(e.meta["due_at"]) - now_ts
                if left <= 0:
                    due_txt = "【已到点】"
                elif left < 86400:
                    due_txt = f"【还有{max(1, int(left / 3600))}小时】"
                else:
                    due_txt = f"【还有{int(left / 86400)}天】"
            src = "他让记" if e.kind in USER_KINDS else "我推断"
            line = f"{due_txt}{src}·{when}: {e.text[:90]}"
            if lines and used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def consolidate(self, now: Optional[float] = None) -> Dict[str, int]:
        """整理：去近重复（跨表述）、抬升置信度、淘汰过期的低置信推断。

        去重范围从原来的「6 小时内一字不差」放宽到「7 天内高度相似」，
        因为同一个意思换个说法是最常见的重复形态，纯字符串相等抓不到。
        带 due_at 的计划一条都不合（不同时间点是两件事）。
        """
        now_ts = now or time.time()
        merged = pruned = 0
        with self._lock:
            survivors: List[TFEvent] = []
            recent: List[TFEvent] = []
            for e in list(self._events):
                if e.kind not in MENTAL_KINDS and e.kind not in USER_KINDS:
                    survivors.append(e)
                    continue
                if (e.kind in MENTAL_KINDS and now_ts - e.ts > 48 * 3600
                        and e.confidence < 0.4
                        and not e.meta.get("pin") and not e.meta.get("due_at")):
                    pruned += 1
                    continue
                if e.meta.get("due_at"):
                    survivors.append(e)
                    continue
                body = _norm(e.text)
                target: Optional[TFEvent] = None
                for p in reversed(recent[-300:]):
                    if p.kind != e.kind or e.ts - p.ts > 7 * 86400:
                        continue
                    pbody = _norm(p.text)
                    if body == pbody or difflib.SequenceMatcher(None, body, pbody).ratio() >= 0.9:
                        target = p
                        break
                if target is not None:
                    target.confidence = min(0.95, max(target.confidence, e.confidence) + 0.03)
                    target.basis = list(dict.fromkeys(target.basis + e.basis))[:4]
                    target.ts = max(target.ts, e.ts)
                    if len(e.text) > len(target.text):
                        target.text = e.text
                    merged += 1
                    continue
                recent.append(e)
                survivors.append(e)
            survivors.sort(key=lambda x: x.ts)
            self._events = deque(survivors, maxlen=MAX_EVENTS)
        return {"merged": merged, "pruned": pruned}

    def prune(self, now: Optional[float] = None, ttl_days: Optional[float] = None,
              max_alive: Optional[int] = None) -> Dict[str, int]:
        """清理桌面感知类节点（只动非推断事件，mental 推断保留给模型自己改/删）。

        规则：
        1. 软删除过的事件不再占内存；
        2. 近重复合并：同 kind + 同文本，DUP_WINDOW_S 内只留最后一条
           （窗口标题/活动切换一天几百条，绝大多数是重复）；
        3. 过期删除：非推断事件超过 TTL 直接丢（video.edit 这类结果保留）；
           带 pin / 有到点时间（due_at）的事件不参与过期，10 天后的提醒不会被清；
        4. mind.silent 这类诊断噪声额外只留最近 30 条；
        5. 总量封顶：按「重要度」从低到高删，噪声类最先，保护对象最后。
        """
        now_ts = now or time.time()
        ttl = float(ttl_days if ttl_days is not None else PRUNE_TTL_DAYS) * 86400
        cap = int(max_alive if max_alive is not None else PRUNE_MAX_ALIVE)
        dropped_deleted = dropped_dup = dropped_old = dropped_cap = 0
        with self._lock:
            # 1) 软删除 + 过期（pin / due_at 强制保留）
            kept: List[TFEvent] = []
            for e in list(self._events):
                if e.meta.get("deleted"):
                    dropped_deleted += 1
                    continue
                protected = bool(e.meta.get("pin") or e.meta.get("due_at"))
                if (not protected and e.kind not in MENTAL_KINDS
                        and e.kind != "video.edit"
                        and (now_ts - e.ts) > ttl):
                    dropped_old += 1
                    continue
                kept.append(e)

            # 2) 近重复：同 kind + 同文本，DUP_WINDOW_S 内只留最后一条
            #    受保护 / 推断类不参与合并（重复的计划可能内容不同、都要留）
            survivors: List[TFEvent] = []
            last_by_key: Dict[str, TFEvent] = {}
            idx_by_key: Dict[str, int] = {}
            for e in kept:
                if e.kind in MENTAL_KINDS or e.meta.get("pin") or e.meta.get("due_at"):
                    survivors.append(e)
                    continue
                key = f"{e.kind}|{_norm(e.text)}"
                prev = last_by_key.get(key)
                if prev is not None and (e.ts - prev.ts) < DUP_WINDOW_S:
                    survivors[idx_by_key[key]] = e   # 用新的替换旧的
                    dropped_dup += 1
                else:
                    idx_by_key[key] = len(survivors)
                    survivors.append(e)
                last_by_key[key] = e

            # 诊断噪声只留最近 30 条
            silent_idx = [i for i, e in enumerate(survivors) if e.kind == "mind.silent"]
            if len(silent_idx) > 30:
                drop = set(silent_idx[:-30])
                dropped_old += len(drop)
                survivors = [e for i, e in enumerate(survivors) if i not in drop]

            # 总量封顶：按重要度升序删（噪声最先，保护对象最后）
            if len(survivors) > cap:
                over = len(survivors) - cap
                protected_ids = {
                    id(e) for e in survivors
                    if e.meta.get("pin") or e.meta.get("due_at") or e.kind in MENTAL_KINDS
                }
                order = sorted(
                    [e for e in survivors if id(e) not in protected_ids],
                    key=lambda e: (0 if e.kind in NOISE_KINDS else 1,
                                   self.importance(e, now_ts), e.ts),
                )
                drop_ids = {id(e) for e in order[:over]}
                # 机械事件不够删时才动推断/保护对象（挑最不重要的）
                if len(drop_ids) < over:
                    rest = sorted(
                        [e for e in survivors if id(e) not in drop_ids
                         and id(e) in protected_ids
                         and not e.meta.get("due_at")],
                        key=lambda e: self.importance(e, now_ts),
                    )
                    for e in rest[: over - len(drop_ids)]:
                        drop_ids.add(id(e))
                survivors = [e for e in survivors if id(e) not in drop_ids]
                dropped_cap = len(drop_ids)

            survivors.sort(key=lambda x: x.ts)
            changed = (dropped_deleted + dropped_dup + dropped_old + dropped_cap) > 0
            alive_ids = {e.id for e in survivors}
            self._events = deque(survivors, maxlen=MAX_EVENTS)
            if changed:
                self._edges = deque(
                    [ed for ed in self._edges
                     if ed.get("src") in alive_ids and ed.get("dst") in alive_ids],
                    maxlen=MAX_EVENTS * 2,
                )
                self._persist()
        return {
            "deleted": dropped_deleted, "dup": dropped_dup,
            "expired": dropped_old, "capped": dropped_cap,
            "left": len(survivors),
        }

    def link(self, src_id: str, dst_id: str, relation: str = "before") -> None:
        with self._lock:
            self._edges.append({"src": src_id, "dst": dst_id, "rel": relation, "ts": time.time()})

    def _auto_link(self, ev: TFEvent) -> None:
        """同实体且 30 秒内的事件自动加一条 same_topic 边（方向按时间正序）。"""
        recent = list(self._events)[:-1]
        for prev in reversed(recent[-8:]):
            gap = ev.ts - prev.ts
            if gap < 0:
                continue
            if gap > 30:
                continue
            if set(ev.entities) & set(prev.entities):
                self._edges.append({"src": prev.id, "dst": ev.id, "rel": "same_topic", "ts": ev.ts})

    def _append(self, ev: TFEvent) -> None:
        # 同 rag：过滤驱动 / 多实例偶发锁文件，重试几次再认输
        for attempt in range(3):
            try:
                if self._path.exists() and self._path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                    self._rotate()
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(ev.to_json() + "\n")
                return
            except Exception:
                if attempt == 2:
                    logger.warning("[TFGR] 落盘失败（已重试 3 次）", exc_info=True)
                else:
                    time.sleep(0.05 * (attempt + 1))

    def _rotate(self) -> None:
        try:
            old = self._path.with_suffix(".1.jsonl")
            if old.exists():
                old.unlink()
            self._path.rename(old)
        except Exception:
            pass

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()[-MAX_EVENTS:]
            for line in lines:
                if line.strip():
                    self._events.append(TFEvent.from_dict(json.loads(line)))
            logger.info(f"[TFGR] 恢复 {len(self._events)} 条事件")
        except Exception:
            logger.warning("[TFGR] 读取失败", exc_info=True)

    # ---------- 读 ----------

    def recent(self, n: int = 20, window_s: Optional[float] = None) -> List[TFEvent]:
        now = time.time()
        with self._lock:
            items = [e for e in self._events if not e.meta.get("deleted")]
        if window_s is not None:
            items = [e for e in items if now - e.ts <= window_s]
        return items[-n:]

    def around(self, ts: float, window_s: float = 60) -> List[TFEvent]:
        with self._lock:
            return [e for e in self._events if abs(e.ts - ts) <= window_s]

    def interval(self, t0: float, t1: float) -> List[TFEvent]:
        with self._lock:
            return [
                e for e in self._events
                if (e.end_ts and e.end_ts >= t0 and e.ts <= t1) or (t0 <= e.ts <= t1)
            ]

    def search(self, query: str, limit: int = 10) -> List[TFEvent]:
        q = (query or "").strip().lower()
        if not q:
            return []
        words = [w for w in q.split() if w]
        with self._lock:
            hits = []
            for e in reversed(self._events):
                if e.meta.get("deleted"):
                    continue
                hay = (e.text + " " + " ".join(e.entities) + " " + e.kind).lower()
                if all(w in hay for w in words):
                    hits.append(e)
                if len(hits) >= limit:
                    break
            return list(reversed(hits))

    def by_entity(self, name: str, limit: int = 20,
                  since: Optional[float] = None) -> List[TFEvent]:
        """按实体取事件（人物层用：这个人在哪些集里干过什么）。"""
        n = (name or "").strip().lower()
        if not n:
            return []
        with self._lock:
            hits = [e for e in self._events
                    if not e.meta.get("deleted")
                    and n in (x.lower() for x in e.entities)
                    and (since is None or e.ts >= since)]
        return hits[-limit:]

    def rename_entity(self, old: str, new: str) -> int:
        """把旧实体名（比如跟踪器给的 p1）全部改写成真名。

        这是「回溯改名」：不是只影响以后，历史事件和落盘文件一起重写。
        """
        o, n = (old or "").strip(), (new or "").strip()
        if not o or not n or o.lower() == n.lower():
            return 0
        changed = 0
        with self._lock:
            for e in self._events:
                if o in e.entities:
                    e.entities = [n if x == o else x for x in e.entities]
                    changed += 1
            if changed:
                self._rewrite(o, n)
                self._rewrite(o, n, self._path.with_suffix(".1.jsonl"))
        if changed:
            logger.info(f"[TFGR] 实体改名 {o} → {n}，改写 {changed} 条事件")
        return changed

    def _rewrite(self, old: str, new: str, path: Optional[Path] = None) -> None:
        """把事件整份重写到文件（改名后必须做，否则重启就回到旧名）。"""
        p = Path(path) if path else self._path
        if not p.exists():
            return
        try:
            tmp = p.with_suffix(".tmp")
            if p == self._path:
                tmp.write_text("".join(e.to_json() + "\n" for e in self._events),
                               encoding="utf-8")
            else:
                # 归档文件不在内存里，读出来逐行替换（只换 entities 字段）
                lines = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    d["entities"] = [new if x == old else x for x in d.get("entities", [])]
                    lines.append(json.dumps(d, ensure_ascii=False))
                tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.replace(p)
        except Exception:
            logger.warning("[TFGR] 重写失败", exc_info=True)

    def summarize(self, window_s: float = 1800, max_items: int = 12) -> str:
        items = self.recent(max_items, window_s=window_s)
        if not items:
            return "（没有事件记录）"
        now = time.time()
        lines = []
        for e in items:
            dt = now - e.ts
            stamp = f"-{int(dt)}s" if dt < 120 else f"-{int(dt / 60)}m"
            span = ""
            if e.end_ts:
                span = f" (~{(e.end_ts - e.ts) / 60:.0f}m)"
            ent = f" [{','.join(e.entities[:3])}]" if e.entities else ""
            lines.append(f"{stamp} {e.kind}: {e.text[:70]}{span}{ent}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            alive = [e for e in self._events if not e.meta.get("deleted")]
            revised = sum(1 for e in alive if e.meta.get("rev"))
            return {"events": len(alive), "edges": len(self._edges),
                    "revised": revised, "deleted": len(self._events) - len(alive),
                    "path": str(self._path)}


_tfgr: Optional[TimeFlowGraph] = None
_lock = threading.Lock()


def get_tfgr() -> TimeFlowGraph:
    global _tfgr
    with _lock:
        if _tfgr is None:
            _tfgr = TimeFlowGraph()
        return _tfgr
