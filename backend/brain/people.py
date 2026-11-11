"""人物层（2026-11-04）——把「框」变成「人」，让记忆跨集延续。

分工：
    打框/跟踪（本地小模型，别人做）：每帧给出 [p1, p2, p3] + 人脸 embedding。
    这里（agent 层）：把 p1 这种"这一集里的临时编号"钉到一个稳定的人身上，
                     再让 TFGR / ATG 用那个人的真名说话。

三层身份：
    label     跟踪器当次给的临时编号（p1 / p2），跨集会重来
    pid       稳定的人（person_0001），从 embedding 匹配出来，只有一份
    name      用户起的名字（消炎），没有名字时对外就叫 pid

为什么需要 pid：
    剧里同一张脸在第 1 集是 p1、第 2 集可能是 p3（甚至同一集前后也会换号）。
    光靠 label 记不住"上次他干了什么"；光靠名字又记不住"这一集谁是谁"。
    pid 把两者接上：label → pid（本次），pid → name（长期）。

改名会回溯：
    用户说"这是男主，叫消炎" → 这个 pid 的所有历史事件（entities 里的 p1）
    立刻改写成"消炎"，文件也重写。不是只改未来，过去也改。

对外契约（给打框那边用）：
    people.observe("p1", embedding, show="某剧@第3集", ts=..., box=[x,y,w,h])
        返回 {"pid","name","known","new_episode"}
    embedding 传 None 也行（只按 label 记账，不认人）。
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.logger import get_logger

logger = get_logger("brain.people")

MATCH_TH = float(os.getenv("FRIDAY_PERSON_MATCH", "0.42"))   # 余弦相似度，SFace 同人一般 >0.5
ANNOUNCE_GAP_S = float(os.getenv("FRIDAY_PERSON_GAP", "1800"))  # 同一人 30 分钟内不重复播报
MAX_SEEN = 40


def _data_dir() -> Path:
    try:
        from core.config import get_data_dir
        return Path(get_data_dir())
    except Exception:
        return Path("data")


@dataclass
class Person:
    pid: str
    name: str = ""
    aliases: List[str] = field(default_factory=list)     # ["p1@某剧@第3集", "p1"]
    shows: List[str] = field(default_factory=list)       # 出现过的集
    notes: List[str] = field(default_factory=list)       # 用户补的设定："男主""欠债"
    emb: List[float] = field(default_factory=list)       # 质心
    n_emb: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    appearances: int = 0

    @property
    def label(self) -> str:
        return self.name or self.pid


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class PeopleBook:
    """人物总账。单例；落盘 data/people.json。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path else _data_dir() / "people.json"
        self._people: Dict[str, Person] = {}
        self._by_alias: Dict[str, str] = {}       # label / 旧名 → pid
        self._seen: List[Dict[str, Any]] = []     # 最近观察到的人（屏幕/视频）
        self._load()

    # ---------------- 持久化 ----------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for d in raw.get("people", []):
                p = Person(**{k: v for k, v in d.items() if k in Person.__dataclass_fields__})
                self._people[p.pid] = p
                for a in p.aliases:
                    self._by_alias[a.lower()] = p.pid
                if p.name:
                    self._by_alias[p.name.lower()] = p.pid
            logger.info(f"[People] 载入 {len(self._people)} 个人物")
        except Exception:
            logger.warning("[People] 读取失败", exc_info=True)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"people": [asdict(p) for p in self._people.values()]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.warning("[People] 落盘失败", exc_info=True)

    # ---------------- 认人 ----------------

    def _match(self, emb: Sequence[float], th: float = MATCH_TH) -> Optional[Person]:
        best, best_s = None, -1.0
        for p in self._people.values():
            s = _cos(emb, p.emb)
            if s > best_s:
                best, best_s = p, s
        return best if best is not None and best_s >= th else None

    def _new_pid(self) -> str:
        n = 1
        while f"person_{n:04d}" in self._people:
            n += 1
        return f"person_{n:04d}"

    def _touch_emb(self, p: Person, emb: Optional[Sequence[float]]) -> None:
        if not emb:
            return
        if not p.emb or len(p.emb) != len(emb):
            p.emb = [float(x) for x in emb]
            p.n_emb = 1
            return
        # 在线质心：越看越准，但旧样本仍有话语权（不会被一次侧脸带跑）
        p.n_emb += 1
        w = 1.0 / p.n_emb
        p.emb = [(1 - w) * a + w * float(b) for a, b in zip(p.emb, emb)]

    def observe(self, label: str, emb: Optional[Sequence[float]] = None,
                show: str = "", ts: Optional[float] = None, box: Optional[Sequence[float]] = None,
                announce: bool = True) -> Dict[str, Any]:
        """跟踪器报「这一帧里有 p1」，这里回答「他是谁、是不是新的一集」。

        label  这一集的临时号（p1）
        emb    人脸特征（给了才做跨集/跨号识别）
        show   集标识，例如 "某剧@第3集"（只做记录和"新集"判断）
        """
        ts = ts or time.time()
        key = f"{label}@{show}" if show else label
        with self._lock:
            pid = self._by_alias.get(key.lower()) or self._by_alias.get(label.lower())
            p = self._people.get(pid) if pid else None
            matched_by = "label" if p else ""
            if p is None and emb:
                p = self._match(emb)
                matched_by = "face" if p else ""
            if p is None:
                p = Person(pid=self._new_pid(), first_seen=ts)
                self._people[p.pid] = p
                matched_by = "new"
            # 记住这个 label 属于谁（同一集里后续不必再比对）
            for a in {key.lower(), label.lower()}:
                self._by_alias[a] = p.pid
            if key not in p.aliases:
                p.aliases.append(key)
            new_episode = bool(show) and show not in p.shows
            if new_episode:
                p.shows.append(show)
            self._touch_emb(p, emb)
            first_today = (ts - p.last_seen >= ANNOUNCE_GAP_S) if p.last_seen else True
            p.last_seen = ts
            p.appearances += 1
            self._seen.append({"label": key, "pid": p.pid, "name": p.name, "ts": ts,
                               "box": list(box) if box else None, "show": show})
            self._seen = self._seen[-MAX_SEEN:]
            self._save()

        known = bool(p.name)
        if announce and p.name and (first_today or new_episode):
            self._announce(p, show, new_episode)
        return {"pid": p.pid, "name": p.name, "label": key, "known": known,
                "matched_by": matched_by, "new_episode": new_episode,
                "appearances": p.appearances}

    def _announce(self, p: Person, show: str, new_episode: bool) -> None:
        """认出来了就记一笔，让心智有机会主动开口。"""
        why = f"（新的一集 {show}）" if new_episode else ""
        line = f"「{p.name}」又出现了{why}"
        try:
            from brain import atg
            atg.observe(line, source="people")
        except Exception:
            pass
        try:
            from brain.mind import get_mind
            get_mind().publish("person.appear", line, source="people",
                               wake=bool(new_episode), salience=60 if new_episode else 45,
                               meta={"pid": p.pid, "name": p.name, "show": show})
        except Exception:
            pass

    # ---------------- 命名 / 改名 ----------------

    def find(self, who: str) -> Optional[Person]:
        """按 pid / 名字 / label 找一个人。"""
        if not who:
            return None
        w = who.strip().lower()
        with self._lock:
            if w in self._people:
                return self._people[w]
            for p in self._people.values():
                if p.name and p.name.lower() == w:
                    return p
                if w in (a.lower() for a in p.aliases):
                    return p
            for p in self._people.values():        # 模糊：包含
                if (p.name and w in p.name.lower()) or any(w in a.lower() for a in p.aliases):
                    return p
        return None

    def bind(self, who: str, name: str, note: str = "") -> Dict[str, Any]:
        """用户说「这是男主，叫消炎」→ 钉住身份 + 回溯改历史。"""
        p = self.find(who)
        if p is None:
            return {"ok": False, "error": f"没找到 {who}，是不是还没在画面上出现过？"}
        old_labels = list(p.aliases)
        with self._lock:
            p.name = name.strip()
            self._by_alias[p.name.lower()] = p.pid
            if note:
                p.notes.append(note)
            self._save()

        renamed = 0
        try:
            from brain.tfgr import get_tfgr
            tf = get_tfgr()
            for lbl in old_labels:
                for token in {lbl, lbl.split("@")[0], p.pid}:
                    if token and token.lower() != p.name.lower():
                        renamed += tf.rename_entity(token, p.name)
            tf.record("person.name", f"用户告诉我 {who} 是「{p.name}」"
                      + (f"（{note}）" if note else ""),
                      source="people", entities=[p.name],
                      meta={"pid": p.pid, "aliases": old_labels})
        except Exception:
            logger.warning("[People] 回溯改名失败", exc_info=True)

        try:
            from brain import atg
            atg.observe(f"记住：{who} = {p.name}" + (f"，{note}" if note else ""), source="people")
        except Exception:
            pass
        return {"ok": True, "pid": p.pid, "name": p.name, "aliases": old_labels,
                "renamed_events": renamed}

    def canonical(self, label: str) -> str:
        """label → 对外叫法（有名字给名字，没有给 pid）。给写 TFGR 的人用。"""
        p = self.find(label)
        return p.label if p else label

    def canonicalize(self, labels: Sequence[str]) -> List[str]:
        return [self.canonical(x) for x in (labels or [])]

    def note(self, who: str, text: str) -> Dict[str, Any]:
        p = self.find(who)
        if not p:
            return {"ok": False, "error": f"没找到 {who}"}
        with self._lock:
            p.notes.append(text)
            self._save()
        try:
            from brain.tfgr import get_tfgr
            get_tfgr().record("person.note", f"{p.label}：{text}", source="people",
                              entities=[p.label], meta={"pid": p.pid})
        except Exception:
            pass
        return {"ok": True, "pid": p.pid, "name": p.label, "notes": p.notes}

    # ---------------- 读 ----------------

    def seen_now(self, window_s: float = 300) -> List[Dict[str, Any]]:
        """最近 window_s 秒里在画面上出现过的人（按最近一次去重）。"""
        now = time.time()
        out: Dict[str, Dict[str, Any]] = {}
        for s in reversed(self._seen):
            if now - s["ts"] > window_s:
                break
            if s["pid"] in out:
                continue
            p = self._people.get(s["pid"])
            out[s["pid"]] = {"pid": s["pid"], "name": p.name if p else "", "label": s["label"],
                             "last_seen": s["ts"], "box": s["box"], "show": s["show"]}
        return list(out.values())

    def profile(self, who: str, limit: int = 12) -> Dict[str, Any]:
        """这人以前干过什么：跨集拉 TFGR 里挂在他名下的所有事件。"""
        p = self.find(who)
        if not p:
            return {"ok": False, "error": f"没找到 {who}"}
        events: List[Dict[str, Any]] = []
        try:
            from brain.tfgr import get_tfgr
            for e in get_tfgr().by_entity(p.label, limit=max(limit, 20)):
                events.append({"ts": e.ts, "kind": e.kind, "text": e.text,
                               "show": (e.meta or {}).get("show", "")})
        except Exception:
            logger.warning("[People] 取事件失败", exc_info=True)
        events = events[-limit:]
        return {"ok": True, "pid": p.pid, "name": p.label,
                "shows": p.shows, "notes": p.notes, "appearances": p.appearances,
                "events": events, "summary": self._narrate(p, events)}

    @staticmethod
    def _narrate(p: Person, events: List[Dict[str, Any]]) -> str:
        if not events:
            return f"{p.label}：还没有记下他做过什么。"
        lines = [f"{p.label}：" + ("，".join(p.notes) + "。" if p.notes else "")]
        for e in events[-6:]:
            when = time.strftime("%m-%d %H:%M", time.localtime(e["ts"]))
            ep = f"（{e['show']}）" if e.get("show") else ""
            lines.append(f"· {when}{ep} {e['text'][:70]}")
        return "\n".join(x for x in lines if x)

    def stats(self) -> Dict[str, Any]:
        named = [p for p in self._people.values() if p.name]
        return {"people": len(self._people), "named": len(named),
                "names": [p.name for p in named][:20]}


_book: Optional[PeopleBook] = None
_lock = threading.Lock()


def get_people() -> PeopleBook:
    global _book
    with _lock:
        if _book is None:
            _book = PeopleBook()
        return _book
