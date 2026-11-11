"""RAG — 事实骨架（2026-11-04）

三层记忆的分工，这里是第三层：
    ATG = 时间（模型写的分钟级活动叙事）
    TFGR = 思考（belief/story/intent/pattern，带 confidence/basis）
    RAG = 事实（用户原话、字幕/OCR、帧描述、镜头表），只保证「准」，按需召回

刻意不用向量库：
    - 事实量不大（一天几千条），倒排表足够
    - 要的是「原话可追溯」，不是语义相似；语义泛化反而会串味
    - 无依赖、可热重载、删文件就重置

存储：data/rag.jsonl，append-only；>8MB 轮转为 rag.1.jsonl
检索：关键词倒排 + BM25 排序 + 逆文档频率门槛 + 时间衰减 + 可选时间窗/kinds 过滤
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.logger import get_logger

logger = get_logger("brain.rag")

MAX_RECORDS = int(os.getenv("FRIDAY_RAG_MAX", "20000"))
MAX_FILE_MB = 8.0
# 文档频率上限：出现在超过这个比例记录里的词（「的」「在」「我」）不参与打分。
# 否则问一件库里没有的事（"今天天气怎么样"）也会因为「的」而被召回，永远说不了「不记得」。
MAX_DF = float(os.getenv("FRIDAY_RAG_MAX_DF", "0.30"))
# BM25 参数（只影响排序，不影响「是否返回」的阈值）
BM25_K1 = float(os.getenv("FRIDAY_RAG_BM25_K1", "1.2"))
BM25_B = float(os.getenv("FRIDAY_RAG_BM25_B", "0.75"))
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# 虚词/疑问词不携带检索信息：「今天天气怎么样」不该因为「的/是/什么」命中任何记录
_STOP = set("""
的 了 是 在 和 与 就 都 也 把 被 对 给 我 你 他 她 它 们 这 那 个 之 而 且 还 又
吗 呢 啊 呀 吧 着 过 从 到 向 于 为 有 没 不 会 要 能 可 请 帮 让
什么 多少 怎么 怎样 如何 哪 哪个 哪些 几点 为什么 是不是 一下 一些 一个 一次
那个 这个 那次 这次 那么 这么 讲 说
""".split())

try:                                   # 有 jieba 就分词，没有就退化成字级
    import jieba
    jieba.setLogLevel(20)
except Exception:                      # pragma: no cover
    jieba = None


def _data_dir() -> Path:
    try:
        from core.config import get_data_dir
        d = Path(get_data_dir())
    except Exception:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def tokenize(text: str) -> List[str]:
    """中英混合分词 → 去重保序。中文用 jieba，英文/数字用正则。"""
    low = (text or "").lower()
    out: List[str] = []
    if jieba is not None:
        for w in jieba.lcut(low):
            w = w.strip()
            if w:
                out.append(w)
    else:
        for ch in low:
            if "\u4e00" <= ch <= "\u9fff":
                out.append(ch)
    out.extend(_TOKEN_RE.findall(low))
    seen = set()
    keep: List[str] = []
    for w in out:
        if w in _STOP:
            continue
        if len(w) < 2 and w.isascii():
            continue
        if w in seen:
            continue
        seen.add(w)
        keep.append(w)
    return keep[:24]


@dataclass
class RagRecord:
    id: str
    ts: float
    kind: str
    text: str
    tags: List[str] = field(default_factory=list)
    ref: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class RagStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else (
            Path(os.getenv("FRIDAY_RAG_PATH")) if os.getenv("FRIDAY_RAG_PATH")
            else _data_dir() / "rag.jsonl"
        )
        self._records: Dict[str, RagRecord] = {}
        self._order: List[str] = []         # 写入顺序（用于截断与 recency）
        self._index: Dict[str, Dict[str, int]] = {}   # token -> {id: 词频}
        self._dl: Dict[str, int] = {}       # 文档长度（BM25 的长度归一化要用）
        self._lock = threading.RLock()
        self._load()

    # ---------- 写 ----------

    def add(
        self,
        kind: str,
        text: str,
        tags: Optional[Iterable[str]] = None,
        ref: str = "",
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> str:
        rec = RagRecord(
            id=uuid.uuid4().hex[:10],
            ts=ts or time.time(),
            kind=(kind or "fact")[:32],
            text=(text or "").strip()[:1200],
            tags=[str(t)[:32] for t in (tags or [])][:8],
            ref=(ref or "")[:200],
            meta=meta or {},
        )
        if not rec.text:
            return ""
        with self._lock:
            self._records[rec.id] = rec
            self._order.append(rec.id)
            self._index_record(rec)
            self._append(rec)
            self._trim()
        return rec.id

    def _index_record(self, rec: RagRecord) -> None:
        toks = tokenize(rec.text + " " + " ".join(rec.tags))
        self._dl[rec.id] = max(1, len(toks))
        for t in toks:
            posting = self._index.setdefault(t, {})
            posting[rec.id] = posting.get(rec.id, 0) + 1

    def _append(self, rec: RagRecord) -> None:
        # D 盘的过滤驱动 / 别的实例同时写，偶尔会短暂锁住文件；
        # 单次失败就把这条记录丢了太亏，这里小块重试。
        for attempt in range(3):
            try:
                if self._path.exists() and self._path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                    old = self._path.with_suffix(".1.jsonl")
                    if old.exists():
                        old.unlink()
                    self._path.rename(old)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(rec.to_json() + "\n")
                return
            except Exception:
                if attempt == 2:
                    logger.warning("[RAG] 落盘失败（已重试 3 次）", exc_info=True)
                else:
                    time.sleep(0.05 * (attempt + 1))

    def _trim(self) -> None:
        """超过上限时丢最旧的，并同步清理倒排。"""
        while len(self._order) > MAX_RECORDS:
            old_id = self._order.pop(0)
            rec = self._records.pop(old_id, None)
            if rec is None:
                continue
            self._dl.pop(old_id, None)
            for t in tokenize(rec.text + " " + " ".join(rec.tags)):
                posting = self._index.get(t)
                if posting:
                    posting.pop(old_id, None)
                    if not posting:
                        self._index.pop(t, None)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines()[-MAX_RECORDS:]:
                if not line.strip():
                    continue
                d = json.loads(line)
                rec = RagRecord(**{k: d[k] for k in RagRecord.__dataclass_fields__ if k in d})
                if rec.id in self._records:
                    continue
                self._records[rec.id] = rec
                self._order.append(rec.id)
                self._index_record(rec)
            logger.info(f"[RAG] 恢复 {len(self._records)} 条事实")
        except Exception:
            logger.warning("[RAG] 读取失败", exc_info=True)

    # ---------- 读 ----------

    def recall(
        self,
        query: str,
        k: int = 5,
        window_s: Optional[float] = None,
        kinds: Optional[Iterable[str]] = None,
        min_score: float = 0.15,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """关键词召回。不返回任何「推断」，只返回原始事实与其 ref。

        排序用 BM25（长度归一化 + 词频饱和），是否返回仍由原来的 idf 门槛决定：
        门槛是校准过的（决定「敢不敢说不知道」），换排序算法不该动它。
        """
        toks = tokenize(query)
        if not toks:
            return []
        now_ts = now or time.time()
        allow_kinds = set(kinds) if kinds else None
        total = max(1, len(self._records))
        avgdl = max(1.0, sum(self._dl.values()) / max(1, len(self._dl)))
        gate: Dict[str, float] = {}          # 门槛分（沿用原 idf 公式）
        rank: Dict[str, float] = {}          # 排序分（BM25）
        with self._lock:
            for t in toks:
                posting = self._index.get(t)
                if not posting:
                    continue
                df = len(posting)
                if df > MAX_DF * total:              # 太常见，不具区分度
                    continue
                idf_old = 1.0 / (1.0 + df / total * 10.0)
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                for rid, cnt in posting.items():
                    gate[rid] = gate.get(rid, 0.0) + idf_old * min(3, cnt)
                    dl = self._dl.get(rid, 1)
                    norm = BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl)
                    rank[rid] = rank.get(rid, 0.0) + idf * cnt * (BM25_K1 + 1) / (cnt + norm)
            results: List[Dict[str, Any]] = []
            for rid, g in gate.items():
                rec = self._records.get(rid)
                if rec is None:
                    continue
                if allow_kinds and rec.kind not in allow_kinds:
                    continue
                if window_s is not None and now_ts - rec.ts > window_s:
                    continue
                age_h = max(0.0, now_ts - rec.ts) / 3600.0
                decay = 1.0 / (1.0 + age_h / 24.0)          # 时间衰减：一天折半
                if g * decay < min_score:
                    continue
                results.append({
                    "id": rec.id, "ts": rec.ts, "kind": rec.kind,
                    "text": rec.text[:300], "ref": rec.ref,
                    "score": round(rank.get(rid, 0.0) * decay, 3),
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    def interval(self, t0: float, t1: float) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"id": r.id, "ts": r.ts, "kind": r.kind, "text": r.text[:300], "ref": r.ref}
                for r in self._records.values() if t0 <= r.ts <= t1
            ]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"records": len(self._records), "terms": len(self._index),
                    "path": str(self._path)}


_rag: Optional[RagStore] = None
_lock = threading.Lock()


def get_rag() -> RagStore:
    global _rag
    with _lock:
        if _rag is None:
            _rag = RagStore()
        return _rag


def recall(query: str, k: int = 5, **kw) -> List[Dict[str, Any]]:
    return get_rag().recall(query, k=k, **kw)
