"""闭环 03：长期记忆的「挑」（importance + digest）

跑：python 03_digest.py
对应源码：brain/tfgr.py::importance / digest / _query_overlap / _near_dup
"""
import difflib
import time

KIND_W = {"memory": 0.9, "intent": 0.7, "belief": 0.6, "pattern": 0.55, "story": 0.5}
PIN, DUE, HALF, NEG = 1.0, 1.5, 14.0, 0.55
MENTAL = ("belief", "story", "intent", "pattern")
USER = ("memory",)


class E:
    def __init__(self, id, kind, text, ts=None, conf=1.0, pin=False, due=None):
        self.id, self.kind, self.text = id, kind, text
        self.ts = ts or time.time()
        self.conf = conf
        self.pin, self.due = pin, due
        self.last_hit = self.ts
        self.last_inject = 0.0
        self.negated = False
        self.done = False


def importance(e: E, now=None) -> float:
    now = now or time.time()
    s = KIND_W.get(e.kind, 0.35)
    s += 0.25 * min(1.0, max(0.0, e.conf))
    if e.pin:
        s += PIN
    if e.due:
        s += DUE
    age = max(0.0, (now - e.last_hit) / 86400)
    s *= 0.5 ** (age / HALF)
    if e.negated:
        s *= NEG
    return round(s, 3)


def overlap(q: str, t: str) -> float:
    qs = {c for c in q if "\u4e00" <= c <= "\u9fff"}
    if not qs:
        return 0.0
    t = t.lower()
    return sum(1 for c in qs if c in t) / len(qs)


def near_dup(body: str, others, ratio=0.75) -> bool:
    for o in others:
        if not o:
            continue
        if body in o or o in body:
            return True
        if difflib.SequenceMatcher(None, body, o).ratio() >= ratio:
            return True
    return False


def norm(t: str) -> str:
    return "".join(c for c in (t or "").lower() if c.isalnum())[:120]


def digest(items, query="", max_items=10, max_chars=800, now=None):
    now = now or time.time()
    shown, picked = [], []

    def take(e, dedup=False):
        if e in picked or len(picked) >= max_items:
            return
        if dedup and near_dup(norm(e.text), [norm(s) for s in shown]):
            return
        picked.append(e)
        shown.append(e.text)

    def score(e):
        s = importance(e, now) + 0.6 * overlap(query, e.text)
        if now - (e.last_inject or 0) <= 600:
            s += 0.08
        return s

    def rank(e):
        return (-score(e), -e.ts, e.id)

    for e in sorted([x for x in items if x.due and not x.done], key=lambda x: x.due):
        take(e)
    for e in sorted([x for x in items if x.kind in USER], key=rank):
        take(e)
    for e in sorted([x for x in items if x.pin and x.kind not in USER], key=rank):
        take(e, dedup=True)
    for e in sorted([x for x in items if x.kind in MENTAL], key=rank):
        take(e, dedup=True)

    out, used, injected = [], 0, []
    for e in picked:
        e.last_hit = now
        if e.kind in MENTAL:
            e.last_inject = now
        age = now - e.ts
        when = f"{int(age/86400)}天前" if age >= 86400 else f"{max(1, int(age/60))}分钟前"
        due_txt = ""
        if e.due and not e.done:
            left = e.due - now
            due_txt = "【已到点】" if left <= 0 else f"【还有{int(left/3600)}小时】"
        src = "他让记" if e.kind in USER else "我推断"
        line = f"{due_txt}{src}·{when}: {e.text[:90]}"
        if out and used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line)
        injected.append(e.id)
    return "\n".join(out)


if __name__ == "__main__":
    now = time.time()
    items = [
        E("1", "memory", "我不吃辣", ts=now - 5 * 86400, pin=True),
        E("2", "belief", "他大概在赶项目，周末也没休", ts=now - 3 * 86400, conf=0.6),
        E("3", "belief", "他喜欢猫", ts=now - 40 * 86400, conf=0.8),
        E("4", "memory", "周五交房租", ts=now - 1 * 86400, pin=True, due=now + 2 * 86400),
        E("5", "pattern", "他每天 9 点开始写代码", ts=now - 10 * 86400, conf=0.7),
        E("6", "belief", "他大概在赶项目", ts=now - 3 * 86400, conf=0.55),  # 与 2 近重复
    ]
    print(digest(items, query="房租的事"))
    print("\n各条重要度:")
    for e in items:
        print(f"  {e.id} {e.kind:8s} {importance(e, now):.3f}  {e.text[:20]}")
