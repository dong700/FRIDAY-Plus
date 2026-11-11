"""闭环 04：负反馈（用户否定 → 最近注入的推断降权）

跑：python 04_negate.py
对应源码：brain/tfgr.py::penalize_recent
"""
import re
import time

NEG_RE = re.compile(
    r"你记错|记错了|我没说过|没说过|不是这样|别乱说|胡说|搞错|不对吧|没有这回事")


def overlap(q: str, t: str) -> float:
    qs = {c for c in q if "\u4e00" <= c <= "\u9fff"}
    return sum(1 for c in qs if c in t) / len(qs) if qs else 0.0


class E:
    def __init__(self, id, text, conf=0.7, inject_ago=0.0):
        self.id, self.text, self.conf = id, text, conf
        self.last_inject = time.time() - inject_ago
        self.negated = False

    def __repr__(self):
        return f"{self.id}({self.conf}, neg={bool(self.negated)})"


def penalize(items, user_text, window=1800, factor=0.6, max_items=3):
    if not NEG_RE.search(user_text):
        return []
    now = time.time()
    cands = [e for e in items if now - e.last_inject <= window]
    if not cands:
        return []
    cands.sort(key=lambda e: -e.last_inject)
    hit = [e for e in cands if overlap(user_text, e.text) >= 0.3]
    explicit = bool(NEG_RE.search(user_text))
    target = hit[:max_items] if hit else (cands[:2] if explicit else [])
    for e in target:
        e.conf = max(0.1, round(e.conf * factor, 2))
        e.negated = now
    return [e.id for e in target]


if __name__ == "__main__":
    items = [
        E("1", "他周末在赶项目", 0.7, inject_ago=120),
        E("2", "他喜欢猫", 0.8, inject_ago=60),
        E("3", "他在准备答辩", 0.6, inject_ago=300),
    ]
    print("否定前:", items)
    print("精准否定 ->", penalize(items, "你记错了，我周末没赶项目"))
    print("否定后:", items)

    items2 = [E("4", "他喜欢狗", 0.7, 30), E("5", "他在学 rust", 0.7, 50)]
    print("\n泛否定（有指向但无重合）->", penalize(items2, "你记错了"))
    print("结果:", items2)

    items3 = [E("6", "他喜欢狗", 0.7, 30)]
    print("\n纯泛否定「不对」不该乱杀 ->", penalize(items3, "不对"))
    print("结果:", items3, "（空列表 = 没动）")
