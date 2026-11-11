"""闭环 06：ATG 活动合并（同一件事别切碎）

跑：python 06_atg.py
对应源码：brain/activity_timeline.py::ActivityTimeline.observe_activity
"""
import difflib
import time


class Act:
    _seq = 0

    def __init__(self, desc, task="", t=None):
        Act._seq += 1
        self.id = f"A{Act._seq:04d}"
        self.desc, self.task = desc, task
        self.t0 = t or time.time()
        self.t1 = None
        self.paused = False
        self.interrupt_by = None
        self.resume_of = None

    def __repr__(self):
        return (f"{self.id}({self.desc}{'/' + self.task if self.task else ''}, "
                f"paused={self.paused}, resume_of={self.resume_of})")


class ATG:
    def __init__(self, merge=0.6, gap=30.0, timeout=600.0):
        self.acts = []
        self.merge, self.gap, self.timeout = merge, gap, timeout

    def current(self):
        return next((a for a in reversed(self.acts) if a.t1 is None and not a.paused), None)

    def observe(self, desc, task="", now=None):
        now = now or time.time()
        cur = self.current()
        if cur:
            ratio = difflib.SequenceMatcher(None, cur.desc, desc).ratio()
            if ratio >= self.merge:
                if task and not cur.task:
                    cur.task = task
                return cur.id, "merge"
            if now - cur.t0 < self.gap:
                return cur.id, "overlap"
            cur.paused = True
            cur.t1 = now
        for a in reversed(self.acts):
            if a.paused and difflib.SequenceMatcher(None, a.desc, desc).ratio() >= self.merge:
                a.paused = False
                a.t1 = None
                a.resume_of = a.id
                return a.id, "resume"
        a = Act(desc, task, now)
        self.acts.append(a)
        return a.id, "new"

    def snapshot(self, now=None):
        now = now or time.time()
        lines = []
        for a in self.acts[-8:]:
            end = a.t1 or now
            lines.append(f"{time.strftime('%H:%M', time.localtime(a.t0))}-"
                         f"{'进行中' if a.t1 is None and not a.paused else time.strftime('%H:%M', time.localtime(end))}"
                         f" {a.desc}" + (f"（{a.task}）" if a.task else "")
                         + ("（被打断）" if a.paused else ""))
        return "\n".join(lines)


if __name__ == "__main__":
    g = ATG(gap=30.0)
    t = time.time()
    print(g.observe("在写代码", "实现登录", t))        # new
    print(g.observe("在用电脑", "", t + 10))          # 10s 内描述不同 -> overlap
    print(g.observe("在写代码", "", t + 40))          # 描述相同 -> merge
    print(g.observe("在吃饭", "", t + 80))            # 距 A0001 起点 80s -> 打断
    print(g.observe("在写代码", "", t + 200))         # 命中 paused -> resume
    print("\n节点:", g.acts)
    print("\n快照:\n" + g.snapshot(t + 210))
