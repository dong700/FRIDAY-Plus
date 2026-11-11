"""闭环 05：心智循环（事件 → 触发 → 五道闸 → 说）

跑：python 05_mind.py
对应源码：brain/mind.py::_should_speak / _too_similar / _select_trigger / _next_timeout
"""
import difflib
import re
import threading
import time


def norm(t: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffa-z0-9]+", "", (t or "").lower())


# 修复点：主动消息共用「提醒你一下：」这类模板时，只比模板会互相误杀。
TEMPLATE_PREFIX_RE = re.compile(
    r"^(?:提醒你一下|提醒一下你|提醒一下|顺便说一句|顺便说一下|对了|那个)[：:，,\s]*")


def dedup_key(text: str) -> str:
    t = (text or "").strip()
    stripped = TEMPLATE_PREFIX_RE.sub("", t).strip()
    return stripped or t


def too_similar(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.62:
        return True
    ga = {na[i:i + 2] for i in range(len(na) - 1)}
    gb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    return bool(ga and gb and len(ga & gb) / min(len(ga), len(gb)) >= 0.55)


def is_meta_silence(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.fullmatch(r"[（(][^）)]*[）)]", t):
        return True
    return bool(len(t) <= 90 and re.search(r"(不打扰|继续安静|保持沉默|不说话)", t)
                and "？" not in t and "?" not in t)


class Mind:
    def __init__(self, say):
        self.say = say
        self.hooks = []
        self.pending = []
        self.last_says = []
        self.ignored = 0
        self.in_work = False
        self.last_activity = time.time()
        self.last_was_question = False
        self._stop = False

    def add_hook(self, seconds, note):
        self.hooks.append((time.time() + seconds, note))

    def publish(self, kind, text):
        self.pending.append((kind, text))

    def user_said(self):
        self.ignored = 0
        self.last_activity = time.time()

    def on_ignored(self):
        self.ignored += 1

    def should_speak(self, text, user_initiated=False):
        if not text:
            return False, "空"
        if is_meta_silence(text):
            return False, "旁白式沉默"
        for prev in self.last_says[-5:]:
            if too_similar(dedup_key(text), dedup_key(prev)):
                return False, "和上一句太像"
        if self.ignored >= 2 and not user_initiated:
            return False, "连续两次主动说话没被回应"
        is_q = ("？" in text) or ("?" in text)
        if is_q and self.last_was_question and not user_initiated:
            return False, "上一句已经在问了"
        if self.in_work and not user_initiated:
            return False, "他在工作软件里"
        return True, "ok"

    def next_timeout(self):
        now = time.time()
        ds = [h[0] - now for h in self.hooks]
        return max(0.5, min(ds)) if ds else 5.0

    def loop(self):
        while not self._stop:
            now = time.time()
            fires = [h for h in self.hooks if h[0] <= now]
            if fires:
                self.hooks = [h for h in self.hooks if h[0] > now]
                text = fires[0][1]
                ok, why = self.should_speak(text)
                print(f"[mind] 想说（{why}）: {text}")
                if ok:
                    self.say(text)
                    self.last_says.append(text)
                    self.last_was_question = "？" in text
            elif self.pending:
                kind, text = self.pending.pop(0)
                msg = f"你刚才在{text}，还好吗"
                ok, why = self.should_speak(msg)
                print(f"[mind] 事件 {kind} → {why}")
                if ok:
                    self.say(msg)
                    self.last_says.append(msg)
            time.sleep(0.3)


if __name__ == "__main__":
    sent = []
    m = Mind(say=lambda t: (sent.append(t), print(f"[气泡] {t}")))
    threading.Thread(target=m.loop, daemon=True).start()

    print("\n-- 正常：文案各不相同 --")
    m.add_hook(0.6, "该喝水了，你已经两小时没起身")
    m.add_hook(0.9, "三点要开会，记得带电脑")
    time.sleep(1.6)

    print("\n-- 闸 5：工作软件里不打断 --")
    m.in_work = True
    m.add_hook(0.2, "晚饭想吃点什么")
    time.sleep(0.9)

    print("\n-- 闸 4：连续两次没被回应 --")
    m.in_work = False
    m.ignored = 2
    m.add_hook(0.2, "要不要一起看个视频")
    time.sleep(0.9)

    print("\n-- 用户开口后闸门恢复 --")
    m.user_said()
    m.add_hook(0.2, "要不要一起看个视频")
    time.sleep(0.9)

    print("\n-- 回归验证：同模板提醒不再被误杀，完全重复仍被拦 --")
    m.last_says.append("提醒你一下：喝水")
    print("  「提醒你一下：吃药」 ->", m.should_speak("提醒你一下：吃药")[0], "（应 True）")
    print("  「提醒你一下：喝水」 ->", m.should_speak("提醒你一下：喝水")[0], "（应 False）")

    m._stop = True
    print("\n实际说出去:", sent)
