"""闭环 02：提醒（解析 → 落库 → 到点 → 关闭）

跑：python 02_reminder.py
对应源码：api/server.py::_parse_when / _parse_reminder，brain/tfgr.py::remember / due / mark_done

本文件已按 2026-11-11 的修复同步：
  - 无钟点分支会把分钟清零（曾经的 bug：20:43 说「今晚」→ 20:43 而不是 20:00）
  - 支持「N 分钟后」
  - 只说了「晚上」「三点」但没给日期 → 按今天算；纯「帮我记住…」仍然只钉住不到点
"""
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
TRIGGER = ("提醒我", "提醒一下", "提醒下我", "别忘了", "别忘", "帮我记", "记住")
IMPERATIVE = ("提醒我", "提醒一下", "提醒下我", "帮我记", "记住", "记得帮我", "别忘了", "别忘")
CLOCK_WORDS = ("下午", "晚上", "今晚", "傍晚", "中午", "早上", "上午")


def parse_when(raw: str):
    now = datetime.now()

    # 1) 相对量（分钟最具体，先匹配）
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:分钟|分)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(minutes=float(m.group(1)))).timestamp()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:天|日)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(days=float(m.group(1)))).timestamp()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:个)?(?:小时|钟头)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(hours=float(m.group(1)))).timestamp()

    base = None
    if "大后天" in raw:
        base = now + timedelta(days=3)
    elif "后天" in raw:
        base = now + timedelta(days=2)
    elif "明天" in raw:
        base = now + timedelta(days=1)
    elif "今天" in raw or "今晚" in raw:
        base = now

    m = re.search(r"(下)?(?:周|星期)([一二三四五六日天])", raw)
    if m:
        target = WEEKDAY[m.group(2)]
        ahead = (7 - now.weekday()) + target if m.group(1) else (target - now.weekday()) % 7
        base = (now + timedelta(days=ahead)).replace(hour=9, minute=0, second=0, microsecond=0)

    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", raw)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = now.year if (month, day) >= (now.month, now.day) else now.year + 1
        try:
            base = now.replace(year=year, month=month, day=day,
                               hour=9, minute=0, second=0, microsecond=0)
        except ValueError:
            return None

    if base is None:
        # 只说了时段/钟点、没给日期 → 按今天算
        if not (any(k in raw for k in CLOCK_WORDS)
                or re.search(r"(\d{1,2})\s*[点:：]", raw)
                or re.search(r"[一二两三四五六七八九十]{1,3}\s*[点時时]", raw)):
            return None
        base = now

    hh, mm = base.hour, base.minute
    got_clock = False
    tm = re.search(r"(\d{1,2})\s*[点:：]\s*(\d{1,2})?", raw)
    cn = re.search(r"([一二两三四五六七八九十]{1,3})\s*[点時时]", raw)
    if tm:
        hh, mm, got_clock = int(tm.group(1)), int(tm.group(2) or 0), True
    elif cn:
        table = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        s = cn.group(1)
        h = table.get(s)
        if h is None and s.startswith("十"):
            h = 10 + table.get(s[1:], 0)
        elif h is None and len(s) == 2 and s.endswith("十"):
            h = table.get(s[0], 0) * 10
        if h is not None:
            hh, mm, got_clock = h, 0, True
            tail = raw[cn.end():cn.end() + 4]
            if tail.startswith("半"):
                mm = 30
            elif tail.startswith("一刻"):
                mm = 15

    if got_clock:
        if ("下午" in raw or "晚上" in raw or "今晚" in raw) and hh < 12:
            hh += 12
        if hh > 23 or mm > 59:
            return None
    else:
        mm = 0                             # 修复点：分钟必须清零
        if "下午" in raw:
            hh = 14
        elif "晚上" in raw or "今晚" in raw:
            hh = 20
        elif "中午" in raw:
            hh = 12
        elif "早上" in raw or "上午" in raw:
            hh = 9
        else:
            hh = 9

    dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if dt <= now:
        dt += timedelta(days=1)
    return dt.timestamp()


class Reminders:
    def __init__(self, path="rem_demo.jsonl"):
        self.path = Path(path)
        self.items = []
        if self.path.exists():
            for line in self.path.read_text("utf-8").splitlines():
                if line.strip():
                    try:
                        self.items.append(json.loads(line))
                    except Exception:
                        pass

    def _save(self):
        self.path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in self.items), "utf-8")

    def maybe_add(self, text: str):
        if not any(k in text for k in TRIGGER):
            return None
        has_time = bool(
            re.search(r"\d+\s*(?:分钟|分|天|日|小时|钟头)", text)
            or re.search(r"(?:周|星期)[一二三四五六日天]", text)
            or re.search(r"\d{1,2}\s*月\s*\d{1,2}", text)
            or any(k in text for k in CLOCK_WORDS + ("明天", "后天", "大后天", "今天"))
        )
        imperative = any(k in text for k in IMPERATIVE)
        if not (imperative or (has_time and "提醒" in text)):
            return None
        # 反问/求证不算
        if re.search(r"记(得|不记得)[^，。！？]{0,6}[吗么？?]", text):
            return None
        due = parse_when(text)          # 可能为 None：只钉住，不到点
        for it in self.items:
            if it["text"] == text[:200] and not it.get("done"):
                return it
        it = {"id": f"r{len(self.items) + 1}", "text": text[:200],
              "due_at": due, "done": False}
        self.items.append(it)
        self._save()
        return it

    def due(self):
        now = time.time()
        return [x for x in self.items
                if x.get("due_at") and not x["done"] and x["due_at"] <= now]

    def mark_done(self, rid):
        for x in self.items:
            if x["id"] == rid:
                x["done"] = True
        self._save()


if __name__ == "__main__":
    r = Reminders()
    cases = ["10分钟后提醒我喝水", "明天下午三点提醒我交报告",
             "周五提醒我开会", "今晚提醒我吃药", "晚上提醒我散步",
             "帮我记住我不吃辣", "你还记得吗"]
    for c in cases:
        it = r.maybe_add(c)
        if it and it.get("due_at"):
            print(f"{c:20s} -> {datetime.fromtimestamp(it['due_at']):%m-%d %H:%M}")
        elif it:
            print(f"{c:20s} -> 只钉住（不到点）")
        else:
            print(f"{c:20s} -> 不记")
    Path("rem_demo.jsonl").unlink(missing_ok=True)
