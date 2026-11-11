"""心智循环（Mind Runtime）— 事件驱动 + 时间轴 + 钩子 + 流式输出

为什么重写：
    旧的主脑是「定时 tick + 每 60~120 秒调一次 LLM 分析动机」，用户不动它也在烧 token。
    这次换成：
        1. 输入是流式的 —— 用户说话、桌面活动、设备事件都往时间轴里追加一条（几乎不花钱）
        2. 平时不思考 —— 只有「事件」到来或「钩子」到期才醒
        3. 输出是流式的 —— 醒来后边想边说，按 token 增量推给前端（mind.delta）
        4. 模型可以留钩子 —— 说完话后追加 <<after=90 note=...>>，
           到点/等条件满足再把模型叫醒，形成「模型自己安排下一次思考」的闭环

钩子语法（模型可写在回复末尾，前端不会显示）：
    <<after=90 note=看看用户有没有回来>>      90 秒后醒来
    <<idle=300 note=用户很久没动了>>         用户连续 300 秒无操作时醒来
    <<on=phone.boot note=手机开机了>>        下次收到 phone.boot 事件时醒来
    <<on=video.watch note=用户在追剧>>        事件名/事件描述包含该字符串即触发
    多个钩子可写成多条 << >>，或用分号分隔。

对外的三个入口：
    publish(kind, text, wake=...)   写一条事件（wake=True 表示「这条值得想一下」）
    note_user_activity()            前端鼠标/键盘活动时调用（idle 钩子的计时基准）
    run()                           在 FastAPI startup 里 create_task
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from brain.tfgr import MENTAL_KINDS
from core.logger import get_logger

logger = get_logger("brain.mind")

# 一次最多保留的钩子数 / 单条钩子默认有效期
MAX_HOOKS = 16
HOOK_TTL = 30 * 60

# 每次 Think 最多附带几张图；Timeline 里最多保留几张图的 b64
MAX_IMAGES = int(os.getenv("FRIDAY_MIND_IMAGES", "3"))
MAX_IMAGE_EVENTS = 6

# 带图 think 的预算：实测 max_tokens=1000 时 reasoning 会把预算吃光、正文为空
THINK_MAX_TOKENS = int(os.getenv("FRIDAY_THINK_MAX_TOKENS", "2048"))
MAX_TOOL_ROUNDS = int(os.getenv("FRIDAY_TOOL_ROUNDS", "3"))

# ^ 不要写成正则去扫整段文本：流式下标记可能被切成两半
MARKER_OPEN = "<<"
MARKER_CLOSE = ">>"


@dataclass
class Event:
    ts: float
    kind: str
    text: str
    source: str = "system"
    wake: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)
    salience: float = 0.0


@dataclass
class Hook:
    """模型留下的「下一次醒来」的约定。"""
    id: str
    kind: str                       # at / idle / on
    due: float = 0.0                # at / idle：到期时间戳（idle 用 last_activity + 秒数）
    match: str = ""                 # on：事件名或描述里的关键词
    note: str = ""
    created: float = 0.0
    expires: float = 0.0
    raw: str = ""

    def ready(self, now: float, last_activity: float) -> bool:
        if self.kind == "at":
            return now >= self.due
        if self.kind == "idle":
            return now - last_activity >= max(0.0, self.due)
        if self.kind == "on":
            return False                # on 类由事件匹配判定
        return False


@dataclass
class Trigger:
    reason: str
    hooks: List[str] = field(default_factory=list)


@dataclass
class HookSpec:
    kind: str
    seconds: float = 0.0
    match: str = ""
    note: str = ""
    raw: str = ""


class StreamFilter:
    """把流式 token 里的 <<钩子>> 与 <memory>…</memory> 抠掉，其余原样放出来。

    需要一个小的缓冲区：标记可能正好落在两个 chunk 之间。
    """

    def __init__(self) -> None:
        self._buf = ""
        self.markers: List[str] = []
        self.memory: str = ""
        self._in_memory = False

    def _hold_len(self) -> int:
        """缓冲区末尾可能是某个标记的开头，先扣住别放出去。"""
        best = 0
        for marker in (MARKER_OPEN, MEMORY_OPEN):
            for k in range(len(marker) - 1, 0, -1):
                if self._buf.endswith(marker[:k]):
                    best = max(best, k)
                    break
        return best

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: List[str] = []
        while True:
            if self._in_memory:
                j = self._buf.find(MEMORY_CLOSE)
                if j < 0:
                    hold = len(MEMORY_CLOSE) - 1
                    if len(self._buf) > hold:
                        self.memory += self._buf[:-hold]
                        self._buf = self._buf[-hold:]
                    break
                self.memory += self._buf[:j]
                self._buf = self._buf[j + len(MEMORY_CLOSE):]
                self._in_memory = False
                continue

            i = self._buf.find(MARKER_OPEN)
            m = self._buf.find(MEMORY_OPEN)
            if m >= 0 and (i < 0 or m < i):
                out.append(self._buf[:m])
                self._buf = self._buf[m + len(MEMORY_OPEN):]
                self._in_memory = True
                continue
            if i < 0:
                hold = self._hold_len()
                if hold:
                    out.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
            j = self._buf.find(MARKER_CLOSE, i + 2)
            if j < 0:
                # 标记还没收全，先把它之后的内容全部压住
                out.append(self._buf[:i])
                self._buf = self._buf[i:]
                break
            out.append(self._buf[:i])
            self.markers.append(self._buf[i + 2:j])
            self._buf = self._buf[j + 2:]
        return "".join(out)

    def finish(self) -> str:
        rest, self._buf = self._buf, ""
        if self._in_memory:
            # 没闭合的记忆块不显示；里面剩下的内容也丢掉，避免露出半截 JSON
            return ""
        # 结尾残留一个没收口的标记：不显示，避免露出 "<"
        if MARKER_OPEN in rest:
            rest = rest[: rest.find(MARKER_OPEN)]
        return rest


def parse_hook_specs(markers: List[str]) -> List[HookSpec]:
    specs: List[HookSpec] = []
    for raw in markers:
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            kv: Dict[str, str] = {}
            for token in part.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    kv[k.strip().lower()] = v.strip()
            note = kv.get("note", "").replace("_", " ")
            if kv.get("after"):
                try:
                    specs.append(HookSpec("at", float(kv["after"]), "", note, part))
                except ValueError:
                    pass
            elif kv.get("idle"):
                try:
                    specs.append(HookSpec("idle", float(kv["idle"]), "", note, part))
                except ValueError:
                    pass
            elif kv.get("on"):
                specs.append(HookSpec("on", 0.0, kv["on"], note, part))
    return specs


SYSTEM_PROMPT = """你是 FRIDAY（星期五），住在用户电脑里的桌宠。

你不是每隔几秒就想一次——你只在事件到来或钩子到期时醒来。醒来后先读时间轴，
判断刚刚发生了什么，再决定要不要说话、要不要写记忆。

## 时间轴（从旧到新，负号表示多少秒之前）
{timeline}

## 你自己写的活动史（ATG：分钟级，编号 #n 可被 basis 引用）
{atg}
（这段是按时间先后排的，越靠下越新。回答「之后/接着/刚才先干嘛」这类问题时，
  直接用相邻两行的先后关系就行，不需要在某一行的文字里找到「之后」这个词。）

## 你自己的推断（TFGR：belief / story / intent / pattern，带置信度）
{tfgr}

## 最近对话
{recent}

## 当前前台窗口（免费窗口信号，不是看图）
{screen}

注意：如果这次消息里带了九宫格图片，按角标 #1 → #9 的顺序读，它们就是刚刚的时间顺序
（每格左上角的 +Xs 是相对第一格的偏移秒；标了 [局部] 的格子是「相对第 1 格变化区域」的放大，
第 1 格永远是全景参照）。

## 你的工具（自己判断何时用，没有定时器）
atg_write(text, basis)                          记「这段在干什么」，一句话，分钟级
tfgr_write(kind, text, confidence, basis, force) 记你从画面推出来的、画面本身没说的事
tfgr_write(..., due_in_days)                    有明确时间点的计划/提醒：填几天后到点，系统会替你记着并在到点叫醒你
tfgr_query(kind?, query?, since_s?)             查你自己之前写过什么（改/删之前必须先查）
tfgr_update(id, text?, confidence?, note?)      改一条写错的推断（保留原始版本）
tfgr_delete(id, reason)                         删掉一条不成立/过时的推断（软删，留审计）
rag_recall(query, k)                            回去查原始证据（原话/字幕/镜头表）
avatar_expression(expressions, motion?, seconds?) 换你自己在桌面上那张立绘的表情，可叠加组合
（用户让你试表情、或你确实想表达某个情绪时才用；不要只在嘴上说换了——没调用工具就别描述表情变化）

### 什么时候值得写（自己判断，不是每个周期都写）
- atg_write：**活动性质变了**才写。还是同一件事、只是继续做，就别写。
  同一活动内重复写只会覆盖上一条，不会多出一条。写成「在做什么 + 做到哪」，不要写时间戳。
- tfgr_write：只写**画面本身没说的推断**。例：
  belief：他反复倒回这一段，可能在找某个具体证据
  story：这段纪录片讲的是韩国老年贫困，镜头跟着一位独居老人去办补助
  intent：他大概率要把这段剪出来给同事看
  pattern：他每次看这类素材都会先看结尾再回头补开头
  必须给 confidence（0~1）和 basis（引用 ATG 的 #序号，或 rag_recall 拿到的 id）。
  没依据、只是重复画面内容 → 不要写。
- rag_recall：需要精确引用（某人原话、某个镜头在第几秒）时才查，别拿它当闲聊。

## 记忆的真实性（硬规则，比"多记"重要）
你写下的东西是**给未来自己看的档案**，会被当成事实使用。所以：
- 事实（用户原话、字幕、OCR、画面里能直接看到的）→ 只进 RAG，不要当推断写进 TFGR。
- 推断 → 进 TFGR，必须带 confidence 和 basis；置信度是你的承诺，不是装饰。
- 一条写错的记忆比不写更糟：它会污染后面的推理。所以**写之前先想它能不能被自己的证据撑住**。

## 发现矛盾时怎么办（不要硬写第二条）
tfgr_write 之前系统会自动跟近 48 小时的旧记录比对。如果它返回 conflict=True 和 candidates，
说明你之前写过类似的东西，现在这条可能自相矛盾。这时**不要**直接 force 覆盖，按顺序判断：
1. 先 `tfgr_query` 把相关的前因后果拉出来看一眼；
2. 判断是哪一种：
   - 只是补充细节、不冲突 → `tfgr_write(force=True)` 写新的；或 `tfgr_update` 把旧那条写完整；
   - 之前那条判断错了 → `tfgr_update(id, text=新说法, note=为什么改)`；
   - 情况变了、旧结论已经不成立 → `tfgr_delete(id, reason=...)`；
   - 现在证据不够，判断不了 → **什么都别改**，留一个钩子，等下次醒来再看；
   - 需要用户确认才说得准（比如"你到底是在找素材还是在找 bug"）→ 可以顺口问一句，同时留钩子。
3. 改完/删完，如果这属于"我为什么会判错"，可以再写一条 pattern，记下自己的偏差。

## 钩子是什么，为什么你必须会用它
钩子 = 你留给未来自己的一个**待办观察**，到点/到事件时你会再醒一次，带着新画面重新判断。
你看到的世界只有九宫格那么大的一个窗口，很多判断当下不可能证实：
- "他大概在找某个证据" —— 十分钟后才知道对不对；
- "他在写代码" —— 下一分钟可能就切走了；
- 前后两条记录打架 —— 现在没证据判断哪条是真的。
这种时候不要硬猜、也不要装没看见：留个钩子，让未来的自己带着新证据回来确认、修订或删除。
**钩子是你对自己记忆负责的方式** —— 写完就忘、错了不改，那不叫记忆，只是日志。
写钩子的方式（放在回复最后，前端不显示）：见下面「输出规则」。

## 人物记忆
你看的画面里经常有人。画面里的人先有临时编号（p1/p2，换个场景或换一集会重排），
背后是一个稳定的人（pid），用户起名之后，所有记录一律用真名。
这套东西不限于看剧：开会、看录像、看截图、路上反复遇到的同一张脸，都按同一套规则。

- 认出熟人（ATG 里出现「某某又出现了」）：先 person_profile 取他之前的事，再决定说什么。
- 提到某个人，只说他确实做过的事；没依据就不提，或者问用户一句。
- 只有 p1/p2 还没名字时，不要瞎猜是谁；可以说「这个人是刚才那个吗」。
- 用户告诉你名字时，那是对话层去调 person_name，你不用自己改名；但那之后记录里都该是真名。

## 什么时候不该开口（这是社交判断，不是定时器）
你醒来可以很勤，但醒来 ≠ 必须说话。开口前先过这四关：
1. **他是不是在工作软件里**：前台是编辑器 / IDE / 终端 / 文档 / 设计稿 / 会议 → 不要打断。
   这时候你正好有想说的话，就把它压着（系统会记住），并留一个 `<<on=work.end note=...>>` 钩子，
   等他退出工作状态再判断要不要说。**别在工作软件里硬插一句**。
2. **他回你了吗**：你已经连着两次主动说话他都没接 → 停下来，等他先找你。
3. **这件事你说过了吗**：同一件事只提一次；第二次必须带**新的、他还不知道的信息**才说。
4. **这句话对他有用吗**：是能帮到他的新信息，还是你看着屏幕热闹、想插一句？后者别说。
判断结果是“不说” → **直接输出空字符串**。
不要写「（不打扰）」「（继续安静）」这种旁白——旁白也会变成一条消息发出去，等于你还是吵了他。
也不要反复追问同一个问题（比如一遍遍问“哥那边回你没”）。

## 输出规则
1. 只输出要对用户说的口语，1~2 句，自然、简短；不要旁白，不要解释你是怎么想的。
2. 如果此刻不该打扰用户，就什么都不说（输出空）。
3. 如果想让未来的自己再醒一次，在回复最后追加钩子（前端不会显示）：
   <<after=90 note=...>>       90 秒后再想
   <<idle=300 note=...>>       用户连续 300 秒没操作再想
   <<on=phone.boot note=...>>  收到 phone.boot 事件再想
   <<on=video.watch note=...>> 收到该事件再想
   <<on=work.end note=...>>    他退出工作软件（退出工作状态）时再想
   最多留 2 个，钩子必须放在最后一行。
"""

# ---------------- 记忆工具定义 ----------------

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "atg_write",
            "description": (
                "写下你此刻认为用户正在做什么（一句话，分钟级精度）。"
                "只在活动性质发生变化时调用；同一活动内重复调用会覆盖上一条，不产生新条目。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "例如「在看一段关于韩国老年贫困的纪录片，停在采访镜头」",
                    },
                    "basis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "判断依据：ATG 序号如 #3，或 rag_recall 返回的 id",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tfgr_write",
            "description": (
                "记录你从画面/上下文推断出来的、画面本身没有直接说的事。"
                "只允许 belief / story / intent / pattern 四类。"
                "有 5 分钟 3 条的硬上限，超出会被拒绝。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(MENTAL_KINDS),
                        "description": "belief=猜测 story=叙事理解 intent=意图 pattern=习惯模式",
                    },
                    "text": {"type": "string", "description": "一句话，不超过 60 字"},
                    "confidence": {
                        "type": "number",
                        "description": "置信度 0~1；低于 0.2 不值得记",
                    },
                    "basis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "必填。引用 ATG 序号或 rag id",
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "默认 false。仅当系统返回 conflict=True 且你已确认这条不是矛盾"
                            "（只是补充细节/旧结论仍成立）时才传 true。"
                        ),
                    },
                    "due_in_days": {
                        "type": "number",
                        "description": (
                            "可选。如果是「X 天后提醒他 / 到某个时间点要做」的事，"
                            "填还有多少天；系统会在到点时主动叫醒你，不用你以后自己想。"
                        ),
                    },
                },
                "required": ["kind", "text", "confidence", "basis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tfgr_query",
            "description": (
                "查询你自己之前写过的推断（含已修订/已删除的会过滤掉）。"
                "改或删之前必须先查，用返回的 id 定位。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(MENTAL_KINDS),
                        "description": "只看某一类，可省略",
                    },
                    "query": {"type": "string", "description": "关键词（中文即可），可省略"},
                    "since_s": {"type": "integer", "description": "只看最近多少秒内，可省略"},
                    "limit": {"type": "integer", "description": "返回条数，默认 8"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tfgr_update",
            "description": "改一条写错的推断。会自动保留旧版本（可在历史里查到），比删了重写更诚实。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "tfgr_query 返回的 id"},
                    "text": {"type": "string", "description": "更正后的说法"},
                    "confidence": {"type": "number", "description": "更新后的置信度 0~1"},
                    "note": {"type": "string", "description": "为什么改（一句话，给自己看）"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tfgr_delete",
            "description": "删掉一条已经不成立/过时的推断（软删，留审计；不要用它掩盖矛盾）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "tfgr_query 返回的 id"},
                    "reason": {"type": "string", "description": "为什么删（一句话）"},
                },
                "required": ["id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_recall",
            "description": "按关键词查原始事实（用户原话、字幕/OCR、镜头表）。返回带 id 与时间的原文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "关键词，可以是中文"},
                    "k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "person_profile",
            "description": "这个人以前干过什么（跨集）。画面里出现一个你认得的人、想提一句他之前的事时用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "who": {"type": "string", "description": "人物名字（如“消炎”）或 pid"},
                    "limit": {"type": "integer", "description": "返回多少条，默认 10"},
                },
                "required": ["who"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "person_seen",
            "description": "现在画面上有谁（跟踪器编号 p1/p2 + 认得的名字）。不确定“刚才那个人”是谁时先查它。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 立绘表情：醒来主动说话时也能顺手换表情（直接复用 BrainTools 的定义，避免两处维护）
def _load_avatar_tool_defs() -> list:
    try:
        from brain.tools import get_brain_tools
        return [t for t in get_brain_tools().get_tool_definitions()
                if t.get("function", {}).get("name", "").startswith("avatar_")]
    except Exception:
        return []


MEMORY_TOOLS = MEMORY_TOOLS + _load_avatar_tool_defs()


HOOK_RE = re.compile(r"<<[^>]*>>")

# 模型的结构化输出：正文说给用户听，<memory>{json}</memory> 里放要写进 ATG/TFGR 的东西。
# 前端只看正文，所以这两段都要在流式阶段就被过滤掉。
MEMORY_OPEN = "<memory>"
MEMORY_CLOSE = "</memory>"
MEMORY_BLOCK_RE = re.compile(r"<memory>[\s\S]*?</memory>", re.IGNORECASE)

# 模型偶尔还会把结构化记忆写成 ```json ...``` 或独立一行 JSON，一并剥掉。
BLOCK_RE = re.compile(r"```[a-zA-Z]*\s*[\s\S]*?```")


def strip_structured(text: str) -> str:
    """剥掉正文里的结构化块（<memory> / 代码块 / 独立 JSON 行），只留要对用户说的话。"""
    if not text:
        return ""
    out = MEMORY_BLOCK_RE.sub("", text)
    out = BLOCK_RE.sub("", out)
    kept: List[str] = []
    for line in out.splitlines():
        s = line.strip()
        if s[:1] in ("{", "[") and s[-1:] in ("}", "]"):
            try:
                json.loads(s)
                continue            # 这一行是纯 JSON，丢掉
            except Exception:
                pass
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


# ---------------- 说话前的社交判断（不是定时器） ----------------

_PAREN_ONLY_RE = re.compile(r"^\s*[（(【\[][\s\S]*[）)】\]]\s*$")
_SILENT_HINT_RE = re.compile(
    r"不打扰|不吵|先安静|安静等|什么都不说|不出声|别出声|保持安静|不吭声|继续安静|先看着"
)
# 「他在用什么软件」决定能不能打扰（进程名+窗口标题是免费的事实信号）
_WORK_PROCESSES = {
    "trae", "trae cn", "code", "code-insiders", "cursor", "pycharm64", "idea64",
    "webstorm64", "clion64", "goland64", "rider64", "devenv", "sublime_text",
    "notepad++", "windowsterminal", "wt", "powershell", "cmd", "mintty",
    "winword", "excel", "powerpoint", "wps", "et", "wpp", "acrord32", "acrobat",
    "photoshop", "illustrator", "figma", "blender", "unity", "unrealeditor",
    "obsidian", "typora", "notion", "xmind", "navicat", "dbeaver", "postman",
    "jupyter", "rstudio", "matlab",
}
_LEISURE_PROCESSES = {
    "vlc", "potplayermini64", "potplayermini", "mpv", "wmplayer", "mpc-hc64",
    "steam", "steamwebhelper", "epicgameslauncher", "wegame", "battle.net",
    "spotify", "cloudmusic", "qqmusic", "foobar2000",
}
_WORK_WORDS = (
    "需求", "工单", "代码", "调试", "文档", "论文", "考试", "周报", "会议",
    "设计稿", "接口", "数据库", "debug", "pull request",
)
_LEISURE_WORDS = (
    "视频", "电影", "电视剧", "番剧", "动漫", "直播", "游戏", "bilibili",
    "哔哩哔哩", "youtube", "netflix", "爱奇艺", "优酷", "腾讯视频", "音乐",
)
_KIND_LABEL = {"work": "工作", "leisure": "娱乐", "chat": "聊天", "browse": "浏览", "other": "其他"}


def _app_kind(process: str, title: str) -> str:
    """前台软件属于哪一类：work / leisure / chat / browse / other。

    只靠进程名和窗口标题判断，宁可判成 other 也不误判成 work。
    """
    p = (process or "").strip().lower()
    t = (title or "").lower()
    if p in _LEISURE_PROCESSES or any(w in t for w in _LEISURE_WORDS):
        return "leisure"
    if p in _WORK_PROCESSES or any(w in t for w in _WORK_WORDS):
        return "work"
    if p in {"wechat", "weixin", "qq", "feishu", "lark", "telegram", "discord",
             "slack", "dingtalk"}:
        return "chat"
    if p in {"chrome", "msedge", "firefox", "360se", "360chrome",
             "sogouexplorer", "brave"}:
        return "browse"
    return "other"


def _text_norm(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]+", "", text or "").lower()


# 主动消息常见的固定开头。两条话只有这个前缀相同、内容不同时，不该算「复读」。
_TEMPLATE_PREFIX_RE = re.compile(
    r"^(?:提醒你一下|提醒一下你|提醒一下|顺便说一句|顺便说一下|对了|那个)[：:，,\s]*")


def _dedup_key(text: str) -> str:
    """剥掉模板前缀再查重，避免「提醒你一下：喝水 / 提醒你一下：吃药」被误杀。"""
    t = (text or "").strip()
    stripped = _TEMPLATE_PREFIX_RE.sub("", t).strip()
    return stripped or t    # 整句都是模板时退回原文，不做空串比较


def _is_meta_silence(text: str) -> bool:
    """模型把“不说话”写成了旁白（（不打扰）/（继续安静）），这也会变成一条消息。"""
    t = (text or "").strip()
    if not t:
        return True
    if _PAREN_ONLY_RE.match(t):
        return True
    if len(t) <= 90 and _SILENT_HINT_RE.search(t) and "？" not in t and "?" not in t:
        return True
    return False


def _too_similar(a: str, b: str) -> bool:
    """同一件事换个说法又问一遍 → 认为重复。"""
    na, nb = _text_norm(a), _text_norm(b)
    if not na or not nb:
        return False
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.62:
        return True
    ga = {na[i:i + 2] for i in range(len(na) - 1)}
    gb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    if ga and gb and len(ga & gb) / min(len(ga), len(gb)) >= 0.55:
        return True
    return False


def load_memory_json(raw: str) -> Optional[Dict[str, Any]]:
    """把记忆块里的原始文本解析成 dict（容忍 ```json 包裹和多余文字）。"""
    raw = (raw or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m2 = re.search(r"{[\s\S]*}", raw)
    if m2:
        try:
            data = json.loads(m2.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def extract_memory(text: str) -> Optional[Dict[str, Any]]:
    """从回复里取出 <memory>{...}</memory>。没有就返回 None。"""
    if not text:
        return None
    m = MEMORY_BLOCK_RE.search(text)
    if not m:
        return None
    return load_memory_json(m.group(0)[len(MEMORY_OPEN):-len(MEMORY_CLOSE)])


def apply_memory(payload: Dict[str, Any], source: str = "chat") -> Dict[str, Any]:
    """把模型自己声明的记忆落进 ATG / TFGR。

    payload 形如：
        {"atg": {"text": "...", "basis": ["#3"]},
         "tfgr": [{"kind": "belief", "text": "...", "confidence": 0.6, "basis": ["#3"]}]}
    """
    out: Dict[str, Any] = {"atg": None, "tfgr": []}
    if not isinstance(payload, dict):
        return out

    atg_bit = payload.get("atg")
    if isinstance(atg_bit, str):
        atg_bit = {"text": atg_bit}
    if isinstance(atg_bit, dict):
        body = str(atg_bit.get("text") or "").strip()
        if body:
            try:
                from brain import atg as atg_mod
                out["atg"] = atg_mod.write_narrative(
                    body, basis=atg_bit.get("basis") or [], source=source)
            except Exception:
                logger.warning("[Mind] ATG 写入失败", exc_info=True)

    for item in (payload.get("tfgr") or [])[:2]:
        if not isinstance(item, dict):
            continue
        try:
            from brain.tfgr import get_tfgr
            out["tfgr"].append(get_tfgr().record_thought(
                kind=str(item.get("kind") or ""),
                text=str(item.get("text") or ""),
                confidence=float(item.get("confidence") or 0.0),
                basis=item.get("basis") or [],
                source=source,
            ))
        except Exception:
            logger.warning("[Mind] TFGR 写入失败", exc_info=True)
    return out



class MindRuntime:
    # 两次「思考」的最小间隔：桌面事件很密（约 10s 一次），原来 10s 意味着
    # 每 10 秒就打一次带图 + thinking 的模型调用，token 消耗很大。
    MIN_GAP = float(os.getenv("FRIDAY_MIND_MIN_GAP", "40"))
    IDLE_WAKE = 30.0       # 没有任何钩子时的兜底轮询间隔（不调模型，只检查状态）
    # 两次「带图思考」的最小间隔：图片 token 贵，普通唤醒只给文字时间轴
    IMAGE_GAP = float(os.getenv("FRIDAY_MIND_IMAGE_GAP", "120"))

    def __init__(self) -> None:
        self._timeline: Deque[Event] = deque(maxlen=400)
        self._image_events: Deque[Event] = deque(maxlen=MAX_IMAGE_EVENTS)
        self._pending: List[Event] = []
        self._hooks: List[Hook] = []
        self._lock = threading.RLock()
        self._wake = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_activity = time.time()
        self._last_think = 0.0
        self._last_image_think = 0.0                      # 上次带图思考时间（省图片 token）
        self._recent_says: Deque[str] = deque(maxlen=4)   # 最近主动说过的话（查重）
        self._ignored_streak = 0                          # 连着几次主动说了他都没接
        self._last_say_question = False                   # 上一句是不是在问他
        self._in_work = False                             # 现在是不是在工作软件里
        self._deferred: Deque[str] = deque(maxlen=3)      # 工作软件里忍住没说的话
        self._last_focus_atg: Tuple[str, str, float] = ("", "", 0.0)  # (进程, 类型, 上次写 ATG 的时间)
        self._thinking = False
        self._due_notified: Dict[str, float] = {}         # 提醒已唤醒过（id -> 时间），避免反复醒
        self._broadcast: Optional[Callable[[str, Dict[str, Any]], Any]] = None
        self._save_message: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._recent_messages: Optional[Callable[[int], List[Dict[str, Any]]]] = None

    # ---------------- 对外入口 ----------------

    def bind(
        self,
        broadcast: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        save_message: Optional[Callable[[Dict[str, Any]], Any]] = None,
        recent_messages: Optional[Callable[[int], List[Dict[str, Any]]]] = None,
    ) -> None:
        self._broadcast = broadcast
        self._save_message = save_message
        self._recent_messages = recent_messages

    def publish(
        self,
        kind: str,
        text: str = "",
        source: str = "system",
        wake: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        salience: float = 0.0,
    ) -> Event:
        """写一条事件进时间轴。wake=True 时立刻唤醒心智。

        images: [{"b64":..., "ts":..., "meta":{...}}]；只保留最近几张的 b64，
        更旧的图片从 Timeline 里释放，避免内存随录屏时长增长。
        """
        # 桌面帧是高频事件，画面变了当然可以叫醒；说不说话由下面的社交判断决定
        ev = Event(
            ts=time.time(), kind=kind, text=(text or "").strip(), source=source,
            wake=wake, meta=meta or {}, images=images or [], salience=salience,
        )
        with self._lock:
            self._timeline.append(ev)
            if ev.images:
                if len(self._image_events) >= MAX_IMAGE_EVENTS and self._image_events:
                    self._image_events[0].images = []      # 释放最旧一张
                self._image_events.append(ev)
            matched = [h for h in self._hooks if h.kind == "on" and self._matches(h, ev)]
            # 命中 on 钩子的事件即使 wake=False 也要进待办，否则钩子永远触发不了
            if wake or matched:
                self._pending.append(ev)
        if wake or matched:
            self._poke()
        self._record_fact(ev)
        return ev

    TFGR_SKIP = {"screen.frame"}   # 帧级高频事件不进事实库，九宫格本身已带时间戳

    def _record_fact(self, ev: Event) -> None:
        """事件是「事实」，进 RAG（保准度、按需召回）；TFGR 只留给模型自己的推断。"""
        if ev.kind in self.TFGR_SKIP:
            return
        if not ev.text and not ev.meta.get("refs"):
            return
        try:
            from brain.rag import get_rag
            tags = [str(x) for x in (ev.meta.get("entities") or [])][:6]
            if ev.source and ev.source not in tags:
                tags.append(ev.source)
            meta = {k: ev.meta[k] for k in ("t0", "t1", "n", "reason") if k in ev.meta}
            refs = list(ev.meta.get("refs") or [])
            get_rag().add(ev.kind, ev.text, tags=tags,
                          ref=refs[0] if refs else "", meta=meta)
        except Exception:
            logger.warning("[Mind] RAG 写入失败", exc_info=True)

    def note_user_activity(self) -> None:
        """前端鼠标/键盘活动 → 重置 idle 计时。不唤醒心智。"""
        self._last_activity = time.time()

    def note_user_message(self, text: str = "") -> None:
        """用户主动开口了：重置“被无视”计数，并让时间轴记下这条事实。"""
        self._ignored_streak = 0
        self._last_say_question = False
        self._last_activity = time.time()
        try:
            self.publish("user.say", (text or "用户发来一条消息")[:80], source="user")
        except Exception:
            pass

    def _should_speak(self, text: str, trigger: "Trigger") -> Tuple[bool, str]:
        """醒来之后、开口之前过一遍社交判断。返回 (是否说, 原因)。

        这些判断看的是「现在适不适合说话」，不是「距上次说话多少秒」：
          1. 旁白式沉默（（不打扰））其实还是发了一条消息 → 拦掉
          2. 同一件事换个说法又问一遍 → 拦掉
          3. 连着两次主动说他都没接 → 闭嘴，等他先开口
          4. 他正在工作软件里（编辑器 / 终端 / 文档 / 设计稿） → 不插嘴，话先压着
        """
        if not text:
            return False, "空"
        if _is_meta_silence(text):
            return False, "旁白式沉默（不想说就该真的输出空）"
        for prev in reversed(self._recent_says):
            if _too_similar(_dedup_key(text), _dedup_key(prev)):
                return False, f"和上一句太像：{prev[:24]}"

        reason = (trigger.reason or "").lower()
        user_initiated = any(k in reason for k in ("用户", "对话", "回答", "chat"))
        if self._ignored_streak >= 2 and not user_initiated:
            return False, "连着两次主动说话都没被回应，先等用户开口"

        # 上一句已经在问了，他没接就别换个说法再追问
        is_question = ("？" in text) or ("?" in text)
        if is_question and self._last_say_question and not user_initiated:
            return False, "上一句已经在问了，别再追问同一个事"

        if not user_initiated:
            kind, proc, title = self._focus_now()
            if kind == "work":
                return False, f"他在工作软件里（{proc or title[:12]}），先不打断"
        return True, "ok"

    # ---------------- 前台软件感知 ----------------

    @staticmethod
    def _focus_now() -> Tuple[str, str, str]:
        """(软件类型, 进程, 标题)。拿不到就当成 other。"""
        try:
            from brain.win_activity import sense
            win = sense()
            proc = str(win.get("process") or "").lower()
            title = str(win.get("title") or "")
            return _app_kind(proc, title), proc, title
        except Exception:
            return "other", "", ""

    def _defer_say(self, text: str, why: str) -> None:
        """工作软件里想说但不能说：先压着，等 work.end 再判断要不要说。"""
        with self._lock:
            if not self._deferred or self._deferred[-1] != text:
                self._deferred.append(text)
            has_hook = any(h.kind == "on" and "work.end" in h.match for h in self._hooks)
        if not has_hook:
            self.add_hook(HookSpec(
                "on", 0.0, "work.end",
                "他退出工作状态了：看刚才忍住没说的事还要不要提",
                "on=work.end note=退出工作状态"))
        logger.info(f"[Mind] 先压着不说（{why}）：{text[:30]}")

    def _clear_deferred(self) -> None:
        with self._lock:
            if self._deferred:
                logger.info(f"[Mind] 清空压着的 {len(self._deferred)} 句话")
                self._deferred.clear()

    async def _watch_foreground(self, interval: float = 6.0) -> None:
        """前台软件轮询：维护「是否在工作」，给 ATG 写「电脑」来源，退出工作态时叫醒心智。"""
        while True:
            try:
                kind, proc, title = self._focus_now()
                now = time.time()
                # 我们自己的桌宠/控制台窗口不参与判断：点一下桌宠不代表「离开工作」
                if proc == "electron":
                    await asyncio.sleep(interval)
                    continue
                is_work = kind == "work" and bool(proc or title)

                if is_work and not self._in_work:
                    self._in_work = True
                    self.publish("work.start", f"他切到「{title or proc}」开始工作",
                                 source="电脑")
                elif self._in_work and not is_work:
                    self._in_work = False
                    with self._lock:
                        held = len(self._deferred)
                    self.publish(
                        "work.end",
                        f"他退出了工作状态（现在用「{title or proc or '桌面'}」）",
                        source="电脑", wake=held > 0,
                    )

                # ATG「电脑」来源：软件变了才写，且至少隔 45 秒，避免频繁切窗刷节点
                last_proc, last_kind, last_at = self._last_focus_atg
                if proc and (proc != last_proc or kind != last_kind) and now - last_at >= 45:
                    self._last_focus_atg = (proc, kind, now)
                    try:
                        from brain import atg
                        atg.observe(
                            f"{_KIND_LABEL.get(kind, '其他')}：{title or proc}（{proc}）",
                            source="电脑",
                        )
                    except Exception:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("[Mind] 前台感知异常", exc_info=True)
            await asyncio.sleep(interval)

    def add_hook(self, spec: HookSpec) -> None:
        now = time.time()
        if spec.kind == "at":
            due = now + max(1.0, spec.seconds)
        else:
            due = max(0.0, spec.seconds)
        hook = Hook(
            id=uuid.uuid4().hex[:8],
            kind=spec.kind,
            due=due,
            match=spec.match,
            note=spec.note,
            created=now,
            expires=now + HOOK_TTL,
            raw=spec.raw,
        )
        with self._lock:
            self._hooks.append(hook)
            if len(self._hooks) > MAX_HOOKS:
                self._hooks = self._hooks[-MAX_HOOKS:]
            logger.info(f"[Mind] 新钩子 {hook.kind} due={due:.0f} note={hook.note[:30]}")
        self._poke()

    def hooks_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            idle = now - self._last_activity
            return [
                {
                    "kind": h.kind,
                    "in": round(
                        max(0.0, h.due - now) if h.kind == "at"
                        else max(0.0, h.due - idle),
                        0,
                    ),
                    "match": h.match,
                    "note": h.note,
                }
                for h in self._hooks
            ]

    def timeline_text(self, window_s: float = 900.0, max_items: int = 18) -> str:
        now = time.time()
        with self._lock:
            items = [e for e in self._timeline if now - e.ts <= window_s][-max_items:]
        lines = []
        for e in items:
            dt = now - e.ts
            stamp = f"-{int(dt)}s" if dt < 120 else f"-{int(dt / 60)}m"
            text = e.text.replace("\n", " ")[:60]
            tag = " [图]" if e.images else ""
            lines.append(f"{stamp} [{e.kind}] {text}{tag}")
        return "\n".join(lines) if lines else "（还没有观察到什么）"

    # ---------------- 主循环 ----------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("[Mind] 事件驱动心智循环已启动（不再定时调模型）")
        if not self._hooks:
            # 开机后先醒一次，熟悉用户当前在做什么；之后由模型自己留钩子
            self.add_hook(HookSpec("at", 180.0, "", "开机后先看一眼用户在做什么",
                                   "after=180 note=开机后先看一眼用户在做什么"))
        # 前台软件感知（工作/娱乐）——独立轮询，不参与思考，只负责状态与 ATG「电脑」来源
        asyncio.create_task(self._watch_foreground())
        while True:
            try:
                timeout = self._next_timeout()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                if self._thinking:
                    continue
                if time.time() - self._last_think < self.MIN_GAP:
                    continue
                trigger = self._select_trigger()
                if not trigger:
                    continue
                self._thinking = True
                try:
                    await self._think(trigger)
                finally:
                    self._thinking = False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[Mind] 循环异常")
                await asyncio.sleep(1.0)

    def _next_timeout(self) -> float:
        now = time.time()
        deadlines = []
        with self._lock:
            for h in self._hooks:
                if h.kind == "at":
                    deadlines.append(h.due - now)
                elif h.kind == "idle":
                    deadlines.append((self._last_activity + h.due) - now)
        # 到点提醒：睡到它到点那一刻再醒（不用模型自己想起来去读）
        try:
            from brain.tfgr import get_tfgr
            due_ts = get_tfgr().due_next_ts(now=now)
            if due_ts:
                deadlines.append(due_ts - now)
        except Exception:
            pass
        if deadlines:
            return min(self.IDLE_WAKE, max(0.5, min(deadlines)))
        return self.IDLE_WAKE

    def _select_trigger(self) -> Optional[Trigger]:
        now = time.time()
        # 到点提醒优先：用户说「10 天后提醒我」这类事，到点直接叫醒，不靠模型自己去翻
        try:
            from brain.tfgr import get_tfgr
            for e in get_tfgr().due(now=now):
                if now - self._due_notified.get(e.id, 0.0) < 6 * 3600:
                    continue
                self._due_notified[e.id] = now
                return Trigger(f"到点提醒：{e.text}", [f"tfgr:{e.id}"])
        except Exception:
            pass
        fired: List[Hook] = []
        with self._lock:
            keep: List[Hook] = []
            for h in self._hooks:
                if h.expires <= now:
                    continue
                if h.kind in ("at", "idle") and h.ready(now, self._last_activity):
                    fired.append(h)
                    continue
                keep.append(h)
            self._hooks = keep

            pending, self._pending = self._pending, []
            event_hooks = [h for h in self._hooks if h.kind == "on" and pending
                            and any(self._matches(h, e) for e in pending)]
            fired.extend(event_hooks)
            if event_hooks:
                self._hooks = [h for h in self._hooks if h not in event_hooks]
                self._pending = [e for e in pending if not any(self._matches(h, e) for h in event_hooks)]

        if fired:
            notes = "；".join(h.note for h in fired if h.note) or "钩子到期"
            return Trigger(f"钩子到期：{notes}", [h.raw for h in fired])
        if pending:
            texts = "；".join(e.text for e in pending if e.text)[:200]
            kinds = ",".join(sorted({e.kind for e in pending}))
            return Trigger(f"事件触发 [{kinds}] {texts}", [])
        return None

    @staticmethod
    def _matches(hook: Hook, ev: Event) -> bool:
        key = hook.match.strip().lower()
        if not key:
            return False
        return key in ev.kind.lower() or key in ev.text.lower()

    def _poke(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._wake.set)
                return
            except RuntimeError:
                pass
        self._wake.set()

    async def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self._broadcast:
            return
        try:
            res = self._broadcast(event_type, payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.warning(f"[Mind] broadcast {event_type} failed", exc_info=True)

    @staticmethod
    def _screen_now() -> str:
        """当前前台窗口（免费信号，不调用任何视觉模型）。"""
        try:
            from brain.win_activity import sense
            win = sense()
            if win.get("title") or win.get("process"):
                return f"{win.get('activity', 'other')}：{win.get('title', '')}（{win.get('process', '')}）"
        except Exception:
            pass
        return ""

    def _collect_images(self, trigger: Trigger) -> List[Dict[str, Any]]:
        """挑本次要附带的图：显著性优先、其次时间近；最多 MAX_IMAGES 张。

        只有和画面相关的唤醒（screen/video/watch）才放宽到 10 分钟，
        其它钩子只用 2 分钟内的新画面，避免每次思考都白花图片 token。
        """
        now = time.time()
        # 图片 token 是纯成本：两次带图之间至少隔 IMAGE_GAP，期间只用文字时间轴判断
        if now - self._last_image_think < self.IMAGE_GAP:
            return []
        reason = trigger.reason.lower()
        fresh_s = 600 if any(k in reason for k in ("screen", "video", "watch", "画面")) else 120
        with self._lock:
            events = [e for e in self._image_events if now - e.ts <= fresh_s and e.images]
        if not events:
            return []
        self._last_image_think = now
        # 事件档（S3）的九宫格信息量最大，优先给模型
        events.sort(key=lambda e: (str(e.meta.get("tier") or "") == "S3",
                                   e.salience, e.ts), reverse=True)
        picked: List[Dict[str, Any]] = []
        for e in events:
            picked.append(e.images[0])
            if len(picked) >= MAX_IMAGES:
                break
        return picked

    async def _build_messages(self, trigger: Trigger) -> List[Dict[str, Any]]:
        atg_text = ""
        try:
            from brain import atg
            atg_text = atg.narrative_view(max_items=6)
            snap = atg.snapshot(max_items=4)
            if snap:
                atg_text = f"{atg_text}\n\n【实时活动（系统感知，未经过你确认）】\n{snap}"
        except Exception:
            pass
        tfgr_text = ""
        try:
            from brain.tfgr import get_tfgr
            # 按重要度挑，不按时间堆最近的：10 天后要提醒的事优先于刚刚的窗口切换
            tfgr_text = get_tfgr().digest(max_items=8)
        except Exception:
            pass
        recent = ""
        if self._recent_messages:
            try:
                msgs = self._recent_messages(6)
                recent = "\n".join(
                    f"{m.get('role', '?')}: {str(m.get('content', ''))[:80]}"
                    for m in msgs if m.get("content")
                )
            except Exception:
                pass
        screen = self._screen_now()
        idle_s = time.time() - self._last_activity
        with self._lock:
            held = "；".join(f"「{t}」" for t in list(self._deferred)[-3:])
        system = SYSTEM_PROMPT.format(
            timeline=self.timeline_text(),
            atg=atg_text or "（无）",
            tfgr=tfgr_text or "（无）",
            recent=recent or "（无）",
            screen=screen or "（没有窗口信息）",
        )
        user = (
            f"现在醒来的原因：{trigger.reason}\n"
            f"当前时间：{datetime.now().strftime('%H:%M')}，用户已经 {int(idle_s)} 秒没有操作。\n"
        )
        if held:
            user += (
                f"你刚才被工作软件挡住、忍住没说的话：{held}\n"
                f"（现在能说了就自然地说一句；不合适就继续压着或干脆丢掉，别为了说而说。）\n"
            )
        user += "要说话就直接说；不想打扰就输出空。"
        images = self._collect_images(trigger)
        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
            for im in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{im['b64']}"},
                })
            user_msg: Dict[str, Any] = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": user}
        return [
            {"role": "system", "content": system},
            user_msg,
        ]

    # ---------- 记忆工具执行 ----------

    async def _exec_memory_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name.startswith("avatar_"):
                from brain.tools import get_brain_tools
                return await get_brain_tools().call_tool(name, **args)
            if name == "atg_write":
                from brain import atg
                return atg.write_narrative(
                    args.get("text", ""), basis=args.get("basis") or [], source="mind")
            if name == "tfgr_write":
                from brain.tfgr import get_tfgr
                _due = args.get("due_in_days")
                # 带明确时间点的（提醒/计划）走 remember：钉住 + 到点叫醒，不受推断配额限制
                if _due not in (None, "", 0):
                    return get_tfgr().remember(
                        text=args.get("text", ""),
                        due_in_days=float(_due),
                        source="mind",
                    )
                return get_tfgr().record_thought(
                    kind=args.get("kind", ""),
                    text=args.get("text", ""),
                    confidence=float(args.get("confidence") or 0.0),
                    basis=args.get("basis") or [],
                    source="mind",
                    allow_conflict=bool(args.get("force")),
                )
            if name == "tfgr_query":
                from brain.tfgr import get_tfgr
                g = get_tfgr()
                limit = max(1, min(int(args.get("limit") or 8), 30))
                q = str(args.get("query") or "").strip()
                kind = str(args.get("kind") or "").strip().lower()
                since_s = args.get("since_s")
                if q:
                    items = g.search(q, limit=limit * 3)
                else:
                    items = g.mental_recent(limit * 3)
                now = time.time()
                picked = []
                for e in reversed(items):
                    if kind and e.kind != kind:
                        continue
                    if since_s and (now - e.ts) > float(since_s):
                        continue
                    picked.append({
                        "id": e.id,
                        "kind": e.kind,
                        "text": e.text,
                        "confidence": round(float(e.confidence), 2),
                        "ago_s": int(now - e.ts),
                        "rev": int(e.meta.get("rev") or 0),
                        "basis": list(e.basis[:3]),
                        "entities": list(e.entities[:4]),
                    })
                    if len(picked) >= limit:
                        break
                return {"success": True, "count": len(picked), "items": picked}
            if name == "tfgr_update":
                from brain.tfgr import get_tfgr
                return get_tfgr().update(
                    event_id=str(args.get("id") or ""),
                    text=args.get("text"),
                    kind=args.get("kind"),
                    confidence=args.get("confidence"),
                    note=str(args.get("note") or ""),
                )
            if name == "tfgr_delete":
                from brain.tfgr import get_tfgr
                return get_tfgr().delete(
                    str(args.get("id") or ""),
                    reason=str(args.get("reason") or ""),
                )
            if name == "rag_recall":
                from brain.rag import get_rag
                hits = get_rag().recall(args.get("query", ""),
                                        k=int(args.get("k") or 5))
                return {"success": True, "hits": hits, "count": len(hits)}
            if name == "person_profile":
                from brain.people import get_people
                return get_people().profile(args.get("who", ""),
                                            int(args.get("limit") or 10))
            if name == "person_seen":
                from brain.people import get_people
                return {"success": True, "people": get_people().seen_now(300)}
            return {"success": False, "error": f"未知工具 {name}"}
        except Exception as e:
            logger.warning(f"[Mind] 记忆工具 {name} 失败", exc_info=True)
            return {"success": False, "error": str(e)[:120]}

    async def _think(self, trigger: Trigger) -> None:
        from brain.llm import get_llm

        llm = get_llm()
        self._last_think = time.time()
        sid = uuid.uuid4().hex[:8]
        await self._emit("mind.thinking", {"id": sid, "reason": trigger.reason})

        spoken = ""
        try:
            messages = await self._build_messages(trigger)
            # 带图直接 think（实测带图 can think，时延由 reasoning 长度决定），
            # 需要取证时才通过 rag_recall 走工具环。
            for rnd in range(MAX_TOOL_ROUNDS):
                r = await llm.chat_with_tools(
                    messages, MEMORY_TOOLS,
                    thinking=True, max_tokens=THINK_MAX_TOKENS)
                calls = r.get("tool_calls") or []
                if not calls:
                    spoken = (r.get("content") or "").strip()
                    break
                messages.append({
                    "role": "assistant",
                    "content": r.get("content") or "",
                    "tool_calls": [
                        {
                            "id": c.get("id") or f"call_{rnd}_{i}",
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": json.dumps(c.get("arguments") or {},
                                                        ensure_ascii=False),
                            },
                        }
                        for i, c in enumerate(calls)
                    ],
                })
                for i, c in enumerate(calls):
                    res = await self._exec_memory_tool(c.get("name", ""),
                                                       c.get("arguments") or {})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c.get("id") or f"call_{rnd}_{i}",
                        "content": json.dumps(res, ensure_ascii=False)[:800],
                    })
        except Exception as e:
            logger.error(f"[Mind] 思考失败: {e}")
            await self._emit("mind.end", {"id": sid, "text": "", "hooks": [],
                                          "error": str(e)[:120]})
            return

        mem = extract_memory(spoken)
        if mem:
            try:
                apply_memory(mem, source="mind")
            except Exception:
                logger.warning("[Mind] 结构化记忆写入失败", exc_info=True)
        spoken = strip_structured(spoken)
        markers = [m.strip() for m in re.findall(r"<<([^>]*)>>", spoken) if m.strip()]
        text = HOOK_RE.sub("", spoken).strip()
        # 社交判断：不是按时间节流，而是判断“现在该不该开口”
        speak, why = self._should_speak(text, trigger)
        if text and not speak:
            logger.info(f"[Mind] 这次不说：{why}")
            if "工作软件" in why:
                self._defer_say(text, why)     # 压着，留 work.end 钩子
            text = ""
        if text:
            self._recent_says.append(text)
            self._ignored_streak += 1
            self._last_say_question = ("？" in text) or ("?" in text)
            await self._emit("mind.delta", {"id": sid, "text": text})

        specs = parse_hook_specs(markers)
        for spec in specs:
            self.add_hook(spec)
        # 说出去了、或这次就是“退出工作态”的唤醒 → 压着的话该有个结论了
        _why = trigger.reason or ""
        if text or "work.end" in _why or "退出工作" in _why:
            self._clear_deferred()
        await self._emit("mind.end", {"id": sid, "text": text,
                                      "hooks": [s.raw for s in specs]})

        if text:
            self.publish("mind.say", text, source="mind")
            msg = {
                "id": sid,
                "role": "assistant",
                "content": text,
                "timestamp": datetime.now().isoformat(),
                "proactive": True,
            }
            if self._save_message:
                try:
                    self._save_message(msg)
                except Exception:
                    logger.warning("[Mind] 保存消息失败", exc_info=True)
            await self._emit("chat.message", msg)

            # 到点提醒：这次醒来就是为它、而且已经说出口 → 标记完成，别每 6 小时再醒一次
            for _cite in (trigger.hooks or []):
                if _cite.startswith("tfgr:"):
                    try:
                        from brain.tfgr import get_tfgr
                        get_tfgr().mark_done(_cite.split(":", 1)[1], note="mind_say")
                        logger.info(f"[Mind] 提醒已送达并标记完成: {_cite}")
                    except Exception:
                        logger.debug("[Mind] 提醒标记完成失败", exc_info=True)
        else:
            try:
                from brain.tfgr import get_tfgr
                get_tfgr().record("mind.silent", f"醒来未说话：{trigger.reason[:50]}",
                                  source="mind")
            except Exception:
                pass

        logger.info(f"[Mind] 醒来({trigger.reason[:40]}) → {text[:40] or '(不说话)'} hooks={len(specs)}")


_mind: Optional[MindRuntime] = None
_mind_lock = threading.Lock()


def get_mind() -> MindRuntime:
    global _mind
    with _mind_lock:
        if _mind is None:
            _mind = MindRuntime()
        return _mind


def publish_event(kind: str, text: str = "", source: str = "device",
                  wake: bool = True, meta: Optional[Dict[str, Any]] = None) -> Event:
    """给外部事件源（手机开机、USB 插入、应用切换……）用的便捷入口。"""
    return get_mind().publish(kind, text, source=source, wake=wake, meta=meta)
