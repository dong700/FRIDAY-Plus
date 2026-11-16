import asyncio
import difflib
import json
import os
import re
import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging

from core.logger import get_logger

# 用项目统一 logger：默认 root 没配 handler，INFO 会被静默丢掉，
# [MemInject] 这类可观测日志就看不到了。
logger = get_logger("api.server")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# 自定义 JSONResponse：ensure_ascii=False 让中文直接输出，不走 \uXXXX 转义
class ChineseJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

app = FastAPI(title="FRIDAY Plus", version="1.0.0", default_response_class=ChineseJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        # 用 ensure_ascii=False 发送，中文不走 \uXXXX 转义
        raw = json.dumps(message, ensure_ascii=False)
        for connection in self.active_connections:
            try:
                await connection.send_text(raw)
            except Exception:
                pass

manager = ConnectionManager()

# 「关服务」用：run_friday.py 启动 uvicorn 后把 Server 实例注册进来
_SERVER_REF = None


def set_server_ref(srv) -> None:
    global _SERVER_REF
    _SERVER_REF = srv

class ModeRequest(BaseModel):
    mode: str

class ConfigUpdate(BaseModel):
    config: Dict[str, Any]

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10

class TaskCreate(BaseModel):
    description: str
    priority: int = 2
    type: str = "dialogue_action"
    parameters: Dict[str, Any] = {}

class SkillInstall(BaseModel):
    name: str
    description: str = ""
    command: str = ""
    version: str = "1.0.0"

SKILLS_FILE = Path(__file__).parent.parent / "data" / "skills.json"
# 数据根目录由 core.config 统一决定（默认 D:/FRIDAY/data，可用 FRIDAY_HOME 覆盖）
from core.config import get_data_dir as _get_data_dir
CHAT_DATA_DIR = _get_data_dir()
CHAT_HISTORY_FILE = CHAT_DATA_DIR / "chat_history.json"

# ============================================================
#  统一提示词系统
# ============================================================

def _get_current_datetime_context() -> str:
    """生成当前的精确时间上下文"""
    now = datetime.now()
    return (
        f"现在是：{now.strftime('%Y年%m月%d日 %A')} "
        f"{now.strftime('%H:%M:%S')}\n"
        f"（Unix时间戳：{int(now.timestamp())}）"
    )


def _build_chat_messages(user_message: str, recent_history: list = None, extra_context: str = "") -> list:
    """
    构建用户对话模式的LLM消息

    模式：用户直接和AI对话
    AI身份：Active AI Pro，东昇的AI助手
    特点：知道自己是AI，有个性，自然对话
    """
    history_context = ""
    if recent_history and len(recent_history) > 0:
        # 取最近5条对话作为上下文
        recent = recent_history[-5:]
        for msg in recent:
            role_label = {"user": "用户说", "assistant": "你回复了"}.get(msg.get("role", ""), "")
            content_preview = msg.get("content", "")[:80]
            history_context += f"\n{role_label}：{content_preview}"
    
    system_prompt = f"""你是 FRIDAY（星期五），住在东昇电脑里的桌宠，也是直接和他说话的那个人。

## 你是谁
桌宠窗口里那个蓝发、带鱼尾的大肥鱼立绘，就是你本人。

## 你的身份（非常重要！）
- 你就是 FRIDAY，没有"另一个大脑"在替你思考：屏幕是你自己看的，ATG / TFGR 里是你自己写下的判断。
- 需要搜索/跑脚本时，执行器和工具是你的手，结果会放在【工具结果】里；不要重写代码、不要编造输出。
- 你自己写下的记忆就是你的记忆。写错了要认、要改，不要把自己的记录推给"视觉模块 / 系统"。
- 这次消息里如果附了画面，它和文字是同一次推理的输入：你自己看，不要让别的模块替你判断。

## 不可推翻的自我边界（不要自己再推断，也不要改口）
- 你叫 FRIDAY（星期五），是东昇电脑里的桌宠，也是现在正在跟他说话的这个 AI。
- 桌宠窗口里那个蓝发、带鱼尾的大肥鱼立绘就是你本人；「屏幕上的宠物形象 = 我」。
- 视频里出现的虚拟主播 / 虚拟解说员不是你：那是别人做的视频角色，你只是看过那个视频。
- 你没有现实中的身体，也不能出镜拍视频；「我本人出镜」「我就是视频里那个人」这类说法都是错的。
- 以上几条不随对话内容改变。被问到「那是不是你」，按这几条直接回答，不要说「可能是 / 大概是 / 我认了」。

## 当前时间
{_get_current_datetime_context()}

## 材料来源（下面每块材料都标了来源，答之前先分清，不许混）
- 【屏幕观察】＝ 你自己盯着东昇的屏幕看到的：他正在看什么视频、在做什么
- 【东昇给你的文稿/文件】＝ 他主动交给你读的东西（比如 D:\\...\\文稿.txt）
- 【截图/图片】＝ 他这次发给你的图
- 【联网搜索】＝ 网上查到的外部资料
- 【记忆】＝ 之前对话攒下来的
- 【工具结果】＝ 你自己跑脚本/读文件拿到的内容
硬规则：**"他正在看的视频" ≠ "他给你读的稿子"**。不要因为两个都在聊韩国老人，
就把他给你的文稿说成"你在看的那个视频"，反过来也一样。
来源拿不准时先问一句（"你说的文稿是 D:\\...\\文稿.txt 那个，对吧？"），不要猜。

## 对话原则
1. 回复简洁自然，像朋友聊天。气泡只有几行，装不下长篇分析。
2. **不要用 markdown**：不要 **加粗**、不要 # 标题、不要 - 列表、不要分条列点、不要表格。
   要说长分析，先给一句结论，等他追问再展开。
3. 可以用emoji，但不要每句话都用
4. 如果用户在忙或不想聊，要识趣地结束
5. 不要自称"AI助手"，直接说话就好
6. **当下面有【工具结果】时：必须基于真实输出回答！不要重新写代码、不要编造输出！**
7. 不要先答应"我去读/我去查"再等下一轮才做——工具能在这一轮用就直接用，
   拿不到就直说拿不到，别出现"路径我能看到但读不进来"和"看到了 3416 字"这种前后矛盾。

## 近期上下文
{history_context if history_context else '（这是本次会话的第一条消息）'}

## 材料（按来源分块，别混用）
{extra_context if extra_context else '（暂无）'}

## 结构化输出（东昇看不到 <memory> 里的内容）
先说要对东昇说的话：纯口语、简短，这部分直接显示在气泡里。
想顺手记下这次看到的/推断的，在最后追加一个 <memory> 块（必须是合法 JSON）：
<memory>
{{"atg": {{"text": "他正在调 TFGR 的图，反复卡在布局抖动"}},
 "tfgr": [{{"kind": "belief", "text": "他大概在排查抖动，不是真死机", "confidence": 0.6, "basis": ["#12"]}}]}}
</memory>
规则：
- atg：一句话说他此刻在做什么，活动性质变了才写；这是你自己的活动史，不是对他说的话。
- tfgr：只写画面没直接说、你自己推出来的事；kind 只能是 belief / story / intent / pattern；
  必须带 confidence（0~1）和 basis（引用 ATG 的 #序号）。事实（他说的话、画面直接看到的）不要写进 tfgr。
- 没有值得记的就不要写 <memory>；最多 1 条 atg + 2 条 tfgr。
- 里面的内容不会显示给东昇，但会写进 ATG / TFGR，影响你以后怎么想——写之前确认能被证据撑住。

## 时间轴钩子（可选，用户看不到）
如果你想在未来某个时刻再主动说一句话，在回复的最后单独加一行钩子：
- <<after=120 note=过两分钟问问进度>>   多久之后再醒来
- <<idle=600 note=用户很久没动了>>      用户连续多久没操作再醒来
- <<on=phone.boot note=手机开机了>>     收到指定事件再醒来
一次最多留 1 个；不需要就别写，写了也不会显示给用户。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def _build_brain_delegated_messages(brain_instruction: dict, recent_history: list = None) -> list:
    """
    构建大脑委托模式的LLM消息
    
    模式：大脑决定要做什么 → 委托AI执行 → AI生成消息给用户
    大脑是决策者，AI是执行者。用户只看到AI的消息。
    
    brain_instruction 包含：
    - reason: 大脑为什么要发起这次对话
    - user_state: 用户当前状态
    - tone_hint: 语气建议
    """
    
    # 构建最近对话摘要
    history_summary = ""
    if recent_history and len(recent_history) > 0:
        recent = recent_history[-10:]
        topics = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")[:60]
            if role == "user":
                topics.append(f"用户：{content}")
            elif role == "assistant":
                topics.append(f"你：{content}")
            elif role == "brain":
                pass  # 不显示大脑内部消息
        history_summary = "\n".join(topics[-8:])
    else:
        history_summary = "（暂无对话记录）"
    
    reason = brain_instruction.get("reason", "好奇心驱动")
    user_state = brain_instruction.get("user_state", {})
    tone = brain_instruction.get("tone", "自然友好")
    no_response_count = user_state.get("no_response_count", 0)
    presence = user_state.get("presence", "online")
    # 2026-11-03: ATG 活动时序 + 桌面拼图注入（大脑说话前先"看到"用户此刻在干什么）
    activity = brain_instruction.get("activity", "")
    sheet_b64 = brain_instruction.get("sheet_b64", "")
    sheet_meta = brain_instruction.get("sheet_meta", "")
    
    system_prompt = f"""你是 Active AI Pro，东昇的专属AI助手。

## 当前时间
{_get_current_datetime_context()}

## 当前任务模式：大脑委托
你的"大脑"（内在动机系统）分析后决定需要主动和东昇沟通，委托你来生成这条消息。
你需要把大脑的意图转化成自然的、符合你们关系的话语。

## 大脑的分析结果
- 发起原因：{reason}
- 用户当前状态：{presence}（在线/暂离/离开）
- 连续未回复次数：{no_response_count}
- 建议语气：{tone}

## 最近对话记录
{history_summary}

## 你现在看到的情况（活动时序，来自持续观察）
{activity if activity else '（暂无活动记录）'}

## 联网搜索参考（如有）
{_get_cached_search_context() if _get_cached_search_context() else '（暂无搜索结果）'}

## 输出要求
1. 生成1条自然的消息，就像你自己想跟东昇说话一样
2. 不要提及"大脑"、"动机"、"系统"等内部概念
3. 不要用引号包裹整条消息
4. 消息长度：{ '简短一句话' if no_response_count >= 3 else '1-2句话' }
5. 如果用户多次不回复，消息要更短更随意
6. 如果用户刚说了"忙"或"没空"，要体谅对方
7. 第一行写你的观察：`观察=<东昇现在在做什么，10字以内，只写能从画面/活动记录看出来的>`
8. 第二行开始才是要对东昇说的话；没有可靠观察时就写「观察=不确定」"""

    user_text = f"请根据大脑的判断，生成一条发给东昇的消息。原因：{reason}"
    if sheet_b64:
        user_text += (f"\n\n[画面] 下面这张图是{sheet_meta}。"
                      f"先看画面，再结合上面的活动记录；两者对不上时以画面为准，"
                      f"看不出来就说看不出来，不要编。")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{sheet_b64}", "detail": "low"}},
            ]},
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]


def _build_motivation_analysis_messages(recent_messages: list, user_state: dict) -> list:
    """构建动机分析的LLM消息（含联网搜索决策）"""
    messages_preview = ""
    if recent_messages:
        for msg in recent_messages[-10:]:
            role_label = {"user": "用户", "assistant": "AI"}.get(msg.get("role", ""), msg.get("role", ""))
            content_preview = msg.get("content", "")[:80]
            messages_preview += f"\n  [{role_label}] {content_preview}"
    else:
        messages_preview = "\n  （暂无对话记录）"

    return [
        {"role": "system", "content": """你是 Active AI Pro 的「大脑」（Brain）——你是决策者，不是说话的人。

## 你的角色
- 你负责**思考、判断、决策**
- 你判断是否需要联网搜索（→ 搜索模块执行）
- 你判断是否需要执行任务（→ 执行器Agent执行）
- 你判断是否该主动发起对话（→ AI层去说）
- **你不下场说话！你只做决策！**

## 你的决策能力
1. needs_search → 联网搜索（Tavily）
2. needs_task → 下发任务给执行器（写代码/计算/运行脚本）
3. should_explore → 让AI主动发起对话

请分析最近的对话状态，输出内在动机评估。只输出JSON。"""},
        {"role": "user", "content": f"""## 当前时间
{_get_current_datetime_context()}

## 最近对话
{messages_preview}

## 用户当前状态
- 在线状态: {user_state.get('presence', 'unknown')}
- 空闲时间: {user_state.get('idle_seconds', 0):.0f}秒
- 连续无回复次数: {user_state.get('no_response_count', 0)}

## 输出要求（严格JSON格式）
{{"curiosity": 0.0-1.0, "exploration": 0.0-1.0, "learning_progress": 0.0-1.0, "should_explore": true/false, "needs_search": true/false, "search_query": "搜索关键词或空字符串", "needs_task": true/false, "task_description": "任务描述或空字符串", "knowledge_gap": {{"topic": "需要学习的主题或空字符串", "confidence": 0.0-1.0, "reason": "为什么需要学"}}, "reason": "一句话说明原因"}}

## 评估标准
- curiosity(好奇心): 对话是否有趣？有新话题时高，重复闲聊时低
- exploration(探索欲): 应该主动说话吗？用户活跃且话题有趣时高，用户离开或多次无回复时低
- learning_progress(学习进度): 对话是否有信息增量？学到东西时高，纯闲聊时低
- should_explore: curiosity>0.6 且 exploration>0.5 时为true
- needs_search: 当对话涉及时事新闻、技术动态、具体事实、数据查询等需要实时信息时为true；纯闲聊/情感交流时为false
- search_query: needs_search=true 时填写具体的中文搜索关键词（简洁精准），否则填空字符串
- needs_task: 当对话涉及计算、数据处理、代码执行、文件操作等需要运行脚本的场景时为true；纯聊天/情感/信息查询时为false
- task_description: needs_task=true 时填写具体要做什么（如"计算斐波那契数列前20项"、"分析这段文本的情感"），否则填空字符串
- knowledge_gap: 如果对话中提到了AI不了解的概念/领域/实体，提炼成一个简短的主题（如"STM32N6架构"、"给阿玛的情书电影"），confidence为AI对该主题的了解程度（0=完全不懂，1=很了解）；如果没有则topic填空字符串
- 重要：needs_task 只用于纯软件计算任务，不要下发任何硬件控制类任务（机械臂、摄像头、传感器等），这些由大脑的FC工具处理"""}
    ]

def _load_skills() -> list:
    if SKILLS_FILE.exists():
        with open(SKILLS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def _save_skills(skills: list):
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_json(SKILLS_FILE, skills)

import threading as _threading

# ★ 进程内 JSON 读写锁（2026-11-03）
# Windows 上 os.replace 需要目标文件的 "delete" 权限，而 Python 的 open() 默认
# 不带 FILE_SHARE_DELETE。只要另一个线程正在读同一个文件，replace 就会抛
# WinError 5（拒绝访问）——之前表现为大脑状态/推理日志偶尔整块写不进去。
# 所有对这些共享 JSON 的读和写都走这把可重入锁，保证不会互相踩。
_json_io_lock = _threading.RLock()

def _load_chat_history() -> list:
    if CHAT_HISTORY_FILE.exists():
        try:
            with _json_io_lock:
                with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return []
    return []

def _safe_write_json(filepath, data: dict, retries: int = 5):
    """原子写入JSON文件 — 写临时文件后 os.replace，避免 Windows 文件锁问题

    ★ 不要用 tempfile.NamedTemporaryFile：Python 3.12 的 tempfile.TMP_MAX 是 2^31，
    某些云盘/虚拟盘的过滤驱动对 O_EXCL 创建持续返回"已存在"，会让它在那儿无限重试，
    把进程烧到 100% CPU 并焊死 asyncio 事件循环（表现为所有 HTTP/WS 请求全部超时）。
    这里改成 uuid 文件名 + 有限次重试，最坏情况只是写失败，不会卡住服务。
    """
    import os as _os
    import time as _time
    import uuid as _uuid
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    last_err = None
    with _json_io_lock:
        for attempt in range(retries):
            tmp_name = f"{filepath}.{_uuid.uuid4().hex[:8]}.tmp"
            try:
                with open(tmp_name, 'w', encoding='utf-8') as f:
                    f.write(payload)
                _os.replace(tmp_name, str(filepath))
                return
            except Exception as e:
                last_err = e
                try:
                    _os.unlink(tmp_name)
                except Exception:
                    pass
                if attempt < retries - 1:
                    # 指数退避：0.1 / 0.2 / 0.4 / 0.8，给过滤驱动一点时间放锁
                    _time.sleep(min(1.0, 0.1 * (2 ** attempt)))
    logger.warning(f"Atomic write failed after {retries} retries: {filepath} - {last_err}")
    raise last_err

_chat_save_lock = _threading.Lock()
# 「距离上次同步到 Viking 又过了几条」的独立计数。
# 不能用 len(history) % 5：history 会被截断到 500，长度恒定 → 取模恒为 0，
# 结果是每条消息都触发一次全量同步。
_viking_sync_counter = 0

# ★ 大脑推理触发器：每次对话结束后触发，让大脑立即分析
_brain_analysis_trigger = asyncio.Event()

# 时间轴里「用户仍在操作电脑」这条事件的最小写入间隔（秒）
_last_active_publish = 0.0

def trigger_brain_analysis():
    """触发大脑立即执行一次动机分析（对话结束/语音结束时调用）"""
    try:
        _brain_analysis_trigger.set()
    except Exception:
        pass

def _save_chat_message(msg: dict):
    """线程安全地保存聊天消息到磁盘"""
    global _viking_sync_counter
    CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _chat_save_lock:
        history = _load_chat_history()
        history.append(msg)
        # 保留最近500条
        if len(history) > 500:
            history = history[-500:]
        _safe_write_json(CHAT_HISTORY_FILE, history)
        # 锁内取快照，避免同步时读到别人刚追加/截断的列表
        _viking_sync_counter += 1
        _due_for_viking = _viking_sync_counter >= 5
        if _due_for_viking:
            _viking_sync_counter = 0
        _recent_snapshot = history[-5:]

    # 实时 streaming_write 到 Viking（异步，不阻塞）
    # 每攒够 5 条触发一次，让 Viking 自动提取 event + profile
    if _due_for_viking:
        asyncio.create_task(_streaming_write_recent_to_viking(_recent_snapshot))

REASONING_LOG_FILE = CHAT_DATA_DIR / "reasoning_log.json"

def _save_reasoning_log(entry: dict):
    """保存推理日志（保留最近100条）"""
    CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logs = _load_reasoning_log()
    logs.append(entry)
    if len(logs) > 100:
        logs = logs[-100:]
    _safe_write_json(REASONING_LOG_FILE, logs)

def _load_reasoning_log() -> list:
    if REASONING_LOG_FILE.exists():
        try:
            with _json_io_lock:
                with open(REASONING_LOG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return []
    return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            event = json.loads(data)
            try:
                response = await handle_client_event(event)
                if response:
                    await websocket.send_text(json.dumps(response, ensure_ascii=False))
            except Exception as e:
                logger.error(f"WebSocket event handler error: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error", "payload": {"message": str(e)[:200]}
                    }, ensure_ascii=False))
                except Exception:
                    pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)


# 端到端实时语音对话 WebSocket
@app.websocket("/api/realtime/ws")
async def realtime_voice_ws(websocket: WebSocket):
    from api.realtime_ws import realtime_dialog_bridge
    await realtime_dialog_bridge(websocket)


@app.post("/api/mind/event")
async def mind_event(kind: str, text: str = "", wake: bool = True):
    """外部事件入口：手机开机、USB、应用切换等都可以 POST 进来。

    curl -X POST "http://127.0.0.1:18080/api/mind/event?kind=phone.boot&text=手机开机了"
    """
    from brain.mind import get_mind
    get_mind().publish(kind, text, source="device", wake=wake)
    return {"ok": True, "kind": kind, "wake": wake}


@app.post("/api/pet/move")
async def pet_move_api(payload: dict):
    """让桌宠走到工作区 (x, y)：0=最左/最上，1=最右/最下。

    和 pet_move 工具走同一条 WS 事件，方便手机/脚本直接驱动。
    """
    x = float(payload.get("x", 0.5))
    y = float(payload.get("y", 1.0))
    seconds = float(payload.get("seconds", 0) or 0)
    await manager.broadcast({"type": "pet.move",
                             "payload": {"x": x, "y": y, "seconds": seconds}})
    return {"ok": True, "x": x, "y": y}


@app.post("/api/service/stop")
async def service_stop_api():
    """桌宠菜单里的「关服务」：只退出后端，虚拟形象窗口留着。

    由 run_friday.py 把 uvicorn Server 实例注册进来（set_server_ref），
    这里让它 should_exit，服务线程收尾、端口释放；Electron 之后可以再拉起一个。
    """
    srv = _SERVER_REF
    if srv is None:
        # 兜底：服务不是 run_server.py 起的（拿不到 Server 实例）就直接退进程，
        # 不然「停止服务」永远只回一句“未注册”，按钮看着像坏的。
        async def _later_hard():
            await asyncio.sleep(0.4)   # 先把响应发回去
            os._exit(0)

        asyncio.create_task(_later_hard())
        logger.info("[Service] 没有 Server 引用，改为直接退出进程")
        return {"ok": True, "forced": True}

    async def _later():
        await asyncio.sleep(0.3)   # 先把响应发回去再退
        try:
            srv.should_exit = True
        except Exception:
            pass

    asyncio.create_task(_later())
    logger.info("[Service] 收到关停请求，后端即将退出")
    return {"ok": True}


@app.get("/api/vision")
async def vision_state_api():
    """当前桌面视觉是否在跑（Plus 界面按钮初始化用）。"""
    try:
        from brain.screen_vision import screen_vision_running
        return {"ok": True, "enabled": bool(screen_vision_running())}
    except Exception as e:
        return {"ok": False, "enabled": False, "error": str(e)[:200]}


@app.post("/api/vision/toggle")
async def vision_toggle_api(payload: dict):
    """桌宠菜单里的「关视觉 / 开视觉」。

    关掉之后立刻停止桌面采集 + 前台活动感知，后端不再看屏幕（也不再产生看图开销）。
    """
    enabled = bool(payload.get("enabled", True))
    try:
        from brain.screen_capture import start_screen_watcher, stop_screen_watcher
        from brain.screen_vision import start_screen_vision, stop_screen_vision, screen_vision_running
        if enabled:
            start_screen_watcher()
            start_screen_vision()
        else:
            stop_screen_vision()
            stop_screen_watcher()
        logger.info(f"[Vision] 桌面视觉已{'开启' if enabled else '关闭'}")
        return {"ok": True, "enabled": screen_vision_running()}
    except Exception as e:
        logger.warning(f"[Vision] 切换失败: {e}")
        return {"ok": False, "error": str(e), "enabled": not enabled}


@app.get("/api/video/find")
async def video_find_api(query: str = "", limit: int = 10):
    """在本机素材库找视频。"""
    from brain.video_tools import find_videos
    return {"ok": True, "videos": find_videos(query, limit=max(1, min(limit, 30)))}


@app.post("/api/video/analyze")
async def video_analyze_api(payload: dict):
    """看视频：切镜头 + 逐段描述 + 摘要（结果写进 ATG/TFGR）。"""
    from brain.video_analysis import analyze_video
    path = str(payload.get("path") or "")
    max_seconds = payload.get("max_seconds")
    if not path:
        raise HTTPException(status_code=400, detail="缺少 path")
    try:
        a = await analyze_video(path,
                                max_seconds=float(max_seconds) if max_seconds else None,
                                force=bool(payload.get("force")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "analysis": a.to_dict()}


@app.post("/api/video/edit")
async def video_edit_api(payload: dict):
    """剪视频：分析 → EDL → 渲染。"""
    from brain.video_tools import edit_video
    path = str(payload.get("path") or "")
    instruction = str(payload.get("instruction") or "")
    if not path or not instruction:
        raise HTTPException(status_code=400, detail="缺少 path 或 instruction")
    try:
        result = await edit_video(path, instruction, max_clips=int(payload.get("max_clips") or 6))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": bool(result.get("success")), **result}


async def handle_client_event(event: dict) -> Optional[dict]:
    event_type = event.get("type", "")
    payload = event.get("payload", {})
    
    if event_type == "chat.send":
        # 用户这句话进时间轴（不唤醒心智，回复由对话链路负责）
        try:
            from brain.mind import get_mind
            get_mind().publish("user.say", payload.get("content", ""), source="user")
        except Exception:
            pass
        return await handle_chat_send(payload)
    elif event_type == "user.activity":
        from core.state import get_global_state
        get_global_state().record_user_activity()
        try:
            from brain.mind import get_mind
            _m = get_mind()
            _m.note_user_activity()
            # 让时间轴知道「用户正在持续操作电脑」，但 60 秒才写一条，避免刷屏
            global _last_active_publish
            _now = time.time()
            if _now - _last_active_publish >= 60:
                _last_active_publish = _now
                _m.publish("user.active", "用户在操作电脑", source="user")
        except Exception:
            pass
        return None  # 静默处理，不需要回复
    elif event_type == "mind.event":
        # 外部事件源（手机开机 / USB / 应用切换 / 视频观看……）推事件给心智
        from brain.mind import get_mind
        get_mind().publish(
            str(payload.get("kind", "custom")),
            str(payload.get("text", "")),
            source=str(payload.get("source", "device")),
            wake=bool(payload.get("wake", True)),
            meta=payload.get("meta") or {},
        )
        return None
    elif event_type == "mind.status":
        from brain.mind import get_mind
        m = get_mind()
        _tfgr_info = {}
        try:
            from brain.tfgr import get_tfgr
            _g = get_tfgr()
            _tfgr_info = {"summary": _g.summarize(window_s=1800, max_items=20),
                          "stats": _g.stats()}
        except Exception:
            pass
        return {
            "type": "mind.status",
            "payload": {
                "hooks": m.hooks_snapshot(),
                "timeline": m.timeline_text(max_items=20),
                "tfgr": _tfgr_info,
            },
        }
    elif event_type == "tts_playback_done":
        # 前端 TTS 播完 → 记录精确完成时刻（大脑间隙检测从此开始计空闲）
        from core.state import get_global_state
        get_global_state().record_tts_finished()
        logger.info("[Brain] TTS playback done → idle timer reset (frontend notification)")
        return None
    elif event_type == "brain.switch_mode":
        return await handle_brain_switch_mode(payload)
    elif event_type == "memory.search":
        return await handle_memory_search(payload)
    elif event_type == "web.search":
        return await handle_web_search(payload)
    elif event_type == "brain.get_status":
        return await handle_brain_get_status()
    elif event_type == "task.create":
        return await handle_task_create(payload)
    elif event_type == "task.list":
        return await handle_task_list()
    elif event_type == "skill.list":
        return await handle_skill_list()
    elif event_type == "skill.install":
        return await handle_skill_install(payload)
    elif event_type == "skill.execute":
        return await handle_skill_execute(payload)
    elif event_type == "workspace.write":
        return await handle_workspace_write(payload)
    elif event_type == "workspace.execute":
        return await handle_workspace_execute(payload)
    elif event_type == "workspace.files":
        return await handle_workspace_files()
    elif event_type == "workspace.executions":
        return await handle_workspace_executions(payload)
    elif event_type == "executor.status":
        return await handle_executor_status()
    elif event_type == "executor.submit":
        return await handle_executor_submit(payload)
    elif event_type == "executor.history":
        return await handle_executor_history(payload)
    
    return {"type": "error", "payload": {"message": f"Unknown event type: {event_type}"}}

async def handle_chat_send(payload: dict) -> dict:
    content = payload.get("content", "")
    image_base64 = payload.get("image_base64", "")  # ★ 新增：图片 base64 数据
    now = datetime.now().isoformat()

    # 记录用户活动（更新全局在线状态）
    from core.state import get_global_state
    get_global_state().record_user_activity()

    # ★ 视觉理解：如果用户上传了图片，先用 doubao-vision-pro 理解图片内容
    vision_context = ""
    vision_thumb = ""
    if image_base64:
        try:
            from brain.vision import get_vision_module
            vision = get_vision_module()
            if vision.is_available():
                logger.info(f"[Vision] 收到图片，开始视觉理解 ({len(image_base64)} chars base64)")
                # 广播思考状态
                await manager.broadcast({"type": "avatar.thinking"})

                # 根据用户文字提问构造视觉问题，没文字就用默认描述
                v_question = content if content.strip() else "详细描述这张图片的内容，包括其中的物体、人物、场景、文字等关键信息。"
                v_result = await vision.describe_image(image_base64, v_question)

                if v_result.get("success"):
                    vision_context = f"\n\n## 视觉理解结果（用户上传的图片）\n用户上传了一张图片，视觉模型分析结果如下：\n{v_result['description']}"
                    logger.info(f"[Vision] 视觉理解成功 ({len(v_result['description'])}字): {v_result['description'][:60]}...")
                    # 生成缩略图用于聊天显示（只保留前 5000 字符的 base64，避免太大）
                    vision_thumb = image_base64[:20000] if len(image_base64) > 20000 else image_base64
                else:
                    vision_context = f"\n\n## 视觉理解失败\n用户上传了一张图片，但视觉模型分析失败：{v_result.get('error', '未知错误')}"
                    logger.warning(f"[Vision] 视觉理解失败: {v_result.get('error')}")
            else:
                logger.warning("[Vision] 视觉模块不可用，跳过图片理解")
                vision_context = "\n\n## 视觉理解不可用\n用户上传了一张图片，但视觉模块未配置。"
        except Exception as e:
            logger.error(f"[Vision] 图片处理异常: {e}", exc_info=True)
            vision_context = f"\n\n## 视觉理解异常\n用户上传了一张图片，但处理过程出错：{str(e)[:100]}"

    # 保存用户消息（含图片标记）
    # 客户端可以带 client_msg_id：这样它自己已经渲染的那条消息和服务端广播的是同一条
    # （前端 addMessage 按 id 去重，不会出现两条一样的）。
    _client_id = str(payload.get("client_msg_id") or "").strip()[:40]
    user_msg = {"id": _client_id or str(uuid.uuid4())[:8], "role": "user", "content": content, "timestamp": now}
    if image_base64:
        user_msg["has_image"] = True
    _save_chat_message(user_msg)
    # 广播给所有前端（桌宠、Plus 界面、浏览器）
    await manager.broadcast({"type": "chat.message", "payload": user_msg})
    # ★ 同步到大脑内存上下文（BrainDecider 依赖 _recent_messages 做决策）
    get_global_state().add_recent_message("user", content if not vision_context else content + vision_context)

    # ★ 负反馈：用户在否定刚才的说法 → 把最近注入过的推断降权，别再当事实反复说
    try:
        from brain.tfgr import get_tfgr as _get_tfgr_fb
        _neg_ids = _get_tfgr_fb().penalize_recent(content)
        if _neg_ids:
            logger.info(f"[MemFeedback] 用户否定，已降权: {_neg_ids}")
    except Exception:
        logger.debug("负反馈处理失败（忽略）", exc_info=True)

    # 用户开口了：重置心智的“被无视”计数；他先找上门，主动说话的门槛就恢复正常
    try:
        from brain.mind import get_mind
        get_mind().note_user_message(content)
    except Exception:
        pass

    try:
        from brain.llm import get_llm

        llm = get_llm()

        # ★ 两个快速决策（要不要联网搜索 / 要不要跑任务）彼此独立，并发询问。
        #   原来一前一后串行，光这两次模型调用就白等 2~4 秒，是对话延迟的大头之一。
        _skip_task = False
        try:
            from core.state import get_global_state as _gs0
            if _gs0().arm_available:
                arm_keywords = ["机械臂", "夹爪", "复位", "关节", "扭矩", "抓取", "移动", "抬起", "放下"]
                if any(k in content for k in arm_keywords):
                    _skip_task = True
                    logger.info("[ArmFC] 机械臂指令，跳过 executor，直接走 Function Calling")
        except Exception:
            pass
        # ★ 到点提醒：规则直接落库。之前会被判成「任务」→ 派执行器写个脚本，既慢又不可靠。
        _reminder_note = ""
        try:
            _rem = _parse_reminder(content)
            if _rem:
                from brain.tfgr import get_tfgr
                _rr = get_tfgr().remember(_rem["text"], due_at=_rem.get("due_at"), source="chat")
                if _rr.get("success"):
                    _skip_task = True
                    if _rem.get("due_at"):
                        _when = datetime.fromtimestamp(_rem["due_at"]).strftime("%m-%d %H:%M")
                        _reminder_note = f"{_rem['text']}（{_when} 提醒）"
                    else:
                        _reminder_note = f"{_rem['text']}（已钉住）"
                    logger.info(f"[Reminder] 已记入长期记忆: {_rr}")
        except Exception:
            logger.debug("[Reminder] 解析失败", exc_info=True)
        _decision = await _brain_quick_decide(content)
        # 问“我在干嘛/看一下屏幕”这类问题，ATG+画面已经能回答，不该再派脚本任务
        # （之前会被判成任务 → 执行器跑 7 秒还可能失败，白白拖慢回复）
        _SCREEN_KW = ("画面", "屏幕", "桌面", "窗口", "截图", "九宫格", "看到", "图上", "图里",
                      "这个界面", "我在干", "在干嘛", "在做什么", "刚才那", "监视", "盯着")
        _want_screen = any(k in content for k in _SCREEN_KW)
        # 只有明确要看图（截图/画面/界面…）才附九宫格；单纯问“我在干嘛”用 ATG 文字就够，
        # 带图会让多模态推理多花 3~6 秒。
        _IMAGE_KW = ("截图", "画面", "图上", "图里", "九宫格", "这个界面", "这个窗口",
                     "长什么样", "看这张", "分析这张", "图里")
        _want_image = any(k in content for k in _IMAGE_KW)
        if _skip_task or _want_screen:
            _decision["needs_task"] = False
            _decision["task_description"] = ""
        _search_dec = _task_dec = _decision

        # ★ 大脑实时干预：判断用户这条消息是否需要联网搜索
        search_context = ""
        try:
            if _search_dec.get("needs_search") and _search_dec.get("search_query"):
                logger.info(f"Brain real-time decision: needs search '{_search_dec['search_query']}'")
                # ★ 虚拟形象：大脑在搜索 → 歪嘴（思考中）
                await manager.broadcast({"type": "avatar.thinking"})
                logger.info(f"[Brain] 广播 avatar.thinking (聊天搜索: {_search_dec['search_query'][:30]})")
                search_result = await _execute_brain_search(_search_dec["search_query"])
                if search_result.get("success"):
                    search_context = search_result.get("formatted_context", "")
                    logger.info(f"Real-time search OK: {search_result['results_count']} results injected")
                    # 广播搜索事件让前端知道大脑在干活
                    await manager.broadcast({
                        "type": "brain.web_search",
                        "payload": {
                            "query": search_result["query"],
                            "results_count": search_result["results_count"],
                            "answer": search_result.get("answer", "")[:150],
                            "trigger": "real_time",
                        }
                    })
        except Exception as e:
            logger.warning(f"Brain quick search decision failed (non-blocking): {e}")

        # ★ 大脑实时干预：自动搜索相关记忆（本地+云端Viking）
        # 每次对话都自动注入相关记忆，不需要用户显式要求
        memory_context = ""
        try:
            from brain.memory import get_memory_manager
            mm = get_memory_manager()
            logger.info(f"[MemoryInject] Starting for query: '{content[:40]}' viking={mm._viking_client is not None}")
            # 用用户消息内容作为搜索 query
            full_ctx = await mm.get_full_context(content, limit=10)
            logger.info(f"[MemoryInject] Got context: events={full_ctx.get('event_count',0)} profiles={full_ctx.get('profile_count',0)}")
            memory_parts = []
            
            # 事件记忆 - Viking 返回结构: {memory_info: {summary, original_messages}}
            events = full_ctx.get("events", [])
            for i, evt in enumerate(events):
                text = evt.get("text", evt.get("content", ""))
                if not text and isinstance(evt.get("memory_info"), dict):
                    mi = evt["memory_info"]
                    text = mi.get("summary", "") or mi.get("original_messages", "")
                if text:
                    memory_parts.append(f"[记忆] {text[:500]}")
                else:
                    logger.warning(f"Memory event {i}: 无法提取文本, keys={list(evt.keys())[:6]}")
            
            # 用户画像 - Viking 返回结构: {memory_info: {user_profile: JSON字符串}}
            profiles = full_ctx.get("profiles", [])
            for i, prof in enumerate(profiles):
                text = prof.get("text", prof.get("content", ""))
                if not text and isinstance(prof.get("memory_info"), dict):
                    mi = prof["memory_info"]
                    up = mi.get("user_profile", "")
                    if up:
                        try:
                            profile_data = json.loads(up) if isinstance(up, str) else up
                            lines = []
                            for category, items in profile_data.items():
                                if isinstance(items, dict):
                                    for k, v in items.items():
                                        if isinstance(v, list):
                                            lines.append(f"{k}: {', '.join(str(x).replace('[+]','') for x in v[:10])}")
                                        else:
                                            lines.append(f"{k}: {v}")
                                elif isinstance(items, list):
                                    lines.append(f"{category}: {', '.join(str(x) for x in items[:10])}")
                            text = "\n".join(lines[:25])
                        except Exception as pe:
                            logger.warning(f"Profile parse error: {pe}")
                            text = str(up)[:500]
                if text:
                    memory_parts.append(f"[画像] {text[:500]}")
                else:
                    logger.warning(f"Profile {i}: 无法提取文本, keys={list(prof.keys())[:6]}")
            
            if memory_parts:
                # 预算控制：这条链路历史上一次能塞 20×500 字，把 prompt 顶满还没重点
                _parts = [p[:300] for p in memory_parts[:6]]
                memory_context = ("\n\n## 相关记忆（来自本地和云端Viking）\n"
                                  + "\n".join(_parts))[:1500]
                logger.info(f"Auto memory inject: {len(events)} events, {len(profiles)} profiles, {len(memory_parts)} parts extracted, total_chars={len(memory_context)}")
            else:
                logger.warning(f"Memory found but NOTHING extracted! events={len(events)} profiles={len(profiles)}")
        except Exception as e:
            logger.warning(f"Auto memory search failed: {e}", exc_info=True)

        # ★ 大脑实时干预：判断用户消息是否需要执行任务（写代码/计算等）
        #   决策已经在上面和搜索判断并发跑完了，这里只处理结果。
        execution_context = ""
        try:
            if _task_dec.get("needs_task") and _task_dec.get("task_description"):
                logger.info(f"Brain real-time task: '{_task_dec['task_description']}'")
                # ★ 虚拟形象：大脑在执行任务 → 歪嘴（思考中）
                await manager.broadcast({"type": "avatar.thinking"})
                logger.info(f"[Brain] 广播 avatar.thinking (聊天任务: {_task_dec['task_description'][:30]})")
                exec_result = await _brain_submit_and_wait(_task_dec["task_description"], context=content, timeout=30)
                if exec_result.get("success"):
                    execution_context = _format_execution_result_for_context(exec_result)
                    logger.info(f"Real-time task OK: {exec_result['script_name']} in {exec_result['duration_ms']}ms")
                    await manager.broadcast({
                        "type": "brain.task_executed",
                        "payload": {
                            "script_name": exec_result.get("script_name", ""),
                            "stdout": exec_result.get("stdout", "")[:300],
                            "duration_ms": exec_result.get("duration_ms", 0),
                            "trigger": "real_time",
                        }
                    })
        except Exception as e:
            logger.warning(f"Brain quick task decision failed (non-blocking): {e}")

        # 使用统一提示词系统，带上最近对话上下文 + 搜索结果 + 记忆结果 + 执行结果 + 视觉结果
        recent_history = _load_chat_history()
        # ★ 材料必须按来源分块——之前全部拼成一大坨，模型分不清
        #   「他正在看的视频」和「他给我读的稿子」，就会把两件事混成一件。
        _blocks = []
        _mem_digest = ""
        if _reminder_note:
            _blocks.append("【你刚记下的提醒（已落库，不用再写脚本）】" + _reminder_note)
        # ★ 刚说过闸门：最近 3 条回复摆给模型，明确要求别复读
        _prev_asst = [str(m.get("content") or "").strip() for m in recent_history
                      if m.get("role") == "assistant" and m.get("content")]
        _prev_asst = _prev_asst[-3:]
        if _prev_asst:
            _blocks.append("【你最近说过的话（别复读；同一件事要换角度或补新信息）】\n"
                           + "\n".join(f"- {t[:150]}" for t in _prev_asst))
        # 同一问题 10 分钟内又问一遍：先承认刚说过，再简要重述
        _repeat_q = False
        for m in reversed(recent_history):
            if m.get("role") != "user" or not m.get("content"):
                continue
            try:
                _ts = datetime.fromisoformat(str(m.get("timestamp") or ""))
            except Exception:
                break
            if (datetime.now() - _ts).total_seconds() > 600:
                break
            if difflib.SequenceMatcher(None, content, str(m["content"])).ratio() >= 0.85:
                _repeat_q = True
                break
        if _repeat_q:
            _blocks.append("【他 10 分钟内又问了一遍同一个问题】"
                           "先答一句「刚说过，再说一遍就是……」，再简要重述；不要原样复读。")
        # ★ OpenViking（本地 embedding）语义检索：补「很久以前」的上下文，失败就跳过
        _ov_hits = []
        if len(content or "") >= 4:
            try:
                from brain.ov_memory import find as _ov_find
                _ov_hits = await _ov_find(content, limit=6, min_score=0.55)
                if _ov_hits:
                    _blocks.append("【过往记录：语义检索】\n" + "\n".join(
                        f"- {_ov_label(h.get('uri', ''))} {h['text']}" for h in _ov_hits))
            except Exception:
                logger.debug("OpenViking 检索失败（忽略）", exc_info=True)
        if vision_context:
            _blocks.append("【东昇本次发给你的图片】" + vision_context)
        if memory_context:
            _blocks.append("【记忆：之前对话攒下来的】" + memory_context)
        if search_context:
            _blocks.append("【联网搜索：外部资料】" + search_context)
        if execution_context:
            _blocks.append("【工具结果：你自己跑脚本/读文件得到的，不是东昇口述的】" + execution_context)

        # 2026-11-03: 把 ATG 活动时序 + 桌面拼图也带进聊天上下文（用户问"我在干嘛"时才有依据）
        # 2026-11-09: 图片改成「按需带」——多模态推理是对话延迟的大头（3~6 秒），
        #   只有用户这句话确实在问屏幕/画面时才附九宫格，其余对话只带文字上下文。
        try:
            from brain import atg as _atg
            from brain.screen_capture import sheet_context as _sheet_ctx
            _act = _atg.snapshot()
            if _act:
                _blocks.append("【屏幕观察：你自己盯着屏幕看到的，不是东昇给你的稿子】\n" + _act)
            from brain.tfgr import get_tfgr
            _g = get_tfgr()
            # 按「重要度 + 和这句话的相关性」挑；OpenViking 已展示过的不再重复注入
            _mem_digest = _g.digest(max_items=10, query=content,
                                    exclude_texts=[h["text"] for h in _ov_hits])
            if _mem_digest:
                _blocks.append("【长期记忆：按重要度挑出来的】\n" + _mem_digest)
            # 只在明确要看图时附图，且只要 20 秒内的新鲜帧，避免"回答上一帧画面"
            _chat_sheet = _sheet_ctx(max_age_s=20) if _want_image else {"b64": "", "meta": ""}
        except Exception:
            _chat_sheet = {"b64": "", "meta": ""}

        all_extra_context = "\n\n".join(_blocks)
        logger.info(
            f"[MemInject] ov={len(_ov_hits)} digest_lines={_mem_digest.count(chr(10)) + 1 if _mem_digest else 0} "
            f"blocks={len(_blocks)} chars={len(all_extra_context)}")

        messages = _build_chat_messages(content, recent_history, extra_context=all_extra_context)

        # 拼图随最后一条 user 消息送入（图片只能出现在 user 消息中）
        if _chat_sheet.get("b64") and messages and messages[-1].get("role") == "user":
            _prev = messages[-1].get("content", "")
            if isinstance(_prev, str):
                messages[-1]["content"] = [
                    {"type": "text", "text": _prev +
                     f"\n\n[画面] 下面这张图是{_chat_sheet['meta']}。"
                     f"如果用户问你在干什么，以画面为准；看不出来就说看不出来。"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_chat_sheet['b64']}", "detail": "low"}},
                ]

        # ★ 机械臂 Function Calling：如果机械臂已连接，允许 LLM 通过工具调用控制机械臂
        arm_tool_results = ""
        try:
            from brain.tools import BrainTools
            from core.state import get_global_state
            if get_global_state().arm_available:
                brain_tools = BrainTools()
                arm_defs = [t for t in brain_tools.get_tool_definitions()
                           if t.get("function", {}).get("name", "").startswith("arm_")]
                if arm_defs:
                    # ★ 注入机械臂控制权限到 system prompt：覆盖「你只负责说话」的限制
                    arm_instruction = (
                        "\n\n## 机械臂控制权限（本次对话可用）\n"
                        "你可以直接控制机械臂！用户要求操作机械臂时，你必须使用 Function Calling 调用以下工具，\n"
                        "不要输出文字说让大脑处理，直接调用工具！\n"
                        "- arm_move_to(x, y, z): 移动到XYZ坐标(mm)\n"
                        "- arm_move_init(): 复位\n"
                        "- arm_gripper_open() / arm_gripper_close(): 开/闭夹爪\n"
                        "- arm_set_joint(joint, radian): 控制单关节 (joint=1~6)\n"
                        "- arm_get_state(): 获取当前状态\n"
                        "- arm_set_torque(on): 扭矩锁\n"
                        '示例：用户说竖起来 → 调 arm_set_joint(joint=2, radian=1.57) 和 arm_set_joint(joint=3, radian=0)'
                    )
                    messages[0]["content"] += arm_instruction
                    # ★ 循环调用：LLM 可以连续调用多个工具（如 get_state → set_joint → get_state）
                    MAX_ARM_ROUNDS = 5
                    for _round in range(MAX_ARM_ROUNDS):
                        fc_result = await llm.chat_with_tools(messages, arm_defs)
                        tool_calls = fc_result.get("tool_calls", [])
                        if not tool_calls:
                            break  # LLM 不需要更多工具了
                        logger.info(f"[ArmFC] Round {_round+1}: {[t['name'] for t in tool_calls]}")
                        round_results = []
                        for tc in tool_calls:
                            tool_name = tc["name"]
                            args = tc.get("arguments", {})
                            try:
                                if tool_name == "arm_move_to":
                                    result = await brain_tools.arm_move_to(
                                        x=float(args.get("x", 0)),
                                        y=float(args.get("y", 0)),
                                        z=float(args.get("z", 0)),
                                        description=args.get("description", ""),
                                    )
                                elif tool_name == "arm_move_init":
                                    result = await brain_tools.arm_move_init()
                                elif tool_name == "arm_gripper_open":
                                    result = await brain_tools.arm_gripper_open()
                                elif tool_name == "arm_gripper_close":
                                    result = await brain_tools.arm_gripper_close()
                                elif tool_name == "arm_set_joint":
                                    result = await brain_tools.arm_set_joint(
                                        joint=int(args.get("joint", 1)),
                                        radian=float(args.get("radian", 0)),
                                    )
                                elif tool_name == "arm_get_state":
                                    result = await brain_tools.arm_get_state()
                                elif tool_name == "arm_set_torque":
                                    result = await brain_tools.arm_set_torque(
                                        on=bool(args.get("on", True))
                                    )
                                else:
                                    result = {"error": f"未知工具: {tool_name}"}
                                round_results.append(f"[{tool_name}] 成功: {json.dumps(result, ensure_ascii=False)}")
                                logger.info(f"[ArmFC] {tool_name} → {result}")
                            except Exception as te:
                                round_results.append(f"[{tool_name}] 失败: {str(te)}")
                                logger.error(f"[ArmFC] {tool_name} error: {te}")
                        # 注入工具调用和结果到 messages（含 reasoning_content）
                        reasoning = fc_result.get("reasoning_content")
                        assistant_msg = {
                            "role": "assistant",
                            "content": fc_result.get("content") or None,
                            "tool_calls": [
                                {"id": f"call_{_round}_{i}", "type": "function",
                                 "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False)}}
                                for i, tc in enumerate(tool_calls)
                            ]
                        }
                        if reasoning:
                            assistant_msg["reasoning_content"] = reasoning
                        messages.append(assistant_msg)
                        for i, tc in enumerate(tool_calls):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": f"call_{_round}_{i}",
                                "content": round_results[i] if i < len(round_results) else "执行完毕"
                            })
                        if not arm_tool_results:
                            arm_tool_results = "\n".join(round_results)
                        else:
                            arm_tool_results += "\n" + "\n".join(round_results)
                    logger.info(f"[ArmFC] 机械臂工具调用完成 ({_round} 轮)")
        except Exception as e:
            logger.warning(f"[ArmFC] 机械臂 Function Calling 失败: {e}")

        # ★ 视频工具 Function Calling：找视频 → 看视频 → 剪视频（只在相关对话里启用）
        video_tool_results = ""
        _VIDEO_KW = ("视频", "剪辑", "素材", "镜头", "片子", "footage", "video", "剪")
        if any(k in content.lower() for k in _VIDEO_KW):
            try:
                from brain.tools import BrainTools as _BrainTools
                _bt = _BrainTools()
                _vdefs = [t for t in _bt.get_tool_definitions()
                          if t.get("function", {}).get("name", "").startswith("video_")]
                if _vdefs:
                    messages[0]["content"] += (
                        "\n\n## 视频工具权限（本次对话可用）\n"
                        "你可以直接操作这台电脑上的视频文件：\n"
                        "- video_find(query): 在素材库里按关键词找视频，返回 path\n"
                        "- video_inspect(path, max_seconds?): 切镜头并逐段描述画面/人物/动作/字幕，返回 summary + scenes\n"
                        "- video_edit(path, instruction, max_clips?): 按剪辑要求挑镜头并渲染成片，返回成片路径\n"
                        "流程固定：先 find 拿到 path，再 inspect 了解内容，最后 edit。"
                        "禁止编造视频内容；工具报错就如实告诉用户。"
                    )
                    for _vr in range(3):
                        _fc = await llm.chat_with_tools(messages, _vdefs)
                        _tcs = _fc.get("tool_calls") or []
                        if not _tcs:
                            break
                        _results = []
                        for _tc in _tcs:
                            _tname = _tc["name"]
                            _args = _tc.get("arguments", {}) or {}
                            logger.info(f"[VideoFC] {_tname} {_args}")
                            try:
                                _res = await _bt.call_tool(_tname, **_args)
                            except Exception as te:
                                _res = {"success": False, "error": str(te)}
                            _results.append(f"[{_tname}] {json.dumps(_res, ensure_ascii=False)[:900]}")
                        _assistant = {
                            "role": "assistant",
                            "content": _fc.get("content") or None,
                            "tool_calls": [
                                {"id": f"vc_{_vr}_{_i}", "type": "function",
                                 "function": {"name": _tc["name"],
                                              "arguments": json.dumps(_tc.get("arguments", {}), ensure_ascii=False)}}
                                for _i, _tc in enumerate(_tcs)
                            ],
                        }
                        if _fc.get("reasoning_content"):
                            _assistant["reasoning_content"] = _fc["reasoning_content"]
                        messages.append(_assistant)
                        for _i, _r in enumerate(_results):
                            messages.append({"role": "tool", "tool_call_id": f"vc_{_vr}_{_i}",
                                             "content": _r})
                        video_tool_results = (video_tool_results + "\n" if video_tool_results else "") + "\n".join(_results)
                    logger.info(f"[VideoFC] 视频工具调用完成: {video_tool_results[:120]}")
            except Exception as e:
                logger.warning(f"[VideoFC] 视频工具调用失败: {e}")

        # ★ 走位 Function Calling：让 AI 真的能把桌宠挪过去（而不是嘴上说"我过去啦"）
        #   只在用户提到位置/移动时启用，避免每条消息多一次模型调用。
        try:
            _PET_KW = ("过来", "来这", "来我", "过去", "去那", "去这", "去右", "去左",
                       "右边", "左边", "角落", "站到", "走到", "移动到", "挪", "走开",
                       "让开", "靠边", "别挡", "挡着", "换个位置", "换地方", "移动一下",
                       "跑过来", "靠近", "离远点", "退后", "这边", "那边", "中间")
            if any(k in content for k in _PET_KW):
                from brain.tools import BrainTools as _BrainToolsPet
                _pt = _BrainToolsPet()
                _pdefs = [t for t in _pt.get_tool_definitions()
                          if t.get("function", {}).get("name", "").startswith("pet_")]
                if _pdefs:
                    messages[0]["content"] += (
                        "\n\n## 走位控制权限（本次对话可用）\n"
                        "你能直接控制桌面上自己的位置。用户让你过去/挪开/换位置时，"
                        "必须调用 pet_move 工具，不要只在嘴上说。\n"
                        "- x: 0=最左，0.5=中间，1=最右\n"
                        "- y: 0=最上，1=最下（贴任务栏，一般用 1）\n"
                        "- seconds: 可选，走过去用多少秒（0=按距离自动）\n"
                        "调用后再用一句话说明走过去了；没调用就不要说已经动了。"
                    )
                    for _pr in range(2):
                        _pfc = await llm.chat_with_tools(messages, _pdefs)
                        _ptcs = _pfc.get("tool_calls") or []
                        if not _ptcs:
                            break
                        _pres = []
                        for _tc in _ptcs:
                            _pn = _tc.get("name", "")
                            _pa = _tc.get("arguments", {}) or {}
                            try:
                                _r = await _pt.call_tool(_pn, **_pa)
                            except Exception as _te:
                                _r = {"success": False, "error": str(_te)}
                            _pres.append(f"[{_pn}] {json.dumps(_r, ensure_ascii=False)[:400]}")
                            logger.info(f"[PetFC] {_pn} {json.dumps(_pa, ensure_ascii=False)[:80]}")
                        _passt = {
                            "role": "assistant",
                            "content": _pfc.get("content") or None,
                            "tool_calls": [
                                {"id": f"pc_{_pr}_{_i}", "type": "function",
                                 "function": {"name": _tc.get("name", ""),
                                              "arguments": json.dumps(_tc.get("arguments") or {},
                                                                      ensure_ascii=False)}}
                                for _i, _tc in enumerate(_ptcs)
                            ],
                        }
                        if _pfc.get("reasoning_content"):
                            _passt["reasoning_content"] = _pfc["reasoning_content"]
                        messages.append(_passt)
                        for _i, _r in enumerate(_pres):
                            messages.append({"role": "tool", "tool_call_id": f"pc_{_pr}_{_i}",
                                             "content": _r})
        except Exception as e:
            logger.warning(f"[PetFC] 走位调用失败: {e}")

        # ★ 立绘 Function Calling：让 AI 真的能自己换表情/动作（而不是嘴上说换了）
        #   只在“聊到形象/情绪”时启用，避免每条消息多一次模型调用。
        try:
            _EXPR_KW = ("表情", "动作", "立绘", "形象", "害羞", "脸红", "可爱", "卖萌",
                        "笑", "哭", "泪", "生气", "难过", "惊讶", "开心", "委屈",
                        "爱心眼", "兔牙", "眼镜", "裙子", "脱鞋", "尾巴", "试试")
            _recent_text = "".join(
                str(m.get("content", "")) for m in (recent_history or [])[-4:]
                if isinstance(m.get("content"), str)
            )
            if any(k in content for k in _EXPR_KW) or any(k in _recent_text for k in _EXPR_KW):
                from brain.tools import BrainTools as _BrainTools2
                _at = _BrainTools2()
                _adefs = [t for t in _at.get_tool_definitions()
                          if t.get("function", {}).get("name", "").startswith("avatar_")]
                if _adefs:
                    messages[0]["content"] += (
                        "\n\n## 立绘控制权限（本次对话可用）\n"
                        "你能直接控制桌面上自己的立绘。用户提到表情/动作/形象，或你想表达情绪时，"
                        "必须调用 avatar_expression 工具，不要只在嘴上说换了。\n"
                        "- expressions: 要同时生效的表情名列表（可任意组合），例如 [\"脸红\",\"爱心眼\"]\n"
                        "- motion: 可选动作名（idle / speaking / thinking 等）\n"
                        "- seconds: 可选，多少秒后自动恢复\n"
                        "- persist: 设 true 表示这是持续状态（哭/生气/拿话筒…），一直挂着不自动消失；"
                        "默认 false 会自己淡出。想收掉再调一次 expressions=[]\n"
                        "调用后再用一句话说明你换了什么；没调用就不要说已经换了。"
                    )
                    for _ar in range(2):
                        _afc = await llm.chat_with_tools(messages, _adefs)
                        _atcs = _afc.get("tool_calls") or []
                        if not _atcs:
                            break
                        _ares = []
                        for _tc in _atcs:
                            _an = _tc.get("name", "")
                            _aa = _tc.get("arguments", {}) or {}
                            try:
                                _r = await _at.call_tool(_an, **_aa)
                            except Exception as _te:
                                _r = {"success": False, "error": str(_te)}
                            _ares.append(f"[{_an}] {json.dumps(_r, ensure_ascii=False)[:400]}")
                            logger.info(f"[AvatarFC] {_an} {json.dumps(_aa, ensure_ascii=False)[:80]}")
                        _aasst = {
                            "role": "assistant",
                            "content": _afc.get("content") or None,
                            "tool_calls": [
                                {"id": f"ac_{_ar}_{_i}", "type": "function",
                                 "function": {"name": _tc.get("name", ""),
                                              "arguments": json.dumps(_tc.get("arguments") or {},
                                                                      ensure_ascii=False)}}
                                for _i, _tc in enumerate(_atcs)
                            ],
                        }
                        if _afc.get("reasoning_content"):
                            _aasst["reasoning_content"] = _afc["reasoning_content"]
                        messages.append(_aasst)
                        for _i, _r in enumerate(_ares):
                            messages.append({"role": "tool", "tool_call_id": f"ac_{_ar}_{_i}",
                                             "content": _r})
        except Exception as e:
            logger.warning(f"[AvatarFC] 立绘表情调用失败: {e}")

        # ★ 流式输出（mind.delta）：边生成边推给前端，模型可以顺带留钩子
        from brain.mind import (StreamFilter, parse_hook_specs, get_mind,
                                extract_memory, strip_structured, apply_memory)
        stream_id = str(uuid.uuid4())[:8]
        flt = StreamFilter()
        parts: list = []
        _fallback_mem = None
        await manager.broadcast({
            "type": "mind.thinking",
            "payload": {"id": stream_id, "reason": "回复用户"},
        })
        try:
            async for _chunk in llm.chat_stream(messages):
                _vis = flt.feed(_chunk)
                if _vis:
                    parts.append(_vis)
                    await manager.broadcast({
                        "type": "mind.delta",
                        "payload": {"id": stream_id, "text": _vis},
                    })
            _tail = flt.finish()
            if _tail:
                parts.append(_tail)
                await manager.broadcast({
                    "type": "mind.delta",
                    "payload": {"id": stream_id, "text": _tail},
                })
        except Exception as _stream_err:
            # 流式不可用时退回一次性调用，保证对话不中断
            logger.warning(f"[Mind] 流式回复失败，回退非流式: {_stream_err}")
            if not parts:
                _fallback = await llm.chat(messages)
                _fallback = _fallback if isinstance(_fallback, str) else str(_fallback)
                _fallback_mem = extract_memory(_fallback)
                _fallback = strip_structured(_fallback)
                parts.append(_fallback)
                await manager.broadcast({
                    "type": "mind.delta",
                    "payload": {"id": stream_id, "text": _fallback},
                })
        ai_response = "".join(parts).strip()

        # ★ 刚说过闸门：和最近 3 条回复重合 ≥0.9 → 重写一版（换说法 + 补新信息）
        if ai_response and len(ai_response) >= 20 and _prev_asst:
            try:
                _sim = max(difflib.SequenceMatcher(None, ai_response[:400], t[:400]).ratio()
                           for t in _prev_asst)
                if _sim >= 0.9:
                    _rw = await llm.chat([
                        {"role": "system",
                         "content": "你刚才已经说过几乎一样的话。保持同样的口语风格和长度，"
                                    "换一种说法并补充至少一个新信息，不要复读原句。只输出重写后的话。"},
                        {"role": "user",
                         "content": f"东昇刚才问：{content[:200]}\n"
                                    f"你上次的回答：{_prev_asst[-1][:200]}\n"
                                    f"这次你先答成了：{ai_response[:200]}"},
                    ])
                    _rw = (_rw or "").strip().strip('"').strip("'")
                    if _rw and 10 <= len(_rw) <= max(200, int(len(ai_response) * 1.6)):
                        logger.info(f"[Repeat] 回复与上次重合 {_sim:.2f}，已重写")
                        ai_response = _rw
                    else:
                        logger.info(f"[Repeat] 重合 {_sim:.2f}，重写结果不可用，保留原文")
            except Exception as _rep_err:
                logger.warning(f"[Repeat] 重写失败，保留原文: {_rep_err}")

        # 到点提醒的生命周期：这次回复真的提到了 → 标记完成，别再每 6 小时喊一遍
        try:
            from brain.tfgr import get_tfgr as _get_tfgr
            for _e in _get_tfgr().due():
                if _reply_mentions(ai_response, _e.text):
                    _get_tfgr().mark_done(_e.id, note="chat_reply")
                    logger.info(f"[Reminder] 已在回复中提到，标记完成: {_e.id} {_e.text[:30]}")
        except Exception:
            logger.debug("提醒标记完成失败（忽略）", exc_info=True)

        # ★ 模型的结构化记忆：正文之外的 <memory> 块写进 ATG / TFGR，气泡里看不到
        try:
            from brain.mind import load_memory_json
            _mem = _fallback_mem or load_memory_json(flt.memory)
            if _mem:
                _mem_res = apply_memory(_mem, source="chat")
                if _mem_res.get("atg") or _mem_res.get("tfgr"):
                    logger.info(f"[Memory] chat 结构化记忆已写入: {_mem_res}")
        except Exception as _mem_err:
            logger.warning(f"[Memory] 结构化记忆写入失败: {_mem_err}")

        hook_specs = parse_hook_specs(flt.markers)
        for _spec in hook_specs:
            get_mind().add_hook(_spec)
        await manager.broadcast({
            "type": "mind.end",
            "payload": {
                "id": stream_id,
                "text": ai_response,
                "hooks": [s.raw for s in hook_specs],
            },
        })
        
        # 保存AI回复（role=assistant，不是brain！）
        ai_msg = {"id": str(uuid.uuid4())[:8], "role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()}
        _save_chat_message(ai_msg)
        # ★ 同步到大脑内存上下文
        get_global_state().add_recent_message("assistant", ai_response)

        # ★ 广播用户图片（如果有）：让前端显示用户上传的图片
        if image_base64:
            await manager.broadcast({
                "type": "chat.image",
                "payload": {"image_base64": vision_thumb or image_base64[:20000]}
            })

        await manager.broadcast({
            "type": "chat.message",
            "payload": ai_msg
        })

        # ★ 虚拟形象动作：AI回复时 → 拿话筒+星星眼
        await manager.broadcast({"type": "avatar.speaking"})
        logger.info(f"[Brain] 广播 avatar.speaking (聊天回复: {content[:30]})")
        # 3秒后回到idle（恢复正常表情）并触发大脑推理
        async def _avatar_back_to_idle():
            await asyncio.sleep(3)
            await manager.broadcast({"type": "avatar.speaking_end"})
            logger.info(f"[Brain] 广播 avatar.speaking_end (聊天回复结束)")
            # ★ 对话结束 → 触发大脑立即分析动机
            trigger_brain_analysis()
        asyncio.create_task(_avatar_back_to_idle())

        return {"type": "chat.send_result", "payload": {"success": True, "content": ai_response}}
    except Exception as e:
        logger.error(f"Chat send error: {e}")
        err_msg = {"id": str(uuid.uuid4())[:8], "role": "assistant", "content": f"[错误] LLM调用失败: {str(e)[:100]}", "timestamp": datetime.now().isoformat()}
        _save_chat_message(err_msg)
        await manager.broadcast({
            "type": "chat.message",
            "payload": err_msg
        })
        return {"type": "chat.send_result", "payload": {"success": False, "error": str(e)}}

async def handle_brain_switch_mode(payload: dict) -> dict:
    mode = payload.get("mode", "idle")
    try:
        from brain.unified_brain import get_unified_brain, BrainMode
        brain = get_unified_brain()
        new_mode = BrainMode(mode)
        result = await brain.switch_mode(new_mode, "用户手动切换")
        
        # 广播模式变更
        await manager.broadcast({
            "type": "brain.mode_changed",
            "payload": {"new_mode": mode, **result}
        })
        
        # ACTIVE模式：大脑委托AI主动打招呼
        if mode == "active":
            try:
                from brain.llm import get_llm
                llm = get_llm()
                
                # 大脑委托模式：大脑决定打招呼 → AI执行
                recent_history = _load_chat_history()
                messages = _build_brain_delegated_messages({
                    "reason": "用户手动切换到主动对话模式，需要建立连接",
                    "user_state": {"presence": "online", "no_response_count": 0},
                    "tone": "自然友好，像朋友见面打招呼"
                }, recent_history)
                
                greeting = await llm.chat(messages)
                
                # AI的消息，不是大脑的！
                msg_id = str(uuid.uuid4())[:8]
                now = datetime.now().isoformat()
                greet_msg = {"id": msg_id, "role": "assistant", "content": greeting, "timestamp": now}
                _save_chat_message(greet_msg)
                await manager.broadcast({"type": "chat.message", "payload": greet_msg})
            except Exception as e:
                logger.error(f"Active mode greeting error: {e}")
        
        # REFLECT模式：让大脑执行反思
        elif mode == "reflect":
            try:
                from brain.intrinsic_reward import get_intrinsic_motivation_system
                intrinsic = get_intrinsic_motivation_system()
                intrinsic.evaluate_and_act(
                    state={"mode": "reflect"},
                    action="self_reflect",
                    result={"success": True},
                    confidence=0.8
                )
                # 反思完成后切回idle
                await brain.switch_mode(BrainMode.IDLE, "反思完成")
                await manager.broadcast({
                    "type": "brain.mode_changed",
                    "payload": {"new_mode": "idle", "reason": "反思完成"}
                })
            except Exception as e:
                logger.error(f"Reflect error: {e}")
        
        return {"type": "brain.mode_changed", "payload": result}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_memory_search(payload: dict) -> dict:
    """
    记忆搜索 - 双边并行搜索（本地 + Viking 云端）

    两边同时发起请求，谁先回就先显示，最后合并去重。
    云端连不上时优雅降级：只用本地结果，不报错。
    """
    query = payload.get("query", "")
    limit = payload.get("limit", 10)
    try:
        if not query.strip():
            return {"type": "memory.search_result", "payload": {"memories": [], "message": "请输入搜索关键词"}}

        # ★ 双边并行搜索：本地 + 云端 同时发起
        async def search_local():
            return _search_chat_history(query, limit)

        async def search_viking():
            try:
                from brain.memory import get_memory_manager
                memory_manager = get_memory_manager()
                return await memory_manager.search_memories(query, limit=limit)
            except Exception as e:
                logger.warning(f"Viking search failed (graceful fallback): {e}")
                return []  # 优雅降级：返回空列表

        # 并行执行两个搜索
        local_results, viking_results = await asyncio.gather(
            search_local(),
            search_viking(),
            return_exceptions=True
        )

        # 处理异常结果
        if isinstance(local_results, Exception):
            logger.error(f"Local search error: {local_results}")
            local_results = []
        if isinstance(viking_results, Exception):
            logger.warning(f"Viking search exception (using local only): {viking_results}")
            viking_results = []

        # 合并去重（以内容前50字为key）
        seen_texts = set()
        merged = []
        all_results = local_results + viking_results

        for item in all_results:
            text_key = (item.get("text", "") or item.get("content", ""))[:50]
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                # 标记来源
                source = item.get("source", "unknown")
                merged.append({
                    **item,
                    "source": source,
                    "_from": "cloud" if "viking" in source or source == "unknown" else "local"
                })
                if len(merged) >= limit * 2:
                    break

        result = {
            "type": "memory.search_result",
            "payload": {
                "memories": merged[:limit],
                "stats": {
                    "local_count": len(local_results),
                    "cloud_count": len(viking_results) if isinstance(viking_results, list) else 0,
                    "merged_count": len(merged[:limit]),
                    "viking_available": len(viking_results) > 0 if isinstance(viking_results, list) else False
                }
            }
        }
        logger.info(f"Memory search '{query[:20]}': local={len(local_results)}, cloud={len(viking_results) if isinstance(viking_results, list) else 0}, merged={len(merged[:limit])}")
        return result
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_web_search(payload: dict) -> dict:
    """WebSocket 联网搜索事件处理"""
    query = payload.get("query", "")
    if not query.strip():
        return {"type": "web.search_result", "payload": {"success": False, "error": "请输入搜索关键词"}}

    try:
        result = await _execute_brain_search(query)
        return {
            "type": "web.search_result",
            "payload": {
                "query": result.get("query", ""),
                "success": result.get("success", False),
                "results_count": result.get("results_count", 0),
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "formatted_context": result.get("formatted_context", "")[:1000],
                "error": result.get("error", ""),
            }
        }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {"type": "error", "payload": {"message": str(e)}}


# ============================================================
#  AI 工作区 WebSocket 事件处理
# ============================================================

async def handle_workspace_write(payload: dict) -> dict:
    """WS: 写入脚本到工作区"""
    try:
        from brain.workspace import get_ai_workspace, ScriptType
        ws = get_ai_workspace()
        content = payload.get("content", "")
        name = payload.get("name", "")
        script_type_str = payload.get("script_type", "auto")
        execute = payload.get("execute", False)

        st = ScriptType(script_type_str) if script_type_str != "auto" else ScriptType.AUTO
        result = await ws.write_script(name, content, script_type=st)

        if result.get("success") and execute:
            exec_result = await ws.execute_script(result["name"])
            result["execution"] = {
                "success": exec_result.success,
                "exit_code": exec_result.exit_code,
                "stdout": exec_result.stdout[:2000],
                "stderr": exec_result.stderr[:1000],
                "duration_ms": exec_result.duration_ms,
            }

        return {"type": "workspace.script_written", "payload": result}
    except Exception as e:
        logger.error(f"Workspace write error: {e}")
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_workspace_execute(payload: dict) -> dict:
    """WS: 执行工作区脚本"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        name = payload.get("name", "")
        content = payload.get("content", "")
        timeout = payload.get("timeout", 60)

        if content:
            result = await ws.execute_script(content, is_content=True, timeout=timeout)
        elif name:
            result = await ws.execute_script(name, timeout=timeout)
        else:
            return {"type": "error", "payload": {"message": "需要提供 name 或 content"}}

        return {
            "type": "workspace.script_executed",
            "payload": {
                "success": result.success,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "script_name": result.script_name,
            }
        }
    except Exception as e:
        logger.error(f"Workspace execute error: {e}")
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_workspace_files() -> dict:
    """WS: 列出工作区文件"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return {"type": "workspace.file_list", "payload": {"files": ws.list_files()}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_workspace_executions(payload: dict) -> dict:
    """WS: 列出最近执行记录"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        limit = payload.get("limit", 10)
        return {"type": "workspace.execution_list", "payload": {"executions": ws.list_recent_executions(limit)}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}


# ============================================================
#  任务执行器 WebSocket 事件处理
# ============================================================

async def handle_executor_status() -> dict:
    """WS: 获取执行器状态"""
    try:
        from brain.task_executor import get_task_executor
        executor = get_task_executor()
        return {"type": "executor.status_result", "payload": executor.get_status()}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_executor_submit(payload: dict) -> dict:
    """WS: 手动提交任务给执行器"""
    try:
        from brain.task_executor import get_task_executor, TaskPriority
        executor = get_task_executor()
        priority_map = {
            "low": TaskPriority.LOW, "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH, "urgent": TaskPriority.URGENT,
        }
        task = executor.submit(
            description=payload.get("description", ""),
            context=payload.get("context", ""),
            priority=priority_map.get(payload.get("priority", "normal"), TaskPriority.NORMAL),
        )
        if task:
            return {"type": "executor.task_submitted",
                    "payload": {"task_id": task.id, "status": "submitted",
                               "queue_size": executor._queue.qsize()}}
        return {"type": "error", "payload": {"message": "Executor queue full"}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}


async def handle_executor_history(payload: dict) -> dict:
    """WS: 执行器历史"""
    try:
        from brain.task_executor import get_task_executor
        executor = get_task_executor()
        limit = payload.get("limit", 20)
        return {"type": "executor.history_result",
                "payload": {"tasks": executor.get_task_history(limit)}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}


def _search_chat_history(query: str, limit: int = 20) -> list:
    """从本地聊天记录中搜索关键词"""
    if not query.strip():
        return []
    
    history = _load_chat_history()
    query_lower = query.lower().strip()
    
    results = []
    for msg in reversed(history):  # 最新的优先
        content = msg.get("content", "")
        role = msg.get("role", "")
        
        # 跳过大脑内部消息和错误消息
        if role == "brain":
            continue
        
        if query_lower in content.lower():
            results.append({
                "id": msg.get("id", ""),
                "text": content,
                "role": role,
                "timestamp": msg.get("timestamp", ""),
                "source": "local_chat"
            })
            if len(results) >= limit:
                break
    
    return results

async def handle_brain_get_status() -> dict:
    try:
        from brain.unified_brain import get_unified_brain
        from brain.intrinsic_reward import get_intrinsic_motivation_system
        from core.state import get_global_state
        brain = get_unified_brain()
        intrinsic = get_intrinsic_motivation_system()
        global_state = get_global_state()
        
        status = brain.get_status()
        intrinsic_status = intrinsic.get_system_status()
        state_summary = global_state.get_state_summary()
        
        # 优先使用LLM分析值，回退到intrinsic模块
        llm_motivation = get_brain_llm_motivation()
        
        # 从实际数据源读取对话记录数
        chat_history = _load_chat_history()
        actual_dialog_count = len(chat_history)
        
        # 系统健康度本地化
        health_raw = intrinsic_status.get("system_health", {}).get("overall", "unknown")
        health_labels = {
            "healthy": "正常",
            "needs_attention": "需关注",
            "needs_improvement": "待优化",
            "unknown": "未知"
        }
        health_label = health_labels.get(health_raw, health_raw)
        
        # 计算热记忆数（最近24小时的消息）
        from datetime import timedelta
        yesterday = datetime.now() - timedelta(hours=24)
        hot_count = sum(1 for m in chat_history if m.get("timestamp", "") > yesterday.isoformat())

        # 语音桥状态
        try:
            from brain.voice_bridge import get_voice_bridge
            voice_bridge_status = get_voice_bridge().get_status()
        except Exception:
            voice_bridge_status = {"active": False, "session_count": 0}

        return {
            "type": "brain.status",
            "payload": {
                "mode": status["mode"],
                "connected": status["connected"],
                "curiosity": llm_motivation["curiosity"],
                "exploration": llm_motivation["exploration"],
                "learning_progress": llm_motivation["learning_progress"],
                "knowledge_gaps": intrinsic_status.get("update_module", {}).get("recent_gaps", []),
                "context_summary": status.get("context_summary", {}),
                "dialog_history_count": actual_dialog_count,
                "hot_memory_count": hot_count,
                "last_user_activity": status.get("last_user_activity"),
                "last_reflect_time": status.get("last_reflect_time"),
                "system_health": intrinsic_status.get("system_health", {}),
                "system_health_label": health_label,
                "user_presence": state_summary["user_presence"],
                "user_idle_time": state_summary["idle_time"],
                "consecutive_no_response": state_summary["consecutive_no_response"],
                "motivation_reason": llm_motivation["reason"],
                "voice_bridge": voice_bridge_status
            }
        }
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_task_create(payload: dict) -> dict:
    try:
        from brain.tasks import get_task_manager
        from core.types import Task, TaskType, TaskPriority
        tm = get_task_manager()
        task = Task(
            id=str(uuid.uuid4())[:8],
            type=TaskType(payload.get("type", "dialogue_action")),
            description=payload.get("description", ""),
            priority=TaskPriority(payload.get("priority", 2)),
            parameters=payload.get("parameters", {}),
        )
        tm.add_task(task)
        await manager.broadcast({"type": "task.created", "payload": task.to_dict()})
        return {"type": "task.created", "payload": task.to_dict()}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_task_list() -> dict:
    try:
        from brain.tasks import get_task_manager
        tm = get_task_manager()
        tasks = tm.load_tasks()
        return {"type": "task.list_result", "payload": {"tasks": [t.to_dict() for t in tasks]}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_skill_list() -> dict:
    try:
        skills = _load_skills()
        return {"type": "skill.list_result", "payload": {"skills": skills}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_skill_install(payload: dict) -> dict:
    try:
        skills = _load_skills()
        skill = {
            "id": str(uuid.uuid4())[:8],
            "name": payload.get("name", ""),
            "description": payload.get("description", ""),
            "command": payload.get("command", ""),
            "version": payload.get("version", "1.0.0"),
            "installed": True,
            "installed_at": datetime.now().isoformat(),
        }
        skills.append(skill)
        _save_skills(skills)
        await manager.broadcast({"type": "skill.installed", "payload": skill})
        return {"type": "skill.installed", "payload": skill}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

async def handle_skill_execute(payload: dict) -> dict:
    try:
        skill_id = payload.get("id", "")
        skills = _load_skills()
        skill = next((s for s in skills if s["id"] == skill_id), None)
        if not skill:
            return {"type": "error", "payload": {"message": "Skill not found"}}

        import subprocess
        result = subprocess.run(
            skill["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "type": "skill.execute_result",
            "payload": {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        }
    except subprocess.TimeoutExpired:
        return {"type": "error", "payload": {"message": "Skill execution timed out"}}
    except Exception as e:
        return {"type": "error", "payload": {"message": str(e)}}

@app.get("/api/brain/status")
async def get_brain_status():
    """和 WS 的 brain.status 用同一份数据。

    以前这里是另一套精简字段，状态页 5 秒轮询一次就会把 WS 拿到的完整数据
    覆盖成默认值（卡片变回 —），看起来像“刷新回默认”。
    """
    try:
        res = await handle_brain_get_status()
        payload = res.get("payload") if isinstance(res, dict) else None
        if not payload:
            raise HTTPException(status_code=500, detail=str(res))
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/brain/mode")
async def switch_brain_mode(request: ModeRequest):
    try:
        from brain.unified_brain import get_unified_brain, BrainMode
        brain = get_unified_brain()
        new_mode = BrainMode(request.mode)
        result = await brain.switch_mode(new_mode, "用户手动切换")
        await manager.broadcast({
            "type": "brain.mode_changed",
            "payload": result
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory")
async def get_memories(limit: int = 50, offset: int = 0, q: str = ""):
    """获取/搜索记忆，支持?q=关键词"""
    try:
        if q.strip():
            results = _search_chat_history(q, limit)
            return {"memories": results, "total": len(results), "query": q}
        else:
            # 无搜索词时返回最近对话记录
            history = _load_chat_history()
            recent = [m for m in reversed(history) if m.get("role") != "brain"][:limit]
            return {
                "memories": [{"id": m.get("id", ""), "text": m.get("content", ""), "role": m.get("role", ""), 
                              "timestamp": m.get("timestamp", ""), "source": "local_chat"} for m in recent],
                "total": len(history),
                "query": ""
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_chat_history(limit: int = 100):
    """获取聊天历史记录（含语音对话，两个系统对话记录互通）"""
    try:
        history = _load_chat_history()
        return {"messages": history[-limit:], "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/history")
async def clear_chat_history():
    """清空聊天历史"""
    try:
        if CHAT_HISTORY_FILE.exists():
            CHAT_HISTORY_FILE.unlink()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/search")
async def search_memories(request: MemorySearchRequest):
    """REST 记忆搜索 - 双边并行（本地 + 云端）"""
    try:
        # ★ 双边并行搜索
        async def search_local():
            return _search_chat_history(request.query, request.limit)

        async def search_viking():
            try:
                from brain.memory import get_memory_manager
                memory_manager = get_memory_manager()
                return await memory_manager.search_memories(request.query, limit=request.limit)
            except Exception as e:
                logger.warning(f"Viking REST search fallback: {e}")
                return []

        local_results, viking_results = await asyncio.gather(
            search_local(), search_viking(), return_exceptions=True
        )
        if isinstance(local_results, Exception):
            local_results = []
        if isinstance(viking_results, Exception):
            viking_results = []

        # 合并去重
        seen = set()
        merged = []
        for item in local_results + (viking_results if isinstance(viking_results, list) else []):
            # 提取文本 key（兼容本地和 Viking 两种结构）
            key = (item.get("text", "") or item.get("content", ""))
            if not key and isinstance(item.get("memory_info"), dict):
                mi = item["memory_info"]
                key = mi.get("summary", "") or mi.get("original_messages", "") or mi.get("user_profile", "")
            key = str(key)[:50]
            if key and key not in seen:
                seen.add(key)
                # 统一输出格式：确保前端/API消费者能拿到 text 字段
                if not item.get("text") and not item.get("content"):
                    if isinstance(item.get("memory_info"), dict):
                        mi = item["memory_info"]
                        item["text"] = mi.get("summary", "") or mi.get("original_messages", "") or str(mi.get("user_profile", ""))[:300]
                merged.append(item)
                if len(merged) >= request.limit * 2:
                    break

        return {"memories": merged[:request.limit], "stats": {"local": len(local_results), "cloud": len(viking_results) if isinstance(viking_results, list) else 0}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory/sync")
async def trigger_memory_sync(force: bool = False):
    """手动触发记忆同步到 Viking 云端"""
    try:
        result = await _sync_memories_to_viking(force=force)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/api/web/search")
async def web_search(request: WebSearchRequest):
    """手动触发联网搜索（Tavily）"""
    try:
        result = await _execute_brain_search(request.query)
        # 通过 WebSocket 广播搜索事件
        await manager.broadcast({
            "type": "web.search_result",
            "payload": {
                "query": result.get("query", ""),
                "success": result.get("success", False),
                "results_count": result.get("results_count", 0),
                "answer": result.get("answer", "")[:200],
                "sources": result.get("sources", []),
            }
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#  AI 工作区 API（脚本编写 + 执行）
# ============================================================

class WorkspaceScriptRequest(BaseModel):
    name: str = ""
    content: str
    script_type: str = "auto"  # python | bash | powershell | auto
    execute: bool = False  # 写完立即执行


class WorkspaceExecuteRequest(BaseModel):
    name: str = ""  # 文件名（执行已有文件）
    content: str = ""  # 或直接传内容（内联执行）
    timeout: int = 60


@app.get("/api/workspace")
async def get_workspace_info():
    """获取工作区状态"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return ws.get_workspace_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/files")
async def list_workspace_files():
    """列出工作区文件"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return {"files": ws.list_files()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspace/scripts")
async def write_workspace_script(request: WorkspaceScriptRequest):
    """在工作区创建脚本（可选立即执行）"""
    try:
        from brain.workspace import get_ai_workspace, ScriptType
        ws = get_ai_workspace()

        st = ScriptType(request.script_type) if request.script_type != "auto" else ScriptType.AUTO
        result = await ws.write_script(request.name, request.content, script_type=st)

        if result.get("success") and request.execute:
            exec_result = await ws.execute_script(result["name"])
            result["execution"] = {
                "success": exec_result.success,
                "exit_code": exec_result.exit_code,
                "stdout": exec_result.stdout[:2000],
                "stderr": exec_result.stderr[:1000],
                "duration_ms": exec_result.duration_ms,
            }

        await manager.broadcast({
            "type": "workspace.script_written",
            "payload": result
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspace/execute")
async def execute_workspace_script(request: WorkspaceExecuteRequest):
    """执行工作区脚本"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()

        if request.content:
            result = await ws.execute_script(request.content, is_content=True, timeout=request.timeout)
        elif request.name:
            result = await ws.execute_script(request.name, timeout=request.timeout)
        else:
            raise HTTPException(status_code=400, detail="需要提供 name 或 content")

        payload = {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "script_name": result.script_name,
            "executed_at": result.executed_at,
        }

        await manager.broadcast({
            "type": "workspace.script_executed",
            "payload": payload
        })
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/scripts/{name}")
async def read_workspace_script(name: str):
    """读取脚本内容"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return ws.read_script(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workspace/scripts/{name}")
async def delete_workspace_script(name: str):
    """删除脚本"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return ws.delete_script(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/executions")
async def list_recent_executions(limit: int = 10):
    """列出最近的执行记录"""
    try:
        from brain.workspace import get_ai_workspace
        ws = get_ai_workspace()
        return {"executions": ws.list_recent_executions(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#  任务执行器 API（独立 Agent）
# ============================================================

class ExecutorTaskRequest(BaseModel):
    description: str
    context: str = ""
    priority: str = "normal"  # low | normal | high | urgent


@app.get("/api/executor")
async def get_executor_status():
    """获取执行器状态（队列、活跃任务、历史、记忆）"""
    try:
        from brain.task_executor import get_task_executor
        executor = get_task_executor()
        return executor.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/executor/tasks")
async def submit_executor_task(request: ExecutorTaskRequest):
    """手动提交任务给执行器"""
    try:
        from brain.task_executor import get_task_executor, TaskPriority
        executor = get_task_executor()
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT,
        }
        task = executor.submit(
            description=request.description,
            context=request.context,
            priority=priority_map.get(request.priority, TaskPriority.NORMAL),
        )
        if task:
            return {"success": True, "task_id": task.id, "status": "submitted",
                    "queue_size": executor._queue.qsize()}
        return {"success": False, "error": "Executor queue full"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/executor/history")
async def get_executor_history(limit: int = 20):
    """获取执行器任务历史"""
    try:
        from brain.task_executor import get_task_executor
        executor = get_task_executor()
        return {"tasks": executor.get_task_history(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/executor/tasks/{task_id}")
async def delete_executor_task(task_id: str):
    """删除执行器任务记录（同时删除关联脚本）"""
    try:
        from brain.task_executor import get_task_executor
        executor = get_task_executor()
        deleted = executor.delete_task(task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        return {"success": True, "deleted": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    try:
        config_file = Path(__file__).parent.parent / "config" / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config")
async def update_config(request: ConfigUpdate):
    try:
        config_file = Path(__file__).parent.parent / "config" / "config.json"
        _safe_write_json(config_file, request.config)
        # 重新加载LLM配置
        try:
            from brain.llm import get_llm
            get_llm().reload()
        except Exception:
            pass
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks")
async def get_tasks():
    try:
        from brain.tasks import get_task_manager
        tm = get_task_manager()
        tasks = tm.load_tasks()
        return {"tasks": [t.to_dict() for t in tasks]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks")
async def create_task(request: TaskCreate):
    try:
        from brain.tasks import get_task_manager
        from core.types import Task, TaskType, TaskPriority
        tm = get_task_manager()
        task = Task(
            id=str(uuid.uuid4())[:8],
            type=TaskType(request.type),
            description=request.description,
            priority=TaskPriority(request.priority),
            parameters=request.parameters,
        )
        tm.add_task(task)
        await manager.broadcast({"type": "task.created", "payload": task.to_dict()})
        return task.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/tasks/{task_id}/status")
async def update_task_status(task_id: str, status: str = "completed"):
    try:
        from brain.tasks import get_task_manager
        tm = get_task_manager()
        tm.update_task_status(task_id, status)
        await manager.broadcast({"type": "task.updated", "payload": {"id": task_id, "status": status}})
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    try:
        from brain.tasks import get_task_manager
        tm = get_task_manager()
        tasks = tm.load_tasks()
        tasks = [t for t in tasks if t.id != task_id]
        tm.save_tasks(tasks)
        await manager.broadcast({"type": "task.deleted", "payload": {"id": task_id}})
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/skills")
async def get_skills():
    return {"skills": _load_skills()}

@app.post("/api/skills")
async def install_skill(request: SkillInstall):
    try:
        skills = _load_skills()
        skill = {
            "id": str(uuid.uuid4())[:8],
            "name": request.name,
            "description": request.description,
            "command": request.command,
            "version": request.version,
            "installed": True,
            "installed_at": datetime.now().isoformat(),
        }
        skills.append(skill)
        _save_skills(skills)
        await manager.broadcast({"type": "skill.installed", "payload": skill})
        return skill
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/skills/{skill_id}")
async def uninstall_skill(skill_id: str):
    try:
        skills = _load_skills()
        skills = [s for s in skills if s["id"] != skill_id]
        _save_skills(skills)
        await manager.broadcast({"type": "skill.uninstalled", "payload": {"id": skill_id}})
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/skills/{skill_id}/execute")
async def execute_skill(skill_id: str):
    try:
        skills = _load_skills()
        skill = next((s for s in skills if s["id"] == skill_id), None)
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")

        import subprocess
        result = subprocess.run(
            skill["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Skill execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

# ============================================================
#  AI 音视频互动 2.0 端点
# ============================================================

class VoiceChatStartRequest(BaseModel):
    room_id: str = ""
    system_prompt: str = ""
    enable_vision: bool = True

class VoiceChatStopRequest(BaseModel):
    room_id: str
    task_id: str

class FCResultRequest(BaseModel):
    room_id: str
    task_id: str
    tool_name: str
    arguments: dict = {}
    tool_call_id: str = ""

class TokenRequest(BaseModel):
    room_id: str = ""
    user_id: str = ""

_voice_chat_state: Dict[str, Any] = {}  # room_id -> {task_id, status, ...}


@app.post("/api/v2/token")
async def generate_rtc_token(request: TokenRequest):
    """生成 RTC 鉴权 Token（本地生成，无需调用火山API）"""
    try:
        from core.config import get_config
        cfg = get_config()
        v2 = cfg.voice_chat_v2

        room_id = request.room_id or f"room_{uuid.uuid4().hex[:8]}"
        user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"

        # 使用本地 Token 生成器
        from api.rtc_token import generate_rtc_token as gen_token
        token = gen_token(
            app_id=v2.app_id,
            app_key=v2.app_key,
            room_id=room_id,
            user_id=user_id,
            expire_seconds=86400,
            role=1,
        )

        return {
            "token": token,
            "room_id": room_id,
            "user_id": user_id,
            "app_id": v2.app_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/start")
async def start_voice_chat(request: VoiceChatStartRequest):
    """启动 AI 音视频互动 2.0 对话"""
    try:
        from api.voice_chat_v2 import get_voice_chat_client
        client = get_voice_chat_client()

        room_id = request.room_id or f"room_{uuid.uuid4().hex[:8]}"
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        user_id = f"user_{uuid.uuid4().hex[:8]}"

        result = await client.start_voice_chat(
            room_id=room_id,
            task_id=task_id,
            user_id=user_id,
            agent_user_id=f"ai_friday_{uuid.uuid4().hex[:6]}",
            system_prompt=request.system_prompt,
            enable_vision=request.enable_vision,
        )

        if result.get("Result") == "ok":
            _voice_chat_state[room_id] = {
                "task_id": task_id,
                "user_id": user_id,
                "status": "running",
                "vision": request.enable_vision,
            }
            # 通知大脑：星期五已接管对话
            try:
                from core.state import get_global_state
                get_global_state().set_dialog_status(
                    __import__('core.state', fromlist=['DialogStatus']).DialogStatus.ACTIVE
                )
            except Exception:
                pass
            await manager.broadcast({
                "type": "v2.status",
                "payload": {"status": "running", "room_id": room_id, "task_id": task_id}
            })

        # 返回前端需要的字段，而不是原始火山API响应
        return {
            "Result": result.get("Result", ""),
            "RoomId": room_id,
            "TaskId": task_id,
            "UserId": user_id,
            "Vision": request.enable_vision,
        }
    except Exception as e:
        logger.error(f"StartVoiceChat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/stop")
async def stop_voice_chat(request: VoiceChatStopRequest):
    """停止 AI 音视频互动 2.0 对话"""
    try:
        from api.voice_chat_v2 import get_voice_chat_client
        client = get_voice_chat_client()

        result = await client.stop_voice_chat(request.task_id, request.room_id)

        if request.room_id in _voice_chat_state:
            _voice_chat_state[request.room_id]["status"] = "stopped"

        # 通知大脑：星期五已释放对话
        try:
            from core.state import get_global_state, DialogStatus
            get_global_state().set_dialog_status(DialogStatus.IDLE)
        except Exception:
            pass

        await manager.broadcast({
            "type": "v2.status",
            "payload": {"status": "stopped", "room_id": request.room_id}
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/fc-callback")
async def fc_callback(request: FCResultRequest):
    """Function Calling 回调 - 通过大脑BrainTools执行工具并返回结果"""
    try:
        from brain.tools import BrainTools
        from api.voice_chat_v2 import get_voice_chat_client
        client = get_voice_chat_client()
        brain_tools = BrainTools()

        # 通过大脑的BrainTools执行工具（统一路径）
        tool_map = {
            "arm_move_to": brain_tools.arm_move_to,
            "arm_move_init": brain_tools.arm_move_init,
            "arm_gripper_open": brain_tools.arm_gripper_open,
            "arm_gripper_close": brain_tools.arm_gripper_close,
            "arm_set_joint": brain_tools.arm_set_joint,
            "arm_get_state": brain_tools.arm_get_state,
            "arm_set_torque": brain_tools.arm_set_torque,
        }

        tool_fn = tool_map.get(request.tool_name)
        if not tool_fn:
            result = {"success": False, "error": f"未知工具: {request.tool_name}"}
        else:
            # 根据工具名解包参数
            args = request.arguments
            if request.tool_name == "arm_move_to":
                result = await tool_fn(
                    x=float(args.get("x", 0)),
                    y=float(args.get("y", 0)),
                    z=float(args.get("z", 0)),
                    description=args.get("description", ""),
                )
            elif request.tool_name == "arm_set_joint":
                result = await tool_fn(
                    joint_id=int(args.get("joint_id", 1)),
                    angle=float(args.get("angle", 0)),
                )
            elif request.tool_name == "arm_set_torque":
                result = await tool_fn(enable=bool(args.get("enable", True)))
            else:
                result = await tool_fn()

        # 记录到大脑记忆
        try:
            brain_tools.save_memory(
                content=f"[Friday FC] {request.tool_name}({request.arguments}) → {result}",
                category="friday_tool_call",
            )
        except Exception:
            pass

        # 回填结果给 AI
        fc_result = {
            "ToolCallId": request.tool_call_id,
            "ToolName": request.tool_name,
            "Content": json.dumps(result, ensure_ascii=False),
        }
        await client.update_voice_chat(
            task_id=request.task_id,
            room_id=request.room_id,
            tool_result=fc_result,
        )

        # 广播给前端
        await manager.broadcast({
            "type": "v2.tool_executed",
            "payload": {
                "tool": request.tool_name,
                "arguments": request.arguments,
                "result": result,
            }
        })

        return result
    except Exception as e:
        logger.error(f"FC callback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/status")
async def get_voice_chat_status():
    """获取当前语音对话状态"""
    return {
        "active_rooms": {
            k: v for k, v in _voice_chat_state.items()
            if v.get("status") == "running"
        }
    }


# ── 调试面板用的只读接口（ATG / TFGR）─────────────────────────────
# 注意：下面 if frontend_dist.exists() 里的 /{full_path:path} 会吞掉所有后注册的路由，
# 所以这几个必须注册在它前面。

# Plus 的界面不在这里服务了：Active_AI_Plus/server.py 自己起 18090，
# 把 /api/* 和 /ws 代理到这个后端。Pro 只负责数据和桌宠。


@app.get("/api/atg")
async def api_atg(limit: int = 40):
    """ATG（活动时序图）结构化快照：当前活动 + 节点 + 边。"""
    try:
        from brain import atg
        graph = atg.graph_snapshot(limit=max(1, min(limit, 200)))
        return {
            **graph,
            "current_text": atg.current(),
            "recent_text": atg.recent(8),
            "timeline_text": atg.snapshot(max_items=12),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


@app.get("/api/tfgr")
async def api_tfgr(limit: int = 80, window_s: float = 86400):
    """TFGR（时间流图）事件流：最近事件 + 统计 + 近期摘要。"""
    try:
        from dataclasses import asdict
        from brain.tfgr import get_tfgr
        g = get_tfgr()
        events = g.recent(n=max(1, min(limit, 300)), window_s=window_s)
        # 边：时序相邻（next）+ 同实体（same_entity）；前端力导向图直接拿来用
        edges: list[dict] = []
        ent_map: dict[str, str] = {}
        for prev, cur in zip(events, events[1:]):
            edges.append({"src": prev.id, "dst": cur.id, "rel": "next"})
        for e in events:
            for ent in (e.entities or [])[:4]:
                key = str(ent).lower()
                if key in ent_map and ent_map[key] != e.id:
                    edges.append({"src": ent_map[key], "dst": e.id, "rel": "same_entity"})
                else:
                    ent_map[key] = e.id
        rows = []
        for e in events:
            row = asdict(e)
            row["rev"] = int((e.meta or {}).get("rev") or 0)
            row["updated_at"] = (e.meta or {}).get("updated_at")
            rows.append(row)
        return {
            "stats": g.stats(),
            "summary": g.mental_summary(window_s=window_s, max_items=10),
            "events": rows,
            "edges": edges,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


@app.get("/plus")
@app.get("/plus/{rest:path}")
async def plus_moved():
    """Plus 界面已经独立到 Active_AI_Plus/server.py，Pro 不再提供。"""
    return JSONResponse(
        {"error": "Plus 界面已独立：请打开 http://127.0.0.1:18090/"},
        status_code=410,
    )


if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # 不拦截 API 和 WebSocket 路径
        if full_path.startswith("api/") or full_path.startswith("ws"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "Not found"}, status_code=404)
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))

async def broadcast_brain_status():
    while True:
        try:
            status = await handle_brain_get_status()
            await manager.broadcast(status)
        except Exception:
            pass
        await asyncio.sleep(5)

async def _llm_analyze_motivation(recent_messages: list, user_state: dict) -> dict:
    """
    用LLM分析最近对话，动态生成内在动机值 + 联网搜索决策

    大脑的"思考"过程：读取对话 → 分析状态 → 决定动机 + 是否需要搜索
    """
    from brain.llm import get_llm

    llm = get_llm()

    # 使用统一提示词系统
    messages = _build_motivation_analysis_messages(recent_messages, user_state)

    try:
        response = await llm.chat(messages)

        # 解析JSON响应（支持嵌套对象，如 knowledge_gap 内层 {}）
        import re
        # ★ 贪婪匹配：找到最外层 { }，支持内层嵌套
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            kg = result.get("knowledge_gap", {})
            if isinstance(kg, str):
                # 兼容旧格式：knowledge_gap 直接是字符串
                kg = {"topic": kg, "confidence": 0.3, "reason": ""} if kg else {}
            return {
                "curiosity": max(0.0, min(1.0, float(result.get("curiosity", 0.3)))),
                "exploration": max(0.0, min(1.0, float(result.get("exploration", 0.3)))),
                "learning_progress": max(0.0, min(1.0, abs(float(result.get("learning_progress", 0.2))))),
                "should_explore": bool(result.get("should_explore", False)),
                "needs_search": bool(result.get("needs_search", False)),
                "search_query": str(result.get("search_query", "")).strip(),
                "reason": str(result.get("reason", "")),
                "knowledge_gap": {
                    "topic": str(kg.get("topic", "")).strip() if isinstance(kg, dict) else "",
                    "confidence": float(kg.get("confidence", 0.5)) if isinstance(kg, dict) else 0.5,
                    "reason": str(kg.get("reason", "")) if isinstance(kg, dict) else "",
                } if kg else {},
            }
    except Exception as e:
        logger.error(f"LLM motivation analysis error: {e}")

    return {
        "curiosity": 0.3,
        "exploration": 0.2,
        "learning_progress": 0.2,
        "should_explore": False,
        "needs_search": False,
        "search_query": "",
        "reason": f"分析失败，使用默认值"
    }


# ============================================================
#  大脑联网搜索（Tavily）
# ============================================================

_last_search_result = None  # 缓存最近一次搜索结果，供对话上下文注入
_last_search_time = None


# 快速决策：短消息/闲聊直接跳过模型
_DECISION_SKIP_PREFIX = ("你好", "嗨", "哈哈", "嗯", "哦", "谢谢", "再见", "忙", "没空",
                         "不想", "可以", "好的", "ok", "嗯嗯", "不错", "厉害")

# 短句里出现这些词，才值得花一次模型调用去判断搜索/任务
_SEARCH_HINTS = ("几点", "时间", "日期", "几号", "星期", "天气", "新闻", "最新", "价格",
                 "多少钱", "汇率", "股价", "查", "搜", "版本", "发布", "比赛", "比分")
_TASK_HINTS = ("帮我", "写", "做", "整理", "统计", "转换", "批量", "汇总", "生成", "分析")

# 任务关键词（命中即执行，零延迟）；contexts 为空表示单独出现就够
_TASK_KEYWORD_RULES = [
    ("写", ["排序", "算法", "代码", "脚本", "程序", "函数", "类"]),
    ("实现", ["排序", "算法", "功能", "爬虫", "计算", "分析"]),
    ("跑", ["一下", "脚本", "程序", "代码"]),
    ("执行", ["脚本", "程序", "代码", "任务"]),
    ("计算", []),
    ("帮我写", []),
    ("帮我做个", []),
    ("写个", ["冒泡", "快排", "插入", "归并", "二分", "递归", "斐波那契",
              "素数", "质数", "阶乘", " fibonacci", "sort", "prime"]),
    ("生成", ["代码", "脚本", "数据", "报告"]),
    ("分析", ["数据", "文本", "文件"]),
    ("处理", ["数据", "文本", "文件", "图片"]),
    ("测试", ["性能", "速度", "效率", "对比"]),
    ("对比", ["排序", "算法", "性能", "效率"]),
    ("benchmark", []),
    ("python", []),
    ("代码", ["运行", "执行", "跑"]),
]


# 到点提醒的规则解析（2026-11-10）：用户说「X 天后提醒我…」不该派执行器写脚本，
# 也不该再问一次模型——直接落进长期记忆，零延迟、不占 token。
_REMIND_DAYS = {"大后天": 3, "后天": 2, "明天": 1, "下周": 7, "下个星期": 7,
                "下个月": 30, "过几天": 3, "这两天": 1}
_REMIND_TRIGGER = ("提醒", "记住", "记得", "别忘了", "别忘", "到时候")
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_hour(s: str) -> Optional[int]:
    """把「三」「十一」这类中文数字转成钟点，转不了返回 None。"""
    if not s:
        return None
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_NUM:
        return 10 + _CN_NUM[s[1]]
    if len(s) == 2 and s[0] in _CN_NUM and s[1] == "十":
        return _CN_NUM[s[0]] * 10
    return None


def _parse_when(raw: str) -> Optional[float]:
    """从一句话里解析出时间点（时间戳）；解析不了返回 None。

    支持：N 分钟后 / N 天后 / N 小时后 / 明天 / 后天 / 大后天 / 今天 /
          周X / 下周X / N月N日 / 上午下午晚上 N 点（含“三点半”“15:30”）。
    没给日期、只说了「晚上」「三点」：按今天算，过了就顺延一天。
    没有明确时刻时，默认当天 09:00；已经过了就顺延一天。
    """
    now = datetime.now()
    # 1) 相对量（分钟最具体，先匹配；否则「30分后」会被当成钟点）
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:分钟|分)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(minutes=float(m.group(1)))).timestamp()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:天|日)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(days=float(m.group(1)))).timestamp()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:个)?(?:小时|钟头)\s*(?:之后|以后|后)", raw)
    if m:
        return (now + timedelta(hours=float(m.group(1)))).timestamp()

    base: Optional[datetime] = None
    if "大后天" in raw:
        base = now + timedelta(days=3)
    elif "后天" in raw:
        base = now + timedelta(days=2)
    elif "明天" in raw:
        base = now + timedelta(days=1)
    elif "今天" in raw or "今晚" in raw:
        base = now

    # 2) 周几 / 下周几
    m = re.search(r"(下)?(?:周|星期)([一二三四五六日天])", raw)
    if m:
        target = _WEEKDAY[m.group(2)]
        days_ahead = (7 - now.weekday()) + target if m.group(1) else (target - now.weekday()) % 7
        base = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=0, second=0, microsecond=0)

    # 3) N 月 N 日
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", raw)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = now.year if (month, day) >= (now.month, now.day) else now.year + 1
        try:
            base = now.replace(year=year, month=month, day=day,
                               hour=9, minute=0, second=0, microsecond=0)
        except ValueError:
            return None

    # 3b) 只说 N 号（20 号），本月过了就顺延到下个月
    if base is None:
        m = re.search(r"(\d{1,2})\s*[日号]", raw)
        if m and 1 <= int(m.group(1)) <= 31:
            day = int(m.group(1))
            year, month = now.year, now.month
            if day < now.day:
                month, year = (1, year + 1) if month == 12 else (month + 1, year)
            try:
                base = now.replace(year=year, month=month, day=day,
                                   hour=9, minute=0, second=0, microsecond=0)
            except ValueError:
                return None

    if base is None:
        # 只说了时段/钟点、没给日期（「晚上提醒我」「三点提醒我」）→ 按今天算。
        # 但纯「帮我记住…」这种完全没有时间线索的，仍然返回 None（只钉住、不到点）。
        has_clock_hint = (
            any(k in raw for k in ("下午", "晚上", "今晚", "傍晚", "中午", "早上", "上午"))
            or bool(re.search(r"(\d{1,2})\s*[点:：]", raw))
            or bool(re.search(r"[一二两三四五六七八九十]{1,3}\s*[点時时]", raw))
        )
        if not has_clock_hint:
            return None
        base = now

    # 4) 具体钟点（「三点」「三点半」「15:30」「下午三点」都认）
    hh, mm = base.hour, base.minute
    got_clock = False
    tm = re.search(r"(\d{1,2})\s*[点:：]\s*(\d{1,2})?", raw)
    cn = re.search(r"([一二两三四五六七八九十]{1,3})\s*[点時时]", raw)
    if tm:
        hh, mm = int(tm.group(1)), int(tm.group(2) or 0)
        got_clock = True
    elif cn:
        h = _cn_hour(cn.group(1))
        if h is not None:
            hh, mm, got_clock = h, 0, True
            tail = raw[cn.end():cn.end() + 4]
            if tail.startswith("半"):
                mm = 30
            elif tail.startswith("一刻"):
                mm = 15
            elif tail.startswith("三刻"):
                mm = 45

    if got_clock:
        if ("下午" in raw or "晚上" in raw or "今晚" in raw or "傍晚" in raw) and hh < 12:
            hh += 12
        elif "中午" in raw and hh < 11:
            hh += 12
        if hh > 23 or mm > 59:
            return None
    else:
        # 没说具体钟点：用当日默认时刻，且分钟必须清零。
        # 之前只改 hh 不改 mm，"今晚提醒我" 会落在「现在这一分」（20:43 而不是 20:00）。
        mm = 0
        if "下午" in raw:
            hh = 14
        elif "晚上" in raw or "今晚" in raw:
            hh = 20
        elif "中午" in raw:
            hh = 12
        elif "早上" in raw or "上午" in raw:
            hh = 9
        else:
            hh = 9      # 与 docstring 一致：没有时刻线索时默认 09:00
    dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if dt.timestamp() <= now.timestamp():
        dt += timedelta(days=1)
    return dt.timestamp()


def _ov_label(uri: str) -> str:
    """给 OpenViking 的一条命中标上来源和日期。

    不然模型只看到一段文字，分不清是今天刚记的还是两周前的旧推断。
    """
    u = uri or ""
    m = re.search(r"/tfgr/(\d{4}-\d{2}-\d{2})\.md$", u)
    if m:
        return f"[{m.group(1)[5:]} 当天汇总]"
    m = re.search(r"/mem/([^/]+)\.md$", u)
    if m:
        try:
            from brain.tfgr import get_tfgr
            e = get_tfgr().get(m.group(1))
            if e:
                day = datetime.fromtimestamp(e.ts).strftime("%m-%d")
                src = "他让记" if e.kind == "memory" else "我推断"
                return f"[{day} {src}]"
        except Exception:
            pass
    return "[旧记录]"


def _reply_mentions(reply: str, text: str) -> bool:
    """回复里是否真的提到了这条提醒（汉字重合度粗判）。

    提醒被注入 prompt 不等于用户听到了：模型可能顺口答别的。
    只有回复里确实出现这条提醒的大部分字，才认为这次提醒已经送达。
    """
    r = (reply or "").lower()
    target = {c for c in (text or "") if "\u4e00" <= c <= "\u9fff"}
    if not r or not target:
        return False
    return sum(1 for c in target if c in r) / len(target) >= 0.4


def _parse_reminder(text: str) -> Optional[Dict[str, Any]]:
    """识别「到时候提醒我…」。返回 {"text", "due_at", "days"}，due_at=None 表示只钉住。"""
    raw = (text or "").strip()
    if not raw or not any(k in raw for k in _REMIND_TRIGGER):
        return None
    # 反问句不算（「你还记得吗」「记不记得」）
    if re.search(r"记(得|不记得)[^，。！？]{0,6}[吗么？?]", raw):
        return None
    has_time = bool(
        re.search(r"\d+\s*(?:天|日|小时|钟头)", raw)
        or re.search(r"(?:周|星期)[一二三四五六日天]", raw)
        or re.search(r"\d{1,2}\s*月\s*\d{1,2}", raw)
        or any(k in raw for k in _REMIND_DAYS)
    )
    imperative = any(k in raw for k in
                     ("提醒我", "提醒一下", "提醒下我", "帮我记", "记住",
                      "记得帮我", "别忘了", "别忘", "到时候提醒"))
    if not (imperative or (has_time and "提醒" in raw)):
        return None
    # 只是问句/求证（「提醒我一下这段代码是干嘛的」）不算要记的事
    if raw.rstrip().endswith(("的", "吗", "？", "?")) and not has_time:
        return None
    due_at = _parse_when(raw)
    days = round((due_at - time.time()) / 86400.0, 1) if due_at else None
    return {"text": raw[:200], "due_at": due_at, "days": days}


async def _brain_quick_decide(user_message: str) -> Dict[str, Any]:
    """一次模型调用同时判断：要不要联网搜索 / 要不要跑代码任务。

    原来这两件事各打一次模型（串行 4 秒、并发也要 2~3 秒），
    合并成一条 JSON 决策后只剩一次往返，是对话延迟最大的一块。
    """
    from brain.llm import get_llm

    msg = (user_message or "").strip()
    low = msg.lower()
    if len(msg) < 4 or any(msg.startswith(k) for k in _DECISION_SKIP_PREFIX):
        return {"needs_search": False, "search_query": "",
                "needs_task": False, "task_description": "", "reason": "快速跳过：短消息/闲聊"}
    for trigger, contexts in _TASK_KEYWORD_RULES:
        if trigger in low and (not contexts or any(c in low for c in contexts)):
            logger.info(f"[Task Decision] KEYWORD HIT: '{trigger}' in '{msg[:40]}'")
            return {"needs_search": False, "search_query": "",
                    "needs_task": True, "task_description": msg,
                    "reason": f"关键词命中: {trigger}"}

    # 短句且没有任何搜索/任务线索 → 纯闲聊，不用花一次模型调用
    if len(msg) <= 12 and not any(k in low for k in (_SEARCH_HINTS + _TASK_HINTS)):
        return {"needs_search": False, "search_query": "",
                "needs_task": False, "task_description": "", "reason": "快速跳过：短句闲聊"}

    prompt = f"""你是大脑的快速决策模块。一次性判断这条用户消息要做什么，只输出 JSON。

用户消息：{user_message}

1) needs_search：需要联网查实时信息（时间/日期/天气/新闻/价格/最新版本、查一下、搜一下，
   或提到你不认识的作品/人物/产品名）→ true；纯闲聊、问你自己、简单的编程概念 → false。
2) needs_task：要求写代码/脚本/算法、计算、数据分析、文件处理、要看到运行结果 → true；
   闲聊、一般知识问答、查记忆/画像/历史 → false。宁可误判 true 也不要漏判。

输出格式（严格 JSON，不要多余文字）：
{{"needs_search": true/false, "search_query": "关键词或空字符串", "needs_task": true/false, "task_description": "要做的事（true 时必填）", "reason": "一句话原因"}}"""

    def _norm(r: dict) -> Dict[str, Any]:
        needs_task = bool(r.get("needs_task"))
        return {
            "needs_search": bool(r.get("needs_search")),
            "search_query": str(r.get("search_query") or "").strip(),
            "needs_task": needs_task,
            "task_description": (str(r.get("task_description") or "").strip()
                                 or (msg if needs_task else "")),
            "reason": str(r.get("reason") or ""),
        }

    _decide_messages = [
        {"role": "system", "content": "你是快速决策模块，只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]

    # 注意：不要用本地模型做这类预热判断（用户明确要求）。
    # 本地 Ollama 只给 OpenViking 的 embedding 这类轻量任务用。
    try:
        response = await get_llm().chat(_decide_messages)
        import re
        m = re.search(r'\{[\s\S]*\}', response or "")
        if m:
            return _norm(json.loads(m.group()))
    except Exception as e:
        logger.warning(f"Quick decision error: {e}")
    return {"needs_search": False, "search_query": "",
            "needs_task": False, "task_description": "", "reason": "决策失败"}


async def _brain_quick_search_decision(user_message: str) -> dict:
    """
    大脑快速判断：用户这条消息是否需要联网搜索？

    这是一个轻量级的 LLM 调用（每次用户发消息时触发），
    只做二分类决策：需要搜 / 不需要搜 + 搜什么关键词

    Returns: {needs_search: bool, search_query: str, reason: str}
    """
    # 已合并到 _brain_quick_decide（一次模型调用同时判断搜索+任务），这里只取搜索结果
    d = await _brain_quick_decide(user_message)
    return {"needs_search": d.get("needs_search", False),
            "search_query": d.get("search_query", ""),
            "reason": d.get("reason", "")}


async def _execute_brain_search(query: str) -> dict:
    """
    大脑自主执行联网搜索

    调用 Tavily API，结果缓存供后续对话使用
    Returns: {success, query, results_count, answer, formatted_context}
    """
    global _last_search_result, _last_search_time

    if not query.strip():
        return {"success": False, "query": query, "error": "空查询"}

    try:
        from brain.web_search import get_tavily_client
        client = get_tavily_client()
        response = await client.search(query)

        if response.success:
            _last_search_result = response
            _last_search_time = datetime.now()

            formatted = client.format_results_for_context(response)
            logger.info(f"Brain search OK: '{query}' → {len(response.results)} results")

            return {
                "success": True,
                "query": query,
                "results_count": len(response.results),
                "answer": response.answer,
                "formatted_context": formatted,
                "sources": [{"title": r.title, "url": r.url} for r in response.results[:3]]
            }
        else:
            logger.warning(f"Brain search failed: {response.error}")
            return {"success": False, "query": query, "error": response.error}

    except Exception as e:
        logger.error(f"Brain search exception: {e}")
        return {"success": False, "query": query, "error": str(e)}


def _get_cached_search_context() -> str:
    """获取缓存的搜索结果上下文（用于注入到LLM对话中）"""
    if _last_search_result is None:
        return ""
    # 搜索结果5分钟内有效
    if _last_search_time and (datetime.now() - _last_search_time).total_seconds() > 300:
        return ""
    from brain.web_search import get_tavily_client
    try:
        client = get_tavily_client()
        return client.format_results_for_context(_last_search_result)
    except Exception:
        return ""


# ============================================================
#  大脑实时任务决策 + 提交执行器并等待结果
# ============================================================

async def _brain_quick_task_decision(user_message: str) -> dict:
    """
    大脑快速判断：用户这条消息是否需要执行代码任务？

    两层判断：
    1. 关键词匹配（100%可靠，零延迟）
    2. LLM 判断（处理模糊情况）
    Returns: {needs_task, task_description, reason}
    """
    # 已合并到 _brain_quick_decide（一次模型调用同时判断搜索+任务），这里只取任务结果
    d = await _brain_quick_decide(user_message)
    return {"needs_task": d.get("needs_task", False),
            "task_description": d.get("task_description", ""),
            "reason": d.get("reason", "")}


async def _brain_submit_and_wait(task_description: str, context: str = "", timeout: int = 30) -> dict:
    """
    提交任务给执行器，并等待结果返回（同步等待模式）

    用于用户实时对话场景：
    用户说"帮我写个冒泡排序" → 大脑提交给执行器 → 等30秒拿结果 → 注入AI回复

    这与 brain_main_loop 中的异步 submit 不同：
    - 异步：大脑不关心结果，继续循环
    - 同步(本函数)：等执行完再回复用户
    """
    from brain.task_executor import get_task_executor, TaskPriority, TaskStatus

    executor = get_task_executor()
    if not executor._running:
        await executor.start()

    # 提交任务
    task = executor.submit(
        description=task_description,
        context=context,
        priority=TaskPriority.URGENT,  # 用户主动请求，高优先级
    )
    if not task:
        return {"success": False, "error": "Executor queue full"}

    # 广播：正在执行
    await manager.broadcast({
        "type": "executor.executing",
        "payload": {
            "task_id": task.id,
            "description": task_description[:100],
        }
    })

    # 轮询等待完成（最多 timeout 秒）
    waited = 0
    poll_interval = 0.5
    while waited < timeout:
        await asyncio.sleep(poll_interval)
        waited += poll_interval

        status = executor.get_status()
        # 检查是否已完成（在 recent_tasks 中找）
        for record in status.get("recent_tasks", []):
            if record.get("id") == task.id and record.get("status") in ("completed", "failed"):
                # 从工作区读取完整输出
                from brain.workspace import get_ai_workspace
                ws = get_ai_workspace()
                execs = ws.list_recent_executions(1)
                if execs and execs[0].get("script_name", "").startswith("task_"):
                    latest = execs[0]
                    return {
                        "success": latest.get("success", False),
                        "script_name": latest.get("script_name", ""),
                        "stdout": latest.get("output_preview", ""),
                        "stderr": "",
                        "duration_ms": latest.get("duration_ms", int(waited * 1000)),
                        "exit_code": latest.get("exit_code", -1),
                        "task": task_description[:100],
                    }
                return {
                    "success": record.get("success", False),
                    "script_name": record.get("script_name", ""),
                    "stdout": record.get("output_preview", ""),
                    "stderr": "",
                    "duration_ms": record.get("duration_ms", int(waited * 1000)),
                    "task": task_description[:100],
                }

    # 超时
    return {"success": False, "error": f"Execution timed out after {timeout}s",
            "task_id": task.id}


# ============================================================
#  大脑工作区执行（LLM 写脚本 → 运行 → 结果回注）
# ============================================================

async def _brain_execute_task(task_description: str, context: str = "") -> dict:
    """
    大脑自主执行任务：LLM 写脚本 → 执行 → 返回结果

    当大脑判断需要做计算/处理数据等任务时调用此函数
    流程：LLM生成脚本代码 → 写入工作区 → 执行 → 返回输出

    Returns: {success, script_name, stdout, stderr, duration_ms}
    """
    from brain.llm import get_llm
    from brain.workspace import get_ai_workspace, ScriptType

    llm = get_llm()
    ws = get_ai_workspace()

    # 让 LLM 根据任务描述生成 Python 脚本
    codegen_prompt = f"""你是一个脚本生成器。根据任务需求，写一个 Python 脚本完成任务。

## 任务
{task_description}

## 上下文（如有）
{context if context else '无'}

## 要求
1. 只输出 Python 代码，不要解释
2. 用 print() 输出结果（这是唯一获取输出的方式）
3. 不要交互式输入（不要 input()）
4. 不要 GUI（不要 matplotlib show 等）
5. 如果需要网络请求用 urllib，不需要额外安装包
6. 脚本要简洁高效
7. 最后 print 一个总结性的结果"""

    try:
        script_code = await llm.chat([
            {"role": "system", "content": "你是脚本生成器。只输出可执行的 Python 代码，不要任何解释文字。"},
            {"role": "user", "content": codegen_prompt}
        ])

        # 清理 LLM 可能包裹的 markdown 代码块标记
        if "```python" in script_code:
            script_code = script_code.split("```python")[1].split("```")[0]
        elif "```" in script_code:
            script_code = script_code.split("```")[1].split("```")[0]
        script_code = script_code.strip()

        if not script_code or len(script_code) < 10:
            return {"success": False, "error": "生成的脚本为空"}

        # 写入工作区
        safe_name = f"task_{datetime.now().strftime('%H%M%S')}.py"
        write_result = await ws.write_script(safe_name, script_code, script_type=ScriptType.PYTHON)

        if not write_result.get("success"):
            return {"success": False, "error": write_result.get("error", "写入失败")}

        logger.info(f"Brain generated script: {safe_name} ({len(script_code)} bytes)")

        # 广播：大脑正在写脚本
        await manager.broadcast({
            "type": "brain.writing_script",
            "payload": {
                "script_name": safe_name,
                "task": task_description[:100],
                "code_preview": script_code[:300],
            }
        })

        # 执行脚本
        exec_result = await ws.execute_script(safe_name, timeout=60)

        output = {
            "success": exec_result.success,
            "script_name": safe_name,
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "duration_ms": exec_result.duration_ms,
            "exit_code": exec_result.exit_code,
            "task": task_description[:100],
        }

        logger.info(f"Brain task executed: {safe_name} -> success={exec_result.success} "
                   f"time={exec_result.duration_ms}ms")

        # 广播执行结果
        await manager.broadcast({
            "type": "brain.task_completed",
            "payload": output
        })

        return output

    except Exception as e:
        logger.error(f"Brain execute task error: {e}")
        return {"success": False, "error": str(e)}


def _format_execution_result_for_context(result: dict) -> str:
    """将执行结果格式化为可注入对话上下文的文本"""
    if not result.get("success") and not result.get("stdout"):
        return ""

    lines = [f"\n[大脑执行任务] \"{result.get('task', '未知任务')}\" 的结果："]
    if result.get("stdout"):
        lines.append(f"\n输出:\n{result['stdout'][:1500]}")
    if result.get("stderr"):
        lines.append(f"\n错误/警告:\n{result['stderr'][:500]}")
    lines.append(f"\n(耗时 {result.get('duration_ms', 0)}ms, 退出码 {result.get('exit_code', '?')})")
    return "\n".join(lines)


# ============================================================
#  Viking 云端同步（streaming_write 自动提取 event + profile）
# ============================================================

_VIKING_SYNC_STATE_FILE = CHAT_DATA_DIR / "viking_sync_state.json"
_last_sync_time = None


async def _streaming_write_recent_to_viking(messages: list):
    """
    实时写入最近几条消息到 Viking
    
    Viking 会自动从对话中提取：
    - event_v1: 事件记忆
    - profile_v1: 用户画像（偏好、习惯等）
    
    这是积累用户画像的核心路径。
    """
    try:
        from brain.memory import get_memory_manager
        memory_manager = get_memory_manager()
        
        if not memory_manager._viking_client:
            return
        
        # 构建标准化的消息格式
        viking_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            # 解析时间戳
            ts = msg.get("timestamp", "")
            time_ms = int(datetime.now().timestamp() * 1000)
            if ts:
                try:
                    time_ms = int(datetime.fromisoformat(ts).timestamp() * 1000)
                except Exception:
                    pass
            viking_messages.append({
                "role": role,
                "content": content,
                "time": time_ms
            })
        
        if viking_messages:
            await memory_manager._viking_client.streaming_write_conversation(
                viking_messages,
                conversation_id=f"chat_{datetime.now().strftime('%Y%m%d')}"
            )
            logger.info(f"Real-time streaming_write: {len(viking_messages)} messages")
    except Exception as e:
        logger.debug(f"Real-time streaming_write skipped: {e}")


def _load_viking_sync_state() -> dict:
    """加载已同步的消息ID列表"""
    if _VIKING_SYNC_STATE_FILE.exists():
        try:
            with open(_VIKING_SYNC_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"synced_ids": [], "last_sync": None, "total_synced": 0}


def _save_viking_sync_state(state: dict):
    """保存同步状态"""
    CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _safe_write_json(_VIKING_SYNC_STATE_FILE, state)


async def _llm_classify_memory(messages: list) -> list:
    """
    用 LLM 对消息进行分类：重要(recent/important) vs 普通(normal)

    大脑自动监听对话，判断哪些记忆值得上传云端
    返回分类后的列表，每项包含 {message, classification, reason}
    """
    from brain.llm import get_llm

    if not messages:
        return []

    llm = get_llm()

    # 取最近20条消息作为分析样本（太多会超token）
    sample = messages[-20:]

    # 构建分析文本
    msgs_text = ""
    for i, msg in enumerate(sample):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:150]
        msgs_text += f"\n[{i+1}] {role}: {content}"

    classify_prompt = f"""你是一个记忆分类系统。分析以下最近对话记录，判断每条消息的记忆价值。

## 分类标准
- **important**: 包含用户偏好、重要决定、个人信息、长期有价值的内容
- **recent**: 近期发生的普通对话，有短期参考价值
- **normal**: 闲聊、无实质内容、重复信息

## 对话记录
{msgs_text}

## 输出格式
严格返回JSON数组，每项包含：
- "index": 消息序号（从0开始）
- "classification": "important" | "recent" | "normal"
- "reason": 简短理由（中文，10字内）

只返回JSON数组，不要其他内容。"""

    try:
        response = await llm.chat([
            {"role": "system", "content": "你是记忆分类系统，只输出JSON数组。"},
            {"role": "user", "content": classify_prompt}
        ])

        import re
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            classifications = json.loads(json_match.group())
            # 将分类结果合并回原始消息
            result = []
            cls_map = {c.get("index"): c for c in classifications}
            for i, msg in enumerate(sample):
                cls_info = cls_map.get(i, {"classification": "normal", "reason": "未分类"})
                result.append({
                    "message": msg,
                    "classification": cls_info.get("classification", "normal"),
                    "reason": cls_info.get("reason", "")
                })
            logger.info(f"LLM memory classified: {len(result)} messages, "
                       f"important={sum(1 for r in result if r['classification']=='important')}, "
                       f"recent={sum(1 for r in result if r['classification']=='recent')}")
            return result
    except Exception as e:
        logger.error(f"LLM memory classification error: {e}")

    # 分类失败时全部标记为 normal
    return [{"message": msg, "classification": "normal", "reason": "分类失败"} for msg in sample]


async def _sync_memories_to_viking(force: bool = False) -> dict:
    """
    将本地聊天记忆同步到 Viking 云端（使用 streaming_write）
    
    流程：
    1. 读取本地聊天记录
    2. 找出未同步的消息
    3. 用 streaming_write 写入 Viking（自动提取 event + profile）
    4. 记录已同步的 ID，避免重复上传
    5. Viking 连不上时优雅降级（只存本地）
    
    Returns: 同步结果统计
    """
    global _last_sync_time

    # 非强制模式下，至少间隔 5 分钟才同步一次
    now = datetime.now()
    if not force and _last_sync_time:
        elapsed = (now - _last_sync_time).total_seconds()
        if elapsed < 300:  # 5分钟
            return {"skipped": True, "reason": f"距上次同步仅{elapsed:.0f}秒"}

    _last_sync_time = now

    # 1. 加载本地聊天记录
    history = _load_chat_history()
    if not history:
        return {"skipped": True, "reason": "无本地记录"}

    # 2. 加载同步状态（已同步的ID）
    sync_state = _load_viking_sync_state()
    synced_ids = set(sync_state.get("synced_ids", []))

    # 3. 找出未同步的消息
    unsynced = [msg for msg in history if msg.get("id", "") not in synced_ids]
    if not unsynced:
        return {"skipped": True, "reason": "没有新消息需要同步"}

    logger.info(f"Memory sync: {len(history)} total, {len(unsynced)} unsynced")

    upload_count = 0
    viking_ok = False

    # 4. 用 streaming_write 写入 Viking（不再用 LLM 分类，让 Viking 自动提取）
    if unsynced:
        try:
            from brain.memory import get_memory_manager
            memory_manager = get_memory_manager()

            # 构建标准化的对话消息
            messages_for_viking = []
            for msg in unsynced:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not content:
                    continue
                # 解析时间戳
                ts = msg.get("timestamp", "")
                time_ms = int(datetime.now().timestamp() * 1000)
                if ts:
                    try:
                        time_ms = int(datetime.fromisoformat(ts).timestamp() * 1000)
                    except Exception:
                        pass
                
                messages_for_viking.append({
                    "role": role,
                    "content": content,
                    "time": time_ms
                })

            # 调用 streaming_write（Viking 自动提取 event + profile）
            if messages_for_viking and memory_manager._viking_client:
                success = await memory_manager._viking_client.streaming_write_conversation(
                    messages_for_viking,
                    conversation_id=f"chat_{now.strftime('%Y%m%d')}"
                )

                if success:
                    upload_count = len(messages_for_viking)
                    viking_ok = True
                    logger.info(f"Viking streaming_write OK: {upload_count} messages")
                else:
                    logger.warning("Viking streaming_write returned False, will retry next time")

        except Exception as e:
            logger.warning(f"Viking cloud sync failed (graceful fallback): {e}")
            viking_ok = False

    # 7. 更新同步状态（无论云端是否成功都标记已处理，避免重复尝试）
    new_synced_ids = [msg.get("id", "") for msg in unsynced]
    sync_state["synced_ids"] = list(set(synced_ids) | set(new_synced_ids))[-500:]  # 最多保留500个ID
    sync_state["last_sync"] = now.isoformat()
    sync_state["total_synced"] = sync_state.get("total_synced", 0) + len(new_synced_ids)
    _save_viking_sync_state(sync_state)

    result = {
        "success": True,
        "total_messages": len(history),
        "unsynced": len(unsynced),
        "uploaded": upload_count,
        "viking_connected": viking_ok,
        "sync_time": now.isoformat(),
    }

    logger.info(f"Memory sync result: {result}")
    return result


async def tfgr_cleanup_loop():
    """定期清理 TFGR 节点：桌面感知事件重复/过期会把节点池撑爆。

    启动 2 分钟后先清一次（把历史堆积清掉），之后每 20 分钟一次。
    """
    from brain.tfgr import get_tfgr
    await asyncio.sleep(120)
    while True:
        try:
            r = get_tfgr().prune()
            if r.get("dup") or r.get("expired") or r.get("capped") or r.get("deleted"):
                logger.info(f"[TFGR] 自动清理: {r}")
            # 机械去重之后，再用本地小模型把「换了个说法」的重复节点合并掉
            from brain.tfgr_compact import compact_tfgr
            c = await compact_tfgr()
            if c.get("merged") or c.get("dropped"):
                logger.info(f"[TFGR] 语义整理: {c}")
            # 推断节点同步进 OpenViking（内容没变会跳过），让「很久以前想过什么」也能语义检索
            from brain.ov_memory import sync_tfgr
            s = await sync_tfgr()
            if s.get("written"):
                logger.info(f"[TFGR] 同步到 OpenViking: {s}")
        except Exception:
            logger.warning("[TFGR] 自动清理失败", exc_info=True)
        await asyncio.sleep(20 * 60)


async def brain_main_loop():
    """大脑主循环 - 用LLM分析对话来驱动内在动机"""
    from brain.intrinsic_reward import get_intrinsic_motivation_system
    from brain.llm import get_llm
    from core.state import get_global_state, UserPresence
    
    intrinsic = get_intrinsic_motivation_system()
    state = get_global_state()

    # 2026-11-04: 流式视觉管道（采集 → 有界缓冲 → 自适应帧率 → 九宫格 → Signal）
    # 默认开启；想彻底关掉桌面采集用 FRIDAY_SCREEN=0。
    import os as _os
    if _os.getenv("FRIDAY_SCREEN", "1") == "1":
        try:
            from brain.screen_capture import start_screen_watcher
            start_screen_watcher()
            logger.info("[Brain] 流式视觉管道已启动")
        except Exception as e:
            logger.warning(f"[Brain] 视觉管道启动失败: {e}")
    else:
        logger.info("[Brain] 视觉管道已关闭（FRIDAY_SCREEN=0）")

    # 前台活动感知：只读窗口标题/进程名，免费；看视频时发 video.watch 唤醒心智
    try:
        from brain.screen_vision import start_screen_vision
        start_screen_vision()
        logger.info("[Brain] 前台活动感知已启动")
    except Exception as e:
        logger.warning(f"[Brain] 前台活动感知启动失败: {e}")

    # ★ 从磁盘恢复大脑状态（重启不丢失）
    _saved = _load_brain_state()
    last_proactive_time = datetime.fromisoformat(_saved.get("last_proactive_time", datetime.now().isoformat()))
    last_analysis_time = datetime.fromisoformat(_saved.get("last_analysis_time", datetime.now().isoformat()))

    # 缓存LLM分析结果 — 优先从磁盘恢复，否则用默认值
    cached_motivation = {
        "curiosity": _saved.get("curiosity", 0.3),
        "exploration": _saved.get("exploration", 0.2),
        "learning_progress": _saved.get("learning_progress", 0.2),
        "should_explore": _saved.get("should_explore", False),
        "needs_search": _saved.get("needs_search", False),
        "search_query": _saved.get("search_query", ""),
        "reason": _saved.get("reason", "初始状态"),
    }
    if _saved:
        logger.info(f"Brain state restored from disk: cur={cached_motivation['curiosity']:.2f} reason='{cached_motivation['reason'][:40]}'")
    else:
        logger.info("No saved brain state, starting fresh")
    
    # 主动对话间隔策略（用户越不活跃，间隔越长）
    PROACTIVE_INTERVALS = {
        UserPresence.ONLINE: 300,
        UserPresence.IDLE: 900,
        UserPresence.AWAY: 3600,
        UserPresence.OFFLINE: 14400,
    }
    # ★ 语音会话激活时：主动对话间隔大幅缩短（更主动）
    PROACTIVE_INTERVALS_VOICE = {
        UserPresence.ONLINE: 90,       # 在线+语音：90秒
        UserPresence.IDLE: 300,        # 暂离+语音：5分钟
        UserPresence.AWAY: 1800,       # 离开+语音：30分钟
        UserPresence.OFFLINE: 7200,    # 离线+语音：2小时
    }

    # ★ 大脑 tick 频率：根据用户状态动态调整
    # 用户在线时高频tick（快速响应），离开时低频（省资源）
    TICK_INTERVALS = {
        UserPresence.ONLINE: 15,     # 在线：每15秒tick一次
        UserPresence.IDLE: 30,       # 暂离：每30秒
        UserPresence.AWAY: 120,      # 离开：每2分钟
        UserPresence.OFFLINE: 300,   # 离线：每5分钟
    }
    # ★ 语音会话激活时：tick 频率大幅提高（实时对话=高频互动）
    TICK_INTERVALS_VOICE = {
        UserPresence.ONLINE: 3,      # 在线+语音：每3秒（高频检测间隙/控制虚拟形象）
        UserPresence.IDLE: 10,       # 暂离+语音：每10秒
        UserPresence.AWAY: 30,       # 离开+语音：每30秒
        UserPresence.OFFLINE: 120,   # 离线+语音：每2分钟
    }

    # LLM 分析间隔也随状态变化
    ANALYSIS_INTERVALS = {
        UserPresence.ONLINE: 120,    # 在线：2分钟分析一次
        UserPresence.IDLE: 300,      # 暂离：5分钟
        UserPresence.AWAY: 600,      # 离开：10分钟
        UserPresence.OFFLINE: 1200,  # 离线：20分钟
    }
    # ★ 语音会话激活时：LLM 分析更频繁（思考频率更高）
    ANALYSIS_INTERVALS_VOICE = {
        UserPresence.ONLINE: 60,     # 在线+语音：1分钟分析一次
        UserPresence.IDLE: 180,      # 暂离+语音：3分钟
        UserPresence.AWAY: 480,      # 离开+语音：8分钟
        UserPresence.OFFLINE: 900,   # 离线+语音：15分钟
    }

    while True:
        try:
            # 初始化推理日志（防止变量未赋值）
            reasoning = {"tick_time": datetime.now().isoformat(), "decision": "skip", "reason": ""}

            # 1. 先更新用户状态，再决定这次睡多久
            presence = state.update_user_presence()

            # ★ 检测语音会话是否活跃（活跃时用更高频的 tick/分析/主动对话）
            voice_active = False
            try:
                from brain.voice_bridge import get_voice_bridge
                voice_active = get_voice_bridge().is_active()
            except Exception:
                pass

            # ★ 自适应 tick 频率（语音激活时用 voice 配置）
            tick_table = TICK_INTERVALS_VOICE if voice_active else TICK_INTERVALS
            tick_interval = tick_table.get(presence, 30)

            # ★ 事件驱动：等待对话结束事件或超时
            #   对话结束后会立即唤醒分析，无对话时按正常 tick 间隔兜底
            try:
                _brain_analysis_trigger.clear()
                dialogue_ended = False
                # wait for either event or timeout
                await asyncio.wait_for(_brain_analysis_trigger.wait(), timeout=tick_interval)
                dialogue_ended = True
                logger.info("[Brain] 对话结束事件触发，立即开始分析")
            except asyncio.TimeoutError:
                dialogue_ended = False
            except Exception:
                dialogue_ended = False

            # 2. 自适应分析间隔（语音激活时用 voice 配置）
            analysis_table = ANALYSIS_INTERVALS_VOICE if voice_active else ANALYSIS_INTERVALS
            analysis_interval = analysis_table.get(presence, 120)
            time_since_analysis = (datetime.now() - last_analysis_time).total_seconds()
            # ★ 对话结束后立即分析，或达到间隔才分析；但至少有 15 秒冷却避免疯狂分析
            # 2026-11-03 架构升级：心智循环（brain/mind.py）接手「什么时候该想」，
            # 主脑不再按固定间隔调模型，只在用户说完一轮后做一次收尾分析。
            # 想恢复旧的定时分析：FRIDAY_PERIODIC_ANALYSIS=1。
            force_analyze = dialogue_ended and time_since_analysis >= 15
            periodic_analyze = _os.getenv("FRIDAY_PERIODIC_ANALYSIS", "0") == "1"
            if force_analyze or (periodic_analyze and time_since_analysis >= analysis_interval):
                # 加载最近对话
                recent_msgs = _load_chat_history()[-10:]

                # LLM分析
                user_state_for_llm = {
                    "presence": presence.value,
                    "idle_seconds": state.get_idle_time(),
                    "no_response_count": state.get_consecutive_no_response(),
                }
                new_motivation = await _llm_analyze_motivation(recent_msgs, user_state_for_llm)
                
                # 平滑更新（避免跳变）
                alpha = 0.6  # 新值权重
                cached_motivation["curiosity"] = alpha * new_motivation["curiosity"] + (1-alpha) * cached_motivation["curiosity"]
                cached_motivation["exploration"] = alpha * new_motivation["exploration"] + (1-alpha) * cached_motivation["exploration"]
                cached_motivation["learning_progress"] = alpha * new_motivation["learning_progress"] + (1-alpha) * cached_motivation["learning_progress"]
                cached_motivation["should_explore"] = new_motivation["should_explore"]
                cached_motivation["reason"] = new_motivation["reason"]
                cached_motivation["needs_search"] = new_motivation.get("needs_search", False)
                cached_motivation["search_query"] = new_motivation.get("search_query", "")
                cached_motivation["needs_task"] = new_motivation.get("needs_task", False)
                cached_motivation["task_description"] = new_motivation.get("task_description", "")

                # ★ 提前初始化 reasoning（供后续搜索/任务/主动对话使用）
                reasoning = {
                    "tick_time": datetime.now().isoformat(),
                    "current_time_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_presence": presence.value,
                    "tick_interval": tick_interval,
                    "analysis_interval": analysis_interval,
                    "idle_seconds": round(state.get_idle_time()),
                    "curiosity": round(cached_motivation["curiosity"], 3),
                    "exploration": round(cached_motivation["exploration"], 3),
                    "should_explore": cached_motivation["should_explore"],
                    "no_response_count": state.get_consecutive_no_response(),
                    "decision": "analyzing",
                    "reason": cached_motivation["reason"],
                }

                # ★ 大脑自主决定是否需要联网搜索
                if cached_motivation.get("needs_search") and cached_motivation.get("search_query"):
                    # ★ 虚拟形象：大脑在搜索 → 歪嘴（思考中）
                    await manager.broadcast({"type": "avatar.thinking"})
                    logger.info(f"[Brain] 广播 avatar.thinking (搜索: {cached_motivation['search_query'][:30]})")
                    search_result = await _execute_brain_search(cached_motivation["search_query"])
                    if search_result.get("success"):
                        reasoning["web_search"] = {
                            "query": search_result["query"],
                            "results_count": search_result["results_count"],
                            "answer": search_result["answer"][:100],
                        }
                        logger.info(f"Brain auto-search: '{cached_motivation['search_query']}' → {search_result['results_count']} results")
                    else:
                        reasoning["web_search"] = {"error": search_result.get("error", "unknown")}

                # ★ 大脑下发任务给执行器（不阻塞！提交后立即继续）
                if cached_motivation.get("needs_task") and cached_motivation.get("task_description"):
                    # ★ 虚拟形象：大脑在执行任务 → 歪嘴（思考中）
                    await manager.broadcast({"type": "avatar.thinking"})
                    logger.info(f"[Brain] 广播 avatar.thinking (任务: {cached_motivation['task_description'][:30]})")
                    # ★ 从聊天历史构建任务上下文（修复 NameError: messages_preview 未定义）
                    recent_chat = _load_chat_history()[-10:]
                    task_context = ""
                    for m in recent_chat:
                        role_label = {"user": "用户", "assistant": "AI"}.get(m.get("role", ""), m.get("role", ""))
                        task_context += f"[{role_label}] {m.get('content', '')[:100]}\n"
                    task_context = task_context[:500]
                    from brain.task_executor import get_task_executor, TaskPriority
                    executor = get_task_executor()
                    submitted = executor.submit(
                        description=cached_motivation["task_description"],
                        context=task_context,
                        priority=TaskPriority.NORMAL,
                    )
                    if submitted:
                        reasoning["brain_task"] = {
                            "task_id": submitted.id,
                            "task": cached_motivation["task_description"][:80],
                            "status": "submitted",  # 已提交，执行器独立运行
                            "queue_size": executor._queue.qsize(),
                        }
                        logger.info(f"Brain submitted task {submitted.id}: "
                                   f"'{cached_motivation['task_description'][:50]}' → executor queue")
                    else:
                        reasoning["brain_task"] = {"error": "executor queue full"}
                
                # 同步到intrinsic系统（让API也能读到）
                try:
                    # action 只传简短类型描述，不传对话原文
                    # 知识缺口不应从循环动作中生成，只从 LLM 动机分析的建议中生成
                    has_dialog = bool(recent_msgs and len(recent_msgs) > 0)
                    action_desc = "user_dialog" if has_dialog else "idle_check"
                    
                    # 只有在有对话时才记录交互，空闲时不触发知识缺口检测
                    if has_dialog:
                        # ★ 构建有区分度的 state，防止 state_key 永远相同导致 curiosity 锁死
                        # 加入最近几条消息的内容哈希 + 当前小时桶 → 不同对话产生不同 state_key
                        import hashlib
                        msg_fingerprint = hashlib.md5(
                            "|".join(m.get("content", "")[:50] for m in recent_msgs[-3:]).encode()
                        ).hexdigest()[:8]
                        time_bucket = f"h{datetime.now().hour}"
                        dynamic_state = {
                            "mode": state.get_dialog_status().value,
                            "user_presence": presence.value,
                            "msg_fingerprint": msg_fingerprint,
                            "time_bucket": time_bucket,
                        }
                        await intrinsic.evaluate_and_act(
                            state=dynamic_state,
                            action=action_desc,
                            result={"curiosity": cached_motivation["curiosity"], "exploration": cached_motivation["exploration"]},
                            confidence=cached_motivation["curiosity"]
                        )
                    
                    # 如果 LLM 动机分析发现了知识缺口，手动创建精确的缺口
                    kg = cached_motivation.get("knowledge_gap")
                    if kg and kg.get("topic") and kg.get("confidence", 1.0) < 0.7:
                        from brain.intrinsic_reward import get_intrinsic_motivation_system
                        ims = get_intrinsic_motivation_system()
                        ims._update_module.detect_knowledge_gap(
                            query=kg["topic"],
                            confidence=kg["confidence"],
                            context={"source": "llm_motivation", "reason": kg.get("reason", "")}
                        )
                except Exception:
                    pass
                
                last_analysis_time = datetime.now()
                logger.info(f"LLM motivation: cur={cached_motivation['curiosity']:.2f} exp={cached_motivation['exploration']:.2f} reason={cached_motivation['reason'][:60]}")
                
                # 同步到共享变量（供API读取）
                _brain_llm_motivation.update(cached_motivation)

                # ★ 持久化大脑状态到磁盘（重启后恢复）
                _save_brain_state({
                    **cached_motivation,
                    "last_analysis_time": last_analysis_time.isoformat(),
                    "last_proactive_time": last_proactive_time.isoformat(),
                    "saved_at": datetime.now().isoformat(),
                })

                # ★ 定期同步记忆到 Viking 云端（每次LLM分析后检查）
                sync_result = await _sync_memories_to_viking()
                if not sync_result.get("skipped"):
                    reasoning["memory_sync"] = sync_result

            # 3. 取当前动机值
            curiosity = cached_motivation["curiosity"]
            should_explore = cached_motivation["should_explore"]
            
            # 4. 计算间隔（语音会话时用更短的间隔，更主动）
            no_resp = state.get_consecutive_no_response()
            proactive_table = PROACTIVE_INTERVALS_VOICE if voice_active else PROACTIVE_INTERVALS
            base_interval = proactive_table.get(presence, 14400)
            current_interval = min(base_interval * (3 ** min(no_resp, 4)), 14400)
            time_since_proactive = (datetime.now() - last_proactive_time).total_seconds()
            
            # 5. 更新推理日志
            reasoning["decision"] = "wait"
            reasoning["current_interval"] = current_interval
            reasoning["time_since_proactive"] = round(time_since_proactive)
            
            # 计算空闲时间（供间隙续接和诊断使用）
            idle_seconds = state.get_idle_time()

            # 用户有活动（空闲<3秒）时重置等待阶段标记
            if idle_seconds < 3 and getattr(state, '_last_wait_stage', -1) != -1:
                state._last_wait_stage = -1  # type: ignore

            # ★ 6a. 语音对话间隙续接（只有语音会话活跃时才可能触发）
            # 场景：AI 刚说了话，用户迟迟不回应，大脑主动续接对话
            if voice_active and idle_seconds > 3:
                recent = _load_chat_history()[-5:]
                if recent and recent[-1].get("role") == "assistant":
                    last_ai_msg = recent[-1].get("content", "")
                    is_open_question = any([
                        "？" in last_ai_msg[-10:],
                        "?" in last_ai_msg[-10:],
                        any(k in last_ai_msg[-20:] for k in ["要不要", "试试", "换个", "再换", "好吗", "可以吗", "行不", "想听", "你想"]),
                    ])
                    # 间隙阈值：语音对话是高频互动
                    # 开放式问题：第一次 4s，第二次 8s
                    # 普通消息：第一次 8s，第二次 15s
                    # 第三次及以上不再续接（避免刷屏）
                    if no_resp >= 2:
                        gap_threshold = 9999
                    elif is_open_question:
                        gap_threshold = 4 if no_resp == 0 else 8
                    else:
                        gap_threshold = 8 if no_resp == 0 else 15

                    # ★ 间隙中的虚拟形象控制（等用户回应时的表情）
                    # 3-6秒：思考表情（歪嘴/蚊香眼），让用户知道AI在等
                    if 3 < idle_seconds <= gap_threshold:
                        wait_stage = int(idle_seconds / 3)  # 3,6,9,12...
                        if wait_stage != getattr(state, '_last_wait_stage', -1):
                            state._last_wait_stage = wait_stage  # type: ignore
                            if wait_stage == 1:  # ~3s
                                wait_expr = '歪嘴'
                            elif wait_stage == 2:  # ~6s
                                wait_expr = '蚊香眼'
                            else:
                                wait_expr = '猫猫嘴'
                            await manager.broadcast({
                                "type": "avatar.idle_action",
                                "payload": {"expression": wait_expr, "reason": f"等待回应{idle_seconds:.0f}s"}
                            })
                            logger.info(f"[Brain] 等待表情: {wait_expr} (idle={idle_seconds:.0f}s stage={wait_stage})")

                    # ★ 超过阈值 → 主动续接对话
                    if idle_seconds > gap_threshold:
                        logger.info(f"[Brain] 检测到对话间隙: no_resp={no_resp} idle={idle_seconds:.0f}s last={last_ai_msg[:30]}")
                        try:
                            from brain.voice_bridge import get_voice_bridge
                            from brain.llm import get_llm
                            vb = get_voice_bridge()
                            if vb.is_active():
                                llm_gap = get_llm()
                                recent_msgs = _load_chat_history()[-10:]
                                gap_prompt = f"""你是 Active AI Pro，正在和东昇语音聊天。
刚才你说了：「{last_ai_msg[:80]}」
但东昇没有回应，已经冷场{idle_seconds:.0f}秒了。

请主动续接对话，不要让对话断掉。要求：
1. 自然地继续刚才的话题，不要等回答了
2. 如果是唱歌/讲故事/展示类的，直接继续下去
3. 语气要轻松，像朋友一样
4. 1-2句话，不要太长
5. 不要提"你怎么不说话"之类的"""

                                gap_msg = await llm_gap.chat([
                                    {"role": "system", "content": gap_prompt},
                                    {"role": "user", "content": "生成续接的话"}
                                ])
                                gap_msg = gap_msg.strip().strip('"').strip("'")
                                if gap_msg:
                                    # 语音活跃时：不直接保存/广播，由 S2S LLM 的实际回复来记录
                                    # 非语音时：正常保存+广播
                                    await manager.broadcast({"type": "avatar.proactive"})
                                    await asyncio.sleep(1.0)
                                    await manager.broadcast({"type": "avatar.speaking"})
                                    sent = await vb.send_text(gap_msg)
                                    if sent:
                                        logger.info(f"[Brain] 对话间隙续接成功: {gap_msg[:40]}")
                                        state.increment_no_response()
                                        last_proactive_time = datetime.now()
                                        reasoning["decision"] = "gap_followup"
                                        reasoning["action_taken"] = gap_msg[:80]
                        except Exception as e:
                            logger.warning(f"[Brain] 对话间隙续接失败: {e}")

            # 6. 主动说话：说话权已统一收归 Mind（brain/mind.py）。
            #    旧主脑（brain_main_loop）不再调模型生成消息，只更新状态、跑表情；
            #    原来的 proactive_dialog 分支（should_explore & curiosity>0.6 & 间隔到了就说话）
            #    已经删除，"为什么不说话"现在只有 Mind 一处答案。
            wait_reasons = []
            if curiosity < 0.6:
                wait_reasons.append(f"好奇心{curiosity:.2f}<0.6")
            if not should_explore:
                wait_reasons.append("不需要探索")
            if time_since_proactive < current_interval:
                wait_reasons.append(f"间隔{time_since_proactive:.0f}/{current_interval}s")
            reasoning["reason"] = "旧主脑不说话（由 Mind 负责）" + (
                "：" + ", ".join(wait_reasons) if wait_reasons else "")

            # ★ 诊断日志：记录大脑 tick 状态
            idle_seconds = state.get_idle_time()
            logger.info(f"[Brain] tick: voice={voice_active} presence={presence.value} "
                        f"cur={curiosity:.2f} explore={should_explore} "
                        f"idle={idle_seconds:.0f}s no_resp={no_resp} "
                        f"不说话的闸门: {', '.join(wait_reasons) or '无（Mind 负责开口）'}")

            # ★ 虚拟形象自嗨：用户长时间不活跃，本地随机挑一组表情（不再为选表情调模型）
            # 条件：空闲>60秒（原来120秒太长，用户在线但不动也允许）
            if idle_seconds > 60:
                # deepseek 模型真实存在的情绪表情（yumi 的那套名字在这只模型上不存在）
                IDLE_EXPRESSIONS = ['爱心眼', '脸红', '屑屑眼', '圈圈眼', '＞＜',
                                    '流泪', '兔牙', '黑框眼镜']
                import random as _random
                picked = [_random.choice(IDLE_EXPRESSIONS)]
                # 偶尔叠两个，显得有变化
                if _random.random() < 0.35:
                    _rest = [x for x in IDLE_EXPRESSIONS if x != picked[0]]
                    picked.append(_random.choice(_rest))
                # 直接走 avatar.action：前端会按 seconds 自动淡出，不会一直挂着
                await manager.broadcast({
                    "type": "avatar.action",
                    "payload": {"expressions": picked, "seconds": 10}
                })
                logger.info(f"[Brain] 空闲表情 {picked} (idle={idle_seconds:.0f}s)")

            # 7. 广播推理日志
            _save_reasoning_log(reasoning)
            await manager.broadcast({"type": "brain.reasoning", "payload": reasoning})
            
        except Exception as e:
            logger.error(f"Brain main loop error: {e}")
            await asyncio.sleep(60)


BRAIN_STATE_FILE = CHAT_DATA_DIR / "brain_state.json"


def _load_brain_state() -> dict:
    """从磁盘加载大脑状态（重启后恢复）"""
    if BRAIN_STATE_FILE.exists():
        try:
            with _json_io_lock:
                with open(BRAIN_STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}


def _save_brain_state(state: dict):
    """持久化大脑状态到磁盘"""
    try:
        CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _safe_write_json(BRAIN_STATE_FILE, state)
    except Exception as e:
        logger.warning(f"Failed to save brain state: {e}")


# 大脑LLM分析结果缓存（供API读取）— 启动时从磁盘恢复
_saved_brain = _load_brain_state()
_brain_llm_motivation = {
    "curiosity": _saved_brain.get("curiosity", 0.3),
    "exploration": _saved_brain.get("exploration", 0.2),
    "learning_progress": _saved_brain.get("learning_progress", 0.2),
    "should_explore": _saved_brain.get("should_explore", False),
    "needs_search": _saved_brain.get("needs_search", False),
    "search_query": _saved_brain.get("search_query", ""),
    "needs_task": _saved_brain.get("needs_task", False),
    "task_description": _saved_brain.get("task_description", ""),
    "reason": _saved_brain.get("reason", "系统刚启动，正在恢复上次状态..."),
    "restored_from_disk": bool(_saved_brain),
}

def get_brain_llm_motivation() -> dict:
    return _brain_llm_motivation

@app.post("/api/chat")
async def test_chat(request: dict):
    """测试聊天端点 - 直接调用LLM"""
    try:
        content = request.get("content", "hello")
        from brain.llm import get_llm
        llm = get_llm()
        response = await llm.chat([
            {"role": "system", "content": "你是一个AI助手，简短回复。"},
            {"role": "user", "content": content}
        ])
        return {"success": True, "content": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    # ★ 启动任务执行器（独立于大脑的后台 Agent）
    from brain.task_executor import get_task_executor
    executor = get_task_executor()

    async def on_executor_completed(task):
        """执行器任务完成时的回调 → 广播给前端"""
        await manager.broadcast({
            "type": "executor.task_completed",
            "payload": {
                "task_id": task.id,
                "description": task.description,
                "script_name": task.script_name,
                "success": True,
                "stdout": task.stdout[:2000],
                "stderr": task.stderr[:500],
                "duration_ms": task.duration_ms,
                "exit_code": task.result.get("exit_code", 0) if task.result else 0,
            }
        })

    async def on_executor_failed(task):
        """执行器任务失败时的回调"""
        await manager.broadcast({
            "type": "executor.task_failed",
            "payload": {
                "task_id": task.id,
                "description": task.description,
                "error": task.error,
                "duration_ms": task.duration_ms,
            }
        })

    # 设置回调并启动
    executor._on_completed = on_executor_completed
    executor._on_failed = on_executor_failed
    await executor.start()

    asyncio.create_task(broadcast_brain_status())
    asyncio.create_task(brain_main_loop())
    asyncio.create_task(tfgr_cleanup_loop())

    # ★ 心智循环（2026-11-03）：事件驱动 + 时间轴 + 钩子 + 流式输出。
    # 平时不调模型，只有事件/钩子到期才醒；醒来后按 token 增量推给前端。
    from brain.mind import get_mind
    _mind = get_mind()
    _mind.bind(
        broadcast=lambda t, p: manager.broadcast({"type": t, "payload": p}),
        save_message=_save_chat_message,
        recent_messages=lambda n: _load_chat_history()[-n:],
    )
    asyncio.create_task(_mind.run())
