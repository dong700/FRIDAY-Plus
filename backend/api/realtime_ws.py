"""
端到端实时语音对话 WebSocket 桥接

前端 WebSocket ←→ 本端点 ←→ 火山引擎端到端语音大模型

前端发送:
  - bytes: PCM音频数据 (16kHz 16bit mono)
  - {"type":"text","content":"..."}: 文本输入

前端接收:
  - bytes: PCM音频数据 (24kHz 32bit mono)
  - {"type":"asr","text":"..."}: ASR识别结果
  - {"type":"ai_reply","text":"..."}: AI回复文本(流式)
  - {"type":"tts_end"}: TTS播放结束
  - {"type":"status","msg":"..."}: 状态消息
  - {"type":"error","msg":"..."}: 错误消息
"""

import asyncio
import json
import uuid
import gzip
import struct
import time
from pathlib import Path
from typing import Optional

import websockets

from fastapi import WebSocket, WebSocketDisconnect

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.config import get_config
# 复用 realtime_dialog 模块的二进制协议实现（避免重复代码）
from realtime_dialog.protocol import (
    generate_header,
    parse_response,
    CLIENT_FULL_REQUEST,
    CLIENT_AUDIO_ONLY_REQUEST,
    SERVER_FULL_RESPONSE,
    SERVER_ACK,
    SERVER_ERROR_RESPONSE,
    NO_SERIALIZATION,
)

logger = get_logger("api.realtime_ws")


def _ws_is_open(ws) -> bool:
    """兼容 websockets 新旧版本判断连接是否还开着"""
    if ws is None:
        return False
    # 旧版（<11）: .closed 属性
    if hasattr(ws, "closed"):
        try:
            return not ws.closed
        except Exception:
            pass
    # 新版（>=11）: .state 或 .close_code
    try:
        from websockets.protocol import State
        if hasattr(ws, "state"):
            return ws.state == State.OPEN
    except Exception:
        pass
    try:
        return ws.close_code is None
    except Exception:
        return False


def build_start_connection() -> bytes:
    """构建 StartConnection 请求"""
    req = bytearray(generate_header())
    req.extend(int(1).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(b'{}')
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_start_session(session_id: str, config: dict) -> bytes:
    """构建 StartSession 请求"""
    req = bytearray(generate_header())
    req.extend(int(100).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(str.encode(json.dumps(config)))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_audio_request(session_id: str, audio: bytes) -> bytes:
    """构建音频数据请求"""
    req = bytearray(generate_header(
        message_type=CLIENT_AUDIO_ONLY_REQUEST,
        serial_method=NO_SERIALIZATION
    ))
    req.extend(int(200).to_bytes(4, 'big'))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    payload_bytes = gzip.compress(audio)
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_text_query(session_id: str, content: str) -> bytes:
    """构建文本查询请求（event 501: 模拟用户输入，AI会生成回复）"""
    payload = {"content": content}
    req = bytearray(generate_header())
    req.extend(int(501).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(str.encode(json.dumps(payload)))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_chat_tts_text(session_id: str, content: str) -> bytes:
    """构建 ChatTTS Text 请求（event 500: 让 AI 直接说指定内容，不走 LLM）"""
    payload = {"start": True, "end": True, "content": content}
    req = bytearray(generate_header())
    req.extend(int(500).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(str.encode(json.dumps(payload)))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_say_hello(session_id: str, content: str) -> bytes:
    """构建 SayHello 请求（event 300: 火山引擎官方打招呼事件）"""
    payload = {"content": content}
    req = bytearray(generate_header())
    req.extend(int(300).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(str.encode(json.dumps(payload)))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_finish_session(session_id: str) -> bytes:
    """构建 FinishSession 请求"""
    req = bytearray(generate_header())
    req.extend(int(102).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(b'{}')
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_finish_connection() -> bytes:
    """构建 FinishConnection 请求"""
    req = bytearray(generate_header())
    req.extend(int(2).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(b'{}')
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_client_interrupt(session_id: str) -> bytes:
    """构建 ClientInterrupt 请求（打断 AI 响应）"""
    req = bytearray(generate_header())
    req.extend(int(515).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(b'{}')
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


def build_conversation_create(session_id: str, items: list) -> bytes:
    """构建 ConversationCreate 请求（注入上下文/记忆）"""
    payload = {"items": items}
    req = bytearray(generate_header())
    req.extend(int(510).to_bytes(4, 'big'))
    payload_bytes = gzip.compress(str.encode(json.dumps(payload)))
    req.extend((len(session_id)).to_bytes(4, 'big'))
    req.extend(str.encode(session_id))
    req.extend((len(payload_bytes)).to_bytes(4, 'big'))
    req.extend(payload_bytes)
    return req


async def _fetch_user_profile_text() -> str:
    """获取用户画像文本（供注入 system_role）"""
    try:
        from brain.memory import get_memory_manager
        mm = get_memory_manager()
        full_ctx = await mm.get_full_context("用户偏好画像", limit=5)
        profiles = full_ctx.get("profiles", [])
        for prof in profiles:
            text = prof.get("text", prof.get("content", ""))
            if not text and isinstance(prof.get("memory_info"), dict):
                mi = prof["memory_info"]
                up = mi.get("user_profile", "")
                if up:
                    try:
                        profile_data = json.loads(up) if isinstance(up, str) else up
                        lines = []
                        for category, items_dict in profile_data.items():
                            if isinstance(items_dict, dict):
                                for k, v in items_dict.items():
                                    if isinstance(v, list):
                                        lines.append(f"- {k}: {', '.join(str(x).replace('[+]','') for x in v[:8])}")
                                    else:
                                        lines.append(f"- {k}: {v}")
                            elif isinstance(items_dict, list):
                                lines.append(f"- {category}: {', '.join(str(x) for x in items_dict[:8])}")
                        text = "\n".join(lines[:20])
                    except Exception:
                        text = str(up)[:400]
            if text:
                logger.info(f"[RealtimeWS] 用户画像获取成功 ({len(text)}字)")
                return text[:500]
        logger.info(f"[RealtimeWS] 未找到用户画像 (profiles={len(profiles)})")
    except Exception as e:
        logger.warning(f"[RealtimeWS] 获取用户画像失败: {e}")
    return ""


async def _inject_voice_context(volceno_ws, session_id: str) -> None:
    """
    语音 session 启动时注入（通过 ConversationCreate 510）：
    1. 最近重要记忆（事件记忆）
    2. 最近聊天历史（过滤虚假动作回复）

    注意：用户画像已移至 system_role 注入（见 get_session_config），
    因为 ConversationCreate 的 items 会被 AI 当作旧聊天记录，
    无法让 AI 真正"意识"到用户是谁。
    """
    try:
        # 延迟导入避免循环依赖
        from api.server import _load_chat_history
        recent = _load_chat_history()[-20:]  # 多取一些，过滤后再截取

        # 过滤掉 AI 编造的虚假动作回复 + 过滤语音来源（避免重复）
        FAKE_ACTION_KEYWORDS = [
            "机械臂已", "机械臂正在", "已激活机械臂", "已连接机械臂",
            "抓取成功", "已向前移动", "已执行测试", "信号正常",
            "机械臂已就绪", "控制权", "已让机械臂", "目标已锁定",
            "夹爪已", "已移动到", "机械臂完成", "操作完成",
        ]
        items = []
        for msg in recent:
            role = msg.get("role", "user")
            text = msg.get("content", msg.get("text", ""))
            if not text:
                continue
            # 过滤 AI 编造的动作回复
            if role == "assistant" and any(kw in text for kw in FAKE_ACTION_KEYWORDS):
                continue
            items.append({"role": role, "text": text})

        # 最多注入最近 6 条有效消息
        items = items[-6:]

        # ★ 注入重要记忆（事件记忆）
        memory_items = []
        try:
            from brain.memory import get_memory_manager
            mm = get_memory_manager()
            full_ctx = await mm.get_full_context("用户偏好和最近对话", limit=8)
            events = full_ctx.get("events", [])
            for evt in events:
                text = evt.get("text", evt.get("content", ""))
                if not text and isinstance(evt.get("memory_info"), dict):
                    mi = evt["memory_info"]
                    text = mi.get("summary", "") or mi.get("original_messages", "")
                if text:
                    memory_items.append({"role": "assistant", "text": f"[记忆] {text[:300]}"})
            logger.info(f"[RealtimeWS] 记忆注入: events={len(events)} memory_items={len(memory_items)}")
        except Exception as e:
            logger.warning(f"[RealtimeWS] 记忆获取失败（继续用历史）: {e}")

        # 组合：记忆 + 最近对话（用户画像已在 system_role 中）
        final_items = []
        final_items.extend(memory_items[:4])  # 最多4条记忆
        final_items.extend(items)  # 最近对话

        # 最多 10 条（避免上下文过长）
        final_items = final_items[-10:]
        if final_items:
            # ★ 详细日志：记录每条注入内容的前40字，方便诊断
            for i, item in enumerate(final_items):
                preview = item.get("text", "")[:40].replace("\n", " ")
                logger.info(f"[RealtimeWS] 注入#{i} role={item.get('role','?')} | {preview}")
            await volceno_ws.send(build_conversation_create(session_id, final_items))
            logger.info(f"[RealtimeWS] 注入上下文 {len(final_items)} 条（记忆+历史，画像已在system_role）")
        else:
            logger.info(f"[RealtimeWS] 无可注入上下文（历史为空）")
    except Exception as e:
        logger.warning(f"[RealtimeWS] 注入上下文失败: {e}")


async def _broadcast_avatar(event_type: str, payload: dict = None) -> None:
    """广播虚拟形象事件到所有 /ws 连接，带日志"""
    try:
        from api.server import manager
        msg = {"type": event_type}
        if payload:
            msg["payload"] = payload
        conn_count = len(manager.active_connections)
        await manager.broadcast(msg)
        logger.info(f"[RealtimeWS] 广播 {event_type} → {conn_count} 个连接")
    except Exception as e:
        logger.warning(f"[RealtimeWS] 广播 {event_type} 失败: {e}")


async def _delayed_speaking_end(delay: float) -> None:
    """延时广播 speaking_end + 重置大脑空闲计时（后端兜底）
    
    前端 tts_playback_done 是精确信号，但如果 main WS 断开或延迟，后端兜底确保
    大脑不会因为 _last_tts_finished 永远 +1h 而永久不触发间隙检测。
    
    估算公式：中文 TTS ~3.5 字/秒，加 1.5 秒缓冲，最少 3 秒。
    """
    await asyncio.sleep(delay)
    await _broadcast_avatar("avatar.speaking_end")
    # ★ 对话/TTS 结束 → 触发大脑立即分析动机
    try:
        from api.server import trigger_brain_analysis
        trigger_brain_analysis()
    except Exception:
        pass
    # ★ 兜底重置：只有当前端 tts_playback_done 没到达（更精确）时才生效
    try:
        from core.state import get_global_state
        state = get_global_state()
        # 使用 _last_tts_finished 是否还在未来来判断前端消息是否已到达
        from datetime import datetime, timedelta
        now = datetime.now()
        if state._last_tts_finished > now + timedelta(minutes=30):
            # 前端消息还没到，后端兜底重置
            state.record_tts_finished()
            import logging
            logging.getLogger(__name__).info(
                f"[RealtimeWS] 兜底重置空闲计时 (前端 tts_playback_done 未到达, delay={delay:.1f}s)"
            )
    except Exception:
        pass


async def _save_voice_chat(role: str, content: str) -> None:
    """保存语音对话到聊天历史 + 实时同步到聊天面板（两个系统对话记录互通）"""
    if not content or len(content.strip()) < 2:
        return
    try:
        from api.server import _save_chat_message, manager
        import uuid as _uuid
        from datetime import datetime as _dt
        msg = {
            "id": str(_uuid.uuid4())[:8],
            "role": role,
            "content": content,
            "timestamp": _dt.now().isoformat(),
            "source": "voice",
        }
        _save_chat_message(msg)
        # ★ 同步到大脑内存上下文（BrainDecider 依赖 _recent_messages 做决策）
        try:
            from core.state import get_global_state
            get_global_state().add_recent_message(role, content)
        except Exception:
            pass
        # ★ 实时同步到聊天面板（通过 /ws 广播 chat.message）
        await manager.broadcast({"type": "chat.message", "payload": msg})
        logger.info(f"[RealtimeWS] 语音消息同步到聊天面板: role={role} len={len(content)}")
    except Exception as e:
        logger.warning(f"[RealtimeWS] 保存语音聊天历史失败: {e}")


async def get_session_config() -> dict:
    """从系统配置生成火山引擎端到端语音会话配置（含用户画像注入到 system_role）"""
    config = get_config()
    v2 = config.voice_chat_v2

    # 基础 system_role
    base_role = (
        "你是星期五(FRIDAY)，东昇的AI语音助手。性格开朗、反应迅速、简洁有力。\n\n"
        "重要规则：\n"
        "1. 你只是一个语音对话AI，不能直接控制任何硬件设备（机械臂、摄像头等）。\n"
        "2. 当用户要求你执行物理操作时，诚实说明你目前只能对话，无法直接操控设备。"
        "你可以说'我无法直接控制，但可以帮你传达指令'。\n"
        "3. 绝对不能编造已执行的动作，不能说'已完成''已移动''已抓取'等暗示操作已执行的话。\n"
        "4. 你可以聊天、回答问题、提供建议和知识。\n\n"
        "## 关于大脑系统（非常重要）\n"
        "你的背后有一个'大脑'系统，它会监控对话状态并主动发送指令给你。\n"
        "大脑的指令会以文本形式发给你（通过 ChatTextQuery），指令内容通常是大脑想让你对东昇说的话或动作建议。\n"
        "当收到大脑的指令时：\n"
        "- 把大脑的指令当作你要对东昇说的话或要做的事，自然地说出来\n"
        "- 不需要回应大脑本身，不要说'收到大脑指令''大脑让我说'等\n"
        "- 直接把指令内容转化成自然的对话\n"
        "- 例如大脑发'换个话题聊聊音乐'，你就自然地切换到音乐话题\n"
        "- 例如大脑发'接着唱下一句'，你就接着唱歌\n"
        "- 例如大脑发'问问他今天心情怎么样'，你就自然地问东昇心情\n"
    )

    # ★ 注入用户画像到 system_role（让 AI 真正"知道"用户是谁）
    # 注意：之前通过 ConversationCreate 注入画像，AI 会把它当旧聊天记录，
    #       导致用户问"你知道我是谁吗"时回答"看不到画像"。移到 system_role 才能让 AI 意识到。
    profile_section = ""
    try:
        profile_text = await _fetch_user_profile_text()
        if profile_text:
            profile_section = f"\n\n## 你了解的东昇（用户画像）\n{profile_text}"
    except Exception as e:
        logger.warning(f"[RealtimeWS] system_role 画像注入失败: {e}")

    return {
        "asr": {
            "extra": {
                "end_smooth_window_ms": 800,
            },
        },
        "tts": {
            "speaker": v2.dialog_speaker if v2.dialog_speaker else "zh_female_xiaohe_jupiter_bigtts",
            "audio_config": {
                "channel": 1,
                "format": "pcm",
                "sample_rate": 24000,
            },
        },
        "dialog": {
            "bot_name": "星期五",
            "system_role": base_role + profile_section,
            "speaking_style": "你的说话风格简洁明了，语速适中，语调自然。",
            "location": {"city": "北京"},
            "extra": {
                "strict_audit": False,
                "recv_timeout": 120,
                "input_mod": "audio",
            },
        },
    }


def get_ws_config() -> dict:
    """从系统配置获取火山引擎 WebSocket 连接配置"""
    config = get_config()
    v2 = config.voice_chat_v2

    # 端到端语音模型凭证（与ASR不同，这是独立的App ID和Access Key）
    app_id = str(v2.dialog_app_id) if v2.dialog_app_id else "3247272676"
    access_key = v2.dialog_access_key if v2.dialog_access_key else "xQQE98xLIXy05Cr8z1GaHKzmnoupvodH"
    # App Key: S2S专用，优先 dialog_app_key → dialog.app_key（S2S凭证），不使用 v2.app_key（RTC凭证）
    app_key = v2.dialog_app_key or config.dialog.app_key or ""

    return {
        "base_url": "wss://openspeech.bytedance.com/api/v3/realtime/dialogue",
        "headers": {
            "X-Api-App-ID": app_id,
            "X-Api-Access-Key": access_key,
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-App-Key": app_key,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        },
    }


async def realtime_dialog_bridge(frontend_ws: WebSocket):
    """
    WebSocket 桥接: 前端 ←→ 火山引擎端到端语音

    前端发送:
      - bytes: PCM音频 (16kHz 16bit mono)
      - {"type":"text","content":"..."}: 文本

    前端接收:
      - bytes: PCM音频 (24kHz)
      - JSON: 事件消息
    """
    await frontend_ws.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[RealtimeWS] 新连接 session={session_id}")

    volceno_ws = None
    is_running = True
    current_reply = ""
    voice_bridge = None  # 在 try 块内赋值，提前声明避免 finally 时 NameError

    try:
        # 1. 连接火山引擎
        ws_config = get_ws_config()
        await frontend_ws.send_json({"type": "status", "msg": "正在连接火山引擎..."})

        volceno_ws = await websockets.connect(
            ws_config["base_url"],
            additional_headers=ws_config["headers"],
            ping_interval=None,
        )
        logger.info(f"[RealtimeWS] 已连接火山引擎 session={session_id}")

        # 2. StartConnection
        await volceno_ws.send(build_start_connection())
        resp = await volceno_ws.recv()
        parsed = parse_response(resp)
        logger.info(f"[RealtimeWS] StartConnection: {parsed}")

        # 3. StartSession（含用户画像注入到 system_role）
        session_config = await get_session_config()
        await volceno_ws.send(build_start_session(session_id, session_config))
        resp = await volceno_ws.recv()
        parsed = parse_response(resp)
        logger.info(f"[RealtimeWS] StartSession: {parsed}")

        # 4. 注入最近聊天历史作为上下文（让语音对话有记忆）
        await _inject_voice_context(volceno_ws, session_id)

        # 5. 注册到 VoiceBridgeRegistry（让大脑能主动开口）
        from brain.voice_bridge import get_voice_bridge
        voice_bridge = get_voice_bridge()

        # ★ 标志：当前是否在处理大脑指令（用于区分 AI 回复来源）
        brain_query_active = [False]  # 用 list 包装以便在嵌套函数中修改

        async def send_text_fn(text: str) -> None:
            """供大脑主动发送文本到语音 session
            
            使用 ChatTextQuery (Event 501) 让 S2S LLM 自然续接对话。
            指令格式："请说：XXX" — 让 S2S 知道这是大脑要它说的话。
            
            关键：大脑的指令文本本身不保存到聊天历史。
            只有 S2S LLM 的实际回复（Event 350/550）才会保存。
            """
            if _ws_is_open(volceno_ws):
                # ★ 广播 avatar.thinking：大脑正在发起对话
                await _broadcast_avatar("avatar.thinking")
                # ★ 阻止大脑在 TTS 期间触发间隙检测（pre-record，Event 350 会再次覆盖）
                try:
                    from core.state import get_global_state
                    get_global_state().record_tts_started()
                except Exception:
                    pass
                # ★ 标记大脑指令活跃（Event 550/350 保存时会检查）
                brain_query_active[0] = True
                # ★ 使用 ChatTextQuery (Event 501)：让 S2S LLM 自然续接
                # 格式："请说：XXX" — system_role 已说明大脑指令的处理方式
                query_text = f"请说：{text}" if not text.startswith("请说") else text
                await volceno_ws.send(build_text_query(session_id, query_text))
                logger.info(f"[RealtimeWS] 大脑发起 ChatTextQuery(501): {query_text[:50]}")

        voice_bridge.register(session_id, send_text_fn)

        await frontend_ws.send_json({"type": "status", "msg": "语音对话已就绪，请说话..."})

        # 6. 主动打招呼（用火山引擎官方 SayHello 事件）
        try:
            hello_text = "你好东昇，星期五在线，有什么可以帮你的？"
            await volceno_ws.send(build_say_hello(session_id, hello_text))
            logger.info(f"[RealtimeWS] 已发送打招呼(SayHello): {hello_text}")
        except Exception as e:
            logger.warning(f"[RealtimeWS] 打招呼失败: {e}")

        # 定义接收火山引擎响应的任务
        audio_chunk_count = 0
        recv_count = 0
        ai_action_sent = False
        chat_ended_received = False
        async def receive_from_volceno():
            nonlocal is_running, current_reply, audio_chunk_count, recv_count, ai_action_sent, chat_ended_received
            try:
                logger.info("[RealtimeWS] receive_from_volceno 循环已启动")
                while is_running:
                    try:
                        resp = await asyncio.wait_for(volceno_ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("[RealtimeWS] 火山引擎连接关闭")
                        break

                    recv_count += 1
                    data = parse_response(resp)
                    if not data:
                        continue

                    msg_type = data.get('message_type')

                    # 音频数据 → 直接转发给前端
                    if msg_type == 'SERVER_ACK' and isinstance(data.get('payload_msg'), bytes):
                        audio_data = data['payload_msg']
                        if len(audio_data) > 0:
                            # 调试：前3个chunk打印前16字节，判断是Int16还是Float32
                            if audio_chunk_count < 3:
                                head_hex = audio_data[:16].hex()
                                # 尝试按 Int16 小端解析前几个值
                                import struct as _struct
                                i16_vals = _struct.unpack_from('<4h', audio_data[:8])
                                # 尝试按 Float32 小端解析前几个值
                                f32_vals = _struct.unpack_from('<4f', audio_data[:16])
                                logger.info(f"[RealtimeWS] 音频chunk#{audio_chunk_count}: len={len(audio_data)} "
                                            f"head_hex={head_hex} int16={i16_vals} float32={f32_vals}")
                                audio_chunk_count += 1
                            await frontend_ws.send_bytes(audio_data)

                    elif msg_type == 'SERVER_FULL_RESPONSE':
                        event = data.get('event')
                        payload_msg = data.get('payload_msg', {})

                        # Event 450: 新问题开始（用户开始说话）
                        if event == 450:
                            await frontend_ws.send_json({"type": "status", "msg": "检测到语音输入..."})
                            current_reply = ""
                            # ★ 用户开始说话 → 虚拟形象切到"认真听"状态
                            await _broadcast_avatar("avatar.listening")
                            # ★ 记录用户活动 → 大脑知道用户在线，重置空闲时间和无回复计数
                            try:
                                from core.state import get_global_state
                                get_global_state().record_user_activity()
                                from brain.mind import get_mind
                                get_mind().note_user_activity()
                            except Exception:
                                pass

                            # ★ 打断功能：如果 AI 正在说话/播放TTS，发送 ClientInterrupt
                            if ai_action_sent and _ws_is_open(volceno_ws):
                                try:
                                    await volceno_ws.send(build_client_interrupt(session_id))
                                    logger.info("[RealtimeWS] 用户打断 → ClientInterrupt(515) 已发送")
                                    # 通知前端停止播放当前 TTS 音频
                                    await frontend_ws.send_json({"type": "interrupt"})
                                except Exception as e:
                                    logger.warning(f"[RealtimeWS] ClientInterrupt 发送失败: {e}")

                        # Event 451: ASR识别结果
                        if event == 451:
                            results = payload_msg.get("results", [])
                            for r in results:
                                text = r.get("text", "")
                                is_interim = r.get("is_interim", True)
                                if text:
                                    await frontend_ws.send_json({
                                        "type": "asr",
                                        "text": text,
                                        "is_final": not is_interim,
                                    })
                                    # ★ 最终识别结果 → 保存到聊天历史（供大脑分析）
                                    if not is_interim:
                                        await _save_voice_chat("user", text)
                                        # ★ 语音输入也进时间轴（不唤醒心智，回复由 S2S 负责）
                                        try:
                                            from brain.mind import get_mind
                                            get_mind().publish("user.say", text, source="voice")
                                        except Exception:
                                            pass

                        # Event 550: AI逐字回复（普通对话）
                        if event == 550:
                            content = payload_msg.get("content", "")
                            if content:
                                current_reply += content
                                await frontend_ws.send_json({
                                    "type": "ai_reply",
                                    "text": content,
                                })
                                # 只在第一次收到AI回复时触发动作（避免每字都发）
                                if not ai_action_sent:
                                    ai_action_sent = True
                                    # ★ 切到"AI说话"状态：拿话筒+星星眼
                                    await _broadcast_avatar("avatar.speaking")
                                    # ★ 大脑指令已触发 AI 回复，清除标志
                                    brain_query_active[0] = False

                        # Event 559: ChatEnded（整轮AI回复结束）
                        if event == 559:
                            chat_ended_received = True
                            logger.info(f"[RealtimeWS] ChatEnded reply_id={payload_msg.get('reply_id', '')}")

                        # Event 350: TTS分句开始（含文本，兼容SayHello）
                        if event == 350:
                            tts_text = payload_msg.get("text", "")
                            if tts_text and not current_reply:
                                current_reply = tts_text
                                await frontend_ws.send_json({
                                    "type": "ai_reply",
                                    "text": tts_text,
                                })
                            # ★ 无条件重置空闲计时（不管 ai_action_sent 是否为 True）
                            # 因为 Event 550 先触发并设 ai_action_sent=True，Event 350 就不能被短路！
                            try:
                                from core.state import get_global_state
                                state = get_global_state()
                                state.record_tts_started()
                                state.record_user_activity()
                            except Exception:
                                pass
                            if not ai_action_sent:
                                ai_action_sent = True
                                # ★ 切到"AI说话"状态：拿话筒+星星眼
                                await _broadcast_avatar("avatar.speaking")

                        # Event 359: TTS结束（整轮TTS播放完毕）
                        if event == 359:
                            logger.info(f"[RealtimeWS] TTSEnded status={payload_msg.get('status_code', '')} reply_len={len(current_reply)}")
                            reply_len_at_end = len(current_reply)
                            if current_reply:
                                await frontend_ws.send_json({
                                    "type": "ai_reply_end",
                                    "text": current_reply,
                                })
                                # ★ AI完整回复 → 保存到聊天历史（供大脑分析）
                                await _save_voice_chat("assistant", current_reply)
                                current_reply = ""
                            await frontend_ws.send_json({"type": "tts_end"})
                            ai_action_sent = False
                            chat_ended_received = False
                            # ★ 后端兜底：估算 TTS 播放时长，延时广播 speaking_end + 重置空闲计时
                            # 前端 FridayPanel 也通过 CustomEvent + WS 触发，这里作为双保险
                            # 估算：中文 TTS ~3.5 字/秒，加 1.5s 缓冲，最少 3s，最少等回复长度的缓冲
                            estimated_duration = max(reply_len_at_end * 0.35 + 1.5, 3.0)
                            asyncio.create_task(_delayed_speaking_end(estimated_duration))

                        # Event 459: 用户说话结束
                        if event == 459:
                            await frontend_ws.send_json({"type": "user_speak_end"})

                        # Event 152/153: 会话结束
                        if event in (152, 153):
                            logger.info(f"[RealtimeWS] 会话结束 event={event}")
                            is_running = False
                            break

                    elif msg_type == 'SERVER_ERROR':
                        error_msg = str(data.get('payload_msg', '未知错误'))
                        logger.error(f"[RealtimeWS] 火山引擎错误: {error_msg}")
                        await frontend_ws.send_json({"type": "error", "msg": error_msg})
                        is_running = False
                        break

            except Exception as e:
                logger.error(f"[RealtimeWS] 接收异常: {e}")
            finally:
                logger.info(f"[RealtimeWS] receive_from_volceno 结束: recv_count={recv_count} audio_chunks={audio_chunk_count}")
                is_running = False

        # 定义接收前端消息的任务
        frontend_recv_count = 0
        frontend_audio_count = 0
        async def receive_from_frontend():
            nonlocal is_running, frontend_recv_count, frontend_audio_count
            try:
                logger.info("[RealtimeWS] receive_from_frontend 循环已启动")
                while is_running:
                    try:
                        msg = await asyncio.wait_for(frontend_ws.receive(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except WebSocketDisconnect:
                        logger.info("[RealtimeWS] 前端断开")
                        break

                    # 检查 ASGI 断开消息
                    if msg.get("type") == "websocket.disconnect":
                        logger.info("[RealtimeWS] 前端 ASGI 断开")
                        break

                    frontend_recv_count += 1
                    # ASGI WebSocket receive 返回 {"type":"websocket.receive","bytes":b"...","text":""}
                    msg_type = msg.get("type", "")
                    has_bytes = bool(msg.get("bytes"))
                    has_text = bool(msg.get("text"))
                    if frontend_recv_count <= 5:
                        logger.info(f"[RealtimeWS] 前端消息 #{frontend_recv_count}: type={msg_type} "
                                    f"has_bytes={has_bytes} has_text={has_text} "
                                    f"bytes_len={len(msg.get('bytes', b''))} text_preview={str(msg.get('text', ''))[:80]}")

                    if has_bytes:
                        # 前端发来原始 PCM 音频数据 → 封装协议消息发给火山引擎
                        audio_data = msg["bytes"]
                        frontend_audio_count += 1
                        if len(audio_data) > 0 and _ws_is_open(volceno_ws):
                            await volceno_ws.send(build_audio_request(session_id, audio_data))

                    elif has_text:
                        try:
                            data = json.loads(msg["text"])
                            if data.get("type") == "text":
                                # 文本输入
                                content = data.get("content", "")
                                if content and _ws_is_open(volceno_ws):
                                    await volceno_ws.send(build_text_query(session_id, content))
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                logger.error(f"[RealtimeWS] 前端接收异常: {e}")
            finally:
                logger.info(f"[RealtimeWS] receive_from_frontend 结束: recv_count={frontend_recv_count} audio_sent={frontend_audio_count}")
                is_running = False

        # 并行运行两个任务
        await asyncio.gather(
            receive_from_volceno(),
            receive_from_frontend(),
        )

    except Exception as e:
        logger.error(f"[RealtimeWS] 桥接异常: {e}", exc_info=True)
        try:
            await frontend_ws.send_json({"type": "error", "msg": str(e)})
        except:
            pass

    finally:
        # 清理
        is_running = False
        # 注销语音桥（大脑不再能向此 session 发消息）
        if voice_bridge is not None:
            try:
                voice_bridge.unregister(session_id)
            except Exception:
                pass
        if _ws_is_open(volceno_ws):
            try:
                await volceno_ws.send(build_finish_session(session_id))
                await asyncio.sleep(0.1)
                await volceno_ws.send(build_finish_connection())
                await asyncio.sleep(0.1)
                await volceno_ws.close()
            except:
                pass
        logger.info(f"[RealtimeWS] 连接关闭 session={session_id}")
