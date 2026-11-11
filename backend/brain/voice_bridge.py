"""
语音桥注册表 - 连接大脑与端到端语音对话

大脑（brain_main_loop）通过此注册表向活跃的语音 session 发送文本，
让 AI 主动开口说话。realtime_ws.py 桥在建立连接时注册，断开时注销。

数据流：
    brain_main_loop 主动开口
        → VoiceBridgeRegistry.send_text(msg)
        → 注册的 send_fn(msg)
        → realtime_ws.py 桥调用 build_text_query() 发给火山引擎
        → 火山引擎 TTS → 前端播放
"""

import asyncio
from typing import Callable, Awaitable, Dict, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("brain.voice_bridge")


class VoiceBridgeRegistry:
    """
    语音桥注册表（单例）

    跟踪当前活跃的语音对话 session，提供大脑主动发送文本的接口。
    """

    _instance: Optional['VoiceBridgeRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # session_id → async send_text_fn(text)
        self._active_sessions: Dict[str, Callable[[str], Awaitable[None]]] = {}

    def register(self, session_id: str, send_text_fn: Callable[[str], Awaitable[None]]) -> None:
        """注册一个活跃的语音会话"""
        self._active_sessions[session_id] = send_text_fn
        logger.info(f"Voice session registered: {session_id} (total={len(self._active_sessions)})")

    def unregister(self, session_id: str) -> None:
        """注销语音会话"""
        removed = self._active_sessions.pop(session_id, None)
        if removed:
            logger.info(f"Voice session unregistered: {session_id} (total={len(self._active_sessions)})")

    def is_active(self) -> bool:
        """是否有活跃的语音会话"""
        return len(self._active_sessions) > 0

    async def send_text(self, text: str) -> bool:
        """
        向活跃的语音会话发送文本（让 AI 说出来）

        Returns: True 如果成功发送，False 如果没有活跃会话
        """
        if not self._active_sessions:
            return False

        # 取一个活跃 session（通常只有一个）
        dead_sessions = []
        sent = False
        for session_id, send_fn in self._active_sessions.items():
            try:
                await send_fn(text)
                sent = True
                logger.info(f"Voice bridge sent text to session {session_id}: {text[:50]}")
                break  # 只发给一个
            except Exception as e:
                logger.error(f"Voice bridge send_text error for {session_id}: {e}")
                dead_sessions.append(session_id)

        for sid in dead_sessions:
            self._active_sessions.pop(sid, None)

        return sent

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active(),
            "session_count": len(self._active_sessions),
            "session_ids": list(self._active_sessions.keys()),
        }


def get_voice_bridge() -> VoiceBridgeRegistry:
    """获取语音桥注册表单例"""
    return VoiceBridgeRegistry()
