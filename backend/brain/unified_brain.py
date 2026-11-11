"""
统一大脑 - 状态机外壳（B路线）

架构说明：
- 原设计：UnifiedBrain 直接管理火山引擎 WebSocket 连接（_connect/_receive_loop）
- 实际情况：start() 从未被调用，整套连接代码是死代码
- B路线：brain_main_loop（server.py）驱动动机分析，realtime_ws.py 桥管理语音连接，
         voice_bridge.py 连接两者

本模块保留：
- BrainMode 状态枚举（供 API 手动切换模式）→ 已提取到 brain/modes.py
- UnifiedContext 上下文管理（供 get_status 返回摘要）→ 已提取到 brain/context.py
- switch_mode / get_status 接口（供 server.py 调用）

已移除（B路线不再需要）：
- _connect / _receive_loop / _reconnect  → 语音连接由 realtime_ws.py 桥管理
- run_forever / _idle_tick / _active_tick / _reflect_tick → 由 brain_main_loop 替代
- send_message / add_context → 语音由 voice_bridge.py 管理
- start / shutdown → 不再需要直接启动大脑
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.state import get_global_state, DialogStatus
from brain.modes import BrainMode
from brain.context import UnifiedContext

logger = get_logger("brain.unified_brain")

# 向后兼容：允许 from brain.unified_brain import BrainMode
__all__ = ["UnifiedBrain", "get_unified_brain", "BrainMode"]


class UnifiedBrain:
    """
    统一大脑外壳 - 状态机 + 上下文（B路线：不直接管理连接）

    连接状态由 voice_bridge.py 跟踪，大脑逻辑由 brain_main_loop 驱动。
    本类仅保留模式切换和状态查询接口供 API 调用。
    """

    _instance: Optional['UnifiedBrain'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._mode: BrainMode = BrainMode.IDLE
        self._context = UnifiedContext()
        self._global_state = get_global_state()

        self._mode_transition_callbacks: Dict[str, List[Callable]] = {
            "idle_to_active": [],
            "active_to_reflect": [],
            "reflect_to_idle": [],
            "reflect_to_active": [],
            "idle_to_reflect": [],
        }

        logger.info("UnifiedBrain shell initialized (B-route: no direct connection)")

    @property
    def mode(self) -> BrainMode:
        return self._mode

    @property
    def context(self) -> UnifiedContext:
        return self._context

    @property
    def is_connected(self) -> bool:
        """B路线：连接状态由 voice_bridge 跟踪"""
        try:
            from brain.voice_bridge import get_voice_bridge
            return get_voice_bridge().is_active()
        except Exception:
            return False

    def on_mode_transition(self, transition: str, callback: Callable) -> None:
        if transition in self._mode_transition_callbacks:
            self._mode_transition_callbacks[transition].append(callback)

    async def _fire_transition(self, transition: str, **kwargs) -> None:
        for callback in self._mode_transition_callbacks.get(transition, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                logger.error(f"Mode transition callback error: {e}")

    async def switch_mode(self, new_mode: BrainMode, reason: str = "") -> Dict[str, Any]:
        """
        切换大脑模式（不重启连接，只改变行为模式）

        B路线下：模式切换主要影响 brain_main_loop 的行为策略，
        不再涉及连接管理。
        """
        old_mode = self._mode

        if old_mode == new_mode:
            return {"success": True, "mode": new_mode.value, "changed": False}

        logger.info(f"Mode switch: {old_mode.value} → {new_mode.value} ({reason})")

        transition = f"{old_mode.value}_to_{new_mode.value}"

        if new_mode == BrainMode.ACTIVE:
            self._global_state.set_dialog_status(DialogStatus.IN_CONVERSATION)
        elif new_mode == BrainMode.IDLE:
            self._global_state.set_dialog_status(DialogStatus.IDLE)
        elif new_mode == BrainMode.REFLECT:
            self._global_state.set_dialog_status(DialogStatus.IDLE)

        self._mode = new_mode

        await self._fire_transition(transition, old_mode=old_mode, new_mode=new_mode, reason=reason)

        return {
            "success": True,
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "reason": reason
        }

    def get_status(self) -> Dict[str, Any]:
        """获取大脑状态"""
        return {
            "mode": self._mode.value,
            "connected": self.is_connected,
            "session_id": None,  # B路线：session由voice_bridge管理
            "dialog_history_count": 0,  # B路线：由server.py从磁盘加载
            "context_summary": self._context.get_summary(),
            "last_user_activity": None,
            "last_reflect_time": None,
        }


def get_unified_brain() -> UnifiedBrain:
    """获取统一大脑单例"""
    return UnifiedBrain()
