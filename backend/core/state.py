"""
全局状态管理器 - 追踪系统状态
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
import time

def _safe_write_json(filepath: Path, data: dict, retries: int = 3) -> None:
    """原子写入JSON文件 — 写临时文件后 os.replace，避免 Windows 文件锁问题

    ★ 不用 tempfile.NamedTemporaryFile：Python 3.12 的 TMP_MAX 是 2^31，
    文件系统过滤驱动若对 O_EXCL 创建持续返回"已存在"，会无限重试烧满 CPU。
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    last_err = None
    for attempt in range(retries):
        tmp_name = f"{filepath}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_name, 'w', encoding='utf-8') as f:
                f.write(payload)
            os.replace(tmp_name, str(filepath))
            return
        except Exception as e:
            last_err = e
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(0.1 * (attempt + 1))
    raise last_err

logger = get_logger("core.state")


class UserPresence(Enum):
    """用户在线状态"""
    ONLINE = "online"          # 用户正在操作
    IDLE = "idle"              # 用户暂时离开（5分钟无操作）
    AWAY = "away"              # 用户离开（30分钟无操作）
    OFFLINE = "offline"        # 用户长时间不在线（2小时无操作）


class DialogStatus(Enum):
    IDLE = "idle"
    IN_CONVERSATION = "in_conversation"
    USER_TYPING = "user_typing"
    PROCESSING = "processing"


class GlobalState:
    """
    全局状态管理器
    
    追踪：
    - 对话状态（是否正在对话）
    - 用户活动
    - 任务执行状态
    - 最近记忆
    """
    
    _instance: Optional['GlobalState'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._dialog_status: DialogStatus = DialogStatus.IDLE
        self._last_user_activity: datetime = datetime.now()
        self._last_assistant_activity: datetime = datetime.now()
        self._last_tts_finished: datetime = datetime.now()  # 前端 TTS 播完时刻（精确）
        self._user_idle_timeout: int = 60
        self._conversation_start: Optional[datetime] = None
        self._pending_task_ids: set = set()
        self._state_file = Path(__file__).parent.parent / "data" / "global_state.json"
        self._recent_messages: List[Dict[str, Any]] = []
        self._max_recent_messages = 20
        
        # 用户在线状态
        self._user_presence: UserPresence = UserPresence.ONLINE
        self._online_timeout: int = 300       # 5分钟无操作 → IDLE
        self._idle_timeout: int = 1800        # 30分钟无操作 → AWAY
        self._away_timeout: int = 7200        # 2小时无操作 → OFFLINE
        self._consecutive_no_response: int = 0  # 连续无回复次数
        
        # 硬件可用性
        self._arm_available: bool = False
        
        self._load_state()
    
    def _load_state(self) -> None:
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._pending_task_ids = set(data.get("pending_task_ids", []))
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "dialog_status": self._dialog_status.value,
                "last_user_activity": self._last_user_activity.isoformat(),
                "pending_task_ids": list(self._pending_task_ids),
            }
            _safe_write_json(self._state_file, data)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def set_dialog_status(self, status: DialogStatus) -> None:
        old_status = self._dialog_status
        self._dialog_status = status
        
        if status == DialogStatus.IN_CONVERSATION and old_status == DialogStatus.IDLE:
            self._conversation_start = datetime.now()
            logger.info("Conversation started")
        elif status == DialogStatus.IDLE and old_status != DialogStatus.IDLE:
            if self._conversation_start:
                duration = (datetime.now() - self._conversation_start).total_seconds()
                logger.info(f"Conversation ended, duration: {duration:.1f}s")
            self._conversation_start = None
        
        self._save_state()
    
    def get_dialog_status(self) -> DialogStatus:
        return self._dialog_status
    
    def is_in_conversation(self) -> bool:
        return self._dialog_status in [DialogStatus.IN_CONVERSATION, DialogStatus.USER_TYPING, DialogStatus.PROCESSING]
    
    def record_user_activity(self) -> None:
        """记录用户活动 - 任何用户操作都应调用此方法"""
        old_presence = self._user_presence
        self._last_user_activity = datetime.now()
        self._user_presence = UserPresence.ONLINE
        self._consecutive_no_response = 0  # 用户有活动，重置无回复计数
        
        if old_presence != UserPresence.ONLINE:
            logger.info(f"User presence changed: {old_presence.value} → online")
        
        self.set_dialog_status(DialogStatus.IN_CONVERSATION)
    
    def record_assistant_activity(self) -> None:
        self._last_assistant_activity = datetime.now()
    
    def update_user_presence(self) -> UserPresence:
        """根据最后活动时间更新用户在线状态（需定期调用）"""
        idle_seconds = (datetime.now() - self._last_user_activity).total_seconds()
        old_presence = self._user_presence
        
        if idle_seconds >= self._away_timeout:
            self._user_presence = UserPresence.OFFLINE
        elif idle_seconds >= self._idle_timeout:
            self._user_presence = UserPresence.AWAY
        elif idle_seconds >= self._online_timeout:
            self._user_presence = UserPresence.IDLE
        else:
            self._user_presence = UserPresence.ONLINE
        
        if old_presence != self._user_presence:
            logger.info(f"User presence: {old_presence.value} → {self._user_presence.value} (idle {idle_seconds:.0f}s)")
        
        return self._user_presence
    
    def get_user_presence(self) -> UserPresence:
        return self._user_presence
    
    def is_user_online(self) -> bool:
        """用户是否在线（ONLINE 或 IDLE 状态）"""
        return self._user_presence in [UserPresence.ONLINE, UserPresence.IDLE]
    
    def get_consecutive_no_response(self) -> int:
        """获取连续无回复次数"""
        return self._consecutive_no_response
    
    def increment_no_response(self) -> int:
        """增加无回复计数，返回新值"""
        self._consecutive_no_response += 1
        return self._consecutive_no_response
    
    def reset_no_response(self) -> None:
        """重置无回复计数（用户回复时调用）"""
        self._consecutive_no_response = 0
    
    def is_user_idle(self) -> bool:
        idle_time = (datetime.now() - self._last_user_activity).total_seconds()
        return idle_time >= self._user_idle_timeout
    
    def record_tts_started(self) -> None:
        """记录 TTS 开始播放 — 将 finished 时间推到未来，阻止空闲计时增长"""
        self._last_tts_finished = datetime.now() + timedelta(hours=1)
    
    def record_tts_finished(self) -> None:
        """记录前端 TTS 播放完成时刻（用于精确的空闲时间计算）"""
        self._last_tts_finished = datetime.now()
    
    def get_idle_time(self) -> float:
        """获取空闲时间（从最后用户活动或 TTS 播完两者中较晚的开始计算）"""
        ref = max(self._last_user_activity, self._last_tts_finished)
        return (datetime.now() - ref).total_seconds()
    
    def add_recent_message(self, role: str, content: str) -> None:
        self._recent_messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self._recent_messages) > self._max_recent_messages:
            self._recent_messages = self._recent_messages[-self._max_recent_messages:]
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._recent_messages[-limit:]
    
    def register_pending_task(self, task_id: str) -> None:
        self._pending_task_ids.add(task_id)
        self._save_state()
    
    def complete_task(self, task_id: str) -> None:
        self._pending_task_ids.discard(task_id)
        self._save_state()
    
    def get_pending_task_count(self) -> int:
        return len(self._pending_task_ids)
    
    @property
    def arm_available(self) -> bool:
        return self._arm_available
    
    @arm_available.setter
    def arm_available(self, value: bool) -> None:
        self._arm_available = value
    
    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "dialog_status": self._dialog_status.value,
            "is_in_conversation": self.is_in_conversation(),
            "user_presence": self._user_presence.value,
            "is_user_online": self.is_user_online(),
            "is_user_idle": self.is_user_idle(),
            "idle_time": self.get_idle_time(),
            "consecutive_no_response": self._consecutive_no_response,
            "pending_tasks": len(self._pending_task_ids),
            "recent_messages": len(self._recent_messages),
            "arm_available": self._arm_available,
        }


def get_global_state() -> GlobalState:
    return GlobalState()
