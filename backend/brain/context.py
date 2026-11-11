"""
统一上下文管理 - 存算一体

三层上下文架构：
- Hot:  常驻内存，不变化（用户画像、核心记忆、系统提示词）
- Warm: 增量更新，实时变化（对话历史、待通知、动机状态）
- Cold: 按需加载，搜索时获取（Viking云端、历史记录）

从原 unified_brain.py 中提取，供未来 A 路线（大脑直接管理语音连接）复用。
当前 B 路线中，主要用于 get_status() 返回上下文摘要。
"""

from datetime import datetime
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("brain.context")


class UnifiedContext:
    """
    统一上下文管理 - 存算一体

    三层上下文架构：
    - Hot: 常驻内存，不变化（用户画像、核心记忆、系统提示词）
    - Warm: 增量更新，实时变化（对话历史、待通知、动机状态）
    - Cold: 按需加载，搜索时获取（Viking云端、历史记录）
    """

    def __init__(self):
        self._hot: Dict[str, Any] = {
            "user_profile": "",
            "core_memories": [],
            "system_prompt": ""
        }
        self._warm: Dict[str, Any] = {
            "recent_dialog": [],
            "pending_notifications": [],
            "intrinsic_state": {},
            "last_active_time": None
        }
        self._cold_cache: Dict[str, Any] = {}
        self._cold_cache_time: Dict[str, float] = {}
        self._cold_ttl = 300

        self._context_version = 0
        self._pending_updates: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """初始化热上下文（只执行一次）"""
        logger.info("Initializing hot context...")

        try:
            from brain.memory import get_memory_manager
            memory_manager = get_memory_manager()

            profile_memories = await memory_manager.search_memories(
                "用户 东昇 偏好 喜好 身份", limit=10
            )
            for mem in profile_memories:
                text = mem.get("text", mem.get("content", ""))
                if text:
                    self._hot["core_memories"].append(text)

            self._hot["user_profile"] = self._build_user_profile()
            self._hot["system_prompt"] = self._build_system_prompt()

        except Exception as e:
            logger.warning(f"Failed to initialize hot context: {e}")
            self._hot["system_prompt"] = "你是一个AI助手。"

        self._context_version = 1
        logger.info(f"Hot context initialized: {len(self._hot['core_memories'])} core memories")

    def update_warm(self, key: str, value: Any) -> None:
        """更新温上下文（增量）"""
        self._warm[key] = value
        self._context_version += 1
        self._pending_updates.append({
            "layer": "warm",
            "key": key,
            "value": value,
            "version": self._context_version,
            "timestamp": datetime.now().isoformat()
        })

    def append_warm(self, key: str, item: Any, max_len: int = 50) -> None:
        """追加温上下文（滚动窗口）"""
        if key not in self._warm:
            self._warm[key] = []
        self._warm[key].append(item)
        if len(self._warm[key]) > max_len:
            self._warm[key] = self._warm[key][-max_len:]
        self._context_version += 1

    async def get_cold(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取冷上下文（按需加载，带缓存）"""
        import time
        cache_key = f"cold_{query}"

        if cache_key in self._cold_cache:
            cache_time = self._cold_cache_time.get(cache_key, 0)
            if time.time() - cache_time < self._cold_ttl:
                return self._cold_cache[cache_key]

        try:
            from brain.memory import get_memory_manager
            memory_manager = get_memory_manager()
            results = await memory_manager.search_memories(query, limit=limit)

            self._cold_cache[cache_key] = results
            self._cold_cache_time[cache_key] = time.time()

            return results
        except Exception as e:
            logger.warning(f"Failed to load cold context: {e}")
            return []

    def get_dialog_context_items(self) -> List[Dict[str, Any]]:
        """获取需要注入到对话的上下文项"""
        items = []

        items.append({
            "role": "system",
            "text": self._hot["system_prompt"]
        })

        if self._hot["user_profile"]:
            items.append({
                "role": "system",
                "text": f"用户信息：{self._hot['user_profile']}"
            })

        for mem in self._hot["core_memories"][:3]:
            items.append({
                "role": "system",
                "text": f"重要记忆：{mem[:100]}"
            })

        for notification in self._warm.get("pending_notifications", [])[:2]:
            items.append({
                "role": "system",
                "text": f"待通知：{notification}"
            })

        return items

    def get_incremental_updates(self, since_version: int) -> List[Dict[str, Any]]:
        """获取增量更新（自指定版本以来）"""
        return [
            u for u in self._pending_updates
            if u["version"] > since_version
        ]

    def _build_user_profile(self) -> str:
        """构建用户画像"""
        if not self._hot["core_memories"]:
            return "用户：东昇"

        profile_parts = []
        for mem in self._hot["core_memories"][:5]:
            profile_parts.append(mem[:80])

        return "用户：东昇。" + "；".join(profile_parts)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是东昇的AI助手，拥有主动交互能力。
你可以主动与用户对话，也可以在后台进行分析和自我学习。
你拥有内在动机系统，可以自主探索新知识。"""

    @property
    def version(self) -> int:
        return self._context_version

    def get_summary(self) -> Dict[str, Any]:
        return {
            "version": self._context_version,
            "hot_items": len(self._hot["core_memories"]),
            "warm_dialog_count": len(self._warm.get("recent_dialog", [])),
            "warm_pending_count": len(self._warm.get("pending_notifications", [])),
            "cold_cache_count": len(self._cold_cache)
        }
