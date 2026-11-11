"""
实时对话工具 - 将实时对话封装为大脑可调用的工具

功能：
1. 启动/停止实时对话会话
2. 上下文初始化与自动更新
3. 对话状态管理
4. 与大脑记忆系统同步
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pathlib import Path
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.state import get_global_state, DialogStatus
from core.config import get_config

logger = get_logger("brain.realtime_dialog_tool")


class DialogSessionState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    CLOSING = "closing"


class RealtimeDialogTool:
    """
    实时对话工具 - 大脑可调用的对话能力
    
    核心功能：
    1. start_dialog: 启动实时对话会话
    2. stop_dialog: 停止对话会话
    3. send_message: 发送消息
    4. add_context: 添加上下文（记忆注入）
    5. initialize_context: 初始化上下文（从记忆系统加载）
    6. sync_memory: 同步对话记忆到大脑
    """
    
    _instance: Optional['RealtimeDialogTool'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._client = None
        self._session_id: Optional[str] = None
        self._session_state: DialogSessionState = DialogSessionState.IDLE
        self._global_state = get_global_state()
        self._config = get_config()
        self._receive_task: Optional[asyncio.Task] = None
        self._message_handlers: List[Callable] = []
        self._pending_context: List[Dict[str, Any]] = []
        self._dialog_history: List[Dict[str, Any]] = []
        self._max_history = 100
        
        logger.info("RealtimeDialogTool initialized")
    
    @property
    def is_active(self) -> bool:
        return self._session_state == DialogSessionState.ACTIVE and self._client is not None
    
    @property
    def session_id(self) -> Optional[str]:
        return self._session_id
    
    def add_message_handler(self, handler: Callable) -> None:
        self._message_handlers.append(handler)
    
    def remove_message_handler(self, handler: Callable) -> None:
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    async def _notify_handlers(self, message: Dict[str, Any]) -> None:
        for handler in self._message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
    
    async def start_dialog(
        self,
        mod: str = "text",
        greeting: str = None,
        auto_init_context: bool = True,
        purpose: str = None
    ) -> Dict[str, Any]:
        """
        启动实时对话会话
        
        Args:
            mod: 对话模式 ("text" 或 "audio")
            greeting: 自定义问候语
            auto_init_context: 是否自动初始化上下文
            purpose: 本次启动对话的目的（用于智能生成上下文）
        
        Returns:
            {"success": bool, "session_id": str, "error": str}
        """
        if self.is_active:
            logger.warning("Dialog session already active")
            return {"success": True, "session_id": self._session_id, "message": "Session already active"}
        
        logger.info(f"Starting dialog session, mod={mod}, purpose={purpose}")
        self._session_state = DialogSessionState.CONNECTING
        
        try:
            from realtime_dialog.realtime_dialog_client import RealtimeDialogClient
            import realtime_dialog.config as dialog_config
            
            self._session_id = str(uuid.uuid4())
            
            self._client = RealtimeDialogClient(
                config=dialog_config.ws_connect_config,
                session_id=self._session_id,
                output_audio_format="pcm",
                mod=mod,
                recv_timeout=120
            )
            
            await self._client.connect()
            
            if auto_init_context:
                await self.initialize_context(purpose=purpose)
            
            greeting_text = greeting or "你好，我是王明杰，有什么可以帮助你的？"
            await self._client.say_hello()
            
            self._session_state = DialogSessionState.ACTIVE
            self._global_state.set_dialog_status(DialogStatus.IN_CONVERSATION)
            
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            from dialog.connection import get_connection_manager
            get_connection_manager().register(self._client, self._session_id)
            
            logger.info(f"Dialog session started: {self._session_id}")
            
            return {
                "success": True,
                "session_id": self._session_id,
                "message": "Dialog session started"
            }
            
        except Exception as e:
            self._session_state = DialogSessionState.IDLE
            self._client = None
            self._session_id = None
            logger.error(f"Failed to start dialog: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def stop_dialog(self) -> Dict[str, Any]:
        """停止对话会话"""
        if not self.is_active:
            return {"success": True, "message": "No active session"}
        
        logger.info("Stopping dialog session")
        self._session_state = DialogSessionState.CLOSING
        
        try:
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None
            
            if self._client:
                try:
                    await self._client.finish_session()
                    await self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing client: {e}")
            
            from dialog.connection import get_connection_manager
            get_connection_manager().unregister()
            
            await self.sync_memory()
            
            self._client = None
            self._session_id = None
            self._session_state = DialogSessionState.IDLE
            self._global_state.set_dialog_status(DialogStatus.IDLE)
            
            logger.info("Dialog session stopped")
            return {"success": True, "message": "Dialog session stopped"}
            
        except Exception as e:
            logger.error(f"Error stopping dialog: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_message(self, content: str, direct: bool = False) -> Dict[str, Any]:
        """
        发送消息
        
        Args:
            content: 消息内容
            direct: 是否直接发送（不加前缀）
        """
        if not self.is_active:
            return {"success": False, "error": "No active dialog session"}
        
        try:
            message = content if direct else f"请对我说：{content}"
            await self._client.chat_text_query(message)
            
            self._dialog_history.append({
                "role": "brain",
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Message sent: {content[:50]}...")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_context(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        添加对话上下文（注入记忆）
        
        Args:
            items: 上下文项目列表，每项包含 role, text, timestamp
        """
        if not self.is_active:
            self._pending_context.extend(items)
            logger.info(f"Context cached (no active session): {len(items)} items")
            return {"success": True, "cached": True, "count": len(items)}
        
        try:
            formatted_items = []
            for item in items:
                formatted_item = {
                    "role": item.get("role", "user"),
                    "text": item.get("text", ""),
                }
                if item.get("timestamp"):
                    formatted_item["timestamp"] = item["timestamp"]
                formatted_items.append(formatted_item)
            
            await self._client.conversation_create(formatted_items)
            
            logger.info(f"Context added to dialog: {len(items)} items")
            return {"success": True, "count": len(items)}
            
        except Exception as e:
            logger.error(f"Add context error: {e}")
            return {"success": False, "error": str(e)}
    
    async def initialize_context(self, purpose: str = None) -> Dict[str, Any]:
        """
        智能初始化上下文 - 让大脑自己设计注入的上下文
        
        核心设计：
        1. 收集基础信息：用户身份、最近对话、待通知事项
        2. 调用LLM智能生成上下文内容
        3. 将上下文注入到实时对话
        
        Args:
            purpose: 本次启动对话的目的（可选）
        """
        logger.info("Intelligently initializing dialog context...")
        
        context_info = await self._collect_context_info()
        
        smart_context = await self._generate_smart_context(context_info, purpose)
        
        if smart_context and self.is_active:
            result = await self.add_context(smart_context)
            logger.info(f"Smart context initialized: {len(smart_context)} items")
            return result
        
        return {"success": True, "message": "Context initialized", "items_loaded": len(smart_context)}
    
    async def _collect_context_info(self) -> Dict[str, Any]:
        """
        收集上下文信息
        
        Returns:
            {
                "user_profile": 用户身份信息,
                "recent_conversations": 最近对话,
                "pending_notifications": 待通知事项,
                "important_memories": 重要记忆,
                "current_tasks": 当前任务状态
            }
        """
        context_info = {
            "user_profile": {},
            "recent_conversations": [],
            "pending_notifications": [],
            "important_memories": [],
            "current_tasks": []
        }
        
        try:
            from .memory import get_memory_manager
            memory_manager = get_memory_manager()
            
            profile_memories = await memory_manager.search_memories("用户 东昇 偏好 喜好", limit=5)
            for mem in profile_memories:
                text = mem.get("text", mem.get("content", ""))
                if text:
                    context_info["user_profile"][text[:50]] = text
            
            logger.info(f"Collected user profile: {len(context_info['user_profile'])} items")
            
            important_memories = await memory_manager.search_memories("重要 提醒 记住", limit=5)
            for mem in important_memories:
                text = mem.get("text", mem.get("content", ""))
                if text:
                    context_info["important_memories"].append(text)
            
            logger.info(f"Collected important memories: {len(context_info['important_memories'])} items")
            
        except Exception as e:
            logger.warning(f"Failed to collect memories: {e}")
        
        recent_messages = self._global_state.get_recent_messages(5)
        for msg in recent_messages:
            content = msg.get("content", msg.get("text", ""))
            if content:
                context_info["recent_conversations"].append({
                    "role": msg.get("role", "user"),
                    "content": content[:100]
                })
        
        logger.info(f"Collected recent conversations: {len(context_info['recent_conversations'])} items")
        
        try:
            from .executor import get_executor
            executor = get_executor()
            pending_tasks = executor.get_deferred_tasks_for_conversation()
            for task in pending_tasks[:3]:
                if hasattr(task.get("task"), "description"):
                    context_info["pending_notifications"].append(task["task"].description)
                elif task.get("description"):
                    context_info["pending_notifications"].append(task["description"])
            
            logger.info(f"Collected pending notifications: {len(context_info['pending_notifications'])} items")
            
        except Exception as e:
            logger.warning(f"Failed to collect pending tasks: {e}")
        
        return context_info
    
    async def _generate_smart_context(
        self, 
        context_info: Dict[str, Any], 
        purpose: str = None
    ) -> List[Dict[str, Any]]:
        """
        让大脑LLM智能生成上下文
        
        Args:
            context_info: 收集的上下文信息
            purpose: 本次对话的目的
        
        Returns:
            生成的上下文项目列表
        """
        try:
            from .llm import get_llm
            llm = get_llm()
            
            user_profile_str = "\n".join([f"- {k}: {v}" for k, v in context_info["user_profile"].items()]) or "暂无用户信息"
            recent_conv_str = "\n".join([
                f"- {c['role']}: {c['content']}" 
                for c in context_info["recent_conversations"]
            ]) or "暂无最近对话"
            important_mem_str = "\n".join([f"- {m}" for m in context_info["important_memories"]]) or "暂无重要记忆"
            pending_str = "\n".join([f"- {n}" for n in context_info["pending_notifications"]]) or "暂无待通知事项"
            
            prompt = f"""你是一个AI大脑，现在需要为即将启动的实时对话设计上下文。

实时对话系统每次启动都会重置记忆，所以你需要设计一个简洁但完整的上下文，让对话系统能够：
1. 知道用户是谁
2. 了解本次对话的目的
3. 知道需要告诉用户什么消息

【收集到的信息】

用户身份信息：
{user_profile_str}

最近对话历史：
{recent_conv_str}

重要记忆：
{important_mem_str}

待通知用户的事项：
{pending_str}

本次启动对话的目的：
{purpose or "与用户进行日常对话"}

【你的任务】

请设计3-5条上下文消息，以JSON格式返回：
{{
    "context_items": [
        {{
            "role": "system",
            "text": "上下文内容..."
        }}
    ],
    "greeting_suggestion": "建议的问候语（可选）"
}}

要求：
1. 第一条必须是系统消息，介绍用户身份（用户叫什么、有什么偏好）
2. 如果有待通知事项，要包含在内
3. 如果有重要记忆，要包含在内
4. 内容要简洁，每条不超过100字
5. 不要包含技术细节，用自然语言表达

只返回JSON，不要其他内容。"""

            response = await llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            
            import json
            result = json.loads(response)
            
            context_items = []
            for item in result.get("context_items", []):
                context_items.append({
                    "role": item.get("role", "system"),
                    "text": item.get("text", ""),
                    "timestamp": datetime.now().isoformat()
                })
            
            if self._pending_context:
                context_items.extend(self._pending_context)
                self._pending_context.clear()
            
            logger.info(f"Generated {len(context_items)} smart context items")
            return context_items
            
        except Exception as e:
            logger.error(f"Failed to generate smart context: {e}", exc_info=True)
            
            fallback_items = []
            
            if context_info["user_profile"]:
                profile_text = list(context_info["user_profile"].values())[0]
                fallback_items.append({
                    "role": "system",
                    "text": f"用户信息：{profile_text[:100]}",
                    "timestamp": datetime.now().isoformat()
                })
            
            if context_info["pending_notifications"]:
                for notification in context_info["pending_notifications"][:2]:
                    fallback_items.append({
                        "role": "system",
                        "text": f"待通知：{notification[:100]}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"Using fallback context: {len(fallback_items)} items")
            return fallback_items
    
    async def sync_memory(self) -> Dict[str, Any]:
        """
        同步对话记忆到大脑
        
        将本次对话的历史记录保存到大脑记忆系统
        """
        if not self._dialog_history:
            return {"success": True, "message": "No dialog history to sync"}
        
        logger.info(f"Syncing {len(self._dialog_history)} dialog items to memory")
        
        try:
            from .memory import get_memory_manager
            memory_manager = get_memory_manager()
            
            memories_to_save = []
            for item in self._dialog_history[-20:]:
                content = item.get("content", "")
                if content and len(content) > 5:
                    memories_to_save.append({
                        "text": content,
                        "role": item.get("role"),
                        "timestamp": item.get("timestamp")
                    })
            
            if memories_to_save:
                await memory_manager.save_to_viking(memories_to_save)
            
            self._dialog_history = self._dialog_history[-self._max_history:]
            
            logger.info("Dialog memory synced successfully")
            return {"success": True, "synced_count": len(memories_to_save)}
            
        except Exception as e:
            logger.error(f"Failed to sync memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def _receive_loop(self) -> None:
        """接收服务器消息的循环"""
        logger.info("Receive loop started")
        
        try:
            while self.is_active:
                try:
                    response = await self._client.receive_server_response()
                    
                    if response == {}:
                        continue
                    
                    event = response.get('event')
                    message_type = response.get('message_type')
                    payload = response.get('payload_msg', {})
                    
                    if message_type == 'SERVER_FULL_RESPONSE':
                        if event == 450:
                            self._global_state.record_user_activity()
                        
                        elif event == 451:
                            results = payload.get('results', [])
                            for result in results:
                                text = result.get('text', '')
                                if text and not result.get('is_interim', False):
                                    self._dialog_history.append({
                                        "role": "user",
                                        "content": text,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    self._global_state.add_recent_message("user", text)
                        
                        elif event == 550:
                            content = payload.get('content', '')
                            if content:
                                self._dialog_history.append({
                                    "role": "assistant",
                                    "content": content,
                                    "timestamp": datetime.now().isoformat()
                                })
                                self._global_state.add_recent_message("assistant", content)
                        
                        elif event == 359:
                            self._global_state.record_assistant_activity()
                        
                        elif event in [152, 153]:
                            logger.info(f"Session finished event: {event}")
                            break
                    
                    await self._notify_handlers(response)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Receive loop error: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop fatal error: {e}")
        finally:
            logger.info("Receive loop ended")
    
    def get_status(self) -> Dict[str, Any]:
        """获取对话状态"""
        return {
            "is_active": self.is_active,
            "session_state": self._session_state.value,
            "session_id": self._session_id,
            "dialog_history_count": len(self._dialog_history),
            "pending_context_count": len(self._pending_context)
        }
    
    def get_dialog_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self._dialog_history[-limit:]
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """获取工具定义（用于LLM function calling）"""
        return {
            "type": "function",
            "function": {
                "name": "start_realtime_dialog",
                "description": "启动实时语音对话会话。当需要与用户进行实时语音交互时使用。大脑可以控制对话的启动、停止，并在对话开始时自动注入记忆上下文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mod": {
                            "type": "string",
                            "enum": ["text", "audio"],
                            "description": "对话模式：text为文本模式，audio为语音模式"
                        },
                        "greeting": {
                            "type": "string",
                            "description": "自定义问候语（可选）"
                        },
                        "auto_init_context": {
                            "type": "boolean",
                            "description": "是否自动初始化上下文（默认true）"
                        }
                    },
                    "required": ["mod"]
                }
            }
        }


def get_realtime_dialog_tool() -> RealtimeDialogTool:
    return RealtimeDialogTool()
