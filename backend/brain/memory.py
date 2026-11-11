"""
记忆管理 - 对话记忆整理和存储

功能：
1. 读取对话日志
2. 提取重要记忆
3. 存储到 Viking 记忆系统（云端持久化）
4. 启动时从 Viking 加载历史记忆
"""

import json
import hashlib
import os
import time
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config
from core.logger import get_logger

logger = get_logger("brain.memory")


class VikingClient:
    """Viking 记忆系统客户端 - 使用豆包 Viking API"""
    
    def __init__(self, api_key: str, base_url: str, collection_name: str = "Active_AI"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.collection_name = collection_name
        self.project_name = "default"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0))
        return self._session
    
    async def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """添加记忆到 Viking（使用 streaming_write，自动提取事件和画像）"""
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/session/streaming_write"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "conversation_id": metadata.get("conversation_id", "default_conv") if metadata else "default_conv",
            "messages": [
                {
                    "role": "user",
                    "role_name": "东昇",
                    "role_id": "user_01",
                    "content": text,
                    "time": int(time.time() * 1000)
                }
            ],
            "extract_trigger": {
                "token_count": 1000,
                "wait_timeout": 86400,
                "message_count": 20
            }
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Memory streaming_write OK: {text[:50]}...")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to streaming_write: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error streaming_write to Viking: {e}")
            return False
    
    async def batch_add_memories(self, memories: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
        """批量添加记忆到 Viking"""
        if not memories:
            return True
        
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/session/add"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        viking_messages = []
        for memory in memories:
            content = memory.get("text", memory.get("content", ""))
            if content:
                # 正确标记 role：大脑自省/学习笔记用 assistant，用户相关用 user
                role = memory.get("role", "user")
                viking_messages.append({
                    "role": role,
                    "content": content
                })
        
        if not viking_messages:
            return False
        
        default_metadata = {
            "default_user_id": "user_01",
            "default_user_name": "东昇",
            "default_assistant_id": "assistant_01",
            "default_assistant_name": "Brain",
            "time": int(time.time() * 1000)
        }
        if metadata:
            default_metadata.update(metadata)
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "messages": viking_messages,
            "metadata": default_metadata
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Batch added {len(viking_messages)} memories to Viking")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Viking batch add error: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to batch add to Viking: {e}")
            return False
    
    async def get_context(self, query: str, user_id: str = "user_01") -> Dict[str, Any]:
        """从 Viking 获取上下文（事件和画像记忆）"""
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/get_context"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "query": query,
            "event_search_config": {
                "filter": {
                    "user_id": user_id,
                    "memory_type": ["event_v1"]
                },
                "limit": 10,
                "time_decay_config": {
                    "weight": 0.5,
                    "no_decay_period": 3
                }
            },
            "profile_search_config": {
                "filter": {
                    "user_id": user_id,
                    "memory_type": ["profile_v1"]
                },
                "limit": 1
            }
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Retrieved context for query: {query[:30]}...")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Viking get context error: {response.status} - {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to get context from Viking: {e}")
            return {}
    
    async def search_events(self, query: str, user_id: str = "user_01", limit: int = 10) -> List[Dict[str, Any]]:
        """搜索事件记忆"""
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/event/search"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "query": query,
            "filter": {
                "user_id": user_id,
                "memory_type": ["event_v1"]
            },
            "limit": limit,
            "time_decay_config": {
                "weight": 0.5,
                "no_decay_period": 3
            }
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    # Viking API 返回 result_list 而不是 events
                    events = result.get("data", {}).get("result_list", [])
                    logger.info(f"Found {len(events)} events for query: {query[:30]}...")
                    return events
                else:
                    error_text = await response.text()
                    logger.error(f"Viking event search error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search events from Viking: {e}")
            return []
    
    async def search_profiles(self, query: str, user_id: str = "user_01", limit: int = 5) -> List[Dict[str, Any]]:
        """搜索画像记忆（用户偏好、习惯等）"""
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/profile/search"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "query": query,
            "filter": {
                "user_id": user_id,
                "memory_type": ["profile_v1"]
            },
            "limit": limit
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    profiles = result.get("data", {}).get("result_list", [])
                    logger.info(f"Found {len(profiles)} profiles for query: {query[:30]}...")
                    return profiles
                else:
                    error_text = await response.text()
                    logger.error(f"Viking profile search error: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search profiles from Viking: {e}")
            return []
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def streaming_write_conversation(self, messages: List[Dict[str, Any]], 
                                           conversation_id: str = "default_conv") -> bool:
        """
        实时写入对话到 Viking（核心方法）
        
        Viking 会自动从对话中提取：
        - event_v1: 事件记忆（谁说了什么、发生了什么）
        - profile_v1: 用户画像（偏好、习惯、长期特征）
        
        这是让 Viking 积累用户画像的关键接口。
        """
        if not messages:
            return True
        
        session = await self._get_session()
        url = f"{self.base_url}/api/memory/session/streaming_write"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建标准化的 Viking 消息格式
        viking_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            
            viking_msg = {
                "role": role,
                "role_name": "东昇" if role == "user" else "Brain",
                "role_id": "user_01" if role == "user" else "assistant_01",
                "content": content,
                "time": msg.get("time", int(time.time() * 1000))
            }
            viking_messages.append(viking_msg)
        
        if not viking_messages:
            return True
        
        payload = {
            "collection_name": self.collection_name,
            "project_name": self.project_name,
            "conversation_id": conversation_id,
            "messages": viking_messages,
            "extract_trigger": {
                "token_count": 1000,
                "wait_timeout": 86400,
                "message_count": 20
            }
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"streaming_write OK: {len(viking_messages)} messages, conv={conversation_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"streaming_write failed: {response.status} - {error_text[:200]}")
                    return False
        except Exception as e:
            logger.error(f"streaming_write_conversation error: {e}")
            return False


class MemoryManager:
    """
    记忆管理 - 对话记忆整理和存储
    
    功能：
    - 读取对话日志
    - 提取重要记忆
    - 存储到 Viking 记忆系统（云端持久化）
    - 启动时从 Viking 加载历史记忆
    """
    
    _instance: Optional['MemoryManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        config = get_config()
        self._data_dir = Path(__file__).parent.parent / "data"
        self._logs_file = self._data_dir / "conversation_logs.jsonl"
        self._history_file = self._data_dir / "conversation_history.jsonl"
        self._local_memories_file = self._data_dir / "brain_memories.jsonl"
        
        self._viking_client: Optional[VikingClient] = None
        self._memories_cache: List[Dict[str, Any]] = []
        
        if config.viking_memory.api_key and getattr(config.viking_memory, "enabled", True):
            self._viking_client = VikingClient(
                api_key=config.viking_memory.api_key,
                base_url=config.viking_memory.base_url,
                collection_name=getattr(config.viking_memory, 'collection_name', 'Active_AI')
            )
            logger.info(f"Viking client initialized: {config.viking_memory.base_url}")
        elif not getattr(config.viking_memory, "enabled", True):
            logger.info("Viking 云端记忆已关闭（viking_memory.enabled=false），只用本地记忆")
        else:
            logger.warning("Viking memory not configured, using local storage only")
    
    async def initialize(self) -> None:
        """初始化：从 Viking 加载历史记忆"""
        logger.info("Initializing memory manager...")
        
        if self._viking_client:
            try:
                context = await self._viking_client.get_context("用户偏好和重要记忆")
                if context:
                    events = context.get("events", [])
                    profiles = context.get("profiles", [])
                    self._memories_cache = events + profiles
                    logger.info(f"Loaded {len(self._memories_cache)} memories from Viking")
            except Exception as e:
                logger.error(f"Failed to load memories from Viking: {e}")
        
        if not self._memories_cache:
            self._memories_cache = self._load_local_memories()
            logger.info(f"Loaded {len(self._memories_cache)} memories from local file")
    
    def _load_local_memories(self) -> List[Dict[str, Any]]:
        """从本地文件加载记忆"""
        if not self._local_memories_file.exists():
            return []
        
        memories = []
        try:
            with open(self._local_memories_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memories.append(json.loads(line))
            logger.info(f"Loaded {len(memories)} memories from local file")
        except Exception as e:
            logger.error(f"Error loading local memories: {e}")
        return memories
    
    def read_logs(self) -> List[Dict[str, Any]]:
        if not self._logs_file.exists():
            return []
        
        logs = []
        with open(self._logs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        logger.info(f"Read {len(logs)} log entries")
        return logs
    
    async def extract_memories(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not logs:
            return []
        
        from .llm import get_llm
        
        logs_text = json.dumps(logs[-20:], ensure_ascii=False, indent=2)
        
        prompt = f"""请从以下对话日志中提取重要的记忆信息：

{logs_text}

请以JSON格式返回，包含：
- "memories": 记忆列表，每个记忆包含：
  - "content": 记忆内容
  - "importance": "high" | "medium" | "low"
  - "category": "preference" | "fact" | "event" | "knowledge"

只返回JSON。"""

        llm = get_llm()
        # 记忆提取走云端模型：本地模型不参与这类判断（用户明确要求）
        response = await llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        
        try:
            result = json.loads(response)
            return result.get("memories", [])
        except:
            return []
    
    async def save_to_viking(self, memories: List[Dict[str, Any]]) -> bool:
        """保存记忆到 Viking（使用 streaming_write 自动提取 event + profile）"""
        if not memories:
            return True
        
        if not self._viking_client:
            logger.warning("Viking client not available, saving to local file only")
            return self._save_to_local(memories)
        
        # 构建 streaming_write 的消息格式
        messages = []
        for m in memories:
            content = m.get("text", m.get("content", ""))
            role = m.get("role", "user")
            if content:
                messages.append({
                    "role": role,
                    "content": content,
                    "time": int(time.time() * 1000)
                })
        
        success = await self._viking_client.streaming_write_conversation(messages)
        
        if success:
            # 同时保存到本地（带结构化字段）
            self._save_to_local(memories)
        
        logger.info(f"Saved {len(memories)} memories via streaming_write")
        return success
    
    def _save_to_local(self, memories: List[Dict[str, Any]]) -> bool:
        """保存到本地文件（带结构化字段：type, status, deadline, importance）"""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._local_memories_file, 'a', encoding='utf-8') as f:
                for memory in memories:
                    # 结构化增强
                    content = memory.get("text", memory.get("content", ""))
                    enhanced = {
                        "id": memory.get("id", hashlib.md5(content.encode()).hexdigest()[:12]),
                        "content": content,
                        "type": memory.get("type", "event"),       # 事实/事件/承诺/习惯
                        "importance": memory.get("importance", 0.5),
                        "role": memory.get("role", "user"),
                        "source": memory.get("source", "auto"),
                        "created_at": memory.get("created_at", datetime.now().isoformat()),
                        "viking_synced": True,  # 已通过 streaming_write 同步
                    }
                    # 承诺记忆专属字段
                    if memory.get("deadline"):
                        enhanced["deadline"] = memory["deadline"]
                        enhanced["status"] = memory.get("status", "pending")
                        enhanced["remind_count"] = memory.get("remind_count", 0)
                    f.write(json.dumps(enhanced, ensure_ascii=False) + '\n')
            self._memories_cache.extend(memories)
            logger.info(f"Saved {len(memories)} memories to local file (structured)")
            return True
        except Exception as e:
            logger.error(f"Error saving to local: {e}")
            return False
    
    async def consolidate(self) -> Dict[str, Any]:
        """整理记忆"""
        logger.info("Starting memory consolidation")
        
        logs = self.read_logs()
        memories = await self.extract_memories(logs)
        
        if memories:
            await self.save_to_viking(memories)
        
        if logs:
            self._archive_logs(logs)
        
        logger.info(f"Memory consolidation complete: {len(memories)} memories extracted")
        return {
            "success": True,
            "logs_processed": len(logs),
            "memories_extracted": len(memories)
        }
    
    def _archive_logs(self, logs: List[Dict[str, Any]]) -> None:
        """归档日志"""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self._history_file, 'a', encoding='utf-8') as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        self._logs_file.unlink(missing_ok=True)
        logger.info(f"Archived {len(logs)} logs to history")
    
    def get_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取缓存的记忆"""
        return self._memories_cache[-limit:]
    
    async def search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索记忆（本地关键词 + Viking 云端语义搜索）"""
        results = []
        query_lower = query.lower()
        
        # 本地缓存：关键词匹配
        for memory in self._memories_cache:
            text = memory.get("text", memory.get("content", ""))
            if text and query_lower in text.lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        
        # Viking 云端：语义搜索（本地不够时补充）
        if self._viking_client and len(results) < limit:
            try:
                viking_results = await self._viking_client.search_events(query, limit=limit - len(results))
                results.extend(viking_results[:limit - len(results)])
            except Exception as e:
                logger.error(f"Error searching Viking: {e}")
        
        return results[:limit]
    
    async def search_profiles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索用户画像（偏好、习惯等长期特征）"""
        if not self._viking_client:
            return []
        
        try:
            return await self._viking_client.search_profiles(query, limit=limit)
        except Exception as e:
            logger.error(f"Error searching profiles: {e}")
            return []
    
    async def get_full_context(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        获取完整上下文（事件记忆 + 用户画像）
        
        这是大脑推理时应该调用的主要接口，一次性返回：
        - events: 相关事件记忆（带时间衰减）
        - profiles: 用户画像（偏好、习惯等）
        """
        events = await self.search_memories(query, limit=limit)
        profiles = await self.search_profiles(query, limit=5)
        
        return {
            "events": events,
            "profiles": profiles,
            "event_count": len(events),
            "profile_count": len(profiles)
        }
    
    async def close(self):
        """关闭连接"""
        if self._viking_client:
            await self._viking_client.close()


def get_memory_manager() -> MemoryManager:
    return MemoryManager()
