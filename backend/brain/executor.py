"""
执行器 - 事件驱动的任务执行
"""

import asyncio
import heapq
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Task, TaskType, EventType, Event
from core.logger import get_logger
from core.state import get_global_state, DialogStatus

logger = get_logger("brain.executor")


@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task: Task = field(compare=False)
    event_type: EventType = field(compare=False)


class Executor:
    """
    执行器 - 事件驱动的任务执行
    
    功能：
    - 优先级队列管理
    - 事件触发执行
    - 自动连接处理
    - 上下文添加
    - 用户活动检测
    - 主动对话判断
    """
    
    _instance: Optional['Executor'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._task_queue: List[PrioritizedTask] = []
        self._event_handlers: Dict[EventType, List[callable]] = {}
        self._running = False
        self._global_state = get_global_state()
        self._data_dir = Path(__file__).parent.parent / "data"
        self._completed_tasks_file = self._data_dir / "completed_tasks.jsonl"
        self._deferred_tasks: List[Dict[str, Any]] = []
    
    def record_user_activity(self) -> None:
        self._global_state.record_user_activity()
        logger.debug("User activity recorded")
    
    def is_user_idle(self) -> bool:
        return self._global_state.is_user_idle()
    
    def get_priority(self, task_type: TaskType) -> int:
        priority_map = {
            TaskType.DIALOGUE_ACTION: 1,
            TaskType.MEMORY_OPERATION: 2,
            TaskType.TOOL_CALL: 3,
            TaskType.SELF_IMPROVEMENT: 4,
        }
        return priority_map.get(task_type, 5)
    
    def add_task(self, task: Task, event_type: EventType = EventType.TASK_GENERATED) -> None:
        priority = self.get_priority(task.type)
        prioritized = PrioritizedTask(priority=priority, task=task, event_type=event_type)
        heapq.heappush(self._task_queue, prioritized)
        self._global_state.register_pending_task(task.id)
        logger.info(f"Added task: {task.id} (priority: {priority})")
    
    def register_handler(self, event_type: EventType, handler: callable) -> None:
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: Event) -> None:
        handlers = self._event_handlers.get(event.type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        logger.info(f"Executing task: {task.id} ({task.type.value})")
        
        try:
            if task.type == TaskType.DIALOGUE_ACTION:
                return await self._execute_dialogue(task)
            elif task.type == TaskType.MEMORY_OPERATION:
                return await self._execute_memory(task)
            elif task.type == TaskType.TOOL_CALL:
                return await self._execute_tool(task)
            elif task.type == TaskType.SELF_IMPROVEMENT:
                return await self._execute_self_improvement(task)
            else:
                return {"success": False, "error": "Unknown task type"}
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_dialogue(self, task: Task) -> Dict[str, Any]:
        from dialog.connection import get_connection_manager
        
        conn_manager = get_connection_manager()
        
        if not conn_manager.is_connected():
            success = await conn_manager.ensure_connection()
            if not success:
                return {"success": False, "error": "No connection available"}
        
        from .memory_analyzer import get_memory_analyzer
        analyzer = get_memory_analyzer()
        
        decision = await analyzer.should_initiate_proactive_dialog(task)
        
        if not decision.get("should_proactive"):
            logger.info(f"Task {task.id} deferred: {decision.get('reason')}")
            
            if decision.get("save_for_later"):
                self._deferred_tasks.append({
                    "task": task,
                    "reason": decision.get("reason"),
                    "deferred_at": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "deferred": True,
                "reason": decision.get("reason")
            }
        
        message = decision.get("message_suggestion")
        if not message:
            from .llm import get_llm
            llm = get_llm()
            message = await llm.generate_dialogue_message(
                task_description=task.description,
                context_content=task.context_content
            )
        
        logger.info(f"Proactive dialog message: {message}")
        
        self._global_state.set_dialog_status(DialogStatus.PROCESSING)
        
        from dialog.mcp_tools import DialogTools
        tools = DialogTools()
        
        result = await tools.send_message(message)
        
        self._global_state.record_assistant_activity()
        
        if task.context_content:
            context_result = await tools.add_context([
                {
                    "role": "assistant",
                    "text": message,
                    "timestamp": datetime.now().isoformat()
                }
            ])
            result["context_added"] = context_result.get("success", False)
        
        return result
    
    async def _execute_memory(self, task: Task) -> Dict[str, Any]:
        from .memory import get_memory_manager
        memory = get_memory_manager()
        return await memory.consolidate()
    
    async def _execute_tool(self, task: Task) -> Dict[str, Any]:
        logger.info(f"Tool call: {task.tools_needed} - {task.parameters}")
        return {"success": True, "message": "Tool execution placeholder"}
    
    async def _execute_self_improvement(self, task: Task) -> Dict[str, Any]:
        logger.info(f"Self improvement: {task.description}")
        return {"success": True, "message": "Self improvement placeholder"}
    
    def get_deferred_tasks_for_conversation(self) -> List[Dict[str, Any]]:
        """获取延迟的任务，用于下次对话时提起"""
        tasks = self._deferred_tasks.copy()
        self._deferred_tasks.clear()
        return tasks
    
    async def run_once(self) -> None:
        if not self._task_queue:
            self._load_tasks_from_file()
        
        if not self._task_queue:
            return
        
        prioritized = heapq.heappop(self._task_queue)
        task = prioritized.task
        
        if task.type == TaskType.DIALOGUE_ACTION:
            in_conversation = self._global_state.is_in_conversation()
            is_idle = self._global_state.is_user_idle()
            idle_time = self._global_state.get_idle_time()
            status = self._global_state.get_dialog_status()
            
            logger.info(f"Task {task.id}: status={status.value}, in_conversation={in_conversation}, is_idle={is_idle}, idle_time={idle_time:.0f}s")
            
            if in_conversation:
                logger.info(f"In conversation, postponing dialogue task: {task.id}")
                heapq.heappush(self._task_queue, prioritized)
                return
            
            if not is_idle:
                logger.info(f"User is active, postponing dialogue task: {task.id}")
                heapq.heappush(self._task_queue, prioritized)
                return
        
        logger.info(f"Executing dialogue task: {task.id}")
        result = await self.execute_task(task)
        
        if not result.get("deferred"):
            self._mark_task_completed(task.id, result)
            
            if result.get("success"):
                self._save_completed_task_to_memory(task, result)
                self._global_state.complete_task(task.id)
                logger.info(f"Task {task.id} completed successfully")
            else:
                logger.warning(f"Task {task.id} failed: {result.get('error')}")
    
    def _mark_task_completed(self, task_id: str, result: Dict[str, Any]) -> None:
        try:
            tasks_file = self._data_dir / "active_tasks.json"
            
            if not tasks_file.exists():
                return
            
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for task_data in data:
                if task_data.get("id") == task_id:
                    task_data["status"] = "completed"
                    task_data["completed_at"] = datetime.now().isoformat()
                    task_data["result"] = result
                    break
            
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Marked task {task_id} as completed")
            
        except Exception as e:
            logger.error(f"Failed to mark task completed: {e}")
    
    def _save_completed_task_to_memory(self, task: Task, result: Dict[str, Any]) -> None:
        try:
            self._completed_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            
            record = {
                "task_id": task.id,
                "type": task.type.value,
                "description": task.description,
                "completed_at": datetime.now().isoformat(),
                "result": result
            }
            
            with open(self._completed_tasks_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved completed task {task.id} to memory")
            
        except Exception as e:
            logger.error(f"Failed to save completed task: {e}")
    
    def _load_tasks_from_file(self) -> None:
        try:
            tasks_file = self._data_dir / "active_tasks.json"
            
            if not tasks_file.exists():
                return
            
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for task_data in data:
                if task_data.get("status") == "pending":
                    task = Task.from_dict(task_data)
                    if not any(pt.task.id == task.id for pt in self._task_queue):
                        priority = self.get_priority(task.type)
                        prioritized = PrioritizedTask(priority=priority, task=task, event_type=EventType.TASK_GENERATED)
                        heapq.heappush(self._task_queue, prioritized)
                        logger.info(f"Loaded task from file: {task.id} ({task.type.value})")
                        
        except Exception as e:
            logger.error(f"Failed to load tasks from file: {e}")
    
    async def run_loop(self) -> None:
        self._running = True
        while self._running:
            await self.run_once()
            await asyncio.sleep(1)
    
    def stop(self) -> None:
        self._running = False


def get_executor() -> Executor:
    return Executor()


async def run_executor() -> None:
    executor = get_executor()
    await executor.run_once()


async def on_conversation_ended() -> None:
    """对话结束时调用的函数"""
    logger.info("Conversation ended event triggered")
    
    try:
        executor = get_executor()
        
        deferred_tasks = executor.get_deferred_tasks_for_conversation()
        if deferred_tasks:
            logger.info(f"Found {len(deferred_tasks)} deferred tasks from previous conversation")
            
            for deferred in deferred_tasks:
                task = deferred.get("task")
                if task:
                    executor.add_task(task, EventType.CONVERSATION_ENDED)
                    logger.info(f"Re-added deferred task: {task.id}")
        
        await executor.run_once()
    except Exception as e:
        logger.error(f"Error in on_conversation_ended: {e}")
