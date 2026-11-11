"""
任务管理 - 任务队列和状态管理
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config
from core.types import Task, TaskType, TaskPriority, EventType
from core.logger import get_logger

logger = get_logger("brain.tasks")


class TaskManager:
    """
    任务管理 - 任务队列和状态管理
    
    功能：
    - 加载/保存任务
    - 任务状态更新
    - 任务去重
    - 事件触发
    """
    
    _instance: Optional['TaskManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._data_dir = Path(__file__).parent.parent / "data"
        self._tasks_file = self._data_dir / "active_tasks.json"
    
    def load_tasks(self) -> List[Task]:
        if not self._tasks_file.exists():
            return []
        
        with open(self._tasks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [Task.from_dict(t) for t in data]
    
    def save_tasks(self, tasks: List[Task]) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        data = [t.to_dict() for t in tasks]
        with open(self._tasks_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_pending_tasks(self) -> List[Task]:
        tasks = self.load_tasks()
        return [t for t in tasks if t.status == "pending"]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        tasks = self.load_tasks()
        for task in tasks:
            if task.id == task_id:
                return task
        return None
    
    def update_task_status(self, task_id: str, status: str) -> None:
        tasks = self.load_tasks()
        for task in tasks:
            if task.id == task_id:
                task.status = status
                task.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                })
                break
        self.save_tasks(tasks)
    
    def add_task(self, task: Task) -> bool:
        tasks = self.load_tasks()
        
        existing_descriptions = {t.description.strip().lower() for t in tasks}
        if task.description.strip().lower() in existing_descriptions:
            logger.info(f"Skipping duplicate task: {task.description}")
            return False
        
        tasks.append(task)
        self.save_tasks(tasks)
        
        self._trigger_event(EventType.TASK_GENERATED, task)
        return True
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        tasks = self.load_tasks()
        for task in tasks:
            if task.id == task_id:
                task.status = "completed"
                task.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "result": result,
                })
                break
        self.save_tasks(tasks)
        
        self._trigger_event(EventType.TASK_COMPLETED, task_id)
    
    def _trigger_event(self, event_type: EventType, data: Any) -> None:
        try:
            from .executor import get_executor
            executor = get_executor()
            
            import asyncio
            event = Event(type=event_type, data={"task_id": data if isinstance(data, str) else data.id})
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(executor.emit_event(event))
        except Exception as e:
            logger.warning(f"Failed to trigger event: {e}")


def get_task_manager() -> TaskManager:
    return TaskManager()
