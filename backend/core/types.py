"""
类型定义 - 统一类型系统
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


class EventType(Enum):
    CONVERSATION_NEW = "conversation_new"
    CONVERSATION_ENDED = "conversation_ended"
    TASK_GENERATED = "task_generated"
    TASK_COMPLETED = "task_completed"
    SCHEDULED_CHECK = "scheduled_check"
    USER_ONLINE = "user_online"
    USER_OFFLINE = "user_offline"


class TaskType(Enum):
    DIALOGUE_ACTION = "dialogue_action"
    MEMORY_OPERATION = "memory_operation"
    TOOL_CALL = "tool_call"
    SELF_IMPROVEMENT = "self_improvement"


class TaskPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    tools_needed: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context_content: Optional[str] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "tools_needed": self.tools_needed,
            "parameters": self.parameters,
            "context_content": self.context_content,
            "execution_log": self.execution_log,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            type=TaskType(data["type"]),
            description=data["description"],
            priority=TaskPriority(data.get("priority", 2)),
            status=data.get("status", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            tools_needed=data.get("tools_needed", []),
            parameters=data.get("parameters", {}),
            context_content=data.get("context_content"),
            execution_log=data.get("execution_log", []),
        )


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
