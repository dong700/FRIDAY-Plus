"""
实时记忆分析器 - 分析对话内容并检测任务完成
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.state import get_global_state, DialogStatus
from core.types import Task, TaskType

logger = get_logger("brain.memory_analyzer")


class MemoryAnalyzer:
    """
    实时记忆分析器
    
    功能：
    - 实时分析对话内容
    - 检测任务完成状态
    - 提取重要信息
    - 判断是否值得主动发起对话
    - 识别大脑指令（@大脑 或 [BRAIN]）
    """
    
    _instance: Optional['MemoryAnalyzer'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._state = get_global_state()
        self._tasks_file = Path("data/active_tasks.json")
        self._completed_file = Path("data/completed_tasks.jsonl")
        self._analysis_cache: Dict[str, Any] = {}
    
    async def analyze_message(self, role: str, content: str) -> Dict[str, Any]:
        """
        实时分析消息
        
        返回：
        - is_brain_command: 是否是大脑指令
        - brain_command: 指令内容（如果是指令）
        - task_completed: 是否检测到任务完成
        - important_info: 重要信息
        - should_proactive: 是否值得主动发起对话
        """
        self._state.add_recent_message(role, content)
        
        result = {
            "is_brain_command": False,
            "brain_command": None,
            "task_completed": False,
            "completed_task_id": None,
            "important_info": None,
            "should_proactive": False,
        }
        
        if role == "user":
            from .commands import extract_brain_command, get_command_processor
            brain_cmd = extract_brain_command(content)
            
            if brain_cmd:
                result["is_brain_command"] = True
                result["brain_command"] = brain_cmd.command
                
                processor = get_command_processor()
                processor.add_command(brain_cmd)
                
                logger.info(f"Detected brain command: {brain_cmd.command}")
                return result
            
            completed_task = await self._detect_task_completion(content)
            if completed_task:
                result["task_completed"] = True
                result["completed_task_id"] = completed_task["id"]
                await self._mark_task_completed(completed_task, content)
                self._state.complete_task(completed_task["id"])
        
        if role == "assistant":
            important_info = await self._extract_important_info(content)
            if important_info:
                result["important_info"] = important_info
        
        return result
    
    async def _detect_task_completion(self, user_message: str) -> Optional[Dict[str, Any]]:
        """检测用户消息是否表明某个任务已完成"""
        try:
            if not self._tasks_file.exists():
                return None
            
            with open(self._tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            pending_tasks = [t for t in tasks if t.get("status") == "pending"]
            
            if not pending_tasks:
                return None
            
            from .llm import get_llm
            llm = get_llm()
            
            task_descriptions = [
                {"id": t["id"], "description": t["description"]}
                for t in pending_tasks
            ]
            
            prompt = f"""分析用户消息，判断是否有任务被完成。

用户消息：{user_message}

待处理任务：
{json.dumps(task_descriptions, ensure_ascii=False, indent=2)}

如果有任务被完成，返回JSON：
{{"completed_task_id": "任务ID", "reason": "判断理由"}}

如果没有任务完成，返回：
{{"completed_task_id": null}}

只返回JSON。"""

            response = await llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            result = json.loads(response)
            
            if result.get("completed_task_id"):
                for task in pending_tasks:
                    if task["id"] == result["completed_task_id"]:
                        logger.info(f"Detected task completion: {task['id']} - {result.get('reason')}")
                        return task
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting task completion: {e}")
            return None
    
    async def _mark_task_completed(self, task: Dict[str, Any], trigger_message: str) -> None:
        """标记任务为已完成并保存到记忆"""
        try:
            with open(self._tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            for t in tasks:
                if t["id"] == task["id"]:
                    t["status"] = "completed"
                    t["completed_at"] = datetime.now().isoformat()
                    t["completion_trigger"] = trigger_message
                    break
            
            with open(self._tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, ensure_ascii=False, indent=2)
            
            record = {
                "task_id": task["id"],
                "type": task.get("type"),
                "description": task.get("description"),
                "completed_at": datetime.now().isoformat(),
                "completion_trigger": trigger_message,
                "source": "realtime_detection"
            }
            
            self._completed_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._completed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Task {task['id']} marked as completed and saved to memory")
            
        except Exception as e:
            logger.error(f"Error marking task completed: {e}")
    
    async def _extract_important_info(self, content: str) -> Optional[str]:
        """提取重要信息"""
        keywords = ["记住", "记得", "重要", "别忘了", "提醒我", "我的"]
        
        for keyword in keywords:
            if keyword in content:
                return content
        
        return None
    
    async def should_initiate_proactive_dialog(self, task: Task) -> Dict[str, Any]:
        """
        判断是否应该主动发起对话
        
        返回：
        - should_proactive: 是否应该主动发起
        - reason: 原因
        - message_suggestion: 建议的消息（如果应该发起）
        """
        state = self._state
        
        if state.is_in_conversation():
            return {
                "should_proactive": False,
                "reason": "Already in conversation",
                "save_for_later": True
            }
        
        if not state.is_user_idle():
            return {
                "should_proactive": False,
                "reason": f"User is active (idle: {state.get_idle_time():.0f}s)",
                "save_for_later": True
            }
        
        importance_score = await self._evaluate_task_importance(task)
        
        if importance_score < 0.5:
            return {
                "should_proactive": False,
                "reason": f"Task not important enough (score: {importance_score:.2f})",
                "save_for_later": True
            }
        
        from .llm import get_llm
        llm = get_llm()
        
        prompt = f"""判断这个任务是否值得主动打扰用户。

任务描述：{task.description}
任务类型：{task.type.value}
上下文：{task.context_content or '无'}

用户已经空闲了 {state.get_idle_time():.0f} 秒。

判断标准：
1. 是否有重要信息需要告知用户？
2. 是否是紧急事项？
3. 是否会让用户感兴趣？

返回JSON：
{{
    "should_proactive": true/false,
    "reason": "判断理由",
    "importance": 0-1的分数,
    "message_suggestion": "如果要发起对话，建议说什么（简短自然）"
}}

只返回JSON。"""

        response = await llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        result = json.loads(response)
        
        return {
            "should_proactive": result.get("should_proactive", False),
            "reason": result.get("reason", ""),
            "importance": result.get("importance", 0),
            "message_suggestion": result.get("message_suggestion"),
            "save_for_later": not result.get("should_proactive", False)
        }
    
    async def _evaluate_task_importance(self, task: Task) -> float:
        """评估任务重要性分数"""
        score = 0.0
        
        if task.type == TaskType.DIALOGUE_ACTION:
            score += 0.3
        
        if task.context_content:
            important_keywords = ["重要", "紧急", "提醒", "记住", "喜欢", "兴趣"]
            for kw in important_keywords:
                if kw in task.context_content:
                    score += 0.2
        
        if task.description:
            urgent_keywords = ["完成", "问题", "需要", "必须"]
            for kw in urgent_keywords:
                if kw in task.description:
                    score += 0.1
        
        return min(score, 1.0)


def get_memory_analyzer() -> MemoryAnalyzer:
    return MemoryAnalyzer()
