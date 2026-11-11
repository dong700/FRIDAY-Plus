"""
大脑指令系统 - 处理用户给大脑的直接指令

指令格式：
- @大脑 <指令内容>
- [BRAIN] <指令内容>

当用户使用这个格式时：
1. 实时对话系统正常回复（但大脑知道这不是真正的对话）
2. 大脑看到这个标识后，忽略实时对话的回复
3. 大脑执行指令

示例：
- "@大脑 主动告诉我今天天气怎么样"
- "[BRAIN] 提醒我明天早上开会"
- "@大脑 记住我喜欢喝咖啡"
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("brain.commands")


BRAIN_COMMAND_PATTERNS = [
    r'^@大脑\s*(.+)$',
    r'^\[BRAIN\]\s*(.+)$',
    r'^@brain\s*(.+)$',
    r'^\[大脑\]\s*(.+)$',
]


class BrainCommand:
    """大脑指令"""
    
    def __init__(self, raw_text: str, command: str, timestamp: str = None):
        self.raw_text = raw_text
        self.command = command
        self.timestamp = timestamp or datetime.now().isoformat()
        self.processed = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "command": self.command,
            "timestamp": self.timestamp,
            "processed": self.processed
        }


class BrainCommandProcessor:
    """
    大脑指令处理器
    
    功能：
    - 从对话中识别大脑指令
    - 解析指令内容
    - 存储待处理指令
    """
    
    _instance: Optional['BrainCommandProcessor'] = None
    
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
        self._commands_file = self._data_dir / "brain_commands.jsonl"
        self._pending_commands: List[BrainCommand] = []
        self._load_pending_commands()
    
    def _load_pending_commands(self) -> None:
        """加载待处理的指令"""
        try:
            if self._commands_file.exists():
                import json
                with open(self._commands_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if not data.get("processed"):
                                cmd = BrainCommand(
                                    raw_text=data["raw_text"],
                                    command=data["command"],
                                    timestamp=data.get("timestamp")
                                )
                                self._pending_commands.append(cmd)
                        except:
                            continue
                logger.info(f"Loaded {len(self._pending_commands)} pending commands")
        except Exception as e:
            logger.error(f"Failed to load pending commands: {e}")
    
    def extract_command(self, text: str) -> Optional[BrainCommand]:
        """
        从文本中提取大脑指令
        
        Args:
            text: 用户输入的文本
        
        Returns:
            如果是指令，返回 BrainCommand；否则返回 None
        """
        text = text.strip()
        
        for pattern in BRAIN_COMMAND_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                brain_cmd = BrainCommand(raw_text=text, command=command)
                logger.info(f"Extracted brain command: {command}")
                return brain_cmd
        
        return None
    
    def is_brain_command(self, text: str) -> bool:
        """检查文本是否是大脑指令"""
        return self.extract_command(text) is not None
    
    def add_command(self, command: BrainCommand) -> None:
        """添加待处理的指令"""
        self._pending_commands.append(command)
        self._save_command(command)
        logger.info(f"Added brain command: {command.command}")
    
    def _save_command(self, command: BrainCommand) -> None:
        """保存指令到文件"""
        try:
            self._commands_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(self._commands_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(command.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save command: {e}")
    
    def get_pending_commands(self) -> List[BrainCommand]:
        """获取待处理的指令"""
        return [cmd for cmd in self._pending_commands if not cmd.processed]
    
    def mark_processed(self, command: BrainCommand) -> None:
        """标记指令为已处理"""
        command.processed = True
        self._update_command_status(command)
        if command in self._pending_commands:
            self._pending_commands.remove(command)
        logger.info(f"Marked command as processed: {command.command}")
    
    def _update_command_status(self, command: BrainCommand) -> None:
        """更新指令状态"""
        try:
            if not self._commands_file.exists():
                return
            
            import json
            with open(self._commands_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                try:
                    data = json.loads(line)
                    if data.get("raw_text") == command.raw_text and data.get("timestamp") == command.timestamp:
                        data["processed"] = True
                    updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
                except:
                    updated_lines.append(line)
            
            with open(self._commands_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                
        except Exception as e:
            logger.error(f"Failed to update command status: {e}")
    
    def has_pending_commands(self) -> bool:
        """是否有待处理的指令"""
        return len(self.get_pending_commands()) > 0


def get_command_processor() -> BrainCommandProcessor:
    return BrainCommandProcessor()


def extract_brain_command(text: str) -> Optional[BrainCommand]:
    """快捷函数：从文本中提取大脑指令"""
    processor = get_command_processor()
    return processor.extract_command(text)


def is_brain_command(text: str) -> bool:
    """快捷函数：检查文本是否是大脑指令"""
    processor = get_command_processor()
    return processor.is_brain_command(text)
