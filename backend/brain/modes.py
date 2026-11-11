"""
大脑模式枚举

IDLE:   空闲，监听用户活动，检查内在动机
ACTIVE: 主动对话，处理交互
REFLECT: 反思，整理记忆和自我更新
"""

from enum import Enum


class BrainMode(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    REFLECT = "reflect"
