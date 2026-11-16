"""
工具系统 - 大脑可调用的所有工具

大脑可以自主决定调用以下工具：
- start_realtime_dialog: 启动实时对话会话（大脑核心能力）
- stop_realtime_dialog: 停止实时对话会话
- send_message: 发送消息给用户
- add_context: 添加对话上下文（注入记忆的主要方式，比RAG更快）
- send_tts: 发送TTS文本（直接播放）
- set_greeting: 设置打招呼语
- query_chat: 发起闲聊查询
- get_conversation_history: 获取对话历史
- save_memory: 保存记忆
- search_memory: 搜索记忆
- update_state: 更新大脑状态
- arm_move_to: 控制机械臂移动到指定坐标位置
- arm_move_init: 机械臂复位到初始位置
- arm_gripper_open: 打开夹爪
- arm_gripper_close: 闭合夹爪
- arm_set_joint: 控制机械臂单个关节转动
- arm_get_state: 获取机械臂当前状态
- arm_set_torque: 设置扭矩锁开关

核心设计：
1. 大脑拥有实时对话的启动控制权
2. 启动对话时自动初始化上下文（从记忆系统加载）
3. 对话结束后自动同步记忆到大脑
4. 大脑可以通过自然语言控制机械臂（星期五系统）
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger
from core.state import get_global_state

logger = get_logger("brain.tools")


class BrainTools:
    """
    大脑工具系统 - 所有可调用的工具
    
    大脑可以根据记忆分析自主决定调用哪些工具
    """
    
    _instance: Optional['BrainTools'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._tools: Dict[str, Callable] = {}
        self._arm_tools_registered: bool = False
        self._data_dir = Path(__file__).parent.parent / "data"
        self._register_base_tools()
    
    def _register_base_tools(self) -> None:
        """注册基础工具（不含机械臂）"""
        self._tools = {
            "start_realtime_dialog": self.start_realtime_dialog,
            "stop_realtime_dialog": self.stop_realtime_dialog,
            "send_message": self.send_message,
            "add_context": self.add_context,
            "send_tts": self.send_tts,
            "set_greeting": self.set_greeting,
            "query_chat": self.query_chat,
            "get_conversation_history": self.get_conversation_history,
            "save_memory": self.save_memory,
            "search_memory": self.search_memory,
            "memory_remember": self.memory_remember,
            "update_state": self.update_state,
            "get_intrinsic_status": self.get_intrinsic_status,
            "video_find": self.video_find,
            "video_inspect": self.video_inspect,
            "video_edit": self.video_edit,
            "video_watch": self.video_watch,
            "video_watch_with_me": self.video_watch_with_me,
            "video_moment": self.video_moment,
            "video_watch_stop": self.video_watch_stop,
            "video_status": self.video_status,
            "computer_use": self.computer_use,
            "computer_windows": self.computer_windows,
            "person_seen": self.person_seen,
            "person_name": self.person_name,
            "person_profile": self.person_profile,
            "person_note": self.person_note,
            "pet_move": self.pet_move,
            "avatar_expression": self.avatar_expression,
        }
    
    def _ensure_arm_tools_registered(self) -> None:
        """延迟注册机械臂工具（首次调用时检查硬件可用性）"""
        if self._arm_tools_registered:
            return
        self._arm_tools_registered = True
        
        if self.is_arm_available():
            self._tools.update({
                "arm_move_to": self.arm_move_to,
                "arm_move_init": self.arm_move_init,
                "arm_gripper_open": self.arm_gripper_open,
                "arm_gripper_close": self.arm_gripper_close,
                "arm_set_joint": self.arm_set_joint,
                "arm_get_state": self.arm_get_state,
                "arm_set_torque": self.arm_set_torque,
            })
            logger.info("[BrainTools] 机械臂工具已注册")
        else:
            logger.info("[BrainTools] 机械臂不可用，跳过注册 arm_* 工具")
    
    async def start_realtime_dialog(
        self, 
        mod: str = "text", 
        greeting: str = None,
        auto_init_context: bool = True,
        purpose: str = None
    ) -> Dict[str, Any]:
        """
        启动实时对话会话 - 大脑核心能力
        
        大脑通过此工具获得实时对话的启动控制权，可以：
        1. 根据任务需求智能判断是否需要启动对话
        2. 自动完成上下文的初始化与更新
        3. 智能设计上下文内容（用户身份、对话目的、待通知事项）
        
        Args:
            mod: 对话模式 ("text" 或 "audio")
            greeting: 自定义问候语
            auto_init_context: 是否自动初始化上下文（智能生成）
            purpose: 本次启动对话的目的（用于智能生成上下文）
        
        Returns:
            {"success": bool, "session_id": str, "error": str}
        """
        logger.info(f"start_realtime_dialog called: mod={mod}, purpose={purpose}")
        
        try:
            from .realtime_dialog_tool import get_realtime_dialog_tool
            dialog_tool = get_realtime_dialog_tool()
            
            result = await dialog_tool.start_dialog(
                mod=mod,
                greeting=greeting,
                auto_init_context=auto_init_context,
                purpose=purpose
            )
            
            if result.get("success"):
                logger.info(f"Realtime dialog started: {result.get('session_id')}")
            else:
                logger.error(f"Failed to start dialog: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Start realtime dialog error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def stop_realtime_dialog(self) -> Dict[str, Any]:
        """
        停止实时对话会话
        
        停止对话时会自动：
        1. 同步对话记忆到大脑记忆系统
        2. 清理会话状态
        
        Returns:
            {"success": bool, "error": str}
        """
        logger.info("stop_realtime_dialog called")
        
        try:
            from .realtime_dialog_tool import get_realtime_dialog_tool
            dialog_tool = get_realtime_dialog_tool()
            
            result = await dialog_tool.stop_dialog()
            
            if result.get("success"):
                logger.info("Realtime dialog stopped successfully")
            else:
                logger.error(f"Failed to stop dialog: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Stop realtime dialog error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _get_client(self):
        from dialog.connection import get_connection_manager
        return get_connection_manager().get_client()
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """调用工具"""
        self._ensure_arm_tools_registered()
        logger.info(f"call_tool called: {tool_name}, kwargs: {kwargs}")
        
        if tool_name not in self._tools:
            logger.error(f"Unknown tool: {tool_name}")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        try:
            logger.info(f"Calling tool method: {tool_name}")
            result = await self._tools[tool_name](**kwargs)
            logger.info(f"Tool result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool error {tool_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def send_message(self, content: str, direct: bool = False) -> Dict[str, Any]:
        """
        发送消息给用户
        
        Args:
            content: 消息内容
            direct: 是否直接发送（不加"请对我说"前缀）
        """
        logger.info(f"send_message called: content={content[:50]}..., direct={direct}")
        
        client = self._get_client()
        logger.info(f"Client: {client}")
        
        if not client:
            logger.error("No client available")
            return {"success": False, "error": "Not connected"}
        
        try:
            message = content if direct else f"请对我说：{content}"
            logger.info(f"Sending message via chat_text_query: {message}")
            
            await client.chat_text_query(message)
            
            logger.info(f"Message sent successfully: {message[:50]}...")
            return {"success": True}
        except Exception as e:
            logger.error(f"Send message error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def add_context(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        添加对话上下文（注入记忆的主要方式）
        
        Args:
            items: 上下文项目列表，每项包含 role, text, timestamp
        """
        client = self._get_client()
        if not client:
            return {"success": False, "error": "Not connected"}
        
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
            
            await client.conversation_create(formatted_items)
            logger.info(f"Context added: {len(items)} items")
            return {"success": True, "count": len(items)}
        except Exception as e:
            logger.error(f"Add context error: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_tts(self, content: str) -> Dict[str, Any]:
        """
        发送TTS文本（直接播放，不经过对话模型）
        
        Args:
            content: 要播放的文本
        """
        client = self._get_client()
        if not client:
            return {"success": False, "error": "Not connected"}
        
        try:
            await client.chat_tts_text(content)
            logger.info(f"TTS sent: {content[:50]}...")
            return {"success": True}
        except Exception as e:
            logger.error(f"Send TTS error: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_greeting(self, greeting: str) -> Dict[str, Any]:
        """
        设置打招呼语
        
        Args:
            greeting: 新的打招呼语
        """
        try:
            greeting_file = Path("data/custom_greeting.txt")
            greeting_file.parent.mkdir(parents=True, exist_ok=True)
            with open(greeting_file, 'w', encoding='utf-8') as f:
                f.write(greeting)
            logger.info(f"Greeting set: {greeting}")
            return {"success": True, "greeting": greeting}
        except Exception as e:
            logger.error(f"Set greeting error: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_chat(self, query: str) -> Dict[str, Any]:
        """
        发起闲聊查询
        
        Args:
            query: 查询内容
        """
        client = self._get_client()
        if not client:
            return {"success": False, "error": "Not connected"}
        
        try:
            await client.chat_text_query(query)
            logger.info(f"Chat query: {query[:50]}...")
            return {"success": True}
        except Exception as e:
            logger.error(f"Query chat error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_conversation_history(self, limit: int = 20) -> Dict[str, Any]:
        """
        获取对话历史
        
        Args:
            limit: 最大返回条数
        """
        try:
            state = get_global_state()
            messages = state.get_recent_messages(limit)
            return {"success": True, "messages": messages, "count": len(messages)}
        except Exception as e:
            logger.error(f"Get history error: {e}")
            return {"success": False, "error": str(e)}
    
    async def save_memory(self, content: str, memory_type: str = None) -> Dict[str, Any]:
        """
        保存记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
        """
        try:
            memory_file = self._data_dir / "brain_memories.jsonl"
            memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            record = {
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(memory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Memory saved: {content[:50]}...")
            return {"success": True}
        except Exception as e:
            logger.error(f"Save memory error: {e}")
            return {"success": False, "error": str(e)}
    
    async def memory_remember(
        self,
        text: str,
        due_in_days: float = None,
        important: bool = True,
    ) -> Dict[str, Any]:
        """记进长期记忆（TFGR）。给 due_in_days 就是「到点提醒」。

        和 save_memory 的区别：这里进的是能按重要度排序、能主动叫醒的长期记忆，
        没有明确时间点的偏好/事实用 save_memory。
        """
        try:
            from brain.tfgr import get_tfgr
            return get_tfgr().remember(
                text=text,
                due_in_days=due_in_days,
                pin=bool(important),
                source="chat",
            )
        except Exception as e:
            logger.error(f"memory_remember error: {e}")
            return {"success": False, "error": str(e)}

    async def search_memory(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        搜索记忆（包括本地和 Viking 云端）
        
        Args:
            query: 搜索关键词
            limit: 最大返回条数
        """
        try:
            from .memory import get_memory_manager
            memory_manager = get_memory_manager()
            
            memories = await memory_manager.search_memories(query, limit=limit)
            
            logger.info(f"Memory search for '{query}': found {len(memories)} results")
            return {
                "success": True, 
                "memories": memories, 
                "count": len(memories),
                "query": query
            }
        except Exception as e:
            logger.error(f"Search memory error: {e}")
            return {"success": False, "error": str(e), "memories": []}
    
    async def update_state(self, key: str, value: Any) -> Dict[str, Any]:
        """
        更新大脑状态
        
        Args:
            key: 状态键
            value: 状态值
        """
        try:
            state_file = self._data_dir / "brain_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {}
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            
            state[key] = value
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"State updated: {key}={value}")
            return {"success": True, "state": state}
        except Exception as e:
            logger.error(f"Update state error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_intrinsic_status(self) -> Dict[str, Any]:
        """
        获取内在动机系统状态

        返回内在动机系统的完整状态，包括：
        - 探索建议
        - 知识缺口
        - 学习任务
        - 系统健康度
        """
        try:
            from .intrinsic_reward import get_intrinsic_motivation_system
            intrinsic_system = get_intrinsic_motivation_system()

            status = intrinsic_system.get_system_status()
            guidance = intrinsic_system.get_exploration_guidance({})

            logger.info("Retrieved intrinsic motivation status")

            return {
                "success": True,
                "status": status,
                "exploration_guidance": guidance
            }
        except Exception as e:
            logger.error(f"Get intrinsic status error: {e}")
            return {"success": False, "error": str(e)}

    # ============ 星期五系统 - 机械臂控制工具 ============

    async def arm_move_to(
        self,
        x: float, y: float, z: float,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        控制机械臂移动到指定坐标位置（逆运动学）

        Args:
            x: X坐标 (mm)，机械臂正前方为正
            y: Y坐标 (mm)，机械臂左侧为正
            z: Z坐标 (mm)，垂直向上为正
            description: 移动目的描述（用于日志和记忆）
        """
        logger.info(f"[星期五] arm_move_to: ({x}, {y}, {z}) - {description}")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.move_to_xyz(x=x, y=y, z=z)

            if success:
                logger.info(f"[星期五] 机械臂已移动到 ({x}, {y}, {z})")
                return {
                    "success": True,
                    "message": f"机械臂已移动到坐标({x}, {y}, {z})",
                    "position": {"x": x, "y": y, "z": z}
                }
            else:
                return {"success": False, "error": "移动指令发送失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_move_to 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_move_init(self) -> Dict[str, Any]:
        """机械臂复位到初始位置"""
        logger.info("[星期五] arm_move_init: 复位")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.move_init()

            if success:
                return {"success": True, "message": "机械臂已复位"}
            else:
                return {"success": False, "error": "复位指令发送失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_move_init 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_gripper_open(self) -> Dict[str, Any]:
        """打开机械臂夹爪"""
        logger.info("[星期五] arm_gripper_open: 张开夹爪")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.gripper_open()

            if success:
                return {"success": True, "message": "夹爪已张开"}
            else:
                return {"success": False, "error": "夹爪控制失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_gripper_open 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_gripper_close(self) -> Dict[str, Any]:
        """闭合机械臂夹爪"""
        logger.info("[星期五] arm_gripper_close: 闭合夹爪")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.gripper_close()

            if success:
                return {"success": True, "message": "夹爪已闭合"}
            else:
                return {"success": False, "error": "夹爪控制失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_gripper_close 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_set_joint(
        self,
        joint: int,
        radian: float,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        控制单个关节转动

        Args:
            joint: 关节编号 (1=底座, 2=肩, 3=肘, 4=手腕1, 5=手腕2, 6=夹爪)
            radian: 目标角度(弧度)
            description: 操作描述
        """
        joint_names = {1: "底座", 2: "肩", 3: "肘", 4: "手腕1", 5: "手腕2", 6: "夹爪"}
        name = joint_names.get(joint, f"关节{joint}")
        logger.info(f"[星期五] arm_set_joint: {name} 转动到 {radian:.2f}rad - {description}")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.set_joint(joint=joint, rad=radian)

            if success:
                return {
                    "success": True,
                    "message": f"{name}已转动到{radian:.2f}rad",
                    "joint": joint, "radian": radian
                }
            else:
                return {"success": False, "error": "关节控制失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_set_joint 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_get_state(self) -> Dict[str, Any]:
        """获取机械臂当前状态（位置、角度、负载、电压等）"""
        logger.info("[星期五] arm_get_state: 查询状态")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            state = await controller.get_state()

            if state:
                return {
                    "success": True,
                    "message": "获取状态成功",
                    "state": {
                        "position": {"x": round(state.x, 1), "y": round(state.y, 1), "z": round(state.z, 1)},
                        "joints": {
                            "base": round(state.base_rad, 3),
                            "shoulder": round(state.shoulder_rad, 3),
                            "elbow": round(state.elbow_rad, 3),
                            "wrist": round(state.wrist_rad, 3),
                            "roll": round(state.roll_rad, 3),
                            "gripper": round(state.gripper_rad, 3)
                        },
                        "voltage": round(state.voltage, 2)
                    }
                }
            else:
                return {"success": False, "error": "无法读取机械臂状态"}

        except Exception as e:
            logger.error(f"[星期五] arm_get_state 错误: {e}")
            return {"success": False, "error": str(e)}

    async def arm_set_torque(self, on: bool = True) -> Dict[str, Any]:
        """
        设置机械臂扭矩锁

        Args:
            on: True=开启(锁定不能手动转动), False=关闭(可手动转动)
        """
        action = "开启" if on else "关闭"
        logger.info(f"[星期五] arm_set_torque: {action}扭矩锁")

        try:
            from hardware.roarm_controller import get_roarm_controller
            controller = get_roarm_controller()

            if not controller.is_connected:
                await controller.connect()

            success = await controller.set_torque(on=on)

            if success:
                return {"success": True, "message": f"扭矩锁已{action}"}
            else:
                return {"success": False, "error": "扭矩锁控制失败"}

        except Exception as e:
            logger.error(f"[星期五] arm_set_torque 错误: {e}")
            return {"success": False, "error": str(e)}
    
    async def video_find(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        """在本机素材库里找视频（找视频链路的第一步）。"""
        from brain.video_tools import find_videos
        try:
            items = find_videos(query, limit=max(1, min(limit, 30)))
        except Exception as e:
            logger.error(f"[Video] find 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        return {"success": True, "count": len(items), "videos": items}

    async def video_inspect(self, path: str, max_seconds: Optional[float] = None) -> Dict[str, Any]:
        """看视频：切镜头 + 逐段描述 + 整体摘要（结果同时写进 ATG/TFGR）。"""
        from brain.video_analysis import analyze_video
        try:
            a = await analyze_video(path, max_seconds=max_seconds)
        except Exception as e:
            logger.error(f"[Video] inspect 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        return {
            "success": True, "name": a.name, "duration": a.duration,
            "summary": a.summary, "brief": a.brief(8),
            "scenes": [{"index": s.index + 1, "start": s.start, "end": s.end,
                        "description": s.description} for s in a.scenes],
        }

    async def video_edit(self, path: str, instruction: str, max_clips: int = 6) -> Dict[str, Any]:
        """剪视频：分析 → 规划 EDL → ffmpeg 渲染成片。"""
        from brain.video_tools import edit_video
        try:
            return await edit_video(path, instruction, max_clips=max(1, min(max_clips, 10)))
        except Exception as e:
            logger.error(f"[Video] edit 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def video_watch(self, path: str, max_seconds: Optional[float] = None) -> Dict[str, Any]:
        """整片观影：按 3~4Hz 看完全片，分窗口逐段描述，写进 TFGR/ATG。

        比 video_inspect 细一个量级（727 秒片子 → 94 段，每段看 1~9 帧），
        代价是慢（一分钟级）。用户问"把这集看一遍""扫一下这片"时用。
        """
        from brain.video_watch import describe, dump, publish, segment
        try:
            w = segment(path, 0.0, max_seconds, 4.0,
                        save_grid_dir=Path("data") / "video_watch" / "grids")
            w = await describe(w)
            publish(w)
            out = dump(w)
        except Exception as e:
            logger.error(f"[Video] watch 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        return {
            "success": True, "video": w.name, "duration": w.duration,
            "windows": len(w.windows), "timeline_file": str(out),
            "timeline": w.timeline()[:2000],
        }

    async def video_watch_with_me(self, path: str, from_seconds: float = 0.0,
                                  speed: float = 1.0) -> Dict[str, Any]:
        """开始一起看：跟播放进度实时看这片，之后随时能问"刚才那段"。

        不调模型（只解码+切窗口）；你问的时候它才把最近几十秒的画面交给眼睛。
        """
        from brain.video_companion import get_companion
        try:
            return get_companion().start(path, from_seconds, speed)
        except Exception as e:
            logger.error(f"[Companion] start 失败: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    async def video_moment(self, question: str = "") -> Dict[str, Any]:
        """看着片子的时候回答：不给问题就报告"现在画面在干嘛"。"""
        from brain.video_companion import get_companion
        c = get_companion()
        if not c.path:
            return {"ok": False, "error": "还没开始一起看，先用 video_watch_with_me"}
        try:
            return await c.moment(question)
        except Exception as e:
            logger.error(f"[Companion] moment 失败: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    async def video_watch_stop(self) -> Dict[str, Any]:
        """停止一起看。"""
        from brain.video_companion import get_companion
        return get_companion().stop()

    async def video_status(self) -> Dict[str, Any]:
        """现在是不是在陪看：片子名、播到第几秒、最近几段在讲什么。用户说“这”“刚才”指代不明时先查这个。"""
        from brain.video_companion import get_companion
        c = get_companion()
        s = c.status()
        s["recent"] = c.recent(3)
        return {"success": True, **s}

    async def computer_use(self, action: str, **kw) -> Dict[str, Any]:
        """操作电脑：移动/点击/拖拽/滚轮/打字/按键/组合键/切窗口。

        坐标是屏幕绝对像素（左上角原点）。动作全部走 Win32 SendInput，和真人操作等价。
        """
        from brain.computer_control import run
        r = await asyncio.to_thread(run, action, **kw)
        if r.get("ok"):
            try:
                from brain import atg
                atg.observe(f"自己动手：{action} {', '.join(f'{k}={v}' for k, v in kw.items() if v)}",
                            source="computer")
            except Exception:
                pass
        return r

    async def computer_windows(self, limit: int = 20) -> Dict[str, Any]:
        """列出可见窗口（标题+边框坐标），用来确定该 focus 哪个、点哪儿。"""
        from brain.computer_control import list_windows
        wins = await asyncio.to_thread(list_windows, max(1, min(limit, 60)))
        return {"success": True, "count": len(wins), "windows": wins}

    async def pet_move(self, x: float = 0.5, y: float = 1.0,
                       seconds: float = 0.0) -> Dict[str, Any]:
        """让桌面上的大肥鱼走到屏幕某个位置（x/y 是工作区比例，0=左/上，1=右/下）。

        桌宠默认不再自己随机乱飘，走位全部由你决定：用户说“过来”“去右边”“到角落待着”时调它。
        """
        try:
            from api.server import manager
            await manager.broadcast({
                "type": "pet.move",
                "payload": {"x": float(x), "y": float(y), "seconds": float(seconds)},
            })
            return {"success": True, "x": x, "y": y}
        except Exception as e:
            logger.error(f"[Pet] move 失败: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    async def avatar_expression(self, expressions: list = None, motion: str = "",
                                seconds: float = 0, persist: bool = False) -> Dict[str, Any]:
        """给桌面上的大肥鱼换表情/动作。

        expressions 是一个数组，可以一次叠加多个（比如 ["爱心眼", "脸红", "兔牙"]），
        传空数组表示清空表情。motion 可选（Idle/Gesture/说话动作等）。
        seconds > 0 按指定秒数淡出；不传则约 12 秒自动恢复。
        persist=True 表示这是持续状态（哭/生气/拿话筒…），一直挂着直到再调一次清掉。
        """
        try:
            from api.server import manager
            if isinstance(expressions, str):
                expressions = [x for x in expressions.replace("，", ",").replace("、", ",").split(",")]
            names = [str(x).strip() for x in (expressions or []) if str(x).strip()]
            payload: Dict[str, Any] = {"expressions": names}
            if motion:
                payload["action"] = str(motion)
            if seconds and float(seconds) > 0:
                payload["seconds"] = float(seconds)
            if persist:
                payload["persist"] = True
            await manager.broadcast({"type": "avatar.action", "payload": payload})
            return {"success": True, "expressions": names,
                    "motion": motion or None, "seconds": seconds or 0,
                    "persist": bool(persist)}
        except Exception as e:
            logger.error(f"[Avatar] expression 失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def person_seen(self, window_s: float = 300) -> Dict[str, Any]:
        """现在画面上都有谁：返回跟踪器编号（p1/p2）、是否已经认得、对应真名。"""
        from brain.people import get_people
        return {"success": True, "people": get_people().seen_now(window_s)}

    async def person_name(self, who: str, name: str, note: str = "") -> Dict[str, Any]:
        """用户给人物命名：「这是男主，叫消炎」→ 记住这个人和这张脸，并把历史记录一起改名。"""
        from brain.people import get_people
        return await asyncio.to_thread(get_people().bind, who, name, note)

    async def person_profile(self, who: str, limit: int = 12) -> Dict[str, Any]:
        """这个人以前干过什么（跨集）：第一次出现、各集里的事件、用户补的设定。"""
        from brain.people import get_people
        return await asyncio.to_thread(get_people().profile, who, max(1, min(limit, 40)))

    async def person_note(self, who: str, text: str) -> Dict[str, Any]:
        """补一条人物设定（「他是消炎的弟弟」「欠了高利贷」），以后聊天会带出来。"""
        from brain.people import get_people
        return await asyncio.to_thread(get_people().note, who, text)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有已注册工具的定义（用于LLM function calling）"""
        self._ensure_arm_tools_registered()
        all_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "start_realtime_dialog",
                    "description": "启动实时语音对话会话。这是大脑的核心能力，用于主动与用户进行实时语音交互。启动时会智能生成上下文（包含用户身份、对话目的、待通知事项），避免RAG搜索。适用场景：1.需要与用户进行实时语音对话 2.需要主动询问用户信息 3.需要向用户汇报重要事项",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mod": {
                                "type": "string", 
                                "enum": ["text", "audio"],
                                "description": "对话模式：text为文本模式，audio为语音模式"
                            },
                            "purpose": {
                                "type": "string",
                                "description": "本次启动对话的目的，用于智能生成上下文。例如：'通知用户任务完成'、'询问用户偏好'、'提醒用户重要事项'"
                            },
                            "greeting": {
                                "type": "string",
                                "description": "自定义问候语（可选）"
                            },
                            "auto_init_context": {
                                "type": "boolean",
                                "description": "是否自动初始化上下文（默认true，会智能生成上下文）"
                            }
                        },
                        "required": ["mod"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_realtime_dialog",
                    "description": "停止实时对话会话。停止时会自动将对话记忆同步到大脑记忆系统。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "发送消息给用户。用于主动发起对话或回复用户。注意：需要先启动实时对话会话才能使用此工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "消息内容"},
                            "direct": {"type": "boolean", "description": "是否直接发送不加前缀"}
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_context",
                    "description": "添加对话上下文。用于让对话模型了解之前的对话内容、背景信息或记忆。这是注入记忆的主要方式，比RAG更快。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string", "description": "角色: user 或 assistant"},
                                        "text": {"type": "string", "description": "文本内容"},
                                        "timestamp": {"type": "string", "description": "时间戳"}
                                    }
                                }
                            }
                        },
                        "required": ["items"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_tts",
                    "description": "直接播放TTS语音，不经过对话模型。用于播放提示音或简短通知。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "要播放的文本"}
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_greeting",
                    "description": "设置打招呼语。用于自定义连接时的问候语。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "greeting": {"type": "string", "description": "新的打招呼语"}
                        },
                        "required": ["greeting"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "保存重要记忆。用于记录用户偏好、重要事件等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "记忆内容"},
                            "memory_type": {"type": "string", "description": "记忆类型: preference, event, fact, general"}
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_remember",
                    "description": (
                        "把一件以后还要用到的事记进长期记忆。用户说「记住…」「X 天后提醒我…」"
                        "「别忘了…」时必须调用。有明确时间点就传 due_in_days，"
                        "系统会在到点时主动提醒，不需要用户再问。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "要记住的事，一句话，带上关键细节"},
                            "due_in_days": {
                                "type": "number",
                                "description": "多少天后到点提醒（没有明确时间就不要传）",
                            },
                            "important": {
                                "type": "boolean",
                                "description": "是否钉住保护（默认 true，清理记忆时不会被删）",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "搜索记忆（包括本地记忆和Viking云端记忆）。用于查找用户偏好、历史事件、重要对话、知识记录等。搜索结果可以通过 add_context 注入到对话中。注意：这是大脑自带的工具，不需要写代码。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词或语义描述"},
                            "limit": {"type": "integer", "description": "最大返回条数"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conversation_history",
                    "description": "获取最近的对话历史。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "最大返回条数"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_intrinsic_status",
                    "description": "获取内在动机系统状态。包括探索建议、知识缺口、学习任务、系统健康度等。用于了解AI的自我驱动状态。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            # ===== 星期五系统 - 机械臂控制工具定义 =====
            {
                "type": "function",
                "function": {
                    "name": "arm_move_to",
                    "description": "控制机械臂移动到指定XYZ坐标位置。坐标系：X轴正方向为机械臂正前方，Y轴正方向为左侧，Z轴正方向为垂直向上。单位为毫米(mm)。机械臂会通过逆运动学自动计算各关节角度。这是最常用的机械臂控制工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "X坐标(mm)，机械臂正前方为正，范围约-350到350"},
                            "y": {"type": "number", "description": "Y坐标(mm)，机械臂左侧为正，范围约-350到350"},
                            "z": {"type": "number", "description": "Z坐标(mm)，垂直向上为正，范围约0到350"},
                            "description": {"type": "string", "description": "移动目的描述，如'抓取桌面上的物体'"}
                        },
                        "required": ["x", "y", "z"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_move_init",
                    "description": "机械臂复位到初始位置。所有关节回到开机时的中位。在需要重置机械臂状态时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_gripper_open",
                    "description": "打开机械臂夹爪。在抓取物体前需要先打开夹爪。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_gripper_close",
                    "description": "闭合机械臂夹爪。在需要夹住/抓取物体时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_set_joint",
                    "description": "控制机械臂单个关节转动到指定角度。关节编号：1=底座, 2=肩, 3=肘, 4=手腕1, 5=手腕2, 6=夹爪。角度使用弧度制。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "joint": {"type": "integer", "description": "关节编号：1=底座(±3.14), 2=肩(±1.57), 3=肘(0~3.14), 4=手腕1(±1.57), 5=手腕2(±3.14), 6=夹爪(1.08~3.14)"},
                            "radian": {"type": "number", "description": "目标角度(弧度)。夹爪：3.14=闭合, 1.08=最大张开"},
                            "description": {"type": "string", "description": "操作描述"}
                        },
                        "required": ["joint", "radian"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_get_state",
                    "description": "获取机械臂当前状态，包括末端位置坐标(XYZ)、各关节角度、负载和电压。在执行操作前建议先获取状态了解当前位置。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_set_torque",
                    "description": "设置机械臂扭矩锁。开启后关节被锁定不能手动转动；关闭后可以手动拖动机械臂（适用于示教/编程场景）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "on": {"type": "boolean", "description": "True=开启扭矩锁(锁定), False=关闭扭矩锁(可手动转动)"}
                        },
                        "required": ["on"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_find",
                    "description": "在本机视频素材目录里搜索视频文件（关键词匹配文件名/路径）。用于“帮我找找某个视频”。返回路径、大小、修改时间。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词，可以是项目名/主题/文件名片段，留空则按最近修改排序"},
                            "limit": {"type": "integer", "description": "最多返回多少个，默认 10"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_inspect",
                    "description": "看视频内容：自动切分镜头，逐段描述画面/人物/动作/字幕，再给整体摘要。结果会写进活动时间线（ATG）和事件链（TFGR），之后可以直接按时间引用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "视频文件绝对路径（video_find 返回的 path）"},
                            "max_seconds": {"type": "number", "description": "只看前 N 秒（可选，例如用户问“前几秒”）"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_edit",
                    "description": "剪视频：先看内容，再按剪辑要求挑选镜头并渲染成片（只裁/拼不编造素材）。返回成片路径和用到的片段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "源视频绝对路径"},
                            "instruction": {"type": "string", "description": "剪辑要求，例如“剪出老人独处的镜头，30 秒内”"},
                            "max_clips": {"type": "integer", "description": "最多用几段，默认 6"}
                        },
                        "required": ["path", "instruction"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_watch",
                    "description": "整片观影：按每秒 3~4 帧看完全片，按画面变化切成几十段，每段看 1~9 帧后写描述，结果进事件链（TFGR）和活动时间线（ATG）。比 video_inspect 细一个量级，但慢（一分钟级）。用户说“把这集看一遍”“仔细扫一下这条片”时用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "视频文件绝对路径"},
                            "max_seconds": {"type": "number", "description": "只看前 N 秒（可选）"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_watch_with_me",
                    "description": "开始和用户一起看视频：跟着播放进度实时看，不预先花模型的钱。开了之后随时能用 video_moment 问“刚才那段在干嘛”“那个人说了什么”。用户说“陪我一起看这个视频”时用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "视频文件绝对路径"},
                            "from_seconds": {"type": "number", "description": "从第几秒开始看，默认 0"},
                            "speed": {"type": "number", "description": "播放倍速，默认 1.0（用户开 1.5 倍速就给 1.5）"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_moment",
                    "description": "一起看的时候回答：把最近几十秒的画面交给眼睛，再结合已经攒下的叙事回答。用户问“现在画面里是什么”“刚才那个老头干嘛了”“他为什么走”都用它。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "用户的问题；留空则只报告现在画面里在干什么"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "computer_windows",
                    "description": "列出当前可见窗口（标题 + 边框坐标）。不确定该点哪儿、该切到哪个窗口时先调它。",
                    "parameters": {
                        "type": "object",
                        "properties": {"limit": {"type": "integer", "description": "最多几条，默认 20"}},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "computer_use",
                    "description": "操作电脑，等价于真人用鼠标键盘：移动/点击/双击/右键/拖拽/滚轮/打字/按键/组合键/按标题切窗口。坐标用屏幕绝对像素（左上角原点，1920×1080 屏幕 x 0~1919）。先看屏幕或 computer_windows 再决定点哪里；一次只做一个动作，做完看结果再决定下一步。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["move", "click", "double_click", "right_click", "drag", "scroll", "type", "press", "hotkey", "focus"], "description": "要做的动作"},
                            "x": {"type": "number", "description": "目标 x（move/click/scroll 用）"},
                            "y": {"type": "number", "description": "目标 y"},
                            "x1": {"type": "number", "description": "drag 起点 x"},
                            "y1": {"type": "number", "description": "drag 起点 y"},
                            "x2": {"type": "number", "description": "drag 终点 x"},
                            "y2": {"type": "number", "description": "drag 终点 y"},
                            "duration": {"type": "number", "description": "移动/拖拽耗时秒（默认瞬间，给人看的动作给 0.3~0.6）"},
                            "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "哪个键，默认 left"},
                            "clicks": {"type": "integer", "description": "连点几次，默认 1"},
                            "text": {"type": "string", "description": "type 要打的文本（支持中文）"},
                            "key": {"type": "string", "description": "press 的单个键：enter/esc/tab/up/down/f1 等"},
                            "combo": {"type": "string", "description": "hotkey 的组合，例如 ctrl+s、alt+tab、win+r"},
                            "amount": {"type": "integer", "description": "scroll 格数，正数向下/向右"},
                            "horizontal": {"type": "boolean", "description": "scroll 是否横向"},
                            "title": {"type": "string", "description": "focus 的窗口标题片段"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_watch_stop",
                    "description": "停止一起看视频（用户关掉片子或说“别看了”时）。",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_status",
                    "description": "现在是不是在陪看：片子名、播到第几秒、最近几段在讲什么。用户说“这”“刚才”“这片”指代不明时先查它，别猜。",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "person_seen",
                    "description": "现在画面上都有谁：跟踪器给的临时编号（p1/p2）、这张脸认不认得、对应真名。用户问“这人是谁”“刚才那个人”时先调它。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "window_s": {"type": "number", "description": "往前看多少秒内出现过的人，默认 300"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "person_name",
                    "description": "用户给人物命名时调它，例如“这是男主，叫消炎”。会把这张脸钉到这个名上，并把这个人过去所有记录（p1 的）一起改名，以后换集也认得出。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "who": {"type": "string", "description": "跟踪器编号 p1/p2，或已有的 pid/名字"},
                            "name": {"type": "string", "description": "用户给的名字"},
                            "note": {"type": "string", "description": "顺带记一句设定，例如“男主”“欠了债”"}
                        },
                        "required": ["who", "name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "person_profile",
                    "description": "这个人以前干过什么：跨集取他名下的所有事件和时间线。用户问“他上次怎么了”“这人在前几集干嘛了”时用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "who": {"type": "string", "description": "pid 或名字"},
                            "limit": {"type": "integer", "description": "最多返回多少条事件，默认 12"}
                        },
                        "required": ["who"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "person_note",
                    "description": "补一条人物设定，例如“他是消炎的弟弟”“他欠了高利贷”。以后聊到这个人物会带出来。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "who": {"type": "string", "description": "pid 或名字"},
                            "text": {"type": "string", "description": "设定内容"}
                        },
                        "required": ["who", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pet_move",
                    "description": "让桌面上的大肥鱼走到屏幕指定位置。x/y 是工作区比例：0=最左/最上，1=最右/最下（y 用 1 就是贴任务栏）。桌宠默认不会自己乱飘，走位由你决定：用户说“过来”“去右边”“到角落待着”时调它。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "横向位置 0~1，0.5 居中"},
                            "y": {"type": "number", "description": "纵向位置 0~1，1 贴底（桌宠一般用 1）"},
                            "seconds": {"type": "number", "description": "走过去用多少秒，0 表示按距离自动"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "avatar_expression",
                    "description": (
                        "给桌面上的大肥鱼换表情/动作。expressions 是数组，可以叠加组合，"
                        "比如 [\"爱心眼\",\"脸红\"] 或 [\"黑脸\",\"流冷汗\"]；传 [] 清空。"
                        "多次调用会「叠加」：新加的表情不会把之前挂着的顶掉，想整体收掉就传 []。"
                        "默认是「一次性表情」：约 12 秒后自动淡出（想早点收就传 seconds）。"
                        "如果是持续状态（哭、生气、害羞、拿话筒……），设 persist=true 让它一直挂着，"
                        "之后想收掉再调一次 expressions=[]。"
                        "可用表情：默认体型、体型1、体型2、兔牙、前倾、加载、单片眼镜、圈圈眼、"
                        "尾巴开关、屑屑眼、流冷汗、流泪、爱心眼、竖食指开、竖食指关、胸口衣服、"
                        "脱鞋、脸红、裙子切换、钢盆、黑框眼镜、黑脸、＞＜。"
                        "可以按情绪自由搭配；不要每句话都换。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expressions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "要同时打开的表情名，可任意组合；[] 表示全部关掉"
                            },
                            "motion": {
                                "type": "string",
                                "description": "可选动作：Idle / Gesture / speaking / listening / think"
                            },
                            "seconds": {
                                "type": "number",
                                "description": "多少秒后自动淡出，0/不传 = 默认约 12 秒"
                            },
                            "persist": {
                                "type": "boolean",
                                "description": "true=这是持续状态，不自动消失，直到之后主动清掉；默认 false=过一会儿自己淡出"
                            }
                        },
                        "required": ["expressions"]
                    }
                }
            }
        ]
        
        # 只返回已注册工具的定义
        registered_names = set(self._tools.keys())
        return [d for d in all_definitions if d["function"]["name"] in registered_names]

    def is_arm_available(self) -> bool:
        """检查机械臂硬件是否可用"""
        try:
            from core.state import get_global_state
            return get_global_state().arm_available
        except Exception:
            return False


def get_brain_tools() -> BrainTools:
    return BrainTools()
