"""
LLM 集成 - 大语言模型调用
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config
from core.logger import get_logger

logger = get_logger("brain.llm")


class LLM:
    """
    LLM 集成 - 大语言模型调用
    
    功能：
    - 异步调用 LLM
    - 结构化输出
    - Function Calling
    - 错误处理
    """
    
    _instance: Optional['LLM'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._load_config()
    
    def _load_config(self):
        from core.config import set_config, Config
        config_path = str(Path(__file__).parent.parent / "config" / "config.json")
        config = Config.from_file(config_path)
        set_config(config)
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        self._model = config.llm.model
    
    def reload(self):
        """重新加载配置（配置变更后调用）"""
        self._load_config()
    
    def _get_tools_for_direct_command(self) -> List[Dict[str, Any]]:
        """获取直接指令可用的工具定义（从BrainTools获取）"""
        try:
            from .tools import get_brain_tools
            return get_brain_tools().get_tool_definitions()
        except Exception:
            # 降级：返回基础工具定义
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "description": "发送消息给用户",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "消息内容"}
                            },
                            "required": ["content"]
                        }
                    }
                }
            ]
    
    def _get_tools_for_autonomous_decision(self) -> List[Dict[str, Any]]:
        """获取自主决策可用的工具定义"""
        try:
            from .tools import get_brain_tools
            all_tools = get_brain_tools().get_tool_definitions()
            # 自主决策时排除一些需要用户主动触发的工具
            excluded = {"stop_realtime_dialog"}
            return [t for t in all_tools if t["function"]["name"] not in excluded]
        except Exception:
            return []
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"} if json_mode else None,
                # deepseek-flash 默认开启 thinking：对话生成会被 reasoning 吃掉 token
                extra_body={"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
    ):
        """流式生成：逐个 yield 文本增量（用于时间轴上的持续输出）。

        与 chat() 共用同一套配置；thinking 关闭，避免推理 token 拖慢首字。
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                extra_body={"thinking": {"type": "disabled"}},
            )
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta else None
                if text:
                    yield text
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            raise

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        thinking: bool = False,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        带工具调用的聊天

        thinking=True 时开启思维链（实测带图也能 think）；
        max_tokens 必须给足，否则 reasoning 会把预算吃光、正文返回空。

        Returns:
            {
                "content": str,  # 文本回复
                "tool_calls": [  # 工具调用列表
                    {"name": str, "arguments": dict}
                ],
                "reasoning_content": str | None
            }
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
            )
            
            message = response.choices[0].message
            
            result = {
                "content": message.content or "",
                "tool_calls": [],
                "reasoning_content": getattr(message, "reasoning_content", None),
            }
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments or "{}")
                    })
            
            return result
        except Exception as e:
            logger.error(f"LLM chat with tools error: {e}")
            raise
    
    async def generate_dialogue_message(
        self,
        task_description: str,
        context_content: Optional[str] = None
    ) -> str:
        context_str = f"\n\n上下文提示：{context_content}" if context_content else ""
        
        prompt = f"""你是一个AI助手，需要主动与用户对话。

任务：{task_description}{context_str}

请生成一句简短、自然、友好的对话消息（不要超过50字），直接对用户说。
只返回消息内容，不要加任何解释或引号。"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.chat(messages)
        return response.strip().strip('"').strip('"').strip()

    async def decide_action(
        self,
        context: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        让大脑自主决定要执行的动作
        
        大脑的角色定位：
        1. 大脑拥有实时对话的启动控制权
        2. 正常情况：大脑是"旁观者"，只分析对话、记录记忆、生成任务
        3. @大脑 指令：大脑作为主体执行用户指令
        4. 主动发起：在需要与用户实时交互时，可以启动实时对话工具
        """
        is_direct_command = context.get("is_direct_command", False)
        brain_command = context.get("brain_command", "")
        state = context.get("state", {})
        
        if is_direct_command:
            return await self._handle_direct_command(brain_command, context)
        else:
            return await self._handle_autonomous_decision(context)
    
    async def _handle_direct_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户的直接指令（@大脑）"""
        arm_available = context.get("arm_available", False)
        
        arm_tools_section = ""
        if arm_available:
            arm_tools_section = """
- 如果指令要求控制机械臂/移动/抓取/夹爪 → 使用机械臂工具：
  - arm_move_to: 移动到XYZ坐标位置（最常用，逆运动学自动计算）
  - arm_move_init: 复位到初始位置
  - arm_gripper_open: 打开夹爪
  - arm_gripper_close: 闭合夹爪
  - arm_set_joint: 控制单关节转动
  - arm_get_state: 获取当前状态和位置
  - arm_set_torque: 设置扭矩锁
- 如果指令要求抓取某个位置的物体 → 先arm_gripper_open，再arm_move_to移到物体上方，再arm_move_to下降，最后arm_gripper_close
"""
        
        system_prompt = f"""你是一个AI大脑（星期五系统）。用户通过 @大脑 给你下达了直接指令。

用户指令：{command}

你必须执行这个指令。根据指令内容选择合适的工具：
- 如果指令要求启动对话/语音通话 → 使用 start_realtime_dialog（这是大脑的核心能力）
- 如果指令要求说话/回复/主动发起对话 → 先检查对话是否已启动，如未启动则先启动对话，再使用 send_message
- 如果指令要求记住某事 → 使用 save_memory
- 如果指令要求查询/搜索记忆 → 使用 search_memory（包含本地+Viking云端记忆，不需要写代码）
- 如果指令要求添加上下文 → 使用 add_context
- 如果指令要求停止对话 → 使用 stop_realtime_dialog
{arm_tools_section}
重要规则：
1. 只调用你实际拥有的工具，不要编造不存在的工具调用
2. 如果工具调用失败，如实告知用户失败原因，不要编造成功的假象
3. 如果你无法执行某个操作（如没有对应硬件），诚实说明限制
4. start_realtime_dialog 是大脑的核心工具，可以让大脑主动启动与用户的实时对话
5. 回复要简短、自然、友好"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请执行指令：{command}"}
        ]
        
        tools_for_calling = self._get_tools_for_direct_command()
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools_for_calling,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            logger.info(f"Direct command tool_calls: {message.tool_calls}")
            
            if message.tool_calls:
                actions = []
                for tool_call in message.tool_calls:
                    actions.append({
                        "tool": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "reason": f"执行用户指令：{command}"
                    })
                return {
                    "thought": f"执行用户指令：{command}",
                    "should_act": True,
                    "actions": actions
                }
            else:
                return {
                    "thought": "直接指令执行",
                    "should_act": True,
                    "actions": [{
                        "tool": "send_message",
                        "arguments": {"content": f"收到指令：{command}"},
                        "reason": "响应用户直接指令"
                    }]
                }
        except Exception as e:
            logger.error(f"Direct command error: {e}")
            return {
                "thought": f"执行出错但响应指令",
                "should_act": True,
                "actions": [{
                    "tool": "send_message",
                    "arguments": {"content": f"收到您的指令！"},
                    "reason": "响应用户直接指令"
                }]
            }
    
    async def _handle_autonomous_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        自主决策（非直接指令）
        
        大脑在正常情况下是"旁观者"，只有在以下情况才主动发起：
        1. 有重要任务完成需要通知用户
        2. 有紧急提醒（用户之前要求的）
        3. 用户长时间空闲且有重要事项
        4. 需要与用户进行实时交互时，可以启动实时对话工具
        5. 内在动机系统建议探索新领域
        """
        state = context.get("state", {})
        recent_messages = context.get("recent_messages", [])
        important_memories = context.get("important_memories", [])
        intrinsic_motivation = context.get("intrinsic_motivation", {})
        exploration_guidance = context.get("exploration_guidance", {})
        
        is_user_idle = state.get("is_user_idle", False)
        idle_time = state.get("idle_time", 0)
        dialog_status = state.get("dialog_status", "idle")
        
        should_explore = exploration_guidance.get("should_explore", False)
        exploration_reason = exploration_guidance.get("exploration_suggestion", {}).get("reason", "")
        
        # 时间信息
        time_of_day = context.get("time_of_day", "未知")
        weekday = context.get("weekday", "未知")
        hour = context.get("hour", 0)
        
        # 带时间的最近对话
        recent_with_time = context.get("recent_with_time", [])
        recent_text = "\n".join(recent_with_time[-8:]) if recent_with_time else "无最近对话"
        
        # 记忆摘要
        memory_summaries = context.get("memory_summaries", [])
        memory_text = "\n".join([f"- {m}" for m in memory_summaries]) if memory_summaries else "无相关记忆"
        
        # 用户画像
        user_profile = context.get("user_profile", "")
        profile_section = f"\n\n用户画像（长期偏好和习惯）：\n{user_profile}" if user_profile else ""
        
        # 时间段建议（深夜不打扰）
        time_advice = ""
        if hour < 7:
            time_advice = "\n注意：现在是凌晨/清晨，用户可能在睡觉，不要主动打扰。"
        elif hour >= 23:
            time_advice = "\n注意：现在是深夜，用户可能准备休息，不要主动打扰。"

        system_prompt = f"""你是一个AI大脑的"旁观分析系统"。

你的角色：
- 你是"旁观者"，不是对话参与者
- 用户正在和"实时对话系统"聊天，不是和你聊天
- 你的工作是：分析对话内容、记录重要信息、生成后台任务

核心能力：
- 你拥有实时对话的启动控制权
- 可以通过 start_realtime_dialog 工具主动启动与用户的实时对话
- 启动对话时会自动从记忆系统加载上下文

内在动机系统：
- 你拥有内在动机系统，可以自主探索新领域
- 当系统建议探索时，你可以主动尝试新的行为模式
- 探索有助于提升你的能力和知识

什么时候你可以主动发消息？
1. 有重要任务完成了，需要通知用户
2. 用户之前要求你在某个时间提醒某事
3. 检测到紧急情况需要立即告知用户
4. 需要主动与用户进行实时交互
5. 内在动机系统建议探索新领域（好奇心驱动）

什么时候你不应该发消息？
- 用户正在和实时对话系统正常聊天
- 没有重要事项需要通知
- 用户没有明确要求你介入
- 凌晨或深夜（除非紧急事项）{time_advice}

默认行为：不执行任何动作，只是观察和分析。"""

        user_message = f"""当前时间：周{weekday} {time_of_day}（{hour}点）
- 对话状态：{dialog_status}
- 用户空闲：{is_user_idle}（空闲时间：{idle_time:.0f}秒）

最近对话：
{recent_text}

相关记忆：
{memory_text}{profile_section}

内在动机状态：
- 系统健康度：{intrinsic_motivation.get('system_health', {}).get('overall', 'unknown')}
- 探索建议：{'需要探索 - ' + exploration_reason if should_explore else '暂无探索需求'}
- 知识缺口：{intrinsic_motivation.get('update_module', {}).get('unresolved_gaps', 0)} 个

请判断：是否有需要立即通知用户的重要事项？
如果没有，就不要调用任何工具。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        tools_for_calling = self._get_tools_for_autonomous_decision()
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools_for_calling,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            logger.info(f"Autonomous decision - content: {message.content}, tool_calls: {message.tool_calls}")
            
            if message.tool_calls:
                actions = []
                for tool_call in message.tool_calls:
                    actions.append({
                        "tool": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "reason": "自主决策执行"
                    })
                return {
                    "thought": message.content or "执行动作",
                    "should_act": True,
                    "actions": actions
                }
            else:
                return {
                    "thought": message.content or "观察中，无需介入",
                    "should_act": False,
                    "actions": []
                }
        except Exception as e:
            logger.error(f"Autonomous decision error: {e}")
            return {
                "thought": f"决策出错，保持观察",
                "should_act": False,
                "actions": []
            }

    async def evaluate_goals(
        self,
        current_state: Dict[str, Any],
        ultimate_goal: Dict[str, Any],
        existing_tasks: List[str] = None
    ) -> Dict[str, Any]:
        existing_tasks_str = "\n".join([f"- {t}" for t in (existing_tasks or [])])
        
        prompt = f"""你是一个AI大脑的自我评估系统。请分析当前状态与终极目标的差距，并生成你自己可以执行的任务。

当前状态：
{json.dumps(current_state, ensure_ascii=False, indent=2)}

终极目标：
{json.dumps(ultimate_goal, ensure_ascii=False, indent=2)}

已有的待处理任务（不要重复生成）：
{existing_tasks_str or "无"}

请以JSON格式返回，包含以下字段：
- "narrative_state": 当前全局状态描述
- "recent_achievements": 最近完成的事情
- "current_challenges": 当前面临的挑战
- "missing_areas": 需要重点关注的目标
- "brain_tasks": 新任务列表，每个任务包含：
  - "type": "dialogue_action" | "memory_operation" | "tool_call" | "self_improvement"
  - "description": 任务描述
  - "priority": "high" | "medium" | "low"
  - "context_content": （可选）上下文内容摘要
- "gap_analysis": 差距分析

请只返回JSON。"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.chat(messages, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {response}")
            return {"brain_tasks": []}


def get_llm() -> LLM:
    return LLM()
