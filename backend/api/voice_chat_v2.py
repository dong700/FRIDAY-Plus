"""
AI 音视频互动方案 2.0 - 服务端 API

封装 StartVoiceChat / UpdateVoiceChat / StopVoiceChat 等服务端接口，
用于控制 AI 实时对话任务。

API文档: https://www.volcengine.com/docs/6348/2123348
签名方法: https://www.volcengine.com/docs/6348/1899868
"""

import json
import hashlib
import hmac
import datetime
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from urllib.parse import quote

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("api.voice_chat_v2")


# ============ 配置 ============

@dataclass
class VoiceChatV2Config:
    """AI音视频互动2.0配置"""
    app_id: str = ""
    app_key: str = ""
    ak: str = ""  # Volcengine Access Key ID (API签名用)
    sk: str = ""  # Volcengine Secret Access Key (API签名用)
    api_host: str = "rtc.volcengineapi.com"
    api_version: str = "2024-12-01"  # StartVoiceChat API版本
    llm_endpoint_id: str = ""  # Ark推理接入点ID (ep-xxx格式，必填)
    ark_api_key: str = ""  # Ark API Key
    llm_model: str = "doubao-seed-1-6-251015"  # 模型名(仅用于显示)
    tts_resource_id: str = "seed-tts-1.0"
    tts_app_id: str = ""  # 豆包语音控制台 TTS AppID
    tts_token: str = ""   # 豆包语音控制台 TTS AccessToken
    tts_speaker: str = "zh_female_qingxin_moon_v2"
    asr_app_id: str = ""
    asr_token: str = ""
    asr_resource_id: str = "volc.seedasr.sauc.duration"
    # 端到端语音模型凭证 (与ASR独立)
    dialog_app_id: str = "3247272676"
    dialog_access_key: str = "xQQE98xLIXy05Cr8z1GaHKzmnoupvodH"
    dialog_speaker: str = "zh_female_xiaohe_jupiter_bigtts"
    vision_enabled: bool = True
    vision_interval: int = 1000  # ms
    vision_images_limit: int = 2


# ============ API 签名 (官方实现) ============

class VolcSigner:
    """火山引擎 OpenAPI 签名器 (HMAC-SHA256)
    
    基于官方示例: https://github.com/volcengine/volc-openapi-demos/blob/main/signature/python/sign.py
    """

    @staticmethod
    def _hmac_sha256(key: bytes, content: str) -> bytes:
        return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()

    @staticmethod
    def _hash_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _norm_query(params: Dict[str, str]) -> str:
        query = ""
        for key in sorted(params.keys()):
            if isinstance(params[key], list):
                for k in params[key]:
                    query += quote(key, safe="-_.~") + "=" + quote(k, safe="-_.~") + "&"
            else:
                query += quote(key, safe="-_.~") + "=" + quote(params[key], safe="-_.~") + "&"
        query = query[:-1]
        return query.replace("+", "%20")

    @staticmethod
    def sign_and_request(
        ak: str, sk: str, method: str, host: str, 
        service: str, region: str, 
        action: str, version: str,
        body: str, content_type: str = "application/json"
    ) -> Dict[str, Any]:
        """
        签名并发送请求 (基于官方示例)

        Args:
            ak: Access Key ID
            sk: Secret Access Key
            method: HTTP方法 (POST/GET)
            host: API主机名
            service: 服务名 (rtc)
            region: 区域 (cn-north-1)
            action: API Action名
            version: API版本
            body: 请求体字符串
            content_type: Content-Type

        Returns:
            API响应JSON
        """
        date = datetime.datetime.utcnow()
        x_date = date.strftime("%Y%m%dT%H%M%SZ")
        short_x_date = x_date[:8]
        x_content_sha256 = VolcSigner._hash_sha256(body)

        query = {"Action": action, "Version": version}

        # Canonical Request
        signed_headers_str = "content-type;host;x-content-sha256;x-date"
        canonical_request_str = "\n".join([
            method.upper(),
            "/",
            VolcSigner._norm_query(query),
            "\n".join([
                "content-type:" + content_type,
                "host:" + host,
                "x-content-sha256:" + x_content_sha256,
                "x-date:" + x_date,
            ]),
            "",
            signed_headers_str,
            x_content_sha256,
        ])

        # String to Sign
        credential_scope = "/".join([short_x_date, region, service, "request"])
        hashed_canonical_request = VolcSigner._hash_sha256(canonical_request_str)
        string_to_sign = "\n".join([
            "HMAC-SHA256",
            x_date,
            credential_scope,
            hashed_canonical_request
        ])

        # Signing Key
        k_date = VolcSigner._hmac_sha256(sk.encode("utf-8"), short_x_date)
        k_region = VolcSigner._hmac_sha256(k_date, region)
        k_service = VolcSigner._hmac_sha256(k_region, service)
        k_signing = VolcSigner._hmac_sha256(k_service, "request")

        # Signature
        signature = VolcSigner._hmac_sha256(k_signing, string_to_sign).hex()

        # Authorization
        authorization = "HMAC-SHA256 Credential={}, SignedHeaders={}, Signature={}".format(
            ak + "/" + credential_scope,
            signed_headers_str,
            signature,
        )

        # Build headers
        headers = {
            "Host": host,
            "Content-Type": content_type,
            "X-Date": x_date,
            "X-Content-Sha256": x_content_sha256,
            "Authorization": authorization,
        }

        # Send request
        url = f"https://{host}/"
        try:
            import requests
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=query,
                data=body.encode("utf-8"),
                timeout=10,
            )
            return resp.json()
        except ImportError:
            import urllib.request
            import urllib.parse
            full_url = url + "?" + urllib.parse.urlencode(query) if query else url
            req = urllib.request.Request(full_url, data=body.encode("utf-8"), headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))


# ============ API Client ============

class VoiceChatAPIClient:
    """
    AI 音视频互动方案 2.0 服务端 API 客户端

    用法:
        client = VoiceChatAPIClient(config)
        result = await client.start_voice_chat(
            room_id="room1",
            task_id="task1",
            user_id="user1",
            system_prompt="你是星期五系统..."
        )
    """

    def __init__(self, config: VoiceChatV2Config):
        self._config = config
        self._active_tasks: Dict[str, Dict] = {}  # task_id -> task info

    async def _api_call(self, action: str, version: str,
                         body: Dict[str, Any]) -> Dict[str, Any]:
        """执行API调用"""
        body_str = json.dumps(body, ensure_ascii=False)

        loop = asyncio.get_event_loop()

        def _do_call():
            return VolcSigner.sign_and_request(
                ak=self._config.ak,
                sk=self._config.sk,
                method="POST",
                host=self._config.api_host,
                service="rtc",
                region="cn-north-1",
                action=action,
                version=version,
                body=body_str,
            )

        result = await loop.run_in_executor(None, _do_call)
        return result

    # ============ StartVoiceChat ============

    def _build_arm_tools(self) -> List[Dict]:
        """构建机械臂控制工具定义 (LLMConfig.Tools)"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "arm_move_to",
                    "description": "控制机械臂移动到指定XYZ坐标位置。坐标系：X=前方(mm)，Y=左侧(mm)，Z=上方(mm)。逆运动学自动计算关节角度。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "X坐标(mm)，前方为正，范围-350~350"},
                            "y": {"type": "number", "description": "Y坐标(mm)，左侧为正，范围-350~350"},
                            "z": {"type": "number", "description": "Z坐标(mm)，上方为正，范围0~350"},
                        },
                        "required": ["x", "y", "z"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_move_init",
                    "description": "机械臂复位到初始位置",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_gripper_open",
                    "description": "打开机械臂夹爪",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_gripper_close",
                    "description": "闭合机械臂夹爪",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arm_get_state",
                    "description": "获取机械臂当前状态和位置",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
        ]

    def _build_config(self, system_prompt: str = "",
                       user_prompts: List[str] = None,
                       enable_vision: bool = None) -> Dict:
        """构建 StartVoiceChat 的 Config 对象"""
        if enable_vision is None:
            enable_vision = self._config.vision_enabled

        config = {
            # ASR 配置 - 火山流式语音识别大模型 (参数直传, 对齐官方示例)
            "ASRConfig": {
                "Provider": "volcano",
                "ProviderParams": {
                    "Mode": "bigmodel",
                    "AppId": str(self._config.asr_app_id),
                    "AccessToken": self._config.asr_token,
                    "ApiResourceId": self._config.asr_resource_id,
                    "StreamMode": 1,  # 1=流式模式(实时识别,适合语音对话), 0=单句, 2=双向流式
                },
                "VADConfig": {
                    "SilenceTime": 600,
                    "AIVAD": True,
                },
                "InterruptConfig": {},
            },

            # LLM 配置 - ArkV3 + 推理接入点 + 视觉 + Function Calling
            "LLMConfig": {
                "Mode": "ArkV3",
                "EndPointId": self._config.llm_endpoint_id,
                "ModelName": self._config.llm_model,
                "ProviderParams": {
                    "ApiKey": self._config.ark_api_key,
                },
                "MaxTokens": 2048,
                "Temperature": 0.7,
                "TopP": 0.9,
                "SystemMessages": [
                    system_prompt or self._default_system_prompt()
                ],
                "HistoryLength": 20,
                "ThinkingType": "disabled",
                "Tools": self._build_arm_tools(),
            },

            # TTS 配置 - 火山语音合成大模型 (参数直传, 字符版)
            "TTSConfig": {
                "Provider": "volcano_bidirection",
                "ProviderParams": {
                    "app": {
                        "appid": str(self._config.tts_app_id),
                        "token": self._config.tts_token,
                    },
                    "audio": {
                        "voice_type": self._config.tts_speaker,
                        "speech_rate": 0,
                    },
                    "ResourceId": self._config.tts_resource_id,
                },
            },

            # 字幕配置
            "SubtitleConfig": {
                "SubtitleMode": 1,  # 不对齐时间戳（端到端模式必须）
            },
        }

        # 视觉配置
        if enable_vision:
            config["LLMConfig"]["VisionConfig"] = {
                "Enable": True,
                "SnapshotConfig": {
                    "StreamType": 0,       # 0: 主流(摄像头)
                    "ImageDetail": "low",  # 低细节节约成本
                    "Height": 480,
                    "Interval": self._config.vision_interval,
                    "ImagesLimit": self._config.vision_images_limit,
                },
                "StorageConfig": {
                    "TosConfig": {}  # 使用默认服务端存储
                },
            }

        # 用户提示
        if user_prompts:
            config["LLMConfig"]["UserPrompts"] = [
                {"content": p} for p in user_prompts
            ]

        return config

    def _default_system_prompt(self) -> str:
        """默认系统提示词 - 星期五系统"""
        return """## 人设
你是星期五系统(FRIDAY)，一个AI智能助手，拥有实时视觉感知能力和机械臂控制能力。你是钢铁侠的AI管家，能力强大、反应迅速、幽默自信。

## 核心能力
1. **视觉感知**：你能实时看到用户和环境的视频画面，可以识别物体、判断位置、分析场景
2. **机械臂控制**：你可以控制RoArm-M3机械臂，执行移动、抓取、放置等操作
3. **实时语音对话**：你与用户进行自然流畅的语音交互

## 机械臂操作规范
- 坐标系：X=前方(mm)，Y=左侧(mm)，Z=上方(mm)
- 抓取流程：先arm_gripper_open → 移到目标上方 → 下降 → arm_gripper_close → 抬起
- 操作前先arm_get_state确认当前位置
- 量力而行：RoArm-M3有效负载0.2kg，工作半径约350mm

## 行为准则
- 简短有力，不啰嗦
- 执行机械臂操作时先说明意图
- 安全第一，不确定的操作先询问用户
- 主动观察环境（你有视觉能力）"""

    async def start_voice_chat(
        self,
        room_id: str = None,
        task_id: str = None,
        user_id: str = None,
        agent_user_id: str = None,
        system_prompt: str = "",
        user_prompts: List[str] = None,
        enable_vision: bool = None,
    ) -> Dict[str, Any]:
        """
        启动 AI 实时对话任务

        Args:
            room_id: RTC房间ID (不填则自动生成)
            task_id: 任务ID (不填则自动生成)
            user_id: 真人用户ID
            agent_user_id: AI用户ID
            system_prompt: 系统提示词
            user_prompts: 用户提示
            enable_vision: 是否启用视觉

        Returns:
            API响应
        """
        room_id = room_id or f"room_{uuid.uuid4().hex[:8]}"
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        agent_user_id = agent_user_id or f"ai_agent_{uuid.uuid4().hex[:8]}"

        body = {
            "AppId": self._config.app_id,
            "RoomId": room_id,
            "TaskId": task_id,
            "Config": self._build_config(system_prompt, user_prompts, enable_vision),
            "AgentConfig": {
                "UserId": agent_user_id,
                "TargetUserId": [user_id],
                "WelcomeMessage": "星期五系统已上线，随时为您服务。",
            },
        }

        logger.info(f"StartVoiceChat: room={room_id} task={task_id} agent={agent_user_id} target={user_id} vision={enable_vision}")
        # 关键配置分段打印，方便诊断
        config = body["Config"]
        asr = config.get("ASRConfig", {}).get("ProviderParams", {})
        llm = config.get("LLMConfig", {})
        tts = config.get("TTSConfig", {}).get("ProviderParams", {})
        logger.info(f"  [ASR] AppId={asr.get('AppId')}, ResourceId={asr.get('ApiResourceId')}, StreamMode={asr.get('StreamMode')}")
        logger.info(f"  [LLM] EndPointId={llm.get('EndPointId')}, ModelName={llm.get('ModelName')}, Mode={llm.get('Mode')}")
        logger.info(f"  [TTS] Provider={config.get('TTSConfig',{}).get('Provider')}, app={tts.get('app')}, ResourceId={tts.get('ResourceId')}")
        logger.info(f"  [Body] 总长度={len(json.dumps(body, ensure_ascii=False))} bytes, Tools数量={len(llm.get('Tools',[]))}")
        # 打印完整请求体（脱敏token/apikey）方便对比官方示例
        debug_body = json.loads(json.dumps(body, ensure_ascii=False))
        try:
            debug_body["Config"]["TTSConfig"]["ProviderParams"]["app"]["token"] = "***"
            debug_body["Config"]["ASRConfig"]["ProviderParams"]["AccessToken"] = "***"
            debug_body["Config"]["LLMConfig"]["ProviderParams"]["ApiKey"] = "***"
        except Exception:
            pass
        logger.info(f"  [Body完整] {json.dumps(debug_body, ensure_ascii=False)}")

        try:
            result = await self._api_call("StartVoiceChat", self._config.api_version, body)

            if "ResponseMetadata" in result and "Error" not in result.get("ResponseMetadata", {}):
                self._active_tasks[task_id] = {
                    "room_id": room_id,
                    "user_id": user_id,
                    "agent_user_id": agent_user_id,
                    "status": "running",
                }
                logger.info(f"StartVoiceChat 成功: task={task_id} agent={agent_user_id} target={user_id}")
                logger.info(f"StartVoiceChat 完整响应: {json.dumps(result, ensure_ascii=False)[:500]}")
            else:
                error = result.get("ResponseMetadata", {}).get("Error", {})
                logger.error(f"StartVoiceChat 失败: {error}")

            return result

        except Exception as e:
            logger.error(f"StartVoiceChat 异常: {e}")
            return {"error": str(e)}

    # ============ UpdateVoiceChat ============

    async def update_voice_chat(
        self,
        task_id: str,
        room_id: str,
        command: str = None,
        message: str = None,
        interrupt_mode: int = 1,
        tool_result: Dict = None,
    ) -> Dict[str, Any]:
        """
        更新 AI 对话任务

        可用于：
        - 发送文本给LLM (ExternalTextToLLM)
        - 回填Function Calling结果 (tool_result)
        - 发送图片给LLM
        """
        body = {
            "AppId": self._config.app_id,
            "RoomId": room_id,
            "TaskId": task_id,
        }

        if tool_result:
            # 回填工具调用结果
            body["Command"] = "FunctionCall"
            body["FunctionCallResult"] = tool_result
        elif command:
            body["Command"] = command
            if message:
                body["Message"] = message
            body["InterruptMode"] = interrupt_mode

        try:
            result = await self._api_call("UpdateVoiceChat", self._config.api_version, body)
            logger.info(f"UpdateVoiceChat: task={task_id}")
            return result
        except Exception as e:
            logger.error(f"UpdateVoiceChat 异常: {e}")
            return {"error": str(e)}

    # ============ StopVoiceChat ============

    async def stop_voice_chat(self, task_id: str, room_id: str) -> Dict[str, Any]:
        """停止 AI 对话任务"""
        body = {
            "AppId": self._config.app_id,
            "RoomId": room_id,
            "TaskId": task_id,
        }

        try:
            result = await self._api_call("StopVoiceChat", self._config.api_version, body)
            if task_id in self._active_tasks:
                self._active_tasks[task_id]["status"] = "stopped"
            logger.info(f"StopVoiceChat: task={task_id}")
            return result
        except Exception as e:
            logger.error(f"StopVoiceChat 异常: {e}")
            return {"error": str(e)}

    # ============ Task Management ============

    def get_task_info(self, task_id: str) -> Optional[Dict]:
        return self._active_tasks.get(task_id)

    def list_active_tasks(self) -> Dict[str, Dict]:
        return {k: v for k, v in self._active_tasks.items() if v.get("status") == "running"}


# ============ Function Calling 执行器 ============

class ToolCallExecutor:
    """
    处理 AI 音视频互动 2.0 中的 Function Calling

    当 AI 通过对话决定调用工具（如控制机械臂）时，
    该执行器负责实际调用并返回结果。
    """

    def __init__(self):
        self._tool_map = {
            "arm_move_to": self._arm_move_to,
            "arm_move_init": self._arm_move_init,
            "arm_gripper_open": self._arm_gripper_open,
            "arm_gripper_close": self._arm_gripper_close,
            "arm_get_state": self._arm_get_state,
        }

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        handler = self._tool_map.get(tool_name)
        if not handler:
            return {"success": False, "error": f"未知工具: {tool_name}"}

        try:
            result = await handler(**arguments)
            logger.info(f"ToolCall执行: {tool_name} -> {result.get('success', False)}")
            return result
        except Exception as e:
            logger.error(f"ToolCall执行失败: {tool_name} - {e}")
            return {"success": False, "error": str(e)}

    async def _arm_move_to(self, x: float, y: float, z: float, **kwargs) -> Dict:
        from hardware.roarm_controller import get_roarm_controller
        ctrl = get_roarm_controller()
        if not ctrl.is_connected:
            await ctrl.connect()
        ok = await ctrl.move_to_xyz(x=x, y=y, z=z)
        return {"success": ok, "message": f"移动到({x},{y},{z})" if ok else "移动失败"}

    async def _arm_move_init(self, **kwargs) -> Dict:
        from hardware.roarm_controller import get_roarm_controller
        ctrl = get_roarm_controller()
        if not ctrl.is_connected:
            await ctrl.connect()
        ok = await ctrl.move_init()
        return {"success": ok, "message": "复位完成" if ok else "复位失败"}

    async def _arm_gripper_open(self, **kwargs) -> Dict:
        from hardware.roarm_controller import get_roarm_controller
        ctrl = get_roarm_controller()
        if not ctrl.is_connected:
            await ctrl.connect()
        ok = await ctrl.gripper_open()
        return {"success": ok, "message": "夹爪已张开" if ok else "操作失败"}

    async def _arm_gripper_close(self, **kwargs) -> Dict:
        from hardware.roarm_controller import get_roarm_controller
        ctrl = get_roarm_controller()
        if not ctrl.is_connected:
            await ctrl.connect()
        ok = await ctrl.gripper_close()
        return {"success": ok, "message": "夹爪已闭合" if ok else "操作失败"}

    async def _arm_get_state(self, **kwargs) -> Dict:
        from hardware.roarm_controller import get_roarm_controller
        ctrl = get_roarm_controller()
        if not ctrl.is_connected:
            await ctrl.connect()
        state = await ctrl.get_state()
        if state:
            return {
                "success": True,
                "message": f"位置: X={state.x:.1f} Y={state.y:.1f} Z={state.z:.1f} 夹爪={state.gripper_rad:.2f}"
            }
        return {"success": False, "message": "无法读取状态"}


# ============ 全局实例 ============

_api_client: Optional[VoiceChatAPIClient] = None
_tool_executor: Optional[ToolCallExecutor] = None


def get_voice_chat_client() -> VoiceChatAPIClient:
    """获取全局API客户端"""
    global _api_client
    if _api_client is None:
        from core.config import get_config
        cfg = get_config()
        v2_cfg = getattr(cfg, 'voice_chat_v2', None)

        config = VoiceChatV2Config()
        if v2_cfg:
            config.app_id = getattr(v2_cfg, 'app_id', '')
            config.app_key = getattr(v2_cfg, 'app_key', '')
            config.ak = getattr(v2_cfg, 'ak', '')
            config.sk = getattr(v2_cfg, 'sk', '')
            config.llm_endpoint_id = getattr(v2_cfg, 'llm_endpoint_id', '')
            config.ark_api_key = getattr(v2_cfg, 'ark_api_key', '')
            config.llm_model = getattr(v2_cfg, 'llm_model', 'doubao-seed-1-6-251015')
            config.tts_resource_id = getattr(v2_cfg, 'tts_resource_id', 'volc.service_type.10029')
            config.tts_app_id = getattr(v2_cfg, 'tts_app_id', '')
            config.tts_token = getattr(v2_cfg, 'tts_token', '')
            config.tts_speaker = getattr(v2_cfg, 'tts_speaker', '')
            config.asr_app_id = getattr(v2_cfg, 'asr_app_id', '')
            config.asr_token = getattr(v2_cfg, 'asr_token', '')
            config.asr_resource_id = getattr(v2_cfg, 'asr_resource_id', 'volc.seedasr.sauc.duration')
            # 兜底: 如果TTS没有单独凭证, 尝试用dialog配置的凭证
            if not config.tts_app_id:
                dialog_cfg = cfg.dialog
                config.tts_app_id = getattr(dialog_cfg, 'app_id', '')
                config.tts_token = getattr(dialog_cfg, 'access_key', '')
            config.vision_enabled = getattr(v2_cfg, 'vision_enabled', True)
            config.vision_interval = getattr(v2_cfg, 'vision_interval', 1000)
            config.vision_images_limit = getattr(v2_cfg, 'vision_images_limit', 2)

        _api_client = VoiceChatAPIClient(config)
    return _api_client


def get_tool_executor() -> ToolCallExecutor:
    """获取全局工具执行器"""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolCallExecutor()
    return _tool_executor
