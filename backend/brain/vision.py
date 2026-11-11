"""
视觉模块 - 图像理解与场景感知

使用火山引擎 doubao-vision-pro (Ark API) 实现视觉理解。
DeepSeek 不支持视觉输入，因此视觉走独立的 Ark API 端点。

核心功能：
1. 图片理解 - 描述图片内容、回答关于图片的问题
2. 物体定位 - 识别物体在画面中的位置（为机械臂抓取服务）
3. 场景描述 - 自动描述当前摄像头画面
4. 情绪识别 - 分析用户面部表情

调用方式与 LLM 一致，使用 OpenAI 兼容接口，仅换 base_url 和 model。
"""

import base64
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from core.config import get_config
from core.logger import get_logger

logger = get_logger("brain.vision")


class VisionModule:
    """
    视觉理解模块

    使用 doubao-vision-pro (Ark API) 理解图片内容。
    DeepSeek 不支持图片输入，所以视觉走独立的 API。
    """

    _instance: Optional['VisionModule'] = None

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
        """视觉后端 = 主 LLM（deepseek-flash 现在支持图片输入）。

        不再依赖火山方舟的视觉端点：和对话共用同一套 api_key / base_url / model，
        调用时关掉 thinking，否则 token 会被推理过程吃光、正文为空。
        """
        config = get_config()
        llm = config.llm

        self._api_key = getattr(llm, "api_key", "") or ""
        self._base_url = getattr(llm, "base_url", "") or ""
        self._model = getattr(llm, "model", "") or ""

        if self._api_key and self._base_url and self._model:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            logger.info(f"VisionModule 初始化成功 model={self._model}")
        else:
            self._client = None
            logger.warning("VisionModule 未初始化：主 LLM 配置不完整")

    def is_available(self) -> bool:
        """视觉模块是否可用"""
        return self._client is not None

    def _to_data_url(self, image_data: str) -> str:
        """
        将 base64 图片数据转为 data URL 格式。
        前端传来的格式可能是:
        - "data:image/jpeg;base64,xxx" (已完成)
        - "/9j/4AAQ..." (纯 base64，需要加前缀)
        """
        if image_data.startswith("data:"):
            return image_data
        # 默认按 jpeg 处理
        return f"data:image/jpeg;base64,{image_data}"

    async def describe_image(
        self,
        image_base64: str,
        question: str = "详细描述这张图片的内容，包括其中的物体、人物、场景、文字等关键信息。"
    ) -> Dict[str, Any]:
        """
        理解图片内容

        Args:
            image_base64: base64 编码的图片数据（可带 data: 前缀）
            question: 关于图片的问题或描述要求

        Returns:
            {
                "success": bool,
                "description": str,   # 图片描述文本
                "error": str,         # 错误信息（失败时）
            }
        """
        if not self._client:
            return {"success": False, "error": "视觉模块未初始化（主 LLM 配置不完整）", "description": ""}

        try:
            data_url = self._to_data_url(image_base64)

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
                max_tokens=600,
                extra_body={"thinking": {"type": "disabled"}},
            )

            description = response.choices[0].message.content or ""
            logger.info(f"[Vision] 图片理解成功 ({len(description)}字): {description[:80]}...")
            return {"success": True, "description": description, "error": ""}

        except Exception as e:
            logger.error(f"[Vision] 图片理解失败: {e}")
            return {"success": False, "description": "", "error": str(e)}

    async def find_object(
        self,
        image_base64: str,
        object_name: str
    ) -> Dict[str, Any]:
        """
        在图片中查找物体位置（为机械臂抓取服务）

        Args:
            image_base64: base64 编码的图片
            object_name: 要查找的物体名称

        Returns:
            {
                "success": bool,
                "found": bool,
                "description": str,  # 物体位置描述
                "error": str,
            }
        """
        question = (
            f"请仔细查看这张图片，找到「{object_name}」。\n"
            f"如果找到了，描述它的具体位置（在画面左/中/右，上/中/下，距离镜头远近）。\n"
            f"如果没有找到，明确说「没有找到{object_name}」。\n"
            f"用简洁的语言描述，不要超过100字。"
        )
        result = await self.describe_image(image_base64, question)
        if result["success"]:
            desc = result["description"]
            found = object_name in desc and "没有找到" not in desc
            return {
                "success": True,
                "found": found,
                "description": desc,
                "error": ""
            }
        return result

    async def analyze_scene(self, image_base64: str) -> Dict[str, Any]:
        """
        自动描述场景（用于大脑主动观察）

        Returns:
            {
                "success": bool,
                "description": str,  # 场景描述
                "error": str,
            }
        """
        question = (
            "用一句话简洁描述当前场景（谁在做什么、环境如何），不超过50字。"
            "如果有需要提醒用户的事情（如长时间工作、坐姿不正确等），在末尾用【提醒】标注。"
        )
        return await self.describe_image(image_base64, question)

    async def analyze_emotion(self, image_base64: str) -> Dict[str, Any]:
        """
        分析用户面部表情情绪（用于虚拟形象同步）

        Returns:
            {
                "success": bool,
                "emotion": str,  # happy/sad/neutral/angry/surprised/tired
                "description": str,
                "error": str,
            }
        """
        question = (
            "观察图片中的人的面部表情，判断其主要情绪。"
            "只回复一个词：开心/难过/中性/生气/惊讶/疲惫/专注。"
            "如果没有看到人脸，回复「无人脸」。"
        )
        result = await self.describe_image(image_base64, question)
        if result["success"]:
            emotion_map = {
                "开心": "happy", "难过": "sad", "中性": "neutral",
                "生气": "angry", "惊讶": "surprised", "疲惫": "tired",
                "专注": "focused", "无人脸": "no_face"
            }
            emotion_text = result["description"].strip()
            emotion = "neutral"
            for cn, en in emotion_map.items():
                if cn in emotion_text:
                    emotion = en
                    break
            return {
                "success": True,
                "emotion": emotion,
                "description": emotion_text,
                "error": ""
            }
        return {"success": False, "emotion": "neutral", "description": "", "error": result.get("error", "")}


# 单例获取
def get_vision_module() -> VisionModule:
    return VisionModule()
