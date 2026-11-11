"""
配置管理 - 统一管理所有配置
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# ── 统一运行/存储根目录 ─────────────────────────────
# 聊天记录、推理日志、工作区、日志文件全部落在这个根下，
# 方便整体搬到服务器（部署时用 FRIDAY_HOME 指向数据盘即可，不用改代码）。
def get_friday_home() -> Path:
    return Path(os.environ.get("FRIDAY_HOME", "D:/FRIDAY"))


def get_data_dir() -> Path:
    return get_friday_home() / "data"


def get_log_dir() -> Path:
    return get_friday_home() / "logs"


def get_workspace_dir() -> Path:
    return get_data_dir() / "workspace"



@dataclass
class VikingMemoryConfig:
    # 云端记忆库总开关：false 时完全不建客户端，记忆只走本地 jsonl
    enabled: bool = True
    api_key: str = ""
    base_url: str = ""
    collection_name: str = "Active_AI"
    embedding_model: str = "text-embedding-v2"


@dataclass
class SchedulerConfig:
    memory_interval: int = 60
    goal_interval: int = 14400
    check_interval: int = 3600


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"


@dataclass
class TavilyConfig:
    api_key: str = ""
    max_results: int = 5


@dataclass
class RoArmConfig:
    mode: str = "http"
    http_ip: str = "192.168.4.1"
    http_port: int = 80
    serial_port: str = ""
    serial_baudrate: int = 115200
    timeout: float = 5.0


@dataclass
class VoiceChatV2Config:
    app_id: str = ""
    app_key: str = ""
    ak: str = ""
    sk: str = ""
    api_host: str = "rtc.volcengineapi.com"
    llm_endpoint_id: str = ""
    ark_api_key: str = ""
    llm_model: str = "doubao-seed-1-6-251015"
    # ASR 配置
    asr_app_id: str = ""
    asr_token: str = ""
    asr_resource_id: str = "volc.seedasr.sauc.duration"
    # TTS 配置 (语音合成大模型-字符版)
    tts_app_id: str = ""
    tts_token: str = ""
    tts_resource_id: str = "volc.service_type.10029"
    tts_speaker: str = ""
    # 端到端语音模型 (独立于ASR/TTS)
    dialog_app_id: str = "3247272676"
    dialog_access_key: str = "xQQE98xLIXy05Cr8z1GaHKzmnoupvodH"
    dialog_app_key: str = ""
    dialog_speaker: str = "zh_female_xiaohe_jupiter_bigtts"
    vision_enabled: bool = True
    vision_interval: int = 1000
    vision_images_limit: int = 2
    api_version: str = "2024-12-01"


@dataclass
class DialogConfig:
    app_id: str = ""
    access_key: str = ""
    resource_id: str = "volc.speech.dialog"
    app_key: str = ""
    base_url: str = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"


@dataclass
class Config:
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dialog: DialogConfig = field(default_factory=DialogConfig)
    viking_memory: VikingMemoryConfig = field(default_factory=VikingMemoryConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    roarm: RoArmConfig = field(default_factory=RoArmConfig)
    voice_chat_v2: VoiceChatV2Config = field(default_factory=VoiceChatV2Config)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = cls()
        
        if "scheduler" in data:
            config.scheduler = SchedulerConfig(**data["scheduler"])
        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "dialog" in data:
            config.dialog = DialogConfig(**data["dialog"])
        if "viking_memory" in data:
            vm_data = data["viking_memory"]
            config.viking_memory = VikingMemoryConfig(
                enabled=bool(vm_data.get("enabled", True)),
                api_key=vm_data.get("api_key", ""),
                base_url=vm_data.get("base_url", ""),
                collection_name=vm_data.get("collection_name", "Active_AI"),
                embedding_model=vm_data.get("embedding_model", "text-embedding-v2")
            )
        if "tavily" in data:
            tv_data = data["tavily"]
            config.tavily = TavilyConfig(
                api_key=tv_data.get("api_key", ""),
                max_results=tv_data.get("max_results", 5)
            )
        if "roarm" in data:
            ra_data = data["roarm"]
            config.roarm = RoArmConfig(
                mode=ra_data.get("mode", "http"),
                http_ip=ra_data.get("http_ip", "192.168.4.1"),
                http_port=ra_data.get("http_port", 80),
                serial_port=ra_data.get("serial_port", ""),
                serial_baudrate=ra_data.get("serial_baudrate", 115200),
                timeout=ra_data.get("timeout", 5.0)
            )
        if "voice_chat_v2" in data:
            v2_data = data["voice_chat_v2"]
            config.voice_chat_v2 = VoiceChatV2Config(
                app_id=v2_data.get("app_id", ""),
                app_key=v2_data.get("app_key", ""),
                ak=v2_data.get("ak", ""),
                sk=v2_data.get("sk", ""),
                api_host=v2_data.get("api_host", "rtc.volcengineapi.com"),
                llm_endpoint_id=v2_data.get("llm_endpoint_id", ""),
                ark_api_key=v2_data.get("ark_api_key", ""),
                llm_model=v2_data.get("llm_model", "doubao-seed-1-6-251015"),
                asr_app_id=v2_data.get("asr_app_id", ""),
                asr_token=v2_data.get("asr_token", ""),
                asr_resource_id=v2_data.get("asr_resource_id", "volc.seedasr.sauc.duration"),
                tts_app_id=v2_data.get("tts_app_id", ""),
                tts_token=v2_data.get("tts_token", ""),
                tts_resource_id=v2_data.get("tts_resource_id", "volc.service_type.10029"),
                tts_speaker=v2_data.get("tts_speaker", ""),
                dialog_app_id=v2_data.get("dialog_app_id", "3247272676"),
                dialog_access_key=v2_data.get("dialog_access_key", "xQQE98xLIXy05Cr8z1GaHKzmnoupvodH"),
                dialog_app_key=v2_data.get("dialog_app_key", ""),
                dialog_speaker=v2_data.get("dialog_speaker", "zh_female_xiaohe_jupiter_bigtts"),
                vision_enabled=v2_data.get("vision_enabled", True),
                vision_interval=v2_data.get("vision_interval", 1000),
                vision_images_limit=v2_data.get("vision_images_limit", 2),
            )
        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])
        if "log_level" in data:
            config.log_level = data["log_level"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduler": {
                "memory_interval": self.scheduler.memory_interval,
                "goal_interval": self.scheduler.goal_interval,
                "check_interval": self.scheduler.check_interval,
            },
            "llm": {
                "api_key": self.llm.api_key,
                "base_url": self.llm.base_url,
                "model": self.llm.model,
            },
            "dialog": {
                "app_id": self.dialog.app_id,
                "access_key": self.dialog.access_key,
                "resource_id": self.dialog.resource_id,
                "app_key": self.dialog.app_key,
                "base_url": self.dialog.base_url,
            },
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
        }


_config: Optional[Config] = None


def get_config(config_path: str = None) -> Config:
    global _config
    if _config is None:
        if config_path and os.path.exists(config_path):
            _config = Config.from_file(config_path)
        else:
            # 自动查找配置文件
            _search_paths = [
                config_path,
                "config/config.json",
                Path(__file__).parent.parent / "config" / "config.json",
            ]
            for p in _search_paths:
                if p and os.path.exists(str(p)):
                    _config = Config.from_file(str(p))
                    break
            if _config is None:
                _config = Config()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config
