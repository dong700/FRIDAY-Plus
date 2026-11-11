"""
AI 工作区 - 大脑的脚本执行引擎

大脑（LLM）可以：
1. 在工作区创建脚本文件（Python / Bash / PowerShell）
2. 执行脚本并获取输出
3. 查看工作区文件列表

安全限制：
- 工作区隔离在 D:/ActiveAI_Data/workspace/
- 执行超时 60 秒
- 禁止危险操作（rm -rf /, format 等）
- 输出大小限制 1MB
"""

import asyncio
import os
import json
import re
import subprocess
import sys
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("brain.workspace")

# AI 工作区根目录（由 core.config 统一决定，默认 D:/FRIDAY/data/workspace）
from core.config import get_workspace_dir as _get_workspace_dir
WORKSPACE_ROOT = _get_workspace_dir()

# 安全限制
MAX_OUTPUT_SIZE = 1024 * 1024  # 1MB
EXECUTION_TIMEOUT = 60  # 秒
MAX_SCRIPT_SIZE = 50 * 1024  # 脚本最大 50KB

# 禁止的危险模式（简单黑名单）
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "format ",
    "mkfs",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    ":(){:|:&};:",   # fork bomb
    "dev/null",
    "> /etc/",
    "chmod 777 /",
    "curl.*|.*sh",    # curl pipe to shell
    "wget.*|.*sh",    # wget pipe to shell
]


def _sanitize_output_preview(raw: str, max_length: int = 200) -> str:
    """
    清理脚本输出用于预览显示：
    0. 修复 mojibake（GBK字节被错误当UTF-8解码）
    1. 解码 \\uXXXX 和 \\UXXXXXXXX 转义序列（Python repr 产物）
    2. 去除 Python 三引号/引号包裹
    3. 去除控制字符（保留换行）
    4. 截断到指定长度
    """
    if not raw:
        return ""
    text = raw

    # 0. 修复 mojibake：GBK字节被错误当UTF-8解码后的乱码
    try:
        raw_bytes = text.encode('utf-8')
        has_mojibake = any(0xC0 <= b <= 0xEF for b in raw_bytes)
        if has_mojibake:
            try:
                latin1_str = raw_bytes.decode('latin-1')
                fixed = latin1_str.encode('latin-1').decode('gbk')
                if any('\u4e00' <= c <= '\u9fff' for c in fixed):
                    text = fixed
            except (UnicodeDecodeError, UnicodeEncodeError, LookupError):
                pass
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # 1. 解码 \UXXXXXXXX（8位大写Unicode转义）
    def _replace_u8(match):
        hex_str = match.group(1)
        cp = int(hex_str, 16)
        try:
            return chr(cp)
        except (ValueError, OverflowError):
            return match.group(0)

    text = re.sub(r'\\U([0-9a-fA-F]{8})', _replace_u8, text)

    # unicode_escape 解码（处理 \uXXXX 等Python repr格式转义）
    try:
        text = text.encode('latin-1').decode('unicode_escape')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # 清理残余的 \xXX 转义
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)

    # 去除 Python repr 的三引号包裹
    text = re.sub(r"^'''(.*)'''$", r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^"""(.*)"""$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r"^['\"](.*)['\"]$", r'\1', text, flags=re.DOTALL)

    # 去除控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 清理多余空白行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 截断
    if len(text) > max_length:
        text = text[:max_length - 3] + '...'

    return text.strip()


def _smart_decode(raw_bytes: bytes) -> str:
    """
    智能解码子进程输出：
    1. 尝试 UTF-8（标准编码）
    2. 回退 GBK/GB2312（中文Windows系统默认）
    3. 兜底 Latin-1（不会丢失字节）
    """
    if not raw_bytes:
        return ""

    # 1. UTF-8
    try:
        return raw_bytes.decode('utf-8')
    except (UnicodeDecodeError, ValueError):
        pass

    # 2. GBK (中文Windows常见)
    for enc in ('gbk', 'gb2312', 'gb18030'):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, ValueError, LookupError):
            pass

    # 3. Latin-1 兜底（1:1映射，永不报错）
    return raw_bytes.decode('latin-1')


class ScriptType(Enum):
    PYTHON = "python"
    BASH = "bash"
    POWERSHELL = "powershell"
    AUTO = "auto"  # 自动检测


@dataclass
class ScriptFile:
    """工作区中的脚本文件"""
    name: str
    path: str
    content: str
    script_type: ScriptType
    created_at: str
    size: int


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    script_name: str
    executed_at: str
    error: str = ""


@dataclass
class WorkspaceState:
    """工作区状态"""
    files: List[Dict[str, Any]] = field(default_factory=list)
    recent_executions: List[Dict[str, Any]] = field(default_factory=list)
    total_executions: int = 0


class AIWorkspace:
    """
    AI 工作区 - 大脑的脚本沙箱

    提供安全的脚本编写和执行环境，让大脑能通过写代码完成任务
    """

    def __init__(self, root_dir: Path = None):
        self.root = root_dir or WORKSPACE_ROOT
        self._state_file = self.root / ".workspace_state.json"
        self._state: Optional[WorkspaceState] = None
        self._ensure_workspace()

    def _ensure_workspace(self):
        """确保工作区目录存在"""
        self.root.mkdir(parents=True, exist_ok=True)
        # 创建子目录
        (self.root / "scripts").mkdir(exist_ok=True)
        (self.root / "output").mkdir(exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        logger.info(f"AI Workspace ready at {self.root}")

    def _load_state(self) -> WorkspaceState:
        if self._state is not None:
            return self._state
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._state = WorkspaceState(
                    files=data.get("files", []),
                    recent_executions=data.get("recent_executions", [])[-20:],
                    total_executions=data.get("total_executions", 0),
                )
                return self._state
            except Exception as e:
                logger.warning(f"Failed to load workspace state: {e}")
        self._state = WorkspaceState()
        return self._state

    def _save_state(self):
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "files": self._state.files,
                    "recent_executions": self._state.recent_executions[-20:],
                    "total_executions": self._state.total_executions,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save workspace state: {e}")

    def detect_script_type(self, content: str, filename: str = "") -> ScriptType:
        """自动检测脚本类型"""
        # 先看文件扩展名
        if filename.endswith(".py"):
            return ScriptType.PYTHON
        elif filename.endswith(".sh") or filename.endswith(".bash"):
            return ScriptType.BASH
        elif filename.endswith(".ps1"):
            return ScriptType.POWERSHELL

        # 再看内容特征
        content_stripped = content.strip()
        if content_stripped.startswith("#!/usr/bin/python") or content_stripped.startswith("python"):
            return ScriptType.PYTHON
        if content_stripped.startswith("#!/bin/bash") or content_stripped.startswith("#!/bin/sh"):
            return ScriptType.BASH
        if content_stripped.startswith("# powershell") or "Write-" in content[:200]:
            return ScriptType.POWERSHELL

        # 默认 Python
        return ScriptType.PYTHON

    def _check_safety(self, content: str) -> tuple[bool, str]:
        """检查脚本安全性"""
        content_lower = content.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in content_lower:
                return False, f"安全限制：检测到危险操作 '{pattern}'"
        if len(content) > MAX_SCRIPT_SIZE:
            return False, f"脚本过大：{len(content)} > {MAX_SCRIPT_SIZE} 字节"
        return True, ""

    async def write_script(self, name: str, content: str,
                           script_type: ScriptType = ScriptType.AUTO) -> Dict[str, Any]:
        """
        在工作区创建脚本文件

        Returns: {success, file_path, script_type, name}
        """
        safe, reason = self._check_safety(content)
        if not safe:
            return {"success": False, "error": reason}

        # 清理文件名（防路径穿越）
        safe_name = Path(name).name
        if not safe_name:
            safe_name = f"script_{datetime.now().strftime('%H%M%S')}"

        detected_type = self.detect_script_type(content, name)
        final_type = script_type if script_type != ScriptType.AUTO else detected_type

        # 根据类型加扩展名
        ext_map = {
            ScriptType.PYTHON: ".py",
            ScriptType.BASH: ".sh",
            ScriptType.POWERSHELL: ".ps1",
        }
        if not any(safe_name.endswith(e) for e in ext_map.values()):
            safe_name += ext_map.get(final_type, ".py")

        file_path = self.root / "scripts" / safe_name

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            state = self._load_state()

            file_info = {
                "name": safe_name,
                "path": str(file_path),
                "script_type": final_type.value,
                "size": len(content),
                "created_at": datetime.now().isoformat(),
            }

            # 更新文件列表（去重）
            state.files = [f for f in state.files if f["name"] != safe_name]
            state.files.append(file_info)
            state.files = state.files[-50:]  # 最多保留 50 个文件记录
            self._save_state()

            logger.info(f"Script written: {safe_name} ({final_type.value}, {len(content)} bytes)")
            return {"success": True, **file_info}

        except Exception as e:
            logger.error(f"Failed to write script: {e}")
            return {"success": False, "error": str(e)}

    async def execute_script(self, name_or_content: str,
                              is_content: bool = False,
                              timeout: int = EXECUTION_TIMEOUT) -> ExecutionResult:
        """
        执行脚本

        Args:
            name_or_content: 文件名或脚本内容
            is_content: 如果为 True，直接执行内容（不保存文件）
            timeout: 超时时间（秒）

        Returns: ExecutionResult
        """
        start_time = datetime.now()

        if is_content:
            # 直接执行内容（临时文件）
            safe, reason = self._check_safety(name_or_content)
            if not safe:
                return ExecutionResult(
                    success=False, exit_code=-1, stdout="", stderr=reason,
                    duration_ms=0, script_name="<inline>", executed_at=datetime.now().isoformat(),
                    error=reason
                )

            script_type = self.detect_script_type(name_or_content)
            ext = ".py" if script_type == ScriptType.PYTHON else (
                ".sh" if script_type == ScriptType.BASH else ".ps1")

            tmp_file = self.root / "scripts" / f"_tmp_{datetime.now().strftime('%H%M%S%f')}{ext}"
            try:
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    f.write(name_or_content)
                script_name = "<inline>"
                script_path = tmp_file
            except Exception as e:
                return ExecutionResult(
                    success=False, exit_code=-1, stdout="", stderr=str(e),
                    duration_ms=0, script_name="<inline>", executed_at=datetime.now().isoformat(),
                    error=str(e)
                )
        else:
            # 执行已有文件
            safe_name = Path(name_or_content).name
            script_path = self.root / "scripts" / safe_name
            if not script_path.exists():
                return ExecutionResult(
                    success=False, exit_code=-1, stdout="",
                    stderr=f"脚本不存在: {safe_name}",
                    duration_ms=0, script_name=safe_name,
                    executed_at=datetime.now().isoformat(),
                    error="file_not_found"
                )
            script_type = self.detect_script_type("", safe_name)
            script_name = safe_name

        # 构建执行命令
        if script_type == ScriptType.PYTHON:
            cmd = ["python", "-u", str(script_path)]  # -u 无缓冲输出
            cwd = str(self.root)
        elif script_type == ScriptType.BASH:
            cmd = ["bash", str(script_path)]
            cwd = str(self.root)
        elif script_type == ScriptType.POWERSHELL:
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
            cwd = str(self.root)
        else:
            cmd = ["python", "-u", str(script_path)]
            cwd = str(self.root)

        try:
            # 使用 asyncio 子进程实现异步执行 + 超时
            # Windows 需要 CREATE_NO_WINDOW 防止子进程输出丢失到隐藏控制台
            create_flags = 0
            if sys.platform == 'win32':
                create_flags = subprocess.CREATE_NO_WINDOW

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                creationflags=create_flags,
            )

            # DEBUG: 详细记录执行过程（排查"无输出"问题）
            logger.info(f"[execute] START script={script_name} cmd={cmd} cwd={cwd}")
            logger.info(f"[execute] process pid={process.pid} returncode={process.returncode}")

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                return ExecutionResult(
                    success=False, exit_code=-1, stdout="", stderr=f"执行超时 ({timeout}s)",
                    duration_ms=int(elapsed), script_name=script_name,
                    executed_at=datetime.now().isoformat(), error="timeout"
                )

            logger.info(f"[execute] DONE script={script_name} pid={process.pid} "
                        f"returncode={process.returncode} "
                        f"stdout_bytes={len(stdout_bytes)}b stderr_bytes={len(stderr_bytes)}b")

            # 智能解码：UTF-8优先 → GBK(中文Windows) → Latin-1兜底
            try:
                stdout = _smart_decode(stdout_bytes)
                stderr = _smart_decode(stderr_bytes)
            except Exception as decode_err:
                logger.warning(f"[execute] _smart_decode failed: {decode_err}")
                stdout = stdout_bytes.decode('latin-1', errors='replace')
                stderr = stderr_bytes.decode('latin-1', errors='replace')

            # DEBUG: 记录原始输出（排查"无输出"问题）
            logger.info(f"[execute] DECODED script={script_name} "
                        f"stdout_len={len(stdout)} stderr_len={len(stderr)} "
                        f"stdout_preview={repr(stdout[:100])}")

            # 截断过大的输出
            if len(stdout) > MAX_OUTPUT_SIZE:
                stdout = stdout[:MAX_OUTPUT_SIZE] + f"\n... (输出截断，共 {len(stdout)} 字节)"
            if len(stderr) > MAX_OUTPUT_SIZE:
                stderr = stderr[:MAX_OUTPUT_SIZE] + f"\n... (错误截断，共 {len(stderr)} 字节)"

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            result = ExecutionResult(
                success=(process.returncode == 0),
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=int(elapsed),
                script_name=script_name,
                executed_at=datetime.now().isoformat(),
            )

            # 记录到状态
            state = self._load_state()
            state.total_executions += 1
            try:
                preview = _sanitize_output_preview(result.stdout or result.stderr)
            except Exception as preview_err:
                logger.warning(f"[execute] _sanitize_output_preview failed: {preview_err}")
                preview = (result.stdout or result.stderr)[:200]
            exec_record = {
                "script_name": script_name,
                "success": result.success,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "output_preview": preview,
                "executed_at": result.executed_at,
            }
            state.recent_executions.append(exec_record)
            state.recent_executions = state.recent_executions[-20:]
            self._save_state()

            logger.info(f"Script executed: {script_name} -> exit={result.exit_code} "
                       f"time={result.duration_ms}ms stdout={len(result.stdout)}b")
            return result

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            return ExecutionResult(
                success=False, exit_code=-1, stdout="", stderr=str(e),
                duration_ms=int(elapsed), script_name=script_name,
                executed_at=datetime.now().isoformat(), error=str(e)
            )

    def list_files(self) -> List[Dict[str, Any]]:
        """列出工作区文件"""
        state = self._load_state()
        return state.files

    def list_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """列出最近的执行记录"""
        state = self._load_state()
        return state.recent_executions[-limit:]

    def read_script(self, name: str) -> Dict[str, Any]:
        """读取脚本内容"""
        safe_name = Path(name).name
        path = self.root / "scripts" / safe_name
        if path.exists():
            content = path.read_text(encoding='utf-8')
            return {"success": True, "name": safe_name, "content": content, "size": len(content)}
        return {"success": False, "error": f"文件不存在: {safe_name}"}

    def delete_script(self, name: str) -> Dict[str, Any]:
        """删除脚本"""
        safe_name = Path(name).name
        path = self.root / "scripts" / safe_name
        if path.exists():
            path.unlink()
            state = self._load_state()
            state.files = [f for f in state.files if f["name"] != safe_name]
            self._save_state()
            return {"success": True, "deleted": safe_name}
        return {"success": False, "error": f"文件不存在: {safe_name}"}

    def get_workspace_info(self) -> Dict[str, Any]:
        """获取工作区信息"""
        state = self._load_state()
        script_count = len(list((self.root / "scripts").glob("*")))
        total_size = sum(f.stat().st_size for f in (self.root / "scripts").glob("*") if f.is_file())
        return {
            "root": str(self.root),
            "script_count": script_count,
            "total_size_bytes": total_size,
            "total_executions": state.total_executions,
            "recent_executions": state.recent_executions[-5:],
        }


# 单例
_workspace_instance: Optional[AIWorkspace] = None


def get_ai_workspace() -> AIWorkspace:
    global _workspace_instance
    if _workspace_instance is None:
        _workspace_instance = AIWorkspace()
    return _workspace_instance
