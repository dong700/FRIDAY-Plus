"""
任务执行器 (Task Executor) — 独立于大脑的执行 Agent

架构设计：
  大脑(Brain) ──submit_task()──→ 任务队列 ──→ 执行器(Executor)
       ↑ 不阻塞                        ↓ 独立后台运行
       │                         1. LLM生成脚本（独立对话上下文）
       │                         2. 写入工作区
       │                         3. 执行脚本
       │                         4. 读取输出/文件
       │                         5. 回报结果（广播 + 回调）
       └────── 收到结果后可继续下发新任务 ←──┘

关键特性：
- 与大脑完全解耦，不阻塞大脑主循环
- 有自己独立的 LLM 对话上下文和记忆
- 能读写工作区文件，理解执行历史
- 支持任务优先级、并发控制、结果缓存
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
from uuid import uuid4
import re

logger = logging.getLogger("brain.executor")


def _sanitize_output_preview(raw: str, max_length: int = 200) -> str:
    """清理输出预览：修复mojibake、解码转义、去控制字符、截断"""
    if not raw:
        return ""
    text = raw

    # 修复 mojibake：GBK字节被错误当UTF-8解码后的乱码（如 쳲���� → 斐波那契）
    # 原理：把乱码字符串用latin-1编码回原始字节，再用GBK正确解码
    try:
        raw_bytes = text.encode('utf-8')
        # 检测是否包含mojibake特征：大量0xC0-0xFF开头的UTF-8多字节序列
        # 这些通常是GBK中文字节被误读为UTF-8
        has_mojibake = False
        for b in raw_bytes:
            if 0xC0 <= b <= 0xDF or 0xE0 <= b <= 0xEF:
                has_mojibake = True
                break
        if has_mojibake:
            # 尝试：UTF-8字节 → latin-1字符串 → GBK解码
            try:
                latin1_str = raw_bytes.decode('latin-1')
                fixed = latin1_str.encode('latin-1').decode('gbk')
                # 如果GBK解码成功且结果看起来更合理（包含中文字符），使用修复后的
                if any('\u4e00' <= c <= '\u9fff' for c in fixed):
                    text = fixed
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # 解码 \UXXXXXXXX（8位大写Unicode转义）
    def _replace_u8(match):
        hex_str = match.group(1)
        cp = int(hex_str, 16)
        try:
            return chr(cp)
        except (ValueError, OverflowError):
            return match.group(0)

    text = re.sub(r'\\U([0-9a-fA-F]{8})', _replace_u8, text)

    # unicode_escape 解码处理 \uXXXX
    try:
        text = text.encode('latin-1').decode('unicode_escape')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
    text = re.sub(r"^'''(.*)'''$", r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^"""(.*)"""$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r"^['\"](.*)['\"]$", r'\1', text, flags=re.DOTALL)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    if len(text) > max_length:
        text = text[:max_length - 3] + '...'
    return text.strip()


def _sanitize_task_record(task: dict) -> dict:
    """清理任务记录的所有文本字段（description + output_preview），返回新dict不修改原数据"""
    if not task:
        return task
    cleaned = dict(task)
    if cleaned.get("description"):
        cleaned["description"] = _sanitize_output_preview(cleaned["description"])
    if cleaned.get("output_preview"):
        cleaned["output_preview"] = _sanitize_output_preview(cleaned["output_preview"])
    return cleaned


# 工作区路径（与 workspace.py 共享，根目录由 core.config 决定）
from core.config import get_workspace_dir as _get_workspace_dir
WORKSPACE_ROOT = _get_workspace_dir()
EXECUTOR_STATE_FILE = WORKSPACE_ROOT / ".executor_state.json"
EXECUTOR_LOG_FILE = WORKSPACE_ROOT / ".executor_log.json"


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class TaskStatus(Enum):
    PENDING = "pending"        # 在队列中等待
    GENERATING = "generating"  # LLM 正在生成脚本
    EXECUTING = "executing"    # 脚本正在运行
    COMPLETED = "completed"    # 完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消


@dataclass
class Task:
    """待执行的任务"""
    id: str
    description: str           # 任务描述（大脑下发的命令）
    context: str = ""          # 额外上下文（对话片段等）
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    result: Optional[Dict[str, Any]] = None
    error: str = ""
    script_name: str = ""      # 生成的脚本名
    script_code: str = ""      # 生成的脚本代码
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    retries: int = 0           # 重试次数


@dataclass
class ExecutionMemory:
    """执行器的持久化记忆"""
    recent_tasks: List[Dict[str, Any]] = field(default_factory=list)  # 最近20个任务记录
    workspace_files_known: List[str] = field(default_factory=list)     # 已知的工作区文件
    execution_patterns: Dict[str, str] = field(default_factory=dict)  # 学到的执行模式
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    last_active_at: str = ""


class TaskExecutor:
    """
    独立任务执行器 Agent

    特点：
    1. 异步队列驱动 — 大脑提交任务后立即返回，不阻塞
    2. 独立 LLM 对话 — 有自己的 system prompt 和上下文记忆
    3. 文件感知 — 能读取工作区的已有文件，基于历史优化脚本生成
    4. 结果回报 — 通过回调函数通知大脑和前端
    """

    def __init__(self,
                 max_concurrent: int = 1,      # 同时执行的任务数
                 max_queue_size: int = 50,      # 队列上限
                 on_task_completed: Optional[Callable[[Task], Coroutine]] = None,
                 on_task_failed: Optional[Callable[[Task], Coroutine]] = None):
        self._queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=max_queue_size)
        self._max_concurrent = max_concurrent
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # 回调：任务完成/失败时调用（用于广播 WS 事件等）
        self._on_completed = on_task_completed
        self._on_failed = on_task_failed

        # 执行器自身的状态和记忆
        self._memory: ExecutionMemory = ExecutionMemory()
        self._load_memory()

        # 当前正在处理的任务
        self._active_tasks: Dict[str, Task] = {}
        self._task_history: List[Dict] = []  # 所有已完成任务的历史

        logger.info(f"TaskExecutor initialized (concurrency={max_concurrent}, "
                   f"queue_size={max_queue_size}, memory_loaded={bool(self._memory.last_active_at)})")

    def _load_memory(self):
        """加载执行器记忆"""
        if EXECUTOR_STATE_FILE.exists():
            try:
                with open(EXECUTOR_STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._memory = ExecutionMemory(
                    recent_tasks=data.get("recent_tasks", [])[-20:],
                    workspace_files_known=data.get("workspace_files_known", []),
                    execution_patterns=data.get("execution_patterns", {}),
                    total_tasks_completed=data.get("total_tasks_completed", 0),
                    total_tasks_failed=data.get("total_tasks_failed", 0),
                    last_active_at=data.get("last_active_at", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to load executor memory: {e}")

    def _save_memory(self):
        """保存执行器记忆"""
        try:
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            with open(EXECUTOR_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "recent_tasks": self._memory.recent_tasks[-20:],
                    "workspace_files_known": self._memory.workspace_files_known,
                    "execution_patterns": self._memory.execution_patterns,
                    "total_tasks_completed": self._memory.total_tasks_completed,
                    "total_tasks_failed": self._memory.total_tasks_failed,
                    "last_active_at": self._memory.last_active_at,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save executor memory: {e}")

    async def start(self):
        """启动执行器后台循环"""
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._background_task = asyncio.create_task(self._process_loop())
        logger.info("TaskExecutor background loop started")

    async def stop(self):
        """停止执行器"""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskExecutor stopped")

    def submit(self, description: str, context: str = "",
               priority: TaskPriority = TaskPriority.NORMAL) -> Optional[Task]:
        """
        提交任务到队列（大脑调用此方法，立即返回）

        Args:
            description: 任务描述（大脑的命令）
            context: 额外上下文
            priority: 优先级

        Returns: Task 对象（如果队列未满），None（队列满）
        """
        task = Task(
            id=str(uuid4())[:8],
            description=description,
            context=context,
            priority=priority,
        )
        try:
            self._queue.put_nowait(task)
            logger.info(f"[Executor] Task submitted: {task.id} | {description[:60]} | priority={priority.name}")
            return task
        except asyncio.QueueFull:
            logger.warning(f"[Executor] Queue full! Task rejected: {description[:40]}")
            return None

    async def _process_loop(self):
        """后台处理循环 — 从队列取任务 → 生成脚本 → 执行 → 回报"""
        while self._running:
            try:
                # 从队列获取任务（带超时，方便定期检查 running 状态）
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # 并发控制
                async with self._semaphore:
                    await self._execute_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Executor] Process loop error: {e}", exc_info=True)

    async def _execute_task(self, task: Task):
        """完整执行一个任务：LLM生成 → 写入 → 运行 → 读取结果"""
        task.status = TaskStatus.GENERATING
        task.started_at = datetime.now().isoformat()
        self._active_tasks[task.id] = task

        start_time = datetime.now()
        logger.info(f"[Executor:{task.id}] Starting: {task.description[:60]}")

        try:
            # === Step 1: LLM 生成脚本（执行器自己的对话） ===
            script_code = await self._generate_script(task)

            if not script_code:
                raise RuntimeError("LLM 生成的脚本为空")

            task.script_code = script_code
            task.status = TaskStatus.EXECUTING

            # === Step 2: 写入工作区 ===
            from brain.workspace import get_ai_workspace, ScriptType
            ws = get_ai_workspace()
            safe_name = f"task_{datetime.now().strftime('%H%M%S')}_{task.id}.py"
            write_result = await ws.write_script(safe_name, script_code, script_type=ScriptType.PYTHON)

            if not write_result.get("success"):
                raise RuntimeError(f"写入失败: {write_result.get('error')}")

            task.script_name = safe_name

            # 广播：正在执行
            await self._broadcast_event("executor.executing", {
                "task_id": task.id,
                "script_name": safe_name,
                "description": task.description[:100],
                "code_preview": script_code[:300],
            })

            # === Step 3: 执行脚本 ===
            exec_result = await ws.execute_script(safe_name, timeout=60)

            # === Step 4: 读取工作区产生的文件（如果有新的数据文件） ===
            new_files = self._scan_new_workspace_files()

            # 组装结果
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            task.stdout = exec_result.stdout
            task.stderr = exec_result.stderr
            task.duration_ms = exec_result.duration_ms or int(elapsed)
            task.result = {
                "success": exec_result.success,
                "exit_code": exec_result.exit_code,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "duration_ms": task.duration_ms,
                "script_name": safe_name,
                "new_files": new_files,
            }
            task.status = TaskStatus.COMPLETED if exec_result.success else TaskStatus.FAILED
            task.completed_at = datetime.now().isoformat()

            # 更新记忆
            self._update_memory(task)

            logger.info(f"[Executor:{task.id}] {'OK' if task.status == TaskStatus.COMPLETED else 'FAIL'} "
                       f"| {safe_name} | {task.duration_ms}ms | stdout={len(exec_result.stdout)}b")

            # 回调通知
            if task.status == TaskStatus.COMPLETED and self._on_completed:
                try:
                    await self._on_completed(task)
                except Exception as cb_err:
                    logger.error(f"[Executor] on_completed callback error: {cb_err}")
            elif task.status == TaskStatus.FAILED and self._on_failed:
                try:
                    await self._on_failed(task)
                except Exception as cb_err:
                    logger.error(f"[Executor] on_failed callback error: {cb_err}")

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.duration_ms = int(elapsed)
            task.completed_at = datetime.now().isoformat()
            self._memory.total_tasks_failed += 1
            self._save_memory()

            logger.error(f"[Executor:{task.id}] FAILED: {e}")

            if self._on_failed:
                try:
                    await self._on_failed(task)
                except Exception as cb_err:
                    logger.error(f"[Executor] on_failed callback error: {cb_err}")

        finally:
            self._active_tasks.pop(task.id, None)

    async def _generate_script(self, task: Task) -> str:
        """
        用 LLM 生成脚本代码（执行器的独立对话，不是大脑的分析对话）

        关键区别：
        - 大脑分析 prompt：判断"要不要做"
        - 执行器 prompt：决定"怎么做"并生成代码
        - 执行器有工作区文件列表作为上下文
        - 执行器有自己的执行模式记忆
        """
        from brain.llm import get_llm
        llm = get_llm()

        # 构建执行器上下文（包含已知的工作区文件和历史模式）
        workspace_context = self._build_workspace_context()

        # 检查是否有类似的执行模式可以复用
        pattern_hint = ""
        for pattern_key, pattern_code in self._memory.execution_patterns.items():
            if pattern_key in task.description.lower() or any(kw in task.description.lower() for kw in pattern_key.split(",")):
                pattern_hint = f"\n\n## 参考模式（之前成功的类似任务）\n{pattern_code}\n"
                break

        prompt = f"""你是 Active AI Pro 的任务执行器。你的职责是接收大脑下达的任务，编写 Python 脚本来完成它。

## 当前时间
{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 任务描述
{task.description}

## 上下文信息
{task.context if task.context else '无额外上下文'}

## 工作区现状
{workspace_context}{pattern_hint}

## 要求
1. 只输出 Python 代码，不要解释
2. 用 print() 输出最终结果（这是唯一获取输出的方式）
3. 不要交互式输入（不要 input()）
4. 不要 GUI（不要 matplotlib show / tkinter 等）
5. 如果需要网络请求用 urllib.request，不需要额外安装包
6. 脚本要简洁高效，直接解决问题
7. 最后必须 print 一个总结性的结果
8. 可以读写当前目录下的文件（工作区路径已设为 cwd）
9. 输出格式清晰，便于大脑读取和理解结果

## 严格禁止
1. 不要编写任何硬件控制代码（机械臂、摄像头、传感器等）——这不是你的职责
2. 不要 print 虚假的预期结果——代码必须真正执行并输出真实结果
3. 如果代码无法完成任务，print 明确的错误信息，不要假装成功"""

        response = await llm.chat([
            {"role": "system", "content":
             "你是任务执行器Agent。只输出可执行的Python代码，不要任何解释文字。"
             "你的代码会在隔离的工作区环境中运行。"
             "绝对不要编写硬件控制代码，不要输出虚假的预期结果。"},
            {"role": "user", "content": prompt}
        ])

        # 清理 markdown 包裹
        code = response.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    def _build_workspace_context(self) -> str:
        """构建工作区文件上下文，让 LLM 知道有什么现成资源可用"""
        scripts_dir = WORKSPACE_ROOT / "scripts"
        data_dir = WORKSPACE_ROOT / "data"

        lines = []
        files_in_scripts = list(scripts_dir.glob("*")) if scripts_dir.exists() else []
        files_in_data = list(data_dir.glob("*")) if data_dir.exists() else []

        if files_in_scripts:
            lines.append(f"### 已有脚本 ({len(files_in_scripts)} 个):")
            for f in sorted(files_in_scripts)[:15]:
                size = f.stat().st_size
                lines.append(f"  - {f.name} ({size} bytes)")

        if files_in_data:
            lines.append(f"\n### 数据文件 ({len(files_in_data)} 个):")
            for f in sorted(files_in_data)[:10]:
                size = f.stat().st_size
                lines.append(f"  - {f.name} ({size} bytes)")

        if self._memory.recent_tasks:
            lines.append(f"\n### 最近完成的任务:")
            for t in self._memory.recent_tasks[-5:]:
                status_icon = "✓" if t.get("success") else "✗"
                lines.append(f"  [{status_icon}] {t.get('description','?')[:50]} ({t.get('duration_ms',0)}ms)")

        if not lines:
            return "工作区为空（首次执行）"

        return "\n".join(lines)

    def _scan_new_workspace_files(self) -> List[str]:
        """扫描工作区中新生成的数据文件"""
        data_dir = WORKSPACE_ROOT / "data"
        known = set(self._memory.workspace_files_known)
        new_files = []
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.is_file() and f.name not in known:
                    new_files.append(f.name)
                    self._memory.workspace_files_known.append(f.name)
        return new_files

    def _update_memory(self, task: Task):
        """更新执行器记忆"""
        is_success = task.status == TaskStatus.COMPLETED
        if is_success:
            self._memory.total_tasks_completed += 1
        else:
            self._memory.total_tasks_failed += 1
        self._memory.last_active_at = datetime.now().isoformat()

        # 记录到历史
        record = {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "success": is_success,
            "script_name": task.script_name,
            "duration_ms": task.duration_ms,
            "output_preview": _sanitize_output_preview(task.stdout or task.stderr),
            "completed_at": task.completed_at,
        }
        self._memory.recent_tasks.append(record)
        self._memory.recent_tasks = self._memory.recent_tasks[-30:]  # 保留最近30条

        # 学习执行模式（如果成功且脚本有意义）
        if is_success and len(task.script_code) > 20:
            keywords = self._extract_keywords(task.description)
            if keywords:
                self._memory.execution_patterns[keywords] = task.script_code[:500]

        self._save_memory()

    @staticmethod
    def _extract_keywords(description: str) -> str:
        """从任务描述提取关键词作为模式 key"""
        words = [w for w in description.lower().split() if len(w) >= 2][:5]
        return ",".join(words) if words else ""

    async def _broadcast_event(self, event_type: str, payload: dict):
        """广播事件（通过回调或直接引用 manager）"""
        # 延迟导入避免循环依赖
        try:
            from api.server import manager
            await manager.broadcast({"type": event_type, "payload": payload})
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Broadcast failed: {e}")

    # ========== 查询 API ==========

    def get_status(self) -> dict:
        """获取执行器当前状态"""
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "active_count": len(self._active_tasks),
            "active_tasks": [
                {
                    "id": t.id,
                    "description": _sanitize_output_preview(t.description[:80]),
                    "status": t.status.value, "started_at": t.started_at,
                } for t in self._active_tasks.values()
            ],
            "memory": {
                "total_completed": self._memory.total_tasks_completed,
                "total_failed": self._memory.total_tasks_failed,
                "last_active": self._memory.last_active_at,
                "known_files": len(self._memory.workspace_files_known),
                "patterns_learned": len(self._memory.execution_patterns),
            },
            "recent_tasks": [_sanitize_task_record(t) for t in self._memory.recent_tasks[-10:]],
        }

    def get_task_history(self, limit: int = 20) -> List[dict]:
        """获取任务历史（清理所有文本字段中的乱码）"""
        tasks = self._memory.recent_tasks[-limit:]
        return [_sanitize_task_record(t) for t in tasks]

    def cancel_task(self, task_id: str) -> bool:
        """取消队列中的任务（只能取消还没开始执行的）"""
        # TODO: 实现队列内任务的取消
        return False

    def delete_task(self, task_id: str) -> bool:
        """删除任务记录（从历史中移除 + 删除关联脚本文件 + 更新统计）"""
        # 先找到关联的脚本名和状态
        script_name = None
        was_success = None
        new_tasks = []
        for t in self._memory.recent_tasks:
            if t.get("id") == task_id:
                script_name = t.get("script_name")
                was_success = t.get("success")
            else:
                new_tasks.append(t)

        if len(new_tasks) == len(self._memory.recent_tasks) and not script_name:
            return False  # 没找到该任务

        self._memory.recent_tasks = new_tasks

        # 更新统计计数
        if was_success is True:
            self._memory.total_tasks_completed = max(0, self._memory.total_tasks_completed - 1)
        elif was_success is False:
            self._memory.total_tasks_failed = max(0, self._memory.total_tasks_failed - 1)

        # 删除关联的脚本文件
        if script_name:
            script_path = WORKSPACE_ROOT / "scripts" / script_name
            if script_path.exists():
                try:
                    script_path.unlink()
                    logger.info(f"[Executor] Deleted script: {script_name}")
                except Exception as e:
                    logger.warning(f"[Executor] Failed to delete script {script_name}: {e}")

        self._save_memory()
        logger.info(f"[Executor] Task {task_id} deleted from history")
        return True


# ============================================================
# 全局单例
# ============================================================
_executor_instance: Optional[TaskExecutor] = None


def get_task_executor() -> TaskExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = TaskExecutor(max_concurrent=1)
    return _executor_instance
