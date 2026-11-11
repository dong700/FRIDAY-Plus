"""桌面活动感知（2026-11-03）

两条输入，按成本从低到高：
    1. 前台窗口标题/进程名（Win32，免费、毫秒级）—— 大多数情况已经够用
    2. 看不出来时，才把最近的变化拼图交给主 LLM 看图（deepseek-flash 支持图片）

产出统一写进时间轴 / ATG：
    - 每条 `screen.activity`（不唤醒）
    - 连续 N 次判定在看视频 → 一次 `video.watch` 唤醒心智；之后只刷新事件
    - 超过 M 秒没再看到视频 → `video.stop`
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

from core.logger import get_logger

logger = get_logger("brain.screen_vision")

CFG = {
    "interval": float(os.getenv("FRIDAY_ACTIVITY_INTERVAL", "45")),        # 窗口探测间隔
    "vision_interval": float(os.getenv("FRIDAY_VISION_INTERVAL", "90")),   # 需要看图时的最短间隔
    "vision_idle_interval": float(os.getenv("FRIDAY_VISION_IDLE", "240")), # 画面没变化时
    "video_interval": float(os.getenv("FRIDAY_VIDEO_INTERVAL", "90")),    # 看视频期间刷新
    "video_streak": int(os.getenv("FRIDAY_VIDEO_STREAK", "2")),
    "video_stop_after": float(os.getenv("FRIDAY_VIDEO_STOP", "120")),
    "poll": 5.0,
    "screen_fps": float(os.getenv("FRIDAY_SCREEN_FPS", "3")),
}

LABELS = {
    "video": "正在看视频",
    "coding": "正在写代码",
    "game": "正在玩游戏",
    "chat": "正在聊天",
    "browse": "正在浏览网页",
    "idle": "暂时没在操作",
    "other": "在使用电脑",
}


class ScreenVision:
    def __init__(self) -> None:
        self._last_window = 0.0
        self._last_vision = 0.0
        self._fail_streak = 0
        self._cooldown_until = 0.0
        self._last_activity = ""
        self._video_streak = 0
        self._in_video = False
        self._last_video_seen = 0.0
        self._stats = {"window": 0, "vision": 0, "fail": 0}

    # ---------- 输入 ----------

    @staticmethod
    def _window_info() -> Dict[str, Any]:
        try:
            from brain.win_activity import sense
            return sense()
        except Exception:
            return {"activity": "other", "title": "", "process": "", "confident": False}

    # ---------- 输出 ----------

    def _apply(self, info: Dict[str, Any]) -> None:
        from brain.mind import get_mind

        act = info["activity"]
        title = info.get("title", "")
        detail = info.get("detail", "")
        # 只把「事实」交给心智：前台窗口标题/进程。是不是在看视频、在干什么，
        # 由同一个模型结合画面自己判断，不再由这个模块先行贴标签——
        # 之前它把控制台误标成「正在看视频」并写进 ATG，模型和页面因此对不上。
        text = f"前台窗口：{title}" if title else "前台窗口未知"
        if detail:
            text += f"（{detail}）"

        self._last_activity = act
        try:
            get_mind().publish("screen.activity", text, source="screen")
        except Exception:
            pass

        now = time.time()
        if act == "video":
            self._video_streak += 1
            self._last_video_seen = now
            if self._video_streak >= CFG["video_streak"]:
                first = not self._in_video
                self._in_video = True
                try:
                    get_mind().publish("video.watch", text, source="screen", wake=first)
                except Exception:
                    pass
                if first:
                    logger.info(f"[ActivityWatch] 进入看视频状态: {text}")
        else:
            self._video_streak = 0

        if self._in_video and now - self._last_video_seen > CFG["video_stop_after"]:
            self._in_video = False
            logger.info("[ActivityWatch] 离开看视频状态")
            try:
                get_mind().publish("video.stop", "用户不在看视频了", source="screen")
            except Exception:
                pass

    def _wait(self) -> float:
        if self._in_video:
            return CFG["video_interval"]
        if self._last_activity in ("idle", "other"):
            return max(CFG["interval"], CFG["vision_interval"])
        return CFG["interval"]

    async def observe_once(self) -> Optional[Dict[str, Any]]:
        win = self._window_info()
        activity = str(win.get("activity") or "other")
        if activity not in LABELS:
            activity = "other"
        if win.get("confident") and activity != "other":
            info = {
                "activity": activity,
                "title": str(win.get("title") or "")[:40],
                "detail": str(win.get("process") or "")[:20],
            }
        else:
            # 不再单独叫视觉模型判断。画面会和文字一起送进同一个模型，
            # 这里只保留不花钱的窗口标题/进程作为事实信号。
            info = {
                "activity": activity,
                "title": str(win.get("title") or "")[:40],
                "detail": str(win.get("process") or "")[:20],
            }

        self._apply(info)
        return info

    async def run(self) -> None:
        logger.info(
            f"[ActivityWatch] 启动 window={CFG['interval']}s vision={CFG['vision_interval']}s "
            f"video={CFG['video_interval']}s streak={CFG['video_streak']}"
        )
        while True:
            try:
                await asyncio.sleep(CFG["poll"])
                now = time.time()
                if now - self._last_window < self._wait():
                    continue
                self._last_window = now
                self._stats["window"] += 1
                await self.observe_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[ActivityWatch] 循环异常")
                await asyncio.sleep(2.0)


_vision: Optional[ScreenVision] = None
_task: Optional[asyncio.Task] = None


def get_screen_vision() -> ScreenVision:
    global _vision
    if _vision is None:
        _vision = ScreenVision()
    return _vision


def start_screen_vision() -> None:
    """在事件循环里启动后台任务（重复调用安全）。"""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(get_screen_vision().run())


def stop_screen_vision() -> None:
    """桌宠菜单里「关视觉」：取消后台轮询，不再读窗口标题、不再看图。"""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None


def screen_vision_running() -> bool:
    return _task is not None and not _task.done()
