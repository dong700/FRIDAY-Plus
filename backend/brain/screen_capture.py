"""屏幕采集入口（2026-11-04）

真正的采集/缓冲/融合在 brain/frame_pipeline.py：
    采集线程 → 有界缓冲(latest) → 自适应 FPS → 焦点窗 → 九宫格 → Signal

这里只保留旧的对外 API（start/latest/sheet_context），避免上层改动。
"""
from __future__ import annotations

from typing import Dict, Optional

from brain.frame_pipeline import get_pipeline, start_pipeline


def start_screen_watcher(fps: float = 1.0) -> None:
    """启动流式视觉管道。

    fps 参数保留兼容；实际帧率由自适应控制器在 1/2/4/8 之间调整。
    """
    start_pipeline()


def stop_screen_watcher() -> None:
    get_pipeline().stop()


def latest_sheet(max_age_s: float = 60.0) -> Optional[dict]:
    """最近一张九宫格；超过 max_age_s 视为过期。"""
    return get_pipeline().latest_sheet(max_age_s)


def sheet_context(max_age_s: float = 60.0) -> Dict[str, str]:
    """给上下文构建用的两个字段：图片 base64 + 一句说明。"""
    s = latest_sheet(max_age_s)
    if not s:
        return {"b64": "", "meta": ""}
    t0, t1 = s.get("window", [0, 0])
    return {
        "b64": s["b64"],
        "meta": (
            f"最近 {s.get('span', 0):.1f} 秒的 {s.get('n', 0)} 帧桌面画面"
            f"（九宫格按时间顺序排列，角标 #序号 +偏移秒；t0={t0:.0f}~t1={t1:.0f}）"
        ),
    }
