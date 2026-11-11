"""变化筛帧 + 多帧拼图采样器 (2026-11-03)

灵感来源：SAOS-3624/dsh-see-now（Apache-2.0）的"先判断值不值得看，再决定传不传"。
不抄它的代码，只借思路：
    1. 每帧在 64x36 小图上算 0~255 变化分（HSV 三通道加权差，亚毫秒级）
    2. 只有"硬切"或"运动量攒够"才留帧
    3. 留下的 4~8 帧拼成一张图，一次视觉调用看完整段变化
    4. 完全静止时给一张兜底帧，证明"确实没变"

为什么这么做：
    旧管线是 15 秒抓一帧、一帧调一次模型 → 单个瞬间、看不到变化、token 还贵。
    换成"一段时间拼一张"后，模型看到的是按时间排列的几帧，
    能判断"从站着到坐下""人离开了"这类连续变化——这是 ATG 建模真正缺的输入。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SamplerConfig:
    width: int = 800                 # 出图尺寸
    height: int = 450
    grid_w: int = 64                 # 变化分计算网格
    grid_h: int = 36
    hard_cut: int = 170              # 硬切阈值（画面突然换了一张）
    motion_quota: int = 420          # 运动积分阈值（一直在动、攒够了量）
    max_frames: int = 8              # 一张图最多拼几帧
    min_frames: int = 4              # 攒够几张就出图（少于这个继续等）
    idle_fallback_s: float = 20.0    # 完全静止时多久给一张兜底帧
    max_hold_s: float = 12.0         # 最多攒这么久就强制出图
    tag_frames: bool = True          # 是否在每帧角上标"第几帧/多少秒"


@dataclass
class Frame:
    image: np.ndarray
    ts: float
    score: int


@dataclass
class Sheet:
    png_b64: str
    frames: List[float]              # 每帧时间戳
    span_s: float                    # 覆盖时长
    reason: str                      # hard_cut / motion / idle / timeout
    cols: int = 0
    rows: int = 0


class FrameSampler:
    """把连续帧压成"值得看的几张"，再拼成一张图。"""

    def __init__(self, cfg: Optional[SamplerConfig] = None):
        self.cfg = cfg or SamplerConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hsv: Optional[np.ndarray] = None
        self._kept: List[Frame] = []
        self._acc = 0
        self._last_emit_ts = 0.0
        self._window_start = time.time()
        self._now_ts = self._window_start
        self._last_reason = "motion"

    # ---------- 变化分 ----------

    @staticmethod
    def _small(frame: np.ndarray, gw: int, gh: int) -> Tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
        return gray, hsv

    def score(self, frame: np.ndarray) -> int:
        """0~255 变化分。

        以灰度结构差为主（人动、物体移动、姿势变化都会让结构变化），
        HSV 色相差兜底（整体换色但结构不变的情况）。早先只按色相加权，
        纯色相变化（红衣换蓝衣）会被漏判，实测暴露后改成取两者较大值。
        """
        gray, hsv = self._small(frame, self.cfg.grid_w, self.cfg.grid_h)
        if self._prev_gray is None:
            self._prev_gray, self._prev_hsv = gray, hsv
            return 255                       # 第一帧直接留

        gd = float(np.abs(gray - self._prev_gray).mean()) * 2.2
        d = np.abs(hsv - self._prev_hsv)
        d[..., 0] = np.minimum(d[..., 0], 180 - d[..., 0])   # 色相环形距离
        hd = float((d[..., 0] * 0.5 + d[..., 1] * 0.3 + d[..., 2] * 0.2).mean()) * 4.0

        self._prev_gray, self._prev_hsv = gray, hsv
        return int(min(255, max(0.0, max(gd, hd))))

    # ---------- 推帧 ----------

    def push(self, frame: np.ndarray, ts: Optional[float] = None) -> Optional[Sheet]:
        """喂一帧；如果此刻该出图，返回 Sheet，否则 None。"""
        ts = ts or time.time()
        # 时间基准跳变（测试注入模拟时间 / 系统改时间）时重置窗口，避免窗口永远为负
        if abs(ts - self._window_start) > 3600:
            self._window_start = ts
        self._now_ts = ts          # 统一时间基准，避免混用 time.time() 导致窗口判定失效
        s = self.score(frame)

        if s >= self.cfg.hard_cut:
            self._kept.append(Frame(frame.copy(), ts, s))
            self._acc = 0
            self._last_reason = "hard_cut"
        else:
            self._acc += s
            if self._acc >= self.cfg.motion_quota:
                self._kept.append(Frame(frame.copy(), ts, s))
                self._acc = 0
                self._last_reason = "motion"

        if len(self._kept) >= self.cfg.max_frames:
            return self._emit("max_frames")
        if len(self._kept) >= self.cfg.min_frames and (ts - self._window_start) >= 6.0:
            return self._emit(self._last_reason)
        if (ts - self._window_start) >= self.cfg.max_hold_s and self._kept:
            return self._emit("timeout")
        if not self._kept and (ts - self._last_emit_ts) >= self.cfg.idle_fallback_s:
            self._kept.append(Frame(frame.copy(), ts, s))
            return self._emit("idle")
        return None

    # ---------- 出图 ----------

    def _emit(self, reason: str) -> Sheet:
        frames = self._kept[: self.cfg.max_frames]
        self._kept = []
        self._acc = 0
        self._window_start = self._now_ts
        self._last_emit_ts = self._now_ts
        cols, rows = self._layout(len(frames))
        sheet = self._compose(frames, cols, rows)
        return Sheet(
            png_b64=sheet,
            frames=[f.ts for f in frames],
            span_s=(frames[-1].ts - frames[0].ts) if frames else 0.0,
            reason=reason, cols=cols, rows=rows,
        )

    @staticmethod
    def _layout(n: int) -> Tuple[int, int]:
        if n <= 1:
            return 1, 1
        if n <= 4:
            return 2, 2
        if n <= 6:
            return 3, 2
        return 4, 2

    def _compose(self, frames: List[Frame], cols: int, rows: int) -> str:
        import base64
        W, H = self.cfg.width, self.cfg.height
        cw, ch = W // cols, H // rows
        canvas = np.full((ch * rows, cw * cols, 3), 40, dtype=np.uint8)   # 灰底，空格不冒充黑画面

        for i, f in enumerate(frames[: cols * rows]):
            r, c = divmod(i, cols)
            img = cv2.resize(f.image, (cw, ch), interpolation=cv2.INTER_AREA)
            if self.cfg.tag_frames:
                label = f"#{i + 1}"
                if frames:
                    off = f.ts - frames[0].ts
                    label += f" +{off:.0f}s"
                cv2.rectangle(img, (0, 0), (112, 26), (0, 0, 0), -1)
                cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = img

        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return ""
        return base64.b64encode(buf).decode("ascii")

    @property
    def pending(self) -> int:
        return len(self._kept)
