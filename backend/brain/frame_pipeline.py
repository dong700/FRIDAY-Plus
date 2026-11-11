"""流式视觉管道（2026-11-04，v2：快慢环 + 四档 + 动作帧）

为什么改：
    实测（同一段 9 秒视频，单次非流式调用）
        3×3 九格 960 high  think 开  7.8s
        1 帧 640 low       think 开  7.2s
        3×3 九格 960 high  think 关  2.6s
    → 时延由 thinking 长度决定，跟帧数/分辨率几乎无关。
    所以「减帧降时延」是错的；减帧省的是注意力与上下文预算。
    真正降时延只能靠结构：**采样窗口 = 推理周期**。

结构：
    快环（采集线程，4–8Hz，只算 64x36 差分，从不等模型）
        → TierController 四档（S0 静止 / S1 低频 / S2 中速 / S3 事件）
        → 有界缓冲（latest，丢旧保新）
    慢环（处理线程）
        → ChunkPlanner：窗口长度 = 该档推理预算（S0 8s / S1 6s / S2 4s / S3 10s）
        → select_action_frames：段内取差分能量峰，首/尾必留（覆盖动作起止）
        → 尾帧 overlap：上一 chunk 末帧作为新 chunk 首格，保跨窗口连续性
        → 场景卡 + 差分局部裁剪：第 1 格保留全图，其余只给变化区域放大
        → 九宫格 → Signal{screen.frame}

降档规则（防抖）：
    升档立即（宁可多花钱，别漏掉动作）
    降档要连续 DOWN_CONFIRM 次确认，且必须跌破当前档下沿的 DOWN_MARGIN（死区）
    S3 最短驻留 S3_DWELL 秒
    抖动定义：升档后 2 秒内又降回原来的档位 → 计一次 flap

四档的设计意图：
    S0/S1/S2 是「常态」——画面没大事发生，用便宜快档（think 关，2.6s），
    窗口 8/6/4 秒，慢环占用率远低于 1，永远追得上。
    S3 是「有戏」——用户在做值得理解的事，宁可贵一点也要准：think 开（7.5s），
    窗口放到 10 秒（占用约 0.75），中间 4Hz 继续抓帧，一张图都不丢。
    一句话：慢的时候要快，快的时候要准。
"""
from __future__ import annotations

import base64
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("brain.frame_pipeline")

# ---------------- 配置 ----------------

CFG = {
    # 采集：快环固定 4Hz 抓帧打分（只算 64x36 差分，几乎不花钱），
    # 但「留不留这一帧」由档位决定 —— 否则静止档 0.25fps 会漏掉突发动作（最长 4s 才发现）。
    "capture_fps": float(os.getenv("FRIDAY_CAPTURE_FPS", "4")),
    "buffer_size": int(os.getenv("FRIDAY_FRAME_BUFFER", "3")),
    "capture_w": int(os.getenv("FRIDAY_FRAME_W", "960")),
    "capture_h": int(os.getenv("FRIDAY_FRAME_H", "540")),
    # 出图
    "grid_w": int(os.getenv("FRIDAY_GRID_W", "800")),
    "grid_h": int(os.getenv("FRIDAY_GRID_H", "450")),
    "jpeg_q": int(os.getenv("FRIDAY_GRID_Q", "82")),
    # 处理线程
    "process_tick": 0.05,
    # 单个 chunk 的硬上限（异常情况下防内存暴涨）。S3 满窗 = 4fps×10s ≈ 40 帧，留 60% 余量
    "hard_max_frames": 64,
    # 裁剪：变化面积占比超过这个值就当「场景切换」，直接用全图
    "crop_scene_ratio": 0.55,
    # 裁剪：变化面积小于这个就认为没变，直接用全图
    "crop_min_ratio": 0.02,
    "crop_pad": 0.12,
}

# 四档：fps / 窗口秒数 / 最少帧数 / 送模型几张 / 推理预算
TIER_CFG = {
    "S0": {"fps": 0.25, "window_s": 8.0, "min_frames": 1, "send_k": 1, "infer": "fast"},
    "S1": {"fps": 0.50, "window_s": 6.0, "min_frames": 2, "send_k": 2, "infer": "fast"},
    "S2": {"fps": 1.00, "window_s": 4.0, "min_frames": 3, "send_k": 4, "infer": "fast"},
    "S3": {"fps": 4.00, "window_s": 10.0, "min_frames": 6, "send_k": 9, "infer": "deep"},
}

TIER_RANK = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}

# 变化分阈值（FrameSampler.score 返回 0~255）
S0_MAX = float(os.getenv("FRIDAY_TIER_S0", "8"))
S1_MAX = float(os.getenv("FRIDAY_TIER_S1", "30"))
S2_MAX = float(os.getenv("FRIDAY_TIER_S2", "110"))

DOWN_CONFIRM = int(os.getenv("FRIDAY_TIER_DOWN_CONFIRM", "4"))
S3_DWELL = float(os.getenv("FRIDAY_TIER_S3_DWELL", "8.0"))
# 降档死区：必须明显低于当前档的下沿才降，否则在阈值附近会反复横跳
DOWN_MARGIN = float(os.getenv("FRIDAY_TIER_DOWN_MARGIN", "0.75"))
# 升档：单帧冲上阈值不算，要连续 UP_CONFIRM 帧都够格；硬切（>=170）立即跳
UP_CONFIRM = int(os.getenv("FRIDAY_TIER_UP_CONFIRM", "2"))
HARD_CUT_SCORE = 170.0
# 判档用 EMA 平滑分，压掉单帧尖刺；与原始分差距大的是「剧变」另走硬切
EMA_ALPHA = float(os.getenv("FRIDAY_TIER_EMA", "0.35"))
DOWN_DWELL = float(os.getenv("FRIDAY_TIER_DOWN_DWELL", "3.0"))
UP_DWELL = float(os.getenv("FRIDAY_TIER_UP_DWELL", "1.5"))

# 常驻盯屏的空闲闸门：
#   桌面一直不动 = 没事可看。先停「送模型」（省钱），再久一点连采集也降到 0.5Hz（省 CPU）。
#   这两条都不杀线程：一有变化立刻醒回来，不需要重启。
IDLE_QUIET_S = float(os.getenv("FRIDAY_IDLE_QUIET", "300"))     # 5 分钟没变化 → 不再出九宫格
IDLE_STOP_S = float(os.getenv("FRIDAY_IDLE_STOP", "1200"))      # 20 分钟没变化 → 采集降到 0.5Hz
IDLE_PROBE_FPS = float(os.getenv("FRIDAY_IDLE_PROBE_FPS", "0.5"))

# 各档的分数区间下沿/上沿，用于算降档死区
UP_BOUND = {
    "S0": (0.0, S0_MAX),
    "S1": (S0_MAX, S1_MAX),
    "S2": (S1_MAX, S2_MAX),
    "S3": (S2_MAX, 255.0),
}


def tier_for_score(score: float) -> str:
    """变化分 → 档位。"""
    if score >= S2_MAX:
        return "S3"
    if score >= S1_MAX:
        return "S2"
    if score >= S0_MAX:
        return "S1"
    return "S0"


# ---------------- 数据结构 ----------------


@dataclass
class RawFrame:
    image: np.ndarray
    ts: float
    score: int


@dataclass
class Chunk:
    frames: List[RawFrame] = field(default_factory=list)
    t0: float = 0.0
    t1: float = 0.0
    reason: str = "motion"
    peak: int = 0
    tier: str = "S1"
    carry: int = 0                    # 首格是上一 chunk 的尾巴（overlap 帧数）

    @property
    def span(self) -> float:
        return self.t1 - self.t0


@dataclass
class Sheet:
    b64: str
    t0: float
    t1: float
    n: int
    reason: str
    tier: str = "S1"
    cols: int = 3
    rows: int = 3
    sent: int = 0
    cropped: int = 0


# ---------------- 档位控制器 ----------------


class TierController:
    """四档 AIMD：升档需连续确认（硬切立即），降档需死区 + 连续确认，S3 有最短驻留。"""

    def __init__(self, start: str = "S1") -> None:
        self._tier = start if start in TIER_RANK else "S1"
        self._since = -1e9            # 初始视为「早就稳定」，让开机那几跳不算抖动
        self._calm = 0                # 连续「明显低于当前档」的次数
        self._hot = 0                 # 连续「够格升一档」的次数
        self._ema = -1.0              # 平滑后的变化分（首帧直接取原始值）
        self._lock = threading.Lock()
        self._prev_before_up: Optional[str] = None
        self.transitions: List[Tuple[float, str, str]] = []
        self.flaps = 0                # 升档后 2 秒内又降回原档的次数（抖动指标）

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def fps(self) -> float:
        return TIER_CFG[self._tier]["fps"]

    @property
    def window_s(self) -> float:
        return TIER_CFG[self._tier]["window_s"]

    @property
    def send_k(self) -> int:
        return TIER_CFG[self._tier]["send_k"]

    @property
    def min_frames(self) -> int:
        return TIER_CFG[self._tier]["min_frames"]

    @property
    def infer_budget(self) -> str:
        return TIER_CFG[self._tier]["infer"]

    def update(self, score: float, now: float) -> str:
        """喂一个变化分，返回（可能更新后的）档位。

        判档用平滑后的分（EMA）：单帧尖刺不会把档位顶上去；硬切仍按原始分立即升。
        """
        raw = float(score)
        with self._lock:
            self._ema = raw if self._ema < 0 else self._ema * (1.0 - EMA_ALPHA) + raw * EMA_ALPHA
            s = self._ema
            cur = self._tier

            # 硬切（闪屏/切镜/黑场）看原始分，EMA 会把它抹平；命中就直接跳到对应档
            if raw >= HARD_CUT_SCORE:
                target = tier_for_score(raw)
                if TIER_RANK[target] > TIER_RANK[cur]:
                    self._calm = 0
                    self._hot = 0
                    self._set(target, now)
                    return self._tier

            cand = tier_for_score(s)
            if TIER_RANK[cand] > TIER_RANK[cur]:
                # 升档：连续 UP_CONFIRM 帧都够格才升，且刚切过档先不急着再切
                self._calm = 0
                self._hot += 1
                if self._hot >= UP_CONFIRM and (now - self._since) >= UP_DWELL:
                    self._set(cand, now)
                    self._hot = 0
            elif TIER_RANK[cand] < TIER_RANK[cur]:
                self._hot = 0
                # 死区：必须跌到当前档下沿的 DOWN_MARGIN 以下才算「真的变安静」
                floor = UP_BOUND[cur][0] * DOWN_MARGIN
                if s < floor:
                    self._calm += 1
                    if self._calm >= DOWN_CONFIRM:
                        if (now - self._since) < (S3_DWELL if cur == "S3" else DOWN_DWELL):
                            pass                       # 驻留期内不降
                        else:
                            step = [k for k in ("S0", "S1", "S2", "S3")
                                    if TIER_RANK[k] == TIER_RANK[cur] - 1][0]
                            self._set(step, now)
                            self._calm = 0
                else:
                    self._calm = 0
            else:
                self._calm = 0
                self._hot = 0
            return self._tier

    def _set(self, new: str, now: float) -> None:
        old = self._tier
        if new == old:
            return
        if TIER_RANK[new] > TIER_RANK[old]:
            self._prev_before_up = old          # 记住升之前的档
        elif new == self._prev_before_up and now - self._since < 2.0:
            self.flaps += 1                     # 升了又原路降回 = 抖动
        self.transitions.append((now, old, new))
        self._tier = new
        self._since = now


# ---------------- 动作帧选择 ----------------


def _small_gray(img: np.ndarray, gw: int = 64, gh: int = 36) -> np.ndarray:
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)


def frame_energy(frames: List[RawFrame]) -> List[float]:
    """逐帧差分能量（与前一帧的灰度平均绝对差）。第 0 帧记 0。"""
    if not frames:
        return []
    grays = [_small_gray(f.image) for f in frames]
    out = [0.0] * len(frames)
    for i in range(1, len(frames)):
        out[i] = float(np.abs(grays[i] - grays[i - 1]).mean())
    return out


def select_action_frames(frames: List[RawFrame], k: int) -> List[int]:
    """把 n 帧压成 k 帧，保证覆盖整段时间 + 动作峰值。

    做法：把时间轴切成 k 段；首段取第一帧、尾段取最后一帧（保动作起止），
    中间每段取段内差分能量最大的帧（保加速/峰值/回落）。
    比「全局 Top-K 能量」更稳：不会把 9 帧全挤在同一秒。
    """
    n = len(frames)
    if n == 0:
        return []
    if k <= 1:
        return [n - 1]
    if n <= k:
        return list(range(n))

    energy = frame_energy(frames)
    # 3 帧滑动平滑，避免单帧噪声决定选点
    smooth = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        smooth[i] = sum(energy[lo:hi]) / (hi - lo)

    picks: List[int] = []
    for s in range(k):
        lo = int(round(n * s / k))
        hi = int(round(n * (s + 1) / k))
        hi = max(hi, lo + 1)
        seg = range(lo, min(hi, n))
        if s == 0:
            picks.append(0)
        elif s == k - 1:
            picks.append(n - 1)
        else:
            best = max(seg, key=lambda i: smooth[i])
            picks.append(best)
    # 去重保序（段边界的极端情况）
    out: List[int] = []
    for i in picks:
        if not out or i > out[-1]:
            out.append(i)
    if out and out[-1] != n - 1:
        out.append(n - 1)
    return out


# ---------------- 分块规划 ----------------


class ChunkPlanner:
    """决定「什么时候该出一张九宫格」，以及这一格的档位。

    纯逻辑、可注入虚拟时间，便于暴力测试。窗口长度 = 该档推理预算。
    """

    def __init__(self, tiers: Optional[TierController] = None) -> None:
        self.tiers = tiers or TierController()
        self._frames: List[RawFrame] = []
        self._new_t0: Optional[float] = None      # 本 chunk 第一张「新帧」时间
        self._window = 0.0                        # 本窗口的推理周期（开场时锁定）
        self._tier = self.tiers.tier              # 本窗口的推理档位（开场时锁定）
        self._carry: Optional[RawFrame] = None
        self._peak = 0
        self._reason = "motion"
        self._emitted = 0
        self.dropped = 0                          # 因超时未被消费就被新窗口顶掉的张数

    def feed(self, f: RawFrame) -> Optional[Chunk]:
        """喂一帧；该出图时返回 Chunk，否则 None。

        档位的更新由采集端（快环）负责，这里只读当前档 —— 单一 owner，避免两处互相打架。
        """
        if f.score >= 170:
            self._reason = "hard_cut"
        self._peak = max(self._peak, f.score)
        tier = self.tiers.tier

        if not self._frames and self._carry is not None:
            self._frames.append(self._carry)
        self._frames.append(f)
        if self._new_t0 is None:
            self._new_t0 = f.ts
            # 窗口长度和推理档位在窗口开场时锁定：中途档位变化只影响下一个窗口，
            # 否则同一窗口会随着档位上下跳而越拖越长（真实视频里曾拖到 15s/35 帧）。
            self._window = 4.0 if self._emitted == 0 else self.tiers.window_s
            self._tier = self.tiers.tier
        elif TIER_RANK[tier] > TIER_RANK[self._tier]:
            # 窗口内升档 = 画面发生剧变：立刻收口，别把「剧变前」和「剧变后」
            # 混进同一张九宫格。但至少要跑过半个窗口，避免档位小抖动切出碎窗。
            if f.score >= HARD_CUT_SCORE or (f.ts - self._new_t0) >= 0.5 * self._window:
                return self.flush(self._tier)

        # 内存硬上限：异常情况下隔帧丢弃，保住首尾
        if len(self._frames) > CFG["hard_max_frames"]:
            self._frames = self._frames[::2] + [self._frames[-1]]

        elapsed = f.ts - self._new_t0
        if elapsed >= self._window:
            return self.flush(self._tier)
        return None

    def flush(self, tier: Optional[str] = None) -> Optional[Chunk]:
        if not self._frames:
            return None
        frames = self._frames
        carry = 1 if (self._carry is not None and frames and frames[0] is self._carry) else 0
        chunk = Chunk(
            frames=frames,
            t0=frames[0].ts,
            t1=frames[-1].ts,
            reason=self._reason,
            peak=self._peak,
            tier=tier or self._tier,
            carry=carry,
        )
        # 尾帧 overlap：留给下一窗口做第一格
        self._carry = frames[-1]
        self._frames = []
        self._new_t0 = None
        self._peak = 0
        self._reason = "motion"
        self._emitted += 1
        return chunk

    @property
    def pending(self) -> int:
        return len(self._frames)

    def on_consumed(self) -> None:
        """消费方（模型）取走了一张：清空待消费计数。"""
        self.dropped = 0

    def on_overwritten(self) -> None:
        self.dropped += 1


# ---------------- 出图 ----------------


def _crop_cell(img: np.ndarray, anchor: np.ndarray, cw: int, ch: int) -> Optional[np.ndarray]:
    """把「相对场景卡变化的区域」裁出来铺满一格；不值得裁就返回 None（用全图）。"""
    if img.shape[:2] != anchor.shape[:2]:
        anchor = cv2.resize(anchor, (img.shape[1], img.shape[0]))
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g0 = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY).astype(np.int16)
    mask = (np.abs(g1 - g0) > 14).astype(np.uint8)
    ratio = float(mask.mean())
    if ratio < CFG["crop_min_ratio"] or ratio > CFG["crop_scene_ratio"]:
        return None                                  # 没变 或 整屏都变了 → 给全图

    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    h, w = img.shape[:2]
    pad_x = int((xs.max() - xs.min()) * CFG["crop_pad"]) + 6
    pad_y = int((ys.max() - ys.min()) * CFG["crop_pad"]) + 6
    x0, x1 = max(0, int(xs.min()) - pad_x), min(w, int(xs.max()) + pad_x)
    y0, y1 = max(0, int(ys.min()) - pad_y), min(h, int(ys.max()) + pad_y)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = img[y0:y1, x0:x1]

    # 等比缩放 + 黑边居中，避免拉伸变形（变形会把文字/人脸搞坏）
    scale = min(cw / crop.shape[1], ch / crop.shape[0])
    nw, nh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((ch, cw, 3), 28, dtype=np.uint8)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    cell[oy:oy + nh, ox:ox + nw] = resized
    return cell


def compose_grid(chunk: Chunk, k: int) -> Optional[Sheet]:
    """chunk → 一张九宫格。第 1 格是全图场景卡，其余优先给变化区域局部。"""
    if not chunk.frames:
        return None
    picks = select_action_frames(chunk.frames, k)
    picked = [chunk.frames[i] for i in picks]
    n = len(picked)
    cols, rows = _layout(n)

    W, H = CFG["grid_w"], CFG["grid_h"]
    cw, ch = W // cols, H // rows
    canvas = np.full((ch * rows, cw * cols, 3), 28, dtype=np.uint8)
    anchor = chunk.frames[0].image          # 场景卡：本窗口第一帧

    cropped = 0
    for i, f in enumerate(picked):
        r, c = divmod(i, cols)
        is_crop = False
        cell = None
        if i > 0:                            # 第 1 格永远是全图，给模型空间参照
            cell = _crop_cell(f.image, anchor, cw, ch)
        if cell is not None:
            cropped += 1
            is_crop = True
        else:
            cell = cv2.resize(f.image, (cw, ch), interpolation=cv2.INTER_AREA)

        tag = f"#{i + 1} +{f.ts - chunk.t0:.1f}s"
        if is_crop:
            tag += " [局部]"
        cv2.rectangle(cell, (0, 0), (176, 26), (0, 0, 0), -1)
        cv2.putText(cell, tag, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell

    footer = (f"t0={time.strftime('%H:%M:%S', time.localtime(chunk.t0))}  "
              f"span={chunk.span:.1f}s  n={n}/{len(chunk.frames)}  {chunk.tier}")
    if chunk.carry:
        footer += "  (首格=上一窗口尾帧)"
    cv2.rectangle(canvas, (0, H - 22), (W, H), (0, 0, 0), -1)
    cv2.putText(canvas, footer, (6, H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, CFG["jpeg_q"]])
    if not ok:
        return None
    return Sheet(
        b64=base64.b64encode(buf).decode("ascii"),
        t0=chunk.t0, t1=chunk.t1, n=n, reason=chunk.reason,
        tier=chunk.tier, cols=cols, rows=rows, sent=n, cropped=cropped,
    )


def _layout(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    return 3, 3


# ---------------- 有界缓冲 ----------------


class LatestBuffer:
    """有界环形缓冲：满了丢最旧的，永远保留最新帧。"""

    def __init__(self, size: int) -> None:
        self._dq: Deque[RawFrame] = deque(maxlen=max(1, size))
        self._lock = threading.Lock()

    def put(self, frame: RawFrame) -> None:
        with self._lock:
            self._dq.append(frame)

    def drain(self) -> List[RawFrame]:
        with self._lock:
            items = list(self._dq)
            self._dq.clear()
            return items

    def latest(self) -> Optional[RawFrame]:
        with self._lock:
            return self._dq[-1] if self._dq else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


# ---------------- 管道 ----------------


class FramePipeline:
    def __init__(self) -> None:
        self._buffer = LatestBuffer(CFG["buffer_size"])
        self._planner = ChunkPlanner()
        self._scorer = _Scorer()

        self._latest_sheet: Optional[dict] = None
        self._sheet_lock = threading.Lock()

        self._stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stats = {"frames": 0, "sheets": 0, "scans": 0}
        # 空闲闸门：最后一次「桌面真的动了」的时刻（变化分 >= S0_MAX 才算动）
        self._last_change = time.time()
        self._quiet = False
        self._probe = False

    # ---------- 采集（快环） ----------

    def _grab(self) -> Optional[np.ndarray]:
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(all_screens=False)
        except Exception:
            return None
        arr = np.array(img)[:, :, ::-1]
        w, h = CFG["capture_w"], CFG["capture_h"]
        if arr.shape[1] != w or arr.shape[0] != h:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        return arr

    def _capture_loop(self) -> None:
        tiers = self._planner.tiers
        last_keep = 0.0
        last_rank = TIER_RANK[tiers.tier]
        while not self._stop.is_set():
            t0 = time.time()
            frame = self._grab()
            if frame is not None:
                now = time.time()
                score = self._scorer.score(frame)
                if score >= S0_MAX:                 # 桌面真的在动 → 醒过来
                    self._last_change = now
                    if self._quiet or self._probe:
                        self._quiet = self._probe = False
                        logger.info("[FramePipeline] 桌面有变化，恢复盯着")
                tier = tiers.update(score, now)
                self._stats["scans"] += 1
                rank = TIER_RANK[tier]
                # 升档的那一帧必须留，不能等下一个采样点（否则最长要 4s 才发现突发动作）
                due = now - last_keep >= 1.0 / max(0.05, tiers.fps)
                if due or rank > last_rank:
                    self._buffer.put(RawFrame(frame, now, score))
                    self._stats["frames"] += 1
                    last_keep = now
                last_rank = rank

            idle = time.time() - self._last_change
            if idle >= IDLE_STOP_S:
                if not self._probe:
                    self._probe = True
                    logger.info(f"[FramePipeline] 桌面静止 {idle/60:.0f} 分钟，采集降到 "
                                f"{IDLE_PROBE_FPS}Hz（有变化立刻恢复）")
                period = 1.0 / max(0.1, IDLE_PROBE_FPS)
            else:
                period = 1.0 / max(0.5, CFG["capture_fps"])
            self._stop.wait(max(0.02, period - (time.time() - t0)))

    # ---------- 规划 / 融合（慢环） ----------

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            for f in self._buffer.drain():
                chunk = self._planner.feed(f)
                if chunk:
                    self._flush_chunk(chunk)
            self._stop.wait(CFG["process_tick"])

    def _flush_chunk(self, chunk: Chunk) -> None:
        # 空闲闸门：桌面一直不动，就别再出九宫格、别再叫模型（省钱也省注意力）。
        # 窗口本身照常收口，只是不入库——一有变化下一张就正常发。
        if time.time() - self._last_change >= IDLE_QUIET_S:
            if not self._quiet:
                self._quiet = True
                logger.info(f"[FramePipeline] 桌面静止超过 {IDLE_QUIET_S/60:.0f} 分钟，"
                            f"暂停出图（有变化自动恢复）")
            return

        k = TIER_CFG[chunk.tier]["send_k"]
        sheet = compose_grid(chunk, k)
        if not sheet:
            return
        self._stats["sheets"] += 1

        with self._sheet_lock:
            if self._latest_sheet is not None:
                self._planner.on_overwritten()      # 上一张还没被模型取走就被顶掉
            self._latest_sheet = {
                "b64": sheet.b64,
                "ts": time.time(),
                "span": sheet.t1 - sheet.t0,
                "n": sheet.n,
                "total": len(chunk.frames),
                "reason": sheet.reason,
                "tier": sheet.tier,
                "cols": sheet.cols,
                "rows": sheet.rows,
                "cropped": sheet.cropped,
                "window": [sheet.t0, sheet.t1],
            }

        try:
            from brain.mind import get_mind
            infer = TIER_CFG[chunk.tier]["infer"]
            salience = min(0.7, 0.2 + chunk.peak / 255.0 * 0.5)
            get_mind().publish(
                "screen.frame",
                f"桌面九宫格 {sheet.tier}/{infer}：{chunk.span:.1f}s / "
                f"{sheet.n}帧（{chunk.reason}）",
                source="screen",
                wake=(chunk.tier == "S3"),         # 只有事件档才主动唤醒
                salience=salience,
                meta={"t0": chunk.t0, "t1": chunk.t1, "n": sheet.n,
                      "reason": chunk.reason, "tier": chunk.tier, "infer": infer},
                images=[{"b64": sheet.b64, "ts": chunk.t1,
                         "meta": {"t0": chunk.t0, "t1": chunk.t1, "n": sheet.n,
                                  "tier": chunk.tier}}],
            )
        except Exception:
            logger.warning("[FramePipeline] 发布 screen.frame 失败", exc_info=True)

    # ---------- 对外 ----------

    def latest_sheet(self, max_age_s: float = 60.0) -> Optional[dict]:
        with self._sheet_lock:
            s = self._latest_sheet
            if s:
                self._planner.on_consumed()
        if not s:
            return None
        if time.time() - s.get("ts", 0) > max_age_s:
            return None
        return dict(s)

    def stats(self) -> dict:
        t = self._planner.tiers
        return {
            "frames": self._stats["frames"],
            "scans": self._stats["scans"],
            "sheets": self._stats["sheets"],
            "fps": t.fps,
            "tier": t.tier,
            "infer": t.infer_budget,
            "transitions": len(t.transitions),
            "flaps": t.flaps,
            "dropped": self._planner.dropped,
            "buffer": len(self._buffer),
            "chunk": self._planner.pending,
            "idle_s": round(time.time() - self._last_change, 1),
            "asleep": self._quiet or self._probe,
            "probe": self._probe,
        }

    def start(self) -> None:
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="frame-capture")
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="frame-process")
        self._capture_thread.start()
        self._process_thread.start()
        logger.info(f"[FramePipeline] 启动 tier={self._planner.tiers.tier} "
                    f"fps={self._planner.tiers.fps} buffer={CFG['buffer_size']}")

    def stop(self) -> None:
        self._stop.set()


class _Scorer:
    """64x36 变化分（灰度结构差为主，HSV 色相兜底）。复用 FrameSampler 的算法。"""

    def __init__(self) -> None:
        from brain.frame_sampler import FrameSampler, SamplerConfig
        self._s = FrameSampler(SamplerConfig(width=CFG["grid_w"], height=CFG["grid_h"],
                                             grid_w=64, grid_h=36))
        self._primed = False

    def score(self, frame: np.ndarray) -> int:
        s = self._s.score(frame)
        if not self._primed:
            # 第一帧没有参照，采样器会返回 255；那是「不知道」，不是「巨变」，
            # 否则开机瞬间会被误判成事件档。
            self._primed = True
            return 0
        return s


_pipeline: Optional[FramePipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> FramePipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = FramePipeline()
        return _pipeline


def start_pipeline() -> FramePipeline:
    p = get_pipeline()
    p.start()
    return p
