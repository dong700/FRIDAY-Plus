"""视频基本理解管道（2026-11-04）

参考的 GitHub 方案与这里的落地方式：
    PySceneDetect / TransNetV2  镜头切分        → ffmpeg 低分辨率抽样 + 全向量化直方图/帧差
    PEEK / DynFrame             任务导向选帧    → 每个镜头只取关键帧，按场景数上限自适应
    VideoRAG / EC-RAG           层级摘要        → 段描述 → 全局摘要 → TFGR 事件索引
    VideoMind                   长视频推理      → 用 max_seconds 截断 + 场景上限控制上下文

产出统一写进 ATG / TFGR：
    - 每个镜头 = TFGR 区间事件（带视频内部时间 + 关键帧路径，剪辑时可直接引用）
    - 整片 = ATG 一次活动观察（"在看视频《xxx》"）+ TFGR 一个点事件

缓存：data/video_cache/<key>.json，key = 路径+大小+修改时间。同一文件不重复分析。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger

logger = get_logger("brain.video")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".ts", ".mts"}
SAMPLE_W, SAMPLE_H = 160, 90
MAX_SAMPLE_FRAMES = 4800          # 抽样总帧数上限，长视频自动降 fps
MAX_SCENES = int(__import__("os").getenv("FRIDAY_VIDEO_MAX_SCENES", "14"))
MIN_SCENE_S = 1.2
MAX_SCENE_S = float(__import__("os").getenv("FRIDAY_VIDEO_MAX_SCENE_S", "8"))
VISION_CONCURRENCY = 3


# ---------------------------------------------------------------- 基础设施

def _data_dir() -> Path:
    try:
        from core.config import get_data_dir
        d = Path(get_data_dir())
    except Exception:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for c in (r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin", r"C:\ffmpeg\bin"):
        f = Path(c) / f"{name}.exe"
        if f.exists():
            return str(f)
    raise FileNotFoundError(f"找不到 {name}，请确认 ffmpeg 已安装并在 PATH 中")


def resolve_video(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        for root in (Path("D:/VideoAssets"), Path.cwd()):
            cand = root / p
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise FileNotFoundError(f"视频不存在: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"不支持的视频格式: {p.suffix}")
    return p


def probe(path: Path) -> Dict[str, Any]:
    """读时长/分辨率/编码（ffprobe）。"""
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError((out.stderr or "")[:300] or "ffprobe 失败")
    data = json.loads(out.stdout or "{}")
    fmt = data.get("format") or {}
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    duration = float(fmt.get("duration") or 0.0)
    return {
        "duration": duration,
        "width": int(vs.get("width") or 0),
        "height": int(vs.get("height") or 0),
        "fps": _parse_fps(str(vs.get("r_frame_rate") or "0/1")),
        "size": int(fmt.get("size") or 0),
        "codec": str(vs.get("codec_name") or ""),
        "has_audio": has_audio,
    }


def _parse_fps(s: str) -> float:
    try:
        a, b = s.split("/")
        return round(float(a) / float(b), 2) if float(b) else 0.0
    except Exception:
        return 0.0


def _sample_frames(path: Path, duration: float) -> Tuple[np.ndarray, float]:
    """按自适应 fps 抽样解码成 RGB 小帧（绕开 cv2 对中文路径的限制）。"""
    target_fps = min(2.0, MAX_SAMPLE_FRAMES / max(duration, 1.0))
    target_fps = max(0.2, target_fps)
    vf = f"fps={target_fps:.4f},scale={SAMPLE_W}:{SAMPLE_H}:flags=neighbor"
    out = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-i", str(path), "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=600,
    )
    if out.returncode != 0:
        raise RuntimeError((out.stderr.decode("utf-8", "replace") or "")[:300])
    fs = SAMPLE_W * SAMPLE_H * 3
    n = len(out.stdout) // fs
    if n == 0:
        raise RuntimeError("抽帧为空")
    arr = np.frombuffer(out.stdout[: n * fs], dtype=np.uint8).reshape(n, SAMPLE_H, SAMPLE_W, 3)
    return arr, target_fps


def detect_scenes(frames: np.ndarray, fps: float, duration: float,
                  max_scenes: int = MAX_SCENES) -> List[Tuple[float, float, float]]:
    """镜头切分：返回 [(start, end, 边界分数)]，全向量化。

    分数 = 0.6 * 平均灰度差 + 0.4 * 直方图 L1 距离，阈值按分布自适应（均值+2.5σ，下限 0.10）。
    """
    n = len(frames)
    if n < 3:
        return [(0.0, duration, 1.0)]
    gray = frames.mean(axis=3)                                   # (n, h, w)
    d = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2)) / 255.0    # (n-1,)

    q = (gray // 16).astype(np.uint8)
    hist = np.zeros((n, 16), dtype=np.float32)
    np.add.at(hist, (np.repeat(np.arange(n), SAMPLE_H * SAMPLE_W), q.ravel()), 1)
    hist /= hist.sum(axis=1, keepdims=True)
    hd = np.abs(hist[1:] - hist[:-1]).sum(axis=1) / 2.0

    score = np.concatenate([[0.0], 0.6 * d + 0.4 * hd])
    thr = max(0.10, float(score.mean() + 2.5 * score.std()))

    # 局部峰值才算切点（避免一个渐变过程连出许多边界）
    local_max = np.zeros(n, dtype=bool)
    core = score[1:n - 1]
    local_max[1:n - 1] = (core > thr) & (core >= score[0:n - 2]) & (core > score[2:n])
    cands = [i for i in range(1, n - 1) if local_max[i]]

    # 最小镜头时长约束
    min_gap = max(1, int(MIN_SCENE_S * fps))
    kept: List[int] = []
    for i in cands:
        if i - (kept[-1] if kept else 0) >= min_gap:
            kept.append(i)

    # 镜头太多 → 只留分数最高的若干个边界（对应"任务导向选帧"：优先保留变化最大的）
    if len(kept) > max_scenes - 1:
        top = sorted(kept, key=lambda i: score[i], reverse=True)[: max_scenes - 1]
        kept = sorted(top)

    bounds = [0.0] + [i / fps for i in kept] + [duration]
    segments: List[Tuple[float, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 0.4:
            continue
        segments.append((a, b))

    # 镜头内部细分：长镜头里画面也会变（人物走动/换景别）。把剩余的场景预算
    # 优先分给最长的段，在段内变化最大的位置切开（任务导向选帧的简化版）。
    budget = max(0, max_scenes - len(segments))
    guard = 0
    while budget > 0 and guard < 200:
        guard += 1
        idx = max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
        a, b = segments[idx]
        if b - a <= MAX_SCENE_S:
            break
        lo = max(int((a + MIN_SCENE_S) * fps), int(a * fps) + 1)
        hi = min(int(min(b, a + MAX_SCENE_S) * fps), n - 2)
        if hi <= lo:
            break
        k = lo + int(np.argmax(score[lo:hi + 1]))
        if score[k] < 0.02:          # 段内几乎静止 → 不硬切，避免把静止镜头切碎
            break
        cut = k / fps
        segments[idx:idx + 1] = [(a, cut), (cut, b)]
        budget -= 1

    scenes: List[Tuple[float, float, float]] = []
    for a, b in segments:
        if b - a < 0.4:
            continue
        idx2 = min(n - 1, int(round(b * fps)))
        scenes.append((round(a, 2), round(min(b, duration), 2), float(score[min(idx2, n - 1)])))
    return scenes or [(0.0, duration, 1.0)]


def extract_frame(path: Path, ts: float, out_path: Path, width: int = 640) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-y", "-ss", f"{max(0.0, ts):.3f}",
         "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2",
         "-q:v", "3", str(out_path)],
        capture_output=True, timeout=60,
    )
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


# ---------------------------------------------------------------- 数据结构

@dataclass
class Scene:
    index: int
    start: float
    end: float
    score: float = 0.0
    description: str = ""
    frame_path: str = ""


@dataclass
class VideoAnalysis:
    path: str
    name: str
    duration: float
    fps: float
    width: int
    height: int
    scenes: List[Scene] = field(default_factory=list)
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    analyzed_at: float = 0.0
    cached: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def brief(self, max_scenes: int = 6) -> str:
        head = f"《{self.name}》{self.duration:.0f}s，{len(self.scenes)} 个镜头"
        if self.summary:
            head += f"\n{self.summary}"
        lines = []
        for s in self.scenes[:max_scenes]:
            lines.append(f"[{_mmss(s.start)}-{_mmss(s.end)}] {s.description[:90]}")
        return head + ("\n" + "\n".join(lines) if lines else "")

    def scene_at(self, ts: float) -> Optional[Scene]:
        for s in self.scenes:
            if s.start <= ts < s.end:
                return s
        return None


def _mmss(t: float) -> str:
    m, s = divmod(int(max(0.0, t)), 60)
    return f"{m:02d}:{s:02d}"


def _cache_key(p: Path, info: Dict[str, Any]) -> str:
    raw = f"{p.resolve()}|{info.get('size')}|{int(info.get('duration', 0))}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 主流程

async def analyze_video(
    path: str | Path,
    max_scenes: Optional[int] = None,
    max_seconds: Optional[float] = None,
    force: bool = False,
    publish: bool = True,
) -> VideoAnalysis:
    """分析一个视频：镜头切分 → 关键帧 → 分段描述 → 全局摘要 → 写入 ATG/TFGR。

    max_seconds: 只看前 N 秒（"描述这个视频的前几秒"场景）。
    force: 忽略缓存。
    """
    p = resolve_video(path)
    info = await asyncio.to_thread(probe, p)
    key = _cache_key(p, info)
    cache_file = _data_dir() / "video_cache" / f"{key}.json"
    frame_dir = _data_dir() / "video_frames" / key

    if cache_file.exists() and not force:
        try:
            cached = VideoAnalysis(**json.loads(cache_file.read_text(encoding="utf-8")))
            cached.cached = True
            if max_seconds is None and not max_scenes:
                logger.info(f"[Video] 命中缓存 {p.name}")
                return cached
        except Exception:
            logger.warning("[Video] 缓存损坏，重新分析", exc_info=True)

    duration = info["duration"] or 0.0
    limit = min(duration, max_seconds) if max_seconds else duration
    if limit <= 0:
        raise RuntimeError(f"无法读取视频时长: {p}")

    frames, sfps = await asyncio.to_thread(_sample_frames, p, limit)
    if max_seconds:
        cut = int(round(limit * sfps)) + 1
        if cut >= 3:
            frames = frames[:cut]
    scenes_ts = detect_scenes(frames, sfps, limit, max_scenes or MAX_SCENES)

    scenes = [Scene(index=i, start=a, end=b, score=sc)
              for i, (a, b, sc) in enumerate(scenes_ts)]

    # 关键帧：镜头中点（保证不是黑场/转场帧），每镜头 1 张
    def _frames() -> None:
        for s in scenes:
            mid = (s.start + s.end) / 2.0
            fp = extract_frame(p, mid, frame_dir / f"{s.index:03d}.jpg")
            if fp:
                s.frame_path = str(fp)

    await asyncio.to_thread(_frames)

    # 分段描述：并发 3 路，失败不拖垮整片
    sem = asyncio.Semaphore(VISION_CONCURRENCY)
    from brain.vision import get_vision_module
    vision = get_vision_module()

    async def _describe(s: Scene) -> None:
        if not s.frame_path:
            s.description = "（关键帧提取失败）"
            return
        try:
            b64 = base64.b64encode(Path(s.frame_path).read_bytes()).decode()
        except Exception as e:
            s.description = f"（读图失败: {e}）"
            return
        q = (
            f"这是《{p.stem}》第 {s.index + 1} 个镜头（{_mmss(s.start)}-{_mmss(s.end)}）的代表画面。"
            "客观描述：谁/什么在画面里、在做什么、场景环境、镜头类型（特写/中景/全景）、"
            "画面里的文字或字幕。不要猜测画面以外的前因后果，不超过 70 字。"
        )
        async with sem:
            r = await vision.describe_image(b64, q)
        s.description = (r.get("description") or "").strip().replace("\n", " ") or "（无描述）"

    await asyncio.gather(*[_describe(s) for s in scenes], return_exceptions=True)

    summary = await _make_summary(p.stem, scenes, duration)
    keywords = _keywords(scenes, summary)

    analysis = VideoAnalysis(
        path=str(p), name=p.stem, duration=round(limit, 2), fps=info["fps"],
        width=info["width"], height=info["height"], scenes=scenes,
        summary=summary, keywords=keywords, analyzed_at=time.time(),
    )

    # 缓存（只缓存完整分析；截断分析不覆盖完整缓存）
    if not max_seconds:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        except Exception:
            logger.warning("[Video] 缓存写入失败", exc_info=True)

    if publish:
        _publish(analysis)
    return analysis


async def _make_summary(name: str, scenes: List[Scene], duration: float) -> str:
    try:
        from brain.llm import get_llm
        table = "\n".join(
            f"{i + 1}. [{_mmss(s.start)}-{_mmss(s.end)}] {s.description[:80]}"
            for i, s in enumerate(scenes)
        )
        r = await get_llm().chat([
            {"role": "system", "content":
             "你是视频剪辑助理。根据逐镜头描述，用 2-3 句说明这条视频在讲什么、"
             "画面风格和可用的关键素材，不要编造描述里没有的信息。"},
            {"role": "user", "content": f"视频《{name}》（{duration:.0f} 秒）逐镜头描述：\n{table}"},
        ])
        return (r or "").strip()[:400]
    except Exception as e:
        logger.warning(f"[Video] 摘要失败: {e}")
        return ""


_KEYWORD_STOP = {
    "画面", "镜头", "视频", "描述", "背景", "字幕", "中景", "特写", "近景", "全景",
    "画面中", "一位", "一名", "一个", "这个", "视频中", "同时", "以及", "可以",
    "没有", "进行", "正在", "出现", "显示", "身穿", "背景为", "底部", "顶部",
}


def _keywords(scenes: List[Scene], summary: str) -> List[str]:
    """关键词：优先 jieba 分词（去掉虚词/镜头术语），供 TFGR 检索和素材搜索复用。"""
    text = summary + " " + " ".join(s.description for s in scenes)
    freq: Dict[str, int] = {}
    try:
        import jieba
        words = [w.strip() for w in jieba.cut(text)]
        for w in words:
            if 2 <= len(w) <= 4 and w not in _KEYWORD_STOP:
                freq[w] = freq.get(w, 0) + 1
    except Exception:
        import re
        for w in re.findall(r"[一-龥]{2,4}", text):
            if w not in _KEYWORD_STOP:
                freq[w] = freq.get(w, 0) + 1
    picked = [w for w, c in freq.items() if c >= 2]
    if not picked:
        picked = [w for w in freq if len(w) >= 3]
    return [w for w, _ in sorted(((w, freq[w]) for w in picked),
                                 key=lambda kv: kv[1], reverse=True)[:8]]


def _publish(a: VideoAnalysis) -> None:
    """分析结果写进 ATG / TFGR / 心智时间轴。"""
    base = a.analyzed_at or time.time()
    try:
        from brain.tfgr import get_tfgr
        tfgr = get_tfgr()
        for s in a.scenes:
            tfgr.record_interval(
                "video.scene", f"《{a.name}》{_mmss(s.start)} {s.description[:60]}",
                base + s.start, base + s.end, source="video",
                entities=[a.name] + a.keywords[:2],
                meta={"video": a.path, "in": s.start, "out": s.end,
                      "frame": s.frame_path, "index": s.index},
            )
        tfgr.record("video.analyze", f"分析了《{a.name}》：{a.summary[:120]}",
                    ts=base + a.duration, source="video",
                    entities=[a.name], refs=[a.path],
                    meta={"scenes": len(a.scenes), "duration": a.duration})
    except Exception:
        logger.warning("[Video] TFGR 写入失败", exc_info=True)

    try:
        from brain import atg
        atg.observe(f"在看视频《{a.name}》", source="video")
    except Exception:
        pass

    try:
        from brain.mind import get_mind
        get_mind().publish(
            "video.analyze",
            f"看完了《{a.name}》（{a.duration:.0f}s，{len(a.scenes)} 个镜头）：{a.summary[:100]}",
            source="video", wake=False, salience=60,
            meta={"video": a.path, "entities": [a.name]},
        )
    except Exception:
        pass
