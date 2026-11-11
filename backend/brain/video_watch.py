"""整片观影（2026-11-04）——把视频文件当成直播流喂进 frame_pipeline。

和 brain/video_analysis.py 的区别：
    video_analysis  只切 14 个镜头、每个镜头取一张中间帧 → 那是看几张照片。
    这里            按 3~4Hz 解码全片，走同一套 TierController/ChunkPlanner，
                    一个窗口出一张九宫格（1~9 帧，按动作峰值挑），
                    模型看的是「这一段里发生了什么」，不是「这一秒长什么样」。

产物：
    windows[]  每段 {t0,t1,tier,reason,n,grid,desc,summary}，时间码是原片秒数。
    TFGR       每个窗口一条区间事件（带 in/out），剪辑可直接引用。
    ATG        整片一次观察 + 每个窗口一句叙事（成片的时间线就是从这里长出来的）。
    JSONL      data/video_watch/<key>.jsonl，给 film_builder --timeline 用。

为什么用 base 时间偏移：
    ChunkPlanner/TierController 只用相对秒数（now - since），所以喂
    base + 视频秒数 就能复用实时管道；九宫格页脚的时间也能打印成人话。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.frame_pipeline import ChunkPlanner, RawFrame, TIER_CFG, compose_grid, _Scorer
from brain.video_analysis import _bin, _mmss, probe, resolve_video
from core.logger import get_logger

logger = get_logger("brain.video_watch")

W, H = 640, 360                 # 解码尺寸（比源片略小，九宫格够用且解码快）
DESCRIBE_CONCURRENCY = 3


@dataclass
class Window:
    index: int
    t0: float
    t1: float
    tier: str
    reason: str
    n: int                 # 送进模型的帧数
    total: int = 0         # 窗口内实际抓到的帧数
    peak: int = 0
    grid: str = ""         # 九宫格 jpg 路径
    desc: str = ""
    summary: str = ""


@dataclass
class Watch:
    path: str
    name: str
    duration: float
    t_start: float
    t_end: float
    windows: List[Window] = field(default_factory=list)
    frames_read: int = 0
    elapsed: float = 0.0
    error: str = ""

    def timeline(self) -> str:
        return "\n".join(
            f"{w.index + 1}. [{_mmss(w.t0)}-{_mmss(w.t1)}] {w.summary or w.desc[:60]}"
            for w in self.windows)


def _iter_frames(path: Path, t0: float, t1: float, fps: float):
    """流式解码：ffmpeg 吐 bgr24，逐帧 yield (视频秒数, frame)。不占内存。"""
    dur = max(0.1, t1 - t0)
    proc = subprocess.Popen(
        [_bin("ffmpeg"), "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
         "-i", str(path), "-vf", f"fps={fps:.4f},scale={W}:{H}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=W * H * 3 * 4,
    )
    fs = W * H * 3
    buf = bytearray()
    i = 0
    try:
        while True:
            chunk = proc.stdout.read(fs - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) < fs:
                continue
            frame = np.frombuffer(bytes(buf[:fs]), dtype=np.uint8).reshape(H, W, 3)
            buf = buf[fs:]
            yield t0 + i / fps, frame
            i += 1
    finally:
        try:
            proc.stdout.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def segment(path: str | Path, t0: float = 0.0, t1: Optional[float] = None,
            fps: float = 4.0, publish: bool = False,
            save_grid_dir: Optional[Path] = None) -> Watch:
    """只做切分和出图（不调模型）：快，用来检查窗口划分是否合理。"""
    p = resolve_video(path)
    info = probe(p)
    end = min(info["duration"], t1) if t1 else info["duration"]
    base = time.time()

    planner = ChunkPlanner()
    tc = planner.tiers
    scorer = _Scorer()
    windows: List[Window] = []
    last_keep = -1e9
    last_rank = 0
    read = 0

    def _flush(chunk) -> None:
        k = TIER_CFG[chunk.tier]["send_k"]
        sheet = compose_grid(chunk, k)
        if not sheet:
            return
        idx = len(windows)
        gp = ""
        if save_grid_dir is not None:
            save_grid_dir.mkdir(parents=True, exist_ok=True)
            f = save_grid_dir / f"{idx:03d}.jpg"
            f.write_bytes(base64.b64decode(sheet.b64))
            gp = str(f)
        windows.append(Window(
            index=idx, t0=round(chunk.t0 - base, 2), t1=round(chunk.t1 - base, 2),
            tier=chunk.tier, reason=chunk.reason, n=sheet.n,
            total=len(chunk.frames), peak=chunk.peak, grid=gp,
        ))

    for ts_v, img in _iter_frames(p, t0, end, fps):
        read += 1
        score = scorer.score(img)
        now = base + ts_v
        tier = tc.update(score, now)
        rank = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}[tier]
        due = now - last_keep >= 1.0 / max(0.05, tc.fps)
        if due or rank > last_rank:
            ch = planner.feed(RawFrame(img, now, score))
            if ch:
                _flush(ch)
            last_keep = now
        last_rank = rank

    ch = planner.flush()
    if ch:
        _flush(ch)

    w = Watch(path=str(p), name=p.stem, duration=round(info["duration"], 2),
              t_start=t0, t_end=end, windows=windows, frames_read=read)
    return w


_GRID_PROMPT = (
    "这是一段纪录片《{name}》的抽样九宫格，覆盖原片第 {t0} 秒到第 {t1} 秒。"
    "画面按时间顺序排列（从左到右、从上到下），左上角标着每格相对这一段的偏移秒数；"
    "标了[局部]的格子只是变化区域的放大，不是另一个场景。\n"
    "要求：\n"
    "1) 逐格一句话：谁/什么在画面里、在做什么、景别（特写/中景/全景/空镜）、画面上的文字（照抄原文）。\n"
    "2) 如果相邻格子之间发生了动作或场景变化，写清变化发生在第几格到第几格。\n"
    "3) 最后一行以「概括：」开头，一句话说明这一段在讲什么、适不适合当「{hint}」的画面。\n"
    "只写画面里确实有的东西，不要编造前因后果。"
)


async def describe(w: Watch, hint: str = "人物纪录片", concurrency: int = DESCRIBE_CONCURRENCY) -> Watch:
    """给每个窗口的九宫格写描述（并发，失败不拖垮整片）。"""
    from brain.vision import get_vision_module

    vision = get_vision_module()
    sem = asyncio.Semaphore(concurrency)

    async def one(win: Window) -> None:
        if not win.grid or not Path(win.grid).exists():
            win.desc = "（无图）"
            return
        b64 = base64.b64encode(Path(win.grid).read_bytes()).decode()
        q = _GRID_PROMPT.format(name=w.name, t0=_mmss(win.t0), t1=_mmss(win.t1), hint=hint)
        async with sem:
            try:
                r = await vision.describe_image(b64, q)
                win.desc = (r.get("description") or "").strip()
            except Exception as e:
                win.desc = f"（描述失败: {e}）"
        lines = [x.strip() for x in win.desc.splitlines() if x.strip()]
        win.summary = _pick_summary(lines)

    await asyncio.gather(*[one(x) for x in w.windows], return_exceptions=True)
    return w


def _pick_summary(lines: List[str]) -> str:
    """从「逐格描述 + 概括」里挑一句能当摘要的话。

    模型偶尔只吐编号（"3)"、"#2"），或者「概括」后面是空的——那些都不能当摘要。
    """
    def _strip(s: str) -> str:
        s = re.sub(r"^[\s\d\-\.\)\*\#]+", "", s).strip("*：: ").strip()
        return s

    for ln in reversed(lines):
        if "概括" in ln:
            s = _strip(ln.split("概括", 1)[1])
            if len(s) >= 6:
                return s[:160]
    for ln in reversed(lines):
        s = _strip(ln)
        if len(s) >= 6 and not s.startswith("#") and "概括" not in s:
            return s[:120]
    return ""


def publish(w: Watch) -> None:
    """写 TFGR（每窗口一条区间事件，带 in/out）+ ATG（整片叙事）。"""
    base = time.time() - w.t_end
    try:
        from brain.tfgr import get_tfgr
        tfgr = get_tfgr()
        for win in w.windows:
            tfgr.record_interval(
                "video.window", f"《{w.name}》{_mmss(win.t0)} {win.summary[:70]}",
                base + win.t0, base + win.t1, source="video",
                entities=[w.name],
                meta={"video": w.path, "in": win.t0, "out": win.t1,
                      "tier": win.tier, "grid": win.grid},
            )
    except Exception:
        logger.warning("[Watch] TFGR 写入失败", exc_info=True)
    try:
        from brain import atg
        for win in w.windows:
            atg.observe(f"《{w.name}》{_mmss(win.t0)}-{_mmss(win.t1)}：{win.summary[:80]}",
                        source="video")
        atg.observe(f"看完了《{w.name}》（{w.duration:.0f}s，{len(w.windows)} 段）", source="video")
    except Exception:
        logger.warning("[Watch] ATG 写入失败", exc_info=True)


def dump(w: Watch, out: Optional[Path] = None) -> Path:
    """落 JSONL：一行一个窗口，film_builder --timeline 直接读。"""
    if out is None:
        raw = (f"{w.path}|{Path(w.path).stat().st_size if Path(w.path).exists() else 0}"
               f"|{w.t_start:.1f}|{w.t_end:.1f}")
        key = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        out = Path("data") / "video_watch" / f"{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for win in w.windows:
            fp.write(json.dumps(asdict(win), ensure_ascii=False) + "\n")
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({"path": w.path, "name": w.name, "duration": w.duration,
                                "t_start": w.t_start, "t_end": w.t_end,
                                "frames": w.frames_read, "windows": len(w.windows),
                                "elapsed": round(w.elapsed, 1)}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FrIDAY 整片观影")
    ap.add_argument("video")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--hint", default="人物纪录片", help="这段片子适合当什么画面")
    ap.add_argument("--no-describe", action="store_true", help="只切窗口，不调模型")
    ap.add_argument("--no-publish", action="store_true", help="不写 ATG/TFGR")
    ap.add_argument("--from-jsonl", help="跳过观看，直接拿已有的 jsonl 去写 TFGR/ATG")
    ap.add_argument("--out", help="jsonl 输出路径")
    a = ap.parse_args(argv)

    if a.from_jsonl:
        rows = [json.loads(l) for l in Path(a.from_jsonl).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        meta_f = Path(a.from_jsonl).with_suffix(".meta.json")
        meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
        w = Watch(path=meta.get("path", a.video), name=meta.get("name", ""),
                  duration=meta.get("duration", 0), t_start=meta.get("t_start", 0),
                  t_end=meta.get("t_end", 0))
        w.windows = [Window(**{k: v for k, v in r.items() if k in Window.__dataclass_fields__})
                     for r in rows]
        publish(w)
        print(f"把 {len(w.windows)} 个窗口写进 TFGR/ATG：《{w.name}》")
        return 0

    t = time.time()
    grid_dir = Path("data") / "video_watch" / "grids"
    w = segment(a.video, a.t0, a.t1, a.fps, save_grid_dir=grid_dir)
    w.elapsed = time.time() - t
    print(f"《{w.name}》{w.duration:.0f}s  看了 {w.t_start:.0f}-{w.t_end:.0f}s  "
          f"解码 {w.frames_read} 帧 → {len(w.windows)} 个窗口  {w.elapsed:.1f}s")
    for win in w.windows:
        print(f"  #{win.index:<3} {_mmss(win.t0)}-{_mmss(win.t1)}  {win.tier}  "
              f"{win.n}/{win.total}帧  {win.reason}")

    if not a.no_describe and w.windows:
        t2 = time.time()
        w = asyncio.run(describe(w, a.hint))
        print(f"描述完成 {time.time() - t2:.1f}s")
        for win in w.windows:
            print(f"  #{win.index:<3} {_mmss(win.t0)}  {win.summary[:80]}")

    out = dump(w, Path(a.out) if a.out else None)
    print(f"→ {out}")
    if not a.no_publish and w.windows:
        publish(w)
        print(f"已写进 TFGR（{len(w.windows)} 条区间事件）和 ATG（逐段叙事）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
