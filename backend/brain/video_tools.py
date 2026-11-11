"""视频操作工具：找视频 → 看视频 → 剪视频（计算机控制的最小可用链路）

    find_videos(query)            在本机素材目录里按名字/路径检索
    analyze_video(path)           切镜头 + 逐段描述 + 摘要（走 brain.video_analysis）
    plan_edl(analysis, instruction)  让模型把剪辑要求翻译成 EDL（只允许用真实镜头边界）
    render_edl(path, edl)         ffmpeg 精确裁剪 + 拼接，输出成片

设计约束：
    - EDL 用视频内部时间（秒），和 TFGR 里 video.scene 的 meta.in/out 一致，可互相检索
    - 剪辑不做生成式改写：只裁/拼，避免"AI 编造素材"这类不可控行为
    - 所有路径都经过 resolve_video 校验，不执行任意 shell 字符串
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logger import get_logger
from brain.video_analysis import (
    VIDEO_EXTS, VideoAnalysis, _bin, _data_dir, _mmss, analyze_video, probe, resolve_video,
)

logger = get_logger("brain.video_tools")

DEFAULT_ROOTS = [r"D:\VideoAssets"]
MAX_WALK_FILES = 20000


def video_roots() -> List[Path]:
    env = os.getenv("FRIDAY_VIDEO_ROOTS", "")
    roots = [Path(x.strip()) for x in env.split(";") if x.strip()] or \
            [Path(x) for x in DEFAULT_ROOTS]
    return [r for r in roots if r.exists()]


def _tokens(query: str) -> List[str]:
    q = (query or "").strip().lower()
    parts = [p for p in re.split(r"[\s,;，。]+", q) if p]
    return parts


def find_videos(query: str = "", roots: Optional[List[Path]] = None,
                limit: int = 20) -> List[Dict[str, Any]]:
    """按关键词在本机素材库里找视频。

    匹配规则：路径里出现全部关键词（大小写不敏感）→ 命中；query 为空则按修改时间倒序。
    返回 [{"name","path","size","mtime","duration"}]，duration 只用缓存的，不现场 probe。
    """
    roots = roots or video_roots()
    if not roots:
        return []
    toks = _tokens(query)
    hits: List[Dict[str, Any]] = []
    seen = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if Path(fn).suffix.lower() not in VIDEO_EXTS:
                    continue
                seen += 1
                if seen > MAX_WALK_FILES:
                    break
                p = Path(dirpath) / fn
                hay = str(p).lower()
                if toks and not all(t in hay for t in toks):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                hits.append({
                    "name": p.stem, "path": str(p), "size": st.st_size,
                    "mtime": st.st_mtime,
                })
            if seen > MAX_WALK_FILES:
                break

    hits.sort(key=lambda h: (-h["mtime"], h["size"]))
    if toks:
        # 名字命中优先于仅目录命中
        hits.sort(key=lambda h: -sum(t in h["name"].lower() for t in toks))
    return hits[:limit]


async def plan_edl(analysis: VideoAnalysis, instruction: str,
                   max_clips: int = 6) -> Dict[str, Any]:
    """把"剪成什么样"翻译成 EDL。只允许引用已有镜头，时间会被夹到区间内。"""
    from brain.llm import get_llm

    table = "\n".join(
        f'{i + 1}. [{s.start:.1f}-{s.end:.1f}s] {s.description[:70]}'
        for i, s in enumerate(analysis.scenes)
    )
    prompt = (
        f"视频《{analysis.name}》总时长 {analysis.duration:.1f} 秒，逐镜头如下：\n{table}\n\n"
        f"剪辑要求：{instruction}\n\n"
        f"输出 JSON：{{\"title\": \"成片标题\", \"clips\": ["
        f"{{\"index\": 镜头序号, \"start\": 起秒, \"end\": 止秒, \"reason\": \"选这段的理由\"}}]}}\n"
        f"最多 {max_clips} 段。start/end 必须是上面出现过的数字范围内的真实时间，"
        "不要编造镜头；按叙事顺序排列。"
    )
    raw = await get_llm().chat([
        {"role": "system", "content": "你是严谨的视频剪辑助理，只能使用给定的镜头，输出 JSON。"},
        {"role": "user", "content": prompt},
    ], json_mode=True)

    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        data = json.loads(m.group(0)) if m else {}

    clips: List[Dict[str, Any]] = []
    for c in (data.get("clips") or [])[:max_clips]:
        try:
            start = float(c.get("start", 0))
            end = float(c.get("end", 0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, analysis.duration))
        end = max(0.0, min(end, analysis.duration))
        if end - start < 0.3:
            continue
        clips.append({
            "index": int(c.get("index") or 0),
            "start": round(start, 2), "end": round(end, 2),
            "reason": str(c.get("reason") or "")[:80],
        })

    if not clips:
        # 模型没给出可用 EDL 时退化成"每段选一个代表镜头"，保证链路不空转
        clips = [{"index": s.index + 1, "start": s.start, "end": s.end,
                  "reason": "模型未给出有效 EDL，使用镜头原边界"}
                 for s in sorted(analysis.scenes, key=lambda x: -x.score)[:max_clips]]
        clips.sort(key=lambda c: c["start"])

    return {"title": str(data.get("title") or f"{analysis.name} 剪辑")[:60], "clips": clips}


async def render_edl(src: str | Path, edl: Dict[str, Any],
                     out_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """按 EDL 用 ffmpeg 裁剪并拼接。返回 {"success","output","duration","clips"}。"""
    p = resolve_video(src)
    clips = [c for c in (edl.get("clips") or []) if c.get("end", 0) > c.get("start", 0)]
    if not clips:
        return {"success": False, "error": "EDL 里没有有效片段"}

    info = await asyncio.to_thread(probe, p)
    has_audio = bool(info.get("has_audio"))

    if out_path is None:
        out_dir = _data_dir() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r'[\\/:*?"<>|]', "_", p.stem)[:40]
        out_path = out_dir / f"{safe}_cut_{int(time.time())}.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    chains: List[str] = []
    for i, c in enumerate(clips):
        s, e = float(c["start"]), float(c["end"])
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2[v{i}]"
        )
        if has_audio:
            parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
            chains.append(f"[v{i}][a{i}]")
        else:
            chains.append(f"[v{i}]")
    n = len(clips)
    if has_audio:
        parts.append("".join(chains) + f"concat=n={n}:v=1:a=1[v][a]")
    else:
        parts.append("".join(chains) + f"concat=n={n}:v=1:a=0[v]")

    cmd = [
        _bin("ffmpeg"), "-v", "error", "-y", "-i", str(p),
        "-filter_complex", ";".join(parts),
        "-map", "[v]",
    ]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]

    logger.info(f"[Edit] 渲染 {n} 段 → {out_path}")
    r = await asyncio.to_thread(
        lambda: subprocess.run(cmd, capture_output=True, timeout=900)
    )
    if r.returncode != 0 or not out_path.exists():
        err = (r.stderr or b"").decode("utf-8", "replace")[:400]
        logger.error(f"[Edit] 渲染失败: {err}")
        return {"success": False, "error": err or "ffmpeg 渲染失败"}

    total = sum(float(c["end"]) - float(c["start"]) for c in clips)
    try:
        from brain import atg
        atg.observe(f"在剪视频《{p.stem}》", source="video")
    except Exception:
        pass
    try:
        from brain.tfgr import get_tfgr
        get_tfgr().record("video.edit",
                          f"剪《{p.stem}》→ {n} 段 {total:.0f}s：{edl.get('title', '')}",
                          source="video", entities=[p.stem],
                          refs=[str(out_path)], meta={"clips": len(clips)})
    except Exception:
        pass

    return {
        "success": True, "output": str(out_path), "clips": n,
        "duration": round(total, 2), "title": edl.get("title", ""),
    }


async def edit_video(path: str | Path, instruction: str,
                     max_scenes: int = 14, max_clips: int = 6,
                     verify: bool = True, retries: int = 1) -> Dict[str, Any]:
    """一步到位：分析 → 规划 EDL → 渲染 → 自检（不通过则带着数值证据重剪一次）。"""
    analysis = await analyze_video(path, max_scenes=max_scenes)
    edl = await plan_edl(analysis, instruction, max_clips=max_clips)
    result = await render_edl(analysis.path, edl)
    result["edl"] = edl
    result["analysis"] = analysis.brief(max_scenes=8)
    result["attempts"] = 1
    if not verify or not result.get("success"):
        return result

    from brain.video_verify import brief, verify_render

    v = verify_render(analysis.path, edl, result["output"])
    result["verification"] = v
    while not v["pass"] and result["attempts"] <= retries:
        result["attempts"] += 1
        logger.info(f"[Edit] 自检未过（{brief(v)}），第 {result['attempts']} 次重剪")
        edl = await plan_edl(
            analysis,
            f"{instruction}\n\n上一次剪辑自检没通过：{brief(v)}。请修正 start/end 后重新输出 EDL，"
            "只允许使用真实镜头边界。",
            max_clips=max_clips,
        )
        nxt = await render_edl(analysis.path, edl)
        result = {**nxt, "edl": edl, "analysis": result["analysis"],
                  "attempts": result["attempts"]}
        if not nxt.get("success"):
            return result
        v = verify_render(analysis.path, edl, nxt["output"])
        result["verification"] = v
    return result
