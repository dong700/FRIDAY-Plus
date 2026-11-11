"""剪辑结果自检（2026-04-04）

工作流里「自己看效果」这一步：剪完之后不去问模型「你看对不对」，而是拿数值证据比。
每一项都给 expected / actual，模型只需读差值就能决定改哪里。

    verify_render(src, edl, out) -> {"pass": bool, "items": [...], "warnings": [...]}

七项检查：
    1. 产物存在且非空
    2. 时长：|实测 − Σ(clip 时长)| ≤ tol
    3. 边界：每个 clip 都落在源视频内，end > start
    4. 内容对齐：每个 clip 的抽样帧与产物对应位置帧的 SSIM（证明没剪错段）
    5. 段数：产物场景切分数 ≈ EDL 段数
    6. 音轨：有/无与源视频一致
    7. 配音空间：每段 ≥ min_clip_s（人配音读得完；不达标只警告）
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from brain.video_analysis import _bin, _sample_frames, detect_scenes, probe, resolve_video

GW, GH = 96, 54          # SSIM 比对分辨率（够看结构，不算像素噪声）


def _grab_gray(path: Path, ts: float) -> Optional[np.ndarray]:
    r = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-ss", f"{max(0.0, ts):.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={GW}:{GH}:flags=area,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=120,
    )
    buf = r.stdout
    if r.returncode != 0 or len(buf) < GW * GH:
        return None
    return np.frombuffer(buf[:GW * GH], dtype=np.uint8).reshape(GH, GW)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """灰度 SSIM（7×7 均值窗），返回 [0,1]。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu1 = cv2.blur(a, (7, 7))
    mu2 = cv2.blur(b, (7, 7))
    s11 = cv2.blur(a * a, (7, 7)) - mu1 * mu1
    s22 = cv2.blur(b * b, (7, 7)) - mu2 * mu2
    s12 = cv2.blur(a * b, (7, 7)) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 * mu1 + mu2 * mu2 + c1) * (s11 + s22 + c2)
    return float(np.mean(num / den))


def _item(name: str, expected, actual, ok: bool, level: str = "fail") -> Dict[str, Any]:
    return {"name": name, "expected": expected, "actual": actual, "ok": bool(ok), "level": level}


def verify_render(src: str | Path, edl: Dict[str, Any], out: str | Path,
                  tol: float = 0.20, min_clip_s: float = 2.5,
                  ssim_min: float = 0.85, want_audio: Optional[bool] = None) -> Dict[str, Any]:
    """把剪辑结果和 EDL 逐项对上。绝不抛异常：任何一步失败都变成一条 fail 项。"""
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}
    try:
        src_p = resolve_video(src)
        out_p = Path(out)
    except Exception as e:                                   # noqa: BLE001
        return {"pass": False, "items": [_item("路径", "可解析", str(e), False)],
                "warnings": [], "evidence": {}}

    clips = [c for c in (edl.get("clips") or []) if c.get("end", 0) > c.get("start", 0)]
    expect_s = sum(float(c["end"]) - float(c["start"]) for c in clips)

    # 1. 存在
    ok_file = out_p.exists() and out_p.stat().st_size > 10_000
    items.append(_item("产物存在", "文件 >10KB", out_p.stat().st_size if out_p.exists() else 0, ok_file))
    if not ok_file:
        return {"pass": False, "items": items, "warnings": warnings, "evidence": evidence}

    try:
        si = probe(src_p)
        oi = probe(out_p)
    except Exception as e:                                   # noqa: BLE001
        items.append(_item("ffprobe", "可读", str(e)[:80], False))
        return {"pass": False, "items": items, "warnings": warnings, "evidence": evidence}

    # 2. 时长
    dur_err = abs(oi["duration"] - expect_s)
    items.append(_item("时长误差(s)", f"≤{tol}", round(dur_err, 3), dur_err <= tol))

    # 3. 边界合法
    bad = [c for c in clips
           if not (0 <= float(c["start"]) < float(c["end"]) <= si["duration"] + 1e-6)]
    items.append(_item("边界合法", f"全部在 0~{si['duration']:.1f}s", len(bad), len(bad) == 0))

    # 4. 内容对齐（逐个 clip 抽 2 点比对）
    scores: List[float] = []
    acc = 0.0
    for c in clips:
        s, e = float(c["start"]), float(c["end"])
        seg = e - s
        for frac in (0.25, 0.75):
            a = _grab_gray(src_p, s + seg * frac)
            b = _grab_gray(out_p, acc + seg * frac)
            if a is not None and b is not None:
                scores.append(ssim(a, b))
        acc += seg
    if scores:
        worst = min(scores)
        med = float(np.median(scores))
        evidence["ssim"] = {"min": round(worst, 3), "median": round(med, 3), "n": len(scores)}
        bad_n = sum(1 for v in scores if v < ssim_min)
        items.append(_item("内容对齐(SSIM中位)", f"≥{ssim_min}", round(med, 3), med >= ssim_min))
        items.append(_item("错位采样点", "==0", bad_n, bad_n == 0, level="warn"))
    else:
        items.append(_item("内容对齐(SSIM中位)", f"≥{ssim_min}", "无可用帧", False))

    # 5. 段数（场景切分数）
    try:
        frames, fps = _sample_frames(out_p, oi["duration"])
        scenes = detect_scenes(frames, fps, oi["duration"])
        n_out = len(scenes)
        diff = abs(n_out - len(clips))
        items.append(_item("段数", f"{len(clips)}(±1)", n_out, diff <= 1, level="warn"))
        evidence["scenes_out"] = n_out
    except Exception as e:                                   # noqa: BLE001
        warnings.append(f"段数检查跳过：{e}")

    # 6. 音轨一致
    items.append(_item("音轨一致", si["has_audio"], oi["has_audio"], oi["has_audio"] == si["has_audio"],
                       level="warn"))

    # 7. 完整性：整片必须能解码到底，包数也得够（faststart 的坏文件 ffprobe 照样能读出时长）
    dec = subprocess.run([_bin("ffmpeg"), "-v", "error", "-i", str(out_p), "-f", "null", "-"],
                         capture_output=True, timeout=300)
    dec_err = (dec.stderr or b"").decode("utf-8", "replace").strip()
    items.append(_item("整片解码", "无错误退出", dec_err[:60] or "ok", dec.returncode == 0 and not dec_err))
    try:
        pr = subprocess.run(
            [_bin("ffprobe"), "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "default=nw=1:nk=1", str(out_p)],
            capture_output=True, text=True, timeout=120,
        )
        n_pk = int((pr.stdout or "0").strip() or 0)
        want = max(1, int(oi["duration"] * max(1.0, oi["fps"])))
        ratio = n_pk / want
        evidence["packets"] = {"got": n_pk, "want": want}
        items.append(_item("视频包数比", "≥0.90", round(ratio, 2), ratio >= 0.90))
    except Exception as e:                                   # noqa: BLE001
        warnings.append(f"包数检查跳过：{e}")

    # 8. 配音空间（人要配音，每段得读得完）
    shorts = [round(float(c["end"]) - float(c["start"]), 2) for c in clips
              if float(c["end"]) - float(c["start"]) < min_clip_s]
    if shorts:
        warnings.append(f"{len(shorts)} 段短于 {min_clip_s}s（{shorts}），人配音可能读不完")
    items.append(_item("配音空间", f"每段≥{min_clip_s}s", f"短段 {len(shorts)} 个", not shorts,
                       level="warn"))

    fail = [i for i in items if not i["ok"] and i["level"] == "fail"]
    return {
        "pass": not fail,
        "items": items,
        "warnings": warnings,
        "evidence": evidence,
        "failed": [i["name"] for i in fail],
    }


def brief(v: Dict[str, Any]) -> str:
    """压成一行，方便回灌给模型重规划。"""
    if v.get("pass"):
        return "自检通过：" + "；".join(
            f"{i['name']}={i['actual']}" for i in v["items"] if i["name"] in
            ("时长误差(s)", "内容对齐(SSIM中位)", "段数"))
    bad = [f"{i['name']} 期望{i['expected']} 实际{i['actual']}"
           for i in v.get("items", []) if not i["ok"] and i["level"] == "fail"]
    return "自检失败：" + "；".join(bad) + ("；" + "；".join(v["warnings"]) if v.get("warnings") else "")
