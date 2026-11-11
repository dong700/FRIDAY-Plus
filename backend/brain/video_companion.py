"""一起看视频（2026-11-04）——桌宠陪着看，随时答。

和 video_watch 的区别：
    video_watch  事后把整片看完，落时间线，给剪辑用。
    这里         跟着播放进度实时看。播放头往前走，它才往前看；
                 你问「刚才那个老头说的啥」，它把最近这一段九宫格交给眼睛，
                 再用 ATG 里已经攒下的叙事回答，全程带你原来的时间码。

为什么这样省钱：
    实时看≠实时问模型。播放时只做解码+切窗（4Hz 解码、每窗一张九宫格，不起模型）；
    描述是按需的：你问，或者攒够 45 秒它自己说一句。一集 45 分钟剧大概是
    30 次模型调用，不是 900 次。

为什么能做「刚才」：
    窗口一落就记 {t0,t1,desc}，并同步一份到 ATG 的叙事里。
    所以它答的时候知道「你说的是第几分钟」，而不是只看见当前这一帧。

播放器同步：
    开始给个 from_s + speed 就够用（自己跟着走）。真接了播放器，只要在
    进度条变化时调 sync(seconds)，内部会重开解码器对齐，不需要别的。
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.frame_pipeline import ChunkPlanner, RawFrame, TIER_CFG, compose_grid, _Scorer
from brain.video_analysis import _bin, _mmss, probe, resolve_video
from brain.video_watch import Window, Watch, _iter_frames, describe
from core.logger import get_logger

logger = get_logger("brain.video_companion")

FPS = 3.0                     # 解码帧率（够抓切镜，又不至于把 CPU 吃满）
AUTO_NARRATE_S = 45.0         # 攒这么久，它自己说一句"现在在干嘛"
KEEP_WINDOWS = 40             # 只在内存里留最近 40 个窗口
GRID_DIR = Path("data") / "video_watch" / "companion"

_ASK_SYSTEM = """你是 FRIDAY，坐在用户旁边一起看视频。回答要像人一起看电视时随口说的：
一两句话，先说结论，必要时带时间码（mm:ss）。

给你的信息分三块，用的时候别混：
- 「最近看到的」＝这一段画面的描述（画面里有什么、在做什么）
- 「此刻画面里的人」＝只有真的在认人时才会给。p1/p2 是临时编号，括号里是认得的名字。
  没有这一块，就说明此刻没在认人 —— 不要自己编 p1/p2，也别假装认识谁。
- 「他以前」＝这个人的旧事（跨集攒下来的）。没有这一块，或者写着"没记过"，就照实说没记过。

问"这人是谁""刚才那个人"→ 用「此刻画面里的人」答；那里没有，就说"现在没在认人，只看到画面"。
问"他以前干过什么"→ 用「他以前」答；没有就说"没记过"。
画面描述和人物信息都没有的，才说"这段看不出来"。不要编造剧情，不用敬语，不要重复用户的问题。"""
_NO_QUESTION = "用一句话说现在画面里在干什么，不超过 30 字。"


class VideoCompanion:
    """一路实时观影的会话。单例：桌宠只有一个屏幕，但看片子只需要一路。"""

    _instance: Optional["VideoCompanion"] = None

    def __init__(self) -> None:
        self.path: str = ""
        self.name: str = ""
        self.duration: float = 0.0
        self.speed: float = 1.0
        self.windows: List[Window] = []
        self._start_s = 0.0
        self._wall0 = 0.0
        self._n = 0                 # 已读帧序号（用来算下一帧的视频时间）
        self._planner = ChunkPlanner()
        self._tc = self._planner.tiers
        self._scorer = _Scorer()
        self._gen = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_keep = -1e9
        self._last_rank = 0
        self._last_narrate = 0.0
        self._warm_task: Optional[asyncio.Task] = None
        self.frames = 0
        self.started_at = 0.0

    # ---------------- 对外 ----------------

    @property
    def playhead(self) -> float:
        if not self.started_at:
            return self._start_s
        return round(self._start_s + (time.time() - self._wall0) * self.speed, 2)

    def start(self, path: str | Path, from_s: float = 0.0,
              speed: float = 1.0, fps: float = FPS) -> Dict[str, Any]:
        self.stop()
        p = resolve_video(path)
        info = probe(p)
        self.path, self.name = str(p), p.stem
        self.duration = float(info["duration"])
        self.speed = max(0.1, float(speed))
        self.frames = 0
        self.started_at = time.time()
        self._rewind(from_s, fps)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="video-companion")
        self._thread.start()
        try:
            self._warm_task = asyncio.get_running_loop().create_task(self._warm_loop())
        except RuntimeError:
            self._warm_task = None
        logger.info(f"[Companion] 开始跟看《{self.name}》 from={from_s}s speed={speed}")
        return {"ok": True, "video": self.name, "duration": self.duration,
                "from": from_s, "playhead": self.playhead}

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._warm_task is not None:
            self._warm_task.cancel()
            self._warm_task = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        try:
            if self._gen is not None:
                self._gen.close()
        except Exception:
            pass
        self._gen = None
        self.started_at = 0.0
        return {"ok": True, "windows": len(self.windows)}

    def sync(self, seconds: float) -> Dict[str, Any]:
        """播放器进度变了：重开解码器对齐（-ss 是关键帧跳，很快）。"""
        if not self.path:
            return {"ok": False, "error": "还没开始看"}
        fps = FPS
        self._rewind(max(0.0, float(seconds)), fps)
        return {"ok": True, "playhead": self.playhead}

    def status(self) -> Dict[str, Any]:
        return {
            "watching": bool(self.started_at),
            "video": self.name, "duration": self.duration,
            "playhead": self.playhead, "speed": self.speed,
            "windows": len(self.windows), "frames": self.frames,
            "tier": self._tc.tier, "flaps": self._tc.flaps,
            "last": self._recent(1)[0].summary if self.windows else "",
        }

    def recent(self, n: int = 3) -> List[Dict[str, Any]]:
        return [{"t0": w.t0, "t1": w.t1, "what": w.summary or w.desc,
                 "described": bool(w.summary)} for w in self._recent(n)]

    async def moment(self, question: str = "", look_back: float = 75.0) -> Dict[str, Any]:
        """现在这一刻 + 回头 look_back 秒，凑成上下文；有问题就答，没问题就汇报。

        上下文三块：画面（这一段的九宫格描述）、人物（画面上是谁 + 他以前的事）、
        播放位置。缺人物那块，用户问「这人是谁」就只能答"没交代身份"。
        """
        if not self.path:
            return {"ok": False, "error": "还没开始看视频"}
        t = self.playhead
        picked = [w for w in self.windows if w.t1 >= t - look_back]
        if not picked:
            picked = self._recent(2)
        await self._ensure_described(picked)

        ctx = "\n".join(
            f"[{_mmss(w.t0)}-{_mmss(w.t1)}] {w.summary or w.desc[:80]}" for w in picked)
        people = self._people_ctx()
        head = (f"《{self.name}》，现在播放到 {_mmss(t)}（全长 {_mmss(self.duration)}）。\n"
                f"最近看到的：\n{ctx}\n" + (f"{people}\n" if people else ""))

        from brain.llm import get_llm
        try:
            ans = await get_llm().chat([
                {"role": "system", "content": _ASK_SYSTEM},
                {"role": "user", "content": head + f"\n用户问：{question or _NO_QUESTION}"},
            ])
        except Exception as e:
            logger.error(f"[Companion] 回答失败: {e}", exc_info=True)
            return {"ok": False, "error": str(e), "windows": ctx}
        return {"ok": True, "playhead": t, "answer": (ans or "").strip(),
                "windows": ctx, "people": people,
                "window": [picked[-1].t0, picked[-1].t1]}

    def _people_ctx(self, window_s: float = 600) -> str:
        """画面上是谁 + 这个人的旧事。人物层没接到跟踪器时这里是空的，不影响回答。"""
        try:
            from brain.people import get_people
            book = get_people()
            seen = book.seen_now(window_s)
            if not seen:
                return ""
            line = "此刻画面里的人：" + "；".join(
                f"{p['label'].split('@')[0]}" + (f"（{p['name']}）" if p["name"] else "（还没名字）")
                for p in seen[:4])
            extra = []
            for p in seen[:2]:
                if not p["name"]:
                    continue
                prof = book.profile(p["name"], 6)
                if prof.get("ok") and prof.get("summary"):
                    extra.append(prof["summary"].replace("\n", " ")[:220])
            return line + ("\n他以前：\n" + "\n".join(extra) if extra else "")
        except Exception:
            return ""

    # ---------------- 内部 ----------------

    def _recent(self, n: int) -> List[Window]:
        with self._lock:
            return list(self.windows[-n:])

    def _rewind(self, from_s: float, fps: float) -> None:
        try:
            if self._gen is not None:
                self._gen.close()
        except Exception:
            pass
        self._start_s = max(0.0, min(from_s, max(0.0, self.duration - 1)))
        self._wall0 = time.time()
        self._n = 0
        self._planner = ChunkPlanner()
        self._tc = self._planner.tiers
        self._scorer = _Scorer()
        self._last_keep = -1e9
        self._last_rank = 0
        with self._lock:
            self.windows = []
        self._gen = _iter_frames(Path(self.path), self._start_s, self.duration, fps)

    def _flush(self, chunk) -> None:
        sheet = compose_grid(chunk, TIER_CFG[chunk.tier]["send_k"])
        if not sheet:
            return
        GRID_DIR.mkdir(parents=True, exist_ok=True)
        idx = len(self.windows)
        gp = GRID_DIR / f"{Path(self.path).stem}_{idx:03d}.jpg"
        try:
            import base64
            gp.write_bytes(base64.b64decode(sheet.b64))
        except Exception:
            gp = None
        win = Window(index=idx, t0=round(chunk.t0 - self._wall0 + self._start_s, 2),
                     t1=round(chunk.t1 - self._wall0 + self._start_s, 2),
                     tier=chunk.tier, reason=chunk.reason, n=sheet.n,
                     total=len(chunk.frames), peak=chunk.peak,
                     grid=str(gp) if gp else "")
        with self._lock:
            self.windows.append(win)
            if len(self.windows) > KEEP_WINDOWS:
                self.windows = self.windows[-KEEP_WINDOWS:]

    def _loop(self) -> None:
        """跟着播放头走：只有播放头追上下一帧才读它。暂停=不读=解码器自动停。"""
        while not self._stop.is_set():
            target = self.playhead
            read_one = False
            while self._n / FPS + self._start_s <= target:
                if self._stop.is_set():
                    return
                try:
                    ts_v, img = next(self._gen)
                except StopIteration:
                    self._stop.set()
                    return
                except Exception as e:
                    logger.warning(f"[Companion] 解码结束: {e}")
                    self._stop.set()
                    return
                read_one = True
                self.frames = self._n = self._n + 1
                score = self._scorer.score(img)
                now = self._wall0 + (ts_v - self._start_s)
                tier = self._tc.update(score, now)
                rank = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}[tier]
                if (now - self._last_keep) >= 1.0 / max(0.05, self._tc.fps) or rank > self._last_rank:
                    ch = self._planner.feed(RawFrame(img, now, score))
                    if ch:
                        self._flush(ch)
                    self._last_keep = now
                self._last_rank = rank
            if not read_one:
                time.sleep(0.15)

    async def _ensure_described(self, wins: List[Window]) -> None:
        need = [w for w in wins if not w.summary]
        if not need:
            return
        tmp = Watch(path=self.path, name=self.name, duration=self.duration,
                    t_start=need[0].t0, t_end=need[-1].t1)
        tmp.windows = need
        await describe(tmp, hint="用户正在看的片子")
        # describe 就地写进 Window，上面 need 是同一批对象

    async def _warm_loop(self) -> None:
        """边看边预热：把刚落下的窗口提前描述掉，等你问的时候不用等模型。

        这不是"每帧都调模型"——一轮只补最近 1~2 个没描述的窗口，大约 20~30 秒一次。
        目的是把问答时延从 5~9 秒压到 2 秒左右，同时让 ATG 里始终有"刚才在干嘛"。
        """
        while not self._stop.is_set():
            await asyncio.sleep(6)
            if not self.started_at:
                return
            try:
                todo = [w for w in self._recent(4) if not w.summary]
                if todo:
                    await self._ensure_described(todo[-2:])
                if time.time() - self._last_narrate >= AUTO_NARRATE_S:
                    ws = self._recent(2)
                    last = next((w for w in reversed(ws) if w.summary), None)
                    if last:
                        self._last_narrate = time.time()
                        try:
                            from brain import atg
                            atg.observe(f"在看《{self.name}》{_mmss(last.t0)}："
                                        f"{last.summary[:70]}", source="video")
                        except Exception:
                            pass
            except asyncio.CancelledError:
                return
            except Exception:
                logger.warning("[Companion] 预热失败", exc_info=True)


def get_companion() -> VideoCompanion:
    if VideoCompanion._instance is None:
        VideoCompanion._instance = VideoCompanion()
    return VideoCompanion._instance


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="陪着看视频（命令行自测）")
    ap.add_argument("video")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wait", type=float, default=8.0, help="先跟着看几秒")
    ap.add_argument("--ask", help="看完问它一句")
    a = ap.parse_args(argv)

    c = get_companion()
    print(c.start(a.video, a.t0, a.speed))
    time.sleep(max(1.0, a.wait))
    print(json.dumps(c.status(), ensure_ascii=False, indent=1))
    for r in c.recent(4):
        print(f"  {_mmss(r['t0'])}-{_mmss(r['t1'])}  {r['what'][:60]}")
    if a.ask:
        r = asyncio.run(c.moment(a.ask))
        print("\n答：", r.get("answer") or r.get("windows"))
    c.stop()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
