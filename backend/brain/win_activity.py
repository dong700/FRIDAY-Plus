"""前台窗口活动识别（2026-11-03）

为什么需要它：
    看视频 / 写代码 / 打游戏 / 聊天这些事，窗口标题和进程名就能判断，
    完全免费（Win32 调用，毫秒级），不需要任何视觉模型。

用法：
    from brain.win_activity import sense
    sense() -> {"activity": "video", "title": "...", "process": "...", "confident": True}
"""
from __future__ import annotations

import os
from typing import Dict, Optional

# 进程名（小写，不带 .exe）
VIDEO_PROCESSES = {
    "vlc", "potplayermini64", "potplayermini", "mpv", "wmplayer", "video.ui",
    "video", "movies", "mpc-hc64", "mpc-hc", "kmplayer", "qqplayer", "youku",
    "iqiyi", "qqlive", "mgtv", "bilibili",
}
CODING_PROCESSES = {
    "code", "code - insiders", "devenv", "pycharm64", "idea64", "webstorm64",
    "clion64", "goland64", "rider64", "sublime_text", "notepad++", "windowsterminal",
    "powershell", "cmd", "wt", "cursor", "trae", "trae cn", "zsh", "bash",
}
CHAT_PROCESSES = {"wechat", "weixin", "qq", "feishu", "lark", "telegram", "discord", "slack", "dingtalk"}
GAME_PROCESSES = {"steam", "steamwebhelper", "epicgameslauncher", "wegame", "battle.net"}

VIDEO_WORDS = [
    "播放器", "视频", "电影", "电视剧", "番剧", "动漫", "直播", "bilibili", "哔哩哔哩",
    "youtube", "netflix", "爱奇艺", "优酷", "腾讯视频", "芒果tv", "mp4", "mkv", "avi",
    "vlc", "potplayer", "mpv",
]
CODING_WORDS = [".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".c ", ".go", ".rs", "visual studio", "vs code", "pycharm", "idea"]
CHAT_WORDS = ["微信", "wechat", "qq", "飞书", "feishu", "telegram", "discord", "聊天"]
GAME_WORDS = ["游戏", "steam", "英雄联盟", "league of legends", "原神", "genshin", "cs2", "dota"]


def _foreground() -> Optional[Dict[str, str]]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        import ctypes.wintypes as wt

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value or ""

        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = ""
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if handle:
            name = ctypes.create_unicode_buffer(1024)
            size = wt.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
                proc = os.path.basename(name.value).lower()
                if proc.endswith(".exe"):
                    proc = proc[:-4]
            kernel32.CloseHandle(handle)
        return {"title": title, "process": proc}
    except Exception:
        return None


def _hit(text: str, words) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def idle_seconds() -> float:
    """系统级「上次键鼠输入距今多少秒」（GetLastInputInfo，不依赖前端上报）。"""
    if os.name != "nt":
        return 0.0
    try:
        import ctypes
        import ctypes.wintypes as wt

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(lii)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            ms = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return max(0.0, ms / 1000.0)
    except Exception:
        pass
    return 0.0


def classify(title: str, process: str) -> str:
    t = f"{title} {process}".lower()
    p = process.lower()

    if p in VIDEO_PROCESSES or _hit(t, VIDEO_WORDS):
        return "video"
    if p in GAME_PROCESSES or _hit(t, GAME_WORDS):
        return "game"
    if p in CHAT_PROCESSES or _hit(t, CHAT_WORDS):
        return "chat"
    if p in CODING_PROCESSES or _hit(t, CODING_WORDS):
        return "coding"
    if p in {"chrome", "msedge", "firefox", "360se", "360chrome", "sogouexplorer"}:
        return "browse"
    return "other"


def sense() -> Dict[str, object]:
    """读当前前台窗口并给出一句活动判断。

    confident=True 表示标题/进程已经明确指向某一类活动（例如播放器），
    此时不必再花一次视觉调用。
    """
    info = _foreground()
    if not info:
        return {"activity": "other", "title": "", "process": "", "confident": False}
    activity = classify(info["title"], info["process"])
    confident = activity in ("video", "coding", "game", "chat", "browse")
    return {
        "activity": activity,
        "title": info["title"][:60],
        "process": info["process"],
        "confident": confident,
    }
