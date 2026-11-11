"""操作电脑 / 模拟 UI 输入（2026-11-04）

全部走 Win32 SendInput + SetCursorPos，不依赖 pyautogui（本机没装）。
原则：
    1. 每一步都留痕（log.info），出问题能回放做了什么。
    2. 坐标越界直接拒绝，不做"猜你想点哪儿"。
    3. 权限开关 FRIDAY_ALLOW_INPUT=0 可以一键掐掉（默认开，用户明确给了这个权限）。
    4. 打字用 KEYEVENTF_UNICODE，中文/emoji 都能打，不靠剪贴板。

工具层（brain/tools.py）只暴露一个 computer_use，避免工具面板被十几个动作塞满。
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from typing import Any, Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger("brain.computer")

ENABLED = os.getenv("FRIDAY_ALLOW_INPUT", "1") != "0"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ---- SendInput 结构（64 位下 ULONG_PTR 必须是 8 字节，否则 SendInput 返回 0）----
ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_EXTENDEDKEY = 0x0001
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000

MOUSE_DOWN = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}
MOUSE_UP = {"left": 0x0004, "right": 0x0010, "middle": 0x0040}

VK = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def _send(*inputs: INPUT) -> bool:
    n = user32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))
    return n == len(inputs)


def _key(vk: int, up: bool = False) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD,
                 ki=KEYBDINPUT(wVk=vk, wScan=0,
                               dwFlags=KEYEVENTF_KEYUP if up else 0,
                               time=0, dwExtraInfo=0))


def _uni(ch: str, up: bool = False) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD,
                 ki=KEYBDINPUT(wVk=0, wScan=ord(ch),
                               dwFlags=KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                               time=0, dwExtraInfo=0))


def _mouse(flags: int, data: int = 0) -> INPUT:
    return INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=0, dy=0, mouseData=data,
                                                 dwFlags=flags, time=0, dwExtraInfo=0))


def _guard(action: str) -> Optional[Dict[str, Any]]:
    if not ENABLED:
        return {"ok": False, "error": "输入权限被关掉了（FRIDAY_ALLOW_INPUT=0）"}
    return None


def screen_size() -> Tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _clamp(x: Optional[float], y: Optional[float]) -> Tuple[int, int]:
    w, h = screen_size()
    if x is None or y is None:
        p = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(p))
        x = p.x if x is None else x
        y = p.y if y is None else y
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"坐标 ({x},{y}) 超出屏幕 {w}x{h}")
    return x, y


def move(x: Optional[float] = None, y: Optional[float] = None,
         duration: float = 0.0) -> Dict[str, Any]:
    """移动鼠标。duration>0 会平滑移动（动画里看起来像有人在动）。"""
    bad = _guard("move")
    if bad:
        return bad
    tx, ty = _clamp(x, y)
    if duration <= 0:
        user32.SetCursorPos(tx, ty)
    else:
        p = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(p))
        steps = max(6, int(duration * 60))
        for i in range(1, steps + 1):
            user32.SetCursorPos(int(p.x + (tx - p.x) * i / steps),
                                int(p.y + (ty - p.y) * i / steps))
            time.sleep(duration / steps)
    logger.info(f"[Computer] move → ({tx},{ty})")
    return {"ok": True, "x": tx, "y": ty}


def click(x: Optional[float] = None, y: Optional[float] = None,
          button: str = "left", clicks: int = 1) -> Dict[str, Any]:
    """点击。不给坐标就在当前位置点。"""
    bad = _guard("click")
    if bad:
        return bad
    button = (button or "left").lower()
    if button not in MOUSE_DOWN:
        return {"ok": False, "error": f"不认识的键: {button}"}
    tx, ty = _clamp(x, y)
    user32.SetCursorPos(tx, ty)
    time.sleep(0.03)
    for i in range(max(1, min(3, int(clicks)))):
        _send(_mouse(MOUSE_DOWN[button]), _mouse(MOUSE_UP[button]))
        if i + 1 < clicks:
            time.sleep(0.06)
    logger.info(f"[Computer] click {button}x{clicks} @({tx},{ty})")
    return {"ok": True, "x": tx, "y": ty, "button": button, "clicks": clicks}


def drag(x1: float, y1: float, x2: float, y2: float,
         duration: float = 0.4) -> Dict[str, Any]:
    bad = _guard("drag")
    if bad:
        return bad
    ax, ay = _clamp(x1, y1)
    bx, by = _clamp(x2, y2)
    user32.SetCursorPos(ax, ay)
    time.sleep(0.05)
    _send(_mouse(MOUSE_DOWN["left"]))
    steps = max(8, int(duration * 60))
    for i in range(1, steps + 1):
        user32.SetCursorPos(int(ax + (bx - ax) * i / steps),
                            int(ay + (by - ay) * i / steps))
        time.sleep(duration / steps)
    _send(_mouse(MOUSE_UP["left"]))
    logger.info(f"[Computer] drag ({ax},{ay})→({bx},{by})")
    return {"ok": True, "from": [ax, ay], "to": [bx, by]}


def scroll(amount: int, x: Optional[float] = None, y: Optional[float] = None,
           horizontal: bool = False) -> Dict[str, Any]:
    """滚轮。amount 正数向下/向右。"""
    bad = _guard("scroll")
    if bad:
        return bad
    tx, ty = _clamp(x, y)
    user32.SetCursorPos(tx, ty)
    time.sleep(0.02)
    _send(_mouse(MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL,
                 int(amount) * 120))
    logger.info(f"[Computer] scroll {amount} @({tx},{ty})")
    return {"ok": True, "amount": amount}


def type_text(text: str, interval: float = 0.012) -> Dict[str, Any]:
    """打字：逐字符走 KEYEVENTF_UNICODE，中文/符号都能进。"""
    bad = _guard("type")
    if bad:
        return bad
    if not text:
        return {"ok": True, "chars": 0}
    buf = []
    for ch in text:
        if ch == "\n":
            buf.extend([_key(VK["enter"]), _key(VK["enter"], up=True)])
            continue
        buf.extend([_uni(ch), _uni(ch, up=True)])
    # 一次 SendInput 最多几百个，分批发，避免被系统丢
    sent = 0
    for i in range(0, len(buf), 200):
        if not _send(*buf[i:i + 200]):
            return {"ok": False, "error": f"SendInput 被拒（已发 {sent} 个键）"}
        sent += len(buf[i:i + 200])
        time.sleep(interval * min(60, len(buf[i:i + 200])))
    logger.info(f"[Computer] type {len(text)} 字")
    return {"ok": True, "chars": len(text)}


def press(key: str) -> Dict[str, Any]:
    """按一个键（enter/esc/tab/上下左右/f1-f12/字母数字）。"""
    bad = _guard("press")
    if bad:
        return bad
    k = key.strip().lower()
    vk = VK.get(k)
    if vk is None:
        if len(k) == 1:
            vk = ord(k.upper())
        else:
            return {"ok": False, "error": f"不认识的键: {key}"}
    _send(_key(vk), _key(vk, up=True))
    logger.info(f"[Computer] press {key}")
    return {"ok": True, "key": key}


def hotkey(combo: str) -> Dict[str, Any]:
    """组合键，例如 ctrl+shift+s、alt+tab、win+r。"""
    bad = _guard("hotkey")
    if bad:
        return bad
    keys = [k.strip().lower() for k in combo.replace(" ", "").split("+") if k.strip()]
    vks = []
    for k in keys:
        vk = VK.get(k)
        if vk is None:
            if len(k) == 1:
                vk = ord(k.upper())
            else:
                return {"ok": False, "error": f"不认识的键: {k}"}
        vks.append(vk)
    for vk in vks:
        _send(_key(vk))
        time.sleep(0.02)
    for vk in reversed(vks):
        _send(_key(vk, up=True))
        time.sleep(0.02)
    logger.info(f"[Computer] hotkey {combo}")
    return {"ok": True, "combo": combo}


def list_windows(limit: int = 30) -> List[Dict[str, Any]]:
    """列出可见的顶层窗口（标题 + 位置），用来定位"点哪儿"。"""
    import win32gui

    out: List[Dict[str, Any]] = []

    def _cb(hwnd, _):
        if len(out) >= limit or not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        rect = win32gui.GetWindowRect(hwnd)
        out.append({"hwnd": hwnd, "title": title[:60],
                    "rect": [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]})

    win32gui.EnumWindows(_cb, None)
    return out


def focus(title: str) -> Dict[str, Any]:
    """按标题片段把窗口提到前台（找不到就返回候选）。"""
    bad = _guard("focus")
    if bad:
        return bad
    import win32gui

    t = (title or "").lower()
    cands = [w for w in list_windows(60) if t and t in w["title"].lower()]
    if not cands:
        return {"ok": False, "error": f"没找到标题含「{title}」的窗口"}
    hwnd = cands[0]["hwnd"]
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        logger.info(f"[Computer] focus「{cands[0]['title']}」")
        return {"ok": True, "window": cands[0]["title"], "hwnd": hwnd}
    except Exception as e:
        return {"ok": False, "error": str(e), "candidates": [c["title"] for c in cands[:5]]}


def screenshot(path: str = "") -> Dict[str, Any]:
    """截一张全屏（给模型看"现在屏幕上是什么"）。"""
    from PIL import ImageGrab
    img = ImageGrab.grab(all_screens=True)
    if path:
        img.save(path)
        return {"ok": True, "path": path, "size": img.size}
    return {"ok": True, "size": img.size, "image": img}


ACTIONS = {
    "move": (move, ("x", "y", "duration")),
    "click": (click, ("x", "y", "button", "clicks")),
    "double_click": (click, ("x", "y")),
    "right_click": (click, ("x", "y")),
    "drag": (drag, ("x1", "y1", "x2", "y2", "duration")),
    "scroll": (scroll, ("amount", "x", "y", "horizontal")),
    "type": (type_text, ("text", "interval")),
    "press": (press, ("key",)),
    "hotkey": (hotkey, ("combo",)),
    "focus": (focus, ("title",)),
}


def run(action: str, **kw) -> Dict[str, Any]:
    """工具层唯一入口：computer_use(action=..., ...)。"""
    key = (action or "").strip().lower()
    item = ACTIONS.get(key)
    if not item:
        return {"ok": False, "error": f"不认识的 action: {action}",
                "actions": list(ACTIONS)}
    fn, names = item
    if key == "double_click":
        kw = {**kw, "clicks": 2}
    if key == "right_click":
        kw = {**kw, "button": "right"}
    args = {k: v for k, v in kw.items() if k in names and v is not None}
    try:
        return fn(**args)
    except Exception as e:
        logger.warning(f"[Computer] {action} 失败: {e}")
        return {"ok": False, "error": str(e)}
