"""FRIDAY Plus 自带后端入口。

FastAPI 应用（api/brain/core/hardware/config）就在 Plus/backend 下，由这里直接启动；
Plus/server.py 只负责静态页面 + /api /ws 反代，上游是本文件起的 127.0.0.1:18080。

用法：
    python run_server.py                 # 127.0.0.1:18080
    FRIDAY_PLUS_BACKEND_PORT=18081 python run_server.py

日志：启动时把 fd 1/2 直接接到 D:/FRIDAY/logs/plus_backend.log（可用 FRIDAY_HOME 改根目录），
不依赖外面 shell 的重定向——Windows 下从 .ps1 重定向长驻 python 偶发「拒绝访问」。
"""
import io
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)
# 控制台是 GBK，中文日志不加这个会崩
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import uvicorn  # noqa: E402


def _bind_console_to_log() -> None:
    """把标准输出/错误接到日志文件（进程级），在任何模块配置 logger 之前调用。"""
    try:
        log_dir = Path(os.getenv("FRIDAY_HOME", "D:/FRIDAY")) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / "plus_backend.log", "a", encoding="utf-8", buffering=1)
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False),
                                      encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False),
                                      encoding="utf-8", line_buffering=True)
    except Exception:
        pass  # 日志起不来不影响服务本身


if __name__ == "__main__":
    _bind_console_to_log()
    from api.server import app, set_server_ref  # 放到重定向之后导入，logger 才会绑到日志文件

    port = int(os.getenv("FRIDAY_PLUS_BACKEND_PORT", "18080"))
    # 用 Server 实例而不是 uvicorn.run()：桌宠菜单的「停止服务」要拿到实例设 should_exit，
    # 否则只能杀进程（之前就是这样，导致停止按钮一直无效）。
    server = uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ws="wsproto",   # websockets 16 与旧 uvicorn 的握手不兼容，统一走 wsproto
    ))
    set_server_ref(server)
    server.run()
