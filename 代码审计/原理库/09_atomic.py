"""闭环 09：原子写（Windows 安全版）

跑：python 09_atomic.py
对应源码：api/server.py::_safe_write_json

关键点：uuid 临时文件 + os.replace + 有限重试。
注意：本文件的 time.sleep 是安全的（同步函数）；
      同样的代码放进 async 路由就会阻塞事件循环。
"""
import json
import os
import time
import uuid
from pathlib import Path


def safe_write(path: Path, data, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    last_err = None
    for i in range(retries):
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            try:
                os.unlink(tmp)
            except Exception:
                pass
            if i < retries - 1:
                time.sleep(min(1.0, 0.1 * (2 ** i)))
    raise last_err


def load(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


if __name__ == "__main__":
    p = Path("_demo_atomic.json")
    safe_write(p, {"你好": "世界", "n": 1})
    print("写后读:", load(p))
    safe_write(p, {"你好": "世界", "n": 2})
    print("覆盖后读:", load(p))
    p.unlink()
    print("中文/覆盖/清理 都 OK")
