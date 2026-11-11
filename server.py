"""FRIDAY Plus —— 页面服务 + 自有后端的反代/桥接

它只做三件事：
  1. 服务 ui/ 下的四个单文件页面（对话 / ATG / TFGR / 状态）
  2. 把 /api/* 反向代理到 Plus 自带后端（backend/，默认 127.0.0.1:18080）
  3. 把 /ws 跟后端双向桥接（界面和桌宠发来的消息靠这条线同步）

后端由 scripts/start_plus_backend.ps1 启动；没起的时候页面会显示「后端没起」。
Plus 是独立项目，不再依赖任何外部后端目录。

用法：
    python server.py                 # 默认 127.0.0.1:18090
    FRIDAY_PLUS_PORT=18091 python server.py
    FRIDAY_PRO_URL=http://127.0.0.1:18080 python server.py
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

PRO_URL = os.getenv("FRIDAY_PRO_URL", "http://127.0.0.1:18080").rstrip("/")
PRO_WS = os.getenv("FRIDAY_PRO_WS", "ws://127.0.0.1:18080/ws")
UI_DIR = Path(os.getenv("FRIDAY_PLUS_UI_DIR", str(Path(__file__).parent / "ui")))
PORT = int(os.getenv("FRIDAY_PLUS_PORT", "18090"))
PAGES = {"index.html", "atg.html", "tfgr.html", "status.html"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plus")

app = FastAPI(title="FRIDAY Plus")
_client = httpx.AsyncClient(base_url=PRO_URL, timeout=30.0)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{PRO_URL}/api/brain/status")
        return {"ok": True, "upstream": PRO_URL, "upstream_status": r.status_code}
    except Exception as e:
        return {"ok": False, "upstream": PRO_URL, "error": str(e)[:160]}


def _page(name: str):
    if name not in PAGES:
        return None
    f = UI_DIR / name
    return f if f.exists() else None


@app.get("/")
@app.get("/index.html")
async def index_page():
    f = _page("index.html")
    if f:
        return FileResponse(str(f))
    return JSONResponse({"error": f"页面不存在: {UI_DIR / 'index.html'}"}, status_code=404)


@app.get("/{page}")
async def other_page(page: str):
    if page == "ws" or page.startswith("api"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    f = _page(page)
    if f:
        return FileResponse(str(f))
    return JSONResponse({"error": f"页面不存在: {page}"}, status_code=404)


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_api(path: str, request: Request):
    """原样转发到 Pro 后端，前端代码不用区分 Plus / Pro。"""
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "connection")
    }
    try:
        r = await _client.request(
            request.method,
            f"/api/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
    except Exception as e:
        return JSONResponse({"error": f"Pro 后端没起或连不上：{e}"[:200]}, status_code=502)

    out = {
        k: v for k, v in r.headers.items()
        if k.lower() in ("content-type", "cache-control", "content-disposition")
    }
    return Response(content=r.content, status_code=r.status_code, headers=out)


@app.websocket("/ws")
async def ws_bridge(ws: WebSocket):
    await ws.accept()
    try:
        up = await websockets.connect(PRO_WS, max_size=None, ping_interval=30)
    except Exception as e:
        logger.warning(f"[Plus] 连不上 Pro 的 WS：{e}")
        try:
            await ws.close(code=1013)
        except Exception:
            pass
        return

    async def client_to_pro():
        try:
            while True:
                msg = await ws.receive_text()
                await up.send(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            pass

    async def pro_to_client():
        try:
            async for msg in up:
                await ws.send_text(msg)
        except Exception:
            pass

    t1 = asyncio.create_task(client_to_pro())
    t2 = asyncio.create_task(pro_to_client())
    try:
        await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in (t1, t2):
            t.cancel()
        try:
            await up.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    logger.info(f"[Plus] 页面目录 {UI_DIR}")
    logger.info(f"[Plus] 代理 Pro -> {PRO_URL}  (WS {PRO_WS})")
    logger.info(f"[Plus] 打开 http://127.0.0.1:{PORT}/")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
