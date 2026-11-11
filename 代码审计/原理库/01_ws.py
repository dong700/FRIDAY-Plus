"""闭环 01：WebSocket 一问一答（对话骨架）

跑：
    pip install fastapi uvicorn websockets wsproto
    python 01_ws.py
    # 另开一个终端
    python 01_ws.py client

对应源码：backend/api/server.py::websocket_endpoint（477-500）
关键点：uvicorn 必须 ws="wsproto"（websockets 16 + 旧 uvicorn 握手 403）
"""
import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

PORT = int(os.getenv("DEMO_PORT", "18080"))
app = FastAPI()


async def fake_llm(text: str):
    for ch in f"你说的是「{text}」，我收到了。":
        await asyncio.sleep(0.02)
        yield ch


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            ev = json.loads(await sock.receive_text())
            if ev.get("type") != "chat.send":
                continue
            mid = "m1"
            await sock.send_text(json.dumps({"type": "mind.start", "payload": {"id": mid}}))
            buf = []
            async for ch in fake_llm(ev["payload"]["content"]):
                buf.append(ch)
                await sock.send_text(json.dumps(
                    {"type": "mind.delta", "payload": {"id": mid, "text": ch}},
                    ensure_ascii=False))
            await sock.send_text(json.dumps(
                {"type": "mind.end", "payload": {"id": mid, "text": "".join(buf)}},
                ensure_ascii=False))
    except WebSocketDisconnect:
        pass


async def client():
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws_:
        await ws_.send(json.dumps({"type": "chat.send",
                                   "payload": {"content": "在吗"}}))
        while True:
            ev = json.loads(await ws_.recv())
            if ev["type"] == "mind.delta":
                print(ev["payload"]["text"], end="", flush=True)
            elif ev["type"] == "mind.end":
                print("\n--- end ---")
                break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        asyncio.run(client())
    else:
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", ws="wsproto")
