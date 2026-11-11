"""闭环 08：工具调用（模型说「调工具」，你执行并回填）

跑：python 08_toolcall.py
对应源码：brain/llm.py::chat_with_tools + brain/tools.py::BrainTools

要点：schema 与执行表必须同名；arguments 可能不是合法 JSON；未知工具必须兜住。
"""
import json

TOOLS = {
    "get_disk_free": lambda: {"free_gb": 123.4},
    "open_app": lambda name: {"opened": name},
}

SCHEMA = [
    {"type": "function", "function": {
        "name": "get_disk_free", "description": "查磁盘剩余空间",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "open_app", "description": "打开一个应用",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
]


def fake_model_call(user_text):
    if "磁盘" in user_text:
        return {"content": "", "tool_calls": [
            {"id": "c1", "name": "get_disk_free", "arguments": "{}"}]}
    if "打开" in user_text:
        return {"content": "", "tool_calls": [
            {"id": "c2", "name": "open_app", "arguments": '{"name":"记事本"}'}]}
    if "坏参数" in user_text:
        return {"content": "", "tool_calls": [
            {"id": "c3", "name": "open_app", "arguments": '{"name":'}]}
    if "瞎编" in user_text:
        return {"content": "", "tool_calls": [
            {"id": "c4", "name": "format_disk", "arguments": "{}"}]}
    return {"content": "我不需要工具", "tool_calls": []}


def run_turn(user_text):
    r = fake_model_call(user_text)
    if not r["tool_calls"]:
        return r["content"]

    results = []
    for call in r["tool_calls"]:
        name = call["name"]
        fn = TOOLS.get(name)
        if not fn:
            results.append({"id": call["id"], "error": f"未知工具 {name}"})
            continue
        try:
            args = json.loads(call["arguments"] or "{}")
        except Exception as e:
            results.append({"id": call["id"], "error": f"参数不是合法 JSON: {e}"})
            continue
        try:
            results.append({"id": call["id"], "result": fn(**args)})
        except Exception as e:
            results.append({"id": call["id"], "error": str(e)})

    # 真实场景：把 results 包成 role="tool" 消息，再调一次模型生成自然语言
    return f"工具结果: {json.dumps(results, ensure_ascii=False)}"


if __name__ == "__main__":
    for t in ("磁盘还剩多少", "打开记事本", "坏参数测试", "瞎编一个工具", "你好"):
        print(f"{t:10s} -> {run_turn(t)}")
