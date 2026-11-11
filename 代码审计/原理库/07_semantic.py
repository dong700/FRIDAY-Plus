"""闭环 07：语义检索（embedding + 字面重排）

跑：
    # 需要本地 Ollama 且已 pull nomic-embed-text
    ollama pull nomic-embed-text
    python 07_semantic.py
对应源码：brain/ov_memory.py::find

要点：纯向量在中文短句上排序不稳，加 30% 字面重合度做重排。
"""
import json
import math
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/embeddings"
MODEL = "nomic-embed-text"


def embed(text: str):
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps({"model": MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["embedding"]


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def overlap(q: str, t: str) -> float:
    qs = {c for c in q if "\u4e00" <= c <= "\u9fff"}
    return sum(1 for c in qs if c in t) / len(qs) if qs else 0.0


def find(query, docs, k=3, min_score=0.55):
    qv = embed(query)
    scored = []
    for d in docs:
        s = 0.7 * cosine(qv, d["vec"]) + 0.3 * overlap(query, d["text"])
        scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    return [(round(s, 3), d["text"]) for s, d in scored[:k] if s >= min_score]


if __name__ == "__main__":
    texts = [
        "周五要交房租",
        "我不吃辣",
        "他最近在赶一个项目",
        "他喜欢猫",
        "他每天九点开始写代码",
    ]
    try:
        docs = [{"text": t, "vec": embed(t)} for t in texts]
    except Exception as e:
        print(f"Ollama 连不上（{e}）——这是正式代码里必须兜住的降级点：检索失败不阻塞对话。")
        raise SystemExit(0)

    for q in ("房租的事", "他爱吃什么", "作息规律"):
        print(f"\nQ: {q}")
        for s, t in find(q, docs):
            print(f"  {s:.3f}  {t}")
