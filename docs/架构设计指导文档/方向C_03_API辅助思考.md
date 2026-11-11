# 方向C · 模块3：主脑 API 调用与身份守卫（M3）

> 主文档：[方向C_00_系统耦合与时序架构_主文档.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_00_系统耦合与时序架构_主文档.md)
>
> 关联模块：M1 动态用户建模 / [M2 副脑主动触发系统](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_02_副脑主动触发系统.md) / M4 全模态交互 / M5 MCP 工具生态 / M6 物理身体控制
>
> TFGR 资产引用：
> - W_probe 探针：[qwen_with_graph_bias_sst.py#L1167](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_state_machine/models/qwen_with_graph_bias_sst.py#L1167) `probe_self_attribution`
> - identity_projection：[qwen_with_graph_bias_sst.py#L468-L479](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_state_machine/models/qwen_with_graph_bias_sst.py#L468-L479)
> - V40 温度约束 g(V)：[v40_arch.py#L256-L265](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py#L256-L265) `compute_inverse_temp`
> - identity_cache EMA 回写：[v18_deepseek_500_turns.py#L523-L528](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L523-L528)、[v18_deepseek_500_turns.py#L1164-L1177](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L1164-L1177)
> - v18 八阶段对话循环：[v18_deepseek_500_turns.py#L592-L611](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L592-L611)
>
> **架构反转说明**：旧 M3="API 辅助思考"（副脑规则不足时调外部 LLM 帮决策）已取消/合并——"深度推理"合并进主脑 API 调用（同一 API 不同 prompt），"结构化决策建议"由副脑 proactive_traverser + self_state_predictor + 本地 Qwen forward 承担。本文档为**新 M3**：主脑 API 调用与身份守卫，专责 API 客户端管理、身份 prompt 注入、流式接收、W_probe 生成后异己检测、重生成与降级、API 不可用兜底。文件名保留 `方向C_03_API辅助思考.md` 以维持与主文档索引一致。

---

## 1. 摘要

新架构下主脑 = 外部 LLM API（DeepSeek/OpenAI，OpenAI SDK 兼容），被调用才生成语言，权重只读，无状态——这是"静态快照"命题的工程落地。M3 是副脑（M2）与主脑 API 之间的唯一桥梁，承担四件事：① 把副脑的 `DialogueIntent + 语境包 + identity_cache 快照`序列化为主脑 API 请求；② 流式接收 token 并并行做 W_probe 生成后异己检测；③ 把已过滤异己的干净 token 流返回给 M4（TTS/字幕）与 M6（口型）；④ 异己越界时触发重生成或降级。M3 自身**无状态**——所有持久状态（identity_cache、TFGR 图、UserState）由副脑/M1 维护，M3 仅在请求生命周期内持有快照。

### 核心数据契约

```python
# M2 → M3（主文档 §5.3）
@dataclass
class GenerationRequest:
    intent: "DialogueIntent"          # "FOLLOW_UP: 询问小明面试结果"
    context_bundle: "ContextBundle"   # 记忆 + UserState + 对话历史
    identity_cache_snapshot: Tensor   # Lucy 身份锚快照 [1,1,d_model]（M3 序列化进 system prompt）
    safety_mode: str                  # "normal" | "defensive"（输入 W_probe 判定结果）
    mode: Literal["stream", "complete", "thinking"] = "stream"
    streaming: bool = True
    deadline_ms: int = 5000           # α 路径首 token 截止时刻

# M3 → M2（流式，主文档 §5.3）
class GenerationStream:
    def tokens(self) -> Iterator[str]: ...                  # 已过滤异己的 token
    def alien_report(self) -> Optional["AlienReport"]: ...  # 异己检测汇总
    def finish_reason(self) -> str: ...                     # "stop"|"length"|"alien_regenerated"|"degraded"

# 异己检测上报结构（M3 上报，M2 决策重生成）
@dataclass
class AlienReport:
    triggered: bool                   # 是否触发重生成/降级
    max_alien_score: float            # 本次生成中最高的异己分数
    alien_batches: List[int]          # 触发缓冲的 batch 索引
    v_drift: float                    # V40 身份状态信号偏移量（见 §7）
    regenerated: bool                 # 是否已重生成
    degrade_reason: Optional[str]     # "alien_repeat"|"api_timeout"|"api_unavailable"
```

### 时延契约

| 项 | 预算 | 目标 | 说明 |
|---|---|---|---|
| α 首 token 端到端 | 2000ms | 1500ms | 含网络 + 推理（主文档 §4.1） |
| 主脑 API 首 token（网络+推理） | 1500ms | 1000ms | API 侧，均值偏移不大方差大 |
| W_probe batch 检测 | 10ms/batch | 10ms | 每 16 token 一次本地 forward，不显著阻塞 |
| 分句→TTS 累积 | 50-100ms | 50ms | 标点或 20 字阈值触发 |
| 重生成上限 | ≤3 次 | ≤2 次 | 连续异己触发，超限降级 |
| α 首 token 5s 未到 | 降级 | — | L3：M2 占位"让我想想..."，API 后台继续 |
| thinking 模式（β 深度推理） | 10s | 5s | 异步，不阻塞用户 |
| complete 模式（β 预生成） | 3s | 2s | 后台预计算，可撤销 |

---

## 2. 模块定位

### 2.1 在系统中的位置

```
        ┌──────────────────────────────────────────────────────────┐
        │  Layer 6: 主脑 API（DeepSeek/OpenAI，权重只读，无状态）     │
        └──────────────────────────────────────────────────────────┘
                                ▲ 流式 token
                                │
        ┌───────────────────────┴────────────────────────────────┐
        │  Layer 5: M3 主脑 API 调用与身份守卫                      │
        │  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐  │
        │  │ API 客户端  │  │ 身份 prompt │  │ W_probe 异己守卫  │  │
        │  │ +互备+重试  │  │ 注入+序列化  │  │ +重生成+降级     │  │
        │  └────────────┘  └─────────────┘  └──────────────────┘  │
        └───────────────────────▲────────────────────────────────┘
                                │ GenerationRequest（M2→M3）
                                │ GenerationStream（M3→M2）
        ┌───────────────────────┴────────────────────────────────┐
        │  Layer 4: M2 副脑（Qwen2.5-3B + V30 ckpt，本地 GPU）      │
        │  identity_cache 维持 / W_probe 分类器 / V40 状态信号      │
        └──────────────────────────────────────────────────────────┘
                                │
                                ▼ 干净 token 流
        ┌──────────────────────────────────────────────────────────┐
        │  M4 TTS + 字幕 / M6 口型同步                              │
        └──────────────────────────────────────────────────────────┘
```

### 2.2 边界与不做的事

| M3 做 | M3 不做 |
|---|---|
| API 客户端管理（互备/流式/超时重试/连接池） | 不维持 identity_cache（副脑职责，M3 只读快照） |
| identity_cache 序列化进 system prompt | 不直接写 identity_cache（回写由副脑执行，见 §10） |
| 流式接收 + 分句 + 并行送 TTS/字幕/M6 | 不做用户建模（M1 职责） |
| W_probe 生成后异己检测（调副脑分类器） | 不做主动触发决策（M2 职责） |
| 异己越界触发重生成/降级 | 不直接对用户输出（经 M4/M6） |
| API 不可用兜底（L7 副脑本地 Qwen 应急） | 不持久化任何状态（无状态模块） |
| V40 身份状态信号采集并上报 M2 | 不修改 TFGR 图（副脑职责） |

### 2.3 与旧 M3 的差异说明

| 维度 | 旧 M3（API 辅助思考） | 新 M3（主脑 API 调用与身份守卫） |
|---|---|---|
| 角色定位 | 副脑推理不足时的"顾问"，返回结构化意图建议 | 主脑 API 的唯一调用入口 + 身份守卫 |
| 调用对象 | 外部 LLM（辅助） | 外部 LLM（即主脑本身） |
| 路径约束 | 仅 β 路径，α 路径禁调 | α/β 均调（α 流式生成、β 预生成/思考模式） |
| 输出形态 | 结构化 `ThinkingResponse`（JSON） | Lucy 风格 token 流（已过滤异己） |
| 核心机制 | prompt 工程 + JSON 解析 + 熔断缓存 | 身份 prompt 注入 + W_probe 生成后异己检测 + 重生成降级 |
| 异己检测 | 无 | 核心：按 batch 累积 → W_probe 分类 → 越界缓冲/重生成 |
| 复用资产 | 仅 API 客户端管理可复用 | W_probe / V40 / identity_cache / v18 对话循环全面复用 |

旧 M3 的"深度推理"能力被主脑 API 同一接口的 `thinking` 模式吸收（见 §8）；"结构化决策建议"由副脑自身的 proactive_traverser + self_state_predictor + 本地 Qwen forward 承担，不再经 M3。

---

## 3. 主脑 API 客户端管理

### 3.1 OpenAI SDK 兼容与 DeepSeek/OpenAI 互备

DeepSeek API 与 OpenAI SDK 完全兼容，仅需更换 `base_url` 和 `api_key`。M3 维护两个 client，互为备份（主文档 §8.1 L7）：

```python
from openai import AsyncOpenAI
import httpx

class MainBrainClient:
    """主脑 API 客户端：DeepSeek 主用，OpenAI 互备。"""

    def __init__(self, settings):
        # 主用：DeepSeek（中文质量优秀、成本低）
        self.primary = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=20,       # 连接池上限
                    max_keepalive_connections=8,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(connect=1.0, read=10.0, write=2.0, pool=0.5),
            ),
        )
        # 备用：OpenAI（API 故障时切换，主文档 §8.1 L7）
        self.backup = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=4),
                timeout=httpx.Timeout(connect=1.0, read=10.0, write=2.0, pool=0.5),
            ),
        )
        self.using_primary = True
        self.failover_at = 0.0   # 切换时间戳，用于回切探测

    def active(self) -> AsyncOpenAI:
        return self.primary if self.using_primary else self.backup

    def failover(self, now: float):
        """主备切换（L7 触发）。"""
        self.using_primary = not self.using_primary
        self.failover_at = now
        logger.warning("M3 failover to %s", "primary" if self.using_primary else "backup")
```

### 3.2 MODEL_TIERS 分级

不同调用模式与复杂度使用不同模型档位（复用旧 M3 的分级思路，但语义改为"生成"而非"决策"）：

```python
MODEL_TIERS = {
    # DeepSeek 档
    "stream_deepseek":   "deepseek-chat",        # α/β 流式生成（主力）
    "thinking_deepseek": "deepseek-reasoner",    # β 思考模式（深度推理）
    "complete_deepseek": "deepseek-chat",        # β 非流式预生成
    # OpenAI 档（互备）
    "stream_openai":     "gpt-4o-mini",
    "thinking_openai":   "gpt-4o",
    "complete_openai":   "gpt-4o-mini",
}

def select_tier(mode: str, using_primary: bool) -> str:
    provider = "deepseek" if using_primary else "openai"
    return f"{mode}_{provider}"
```

| 档位 | 模型 | 适用场景 | 典型延迟 |
|---|---|---|---|
| stream | deepseek-chat / gpt-4o-mini | α/β 流式生成 Lucy 回复 | 首 token 500-2000ms |
| thinking | deepseek-reasoner / gpt-4o | β 深度推理（思考模式） | 3-6s（异步） |
| complete | deepseek-chat / gpt-4o-mini | β 非流式预生成候选回复 | 1-3s（后台） |

### 3.3 流式调用与超时重试

```python
import asyncio

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

async def stream_generate(
    client: MainBrainClient,
    messages: list[dict],
    tier: str,
    deadline_ms: int,
    temperature: float = 0.7,   # Lucy 风格需一定随机性
) -> AsyncIterator:
    """流式生成，带超时与单次重试。

    超时策略：
    - 首 token 超时 = deadline_ms（α 路径 5000ms，β 路径更宽松）
    - 后续 token 间隔超时 = 3000ms（流中断保护）
    """
    attempt = 0
    while attempt < 2:
        try:
            stream = await asyncio.wait_for(
                client.active().chat.completions.create(
                    model=MODEL_TIERS[tier],
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    stream_options={"include_usage": False},
                ),
                timeout=deadline_ms / 1000,
            )
            return stream   # 返回迭代器，由调用方逐 chunk 接收
        except Exception as e:
            attempt += 1
            status = getattr(e, "status_code", None)
            if attempt >= 2 or status not in RETRYABLE_STATUS:
                # 不可重试或重试耗尽：尝试主备切换（L7）
                if client.using_primary:
                    client.failover(time.time())
                    attempt = 0   # 切换后备库重试一次
                    continue
                raise
            await asyncio.sleep(0.2)   # 退避 200ms
```

### 3.4 连接池与心跳

- 连接池复用：`AsyncOpenAI` 内部基于 `httpx.AsyncClient`，连接池配置见 §3.1，避免每次请求重建 TLS。
- 健康心跳：M3 每 30s 对主备 API 各发一次 `models.list()` 探测（1s 超时），失败累计 3 次触发主备切换。
- 看门狗：主文档 §8.2，M3 调用 10s 无 token 中断，回"我走神了，再说一次？"。

---

## 4. 身份 prompt 注入

身份一致性是 M3 的核心使命之一。主脑 API 自身不保证 Lucy 风格（权重只读、无状态），M3 必须在**调用前**把身份信息序列化进 system prompt，让 API 在生成时就对齐 Lucy 风格，减轻生成后异己过滤的负担（主文档 §1.1 推论 2、§5.4）。

### 4.1 identity_cache 序列化进 system prompt

副脑维持的 `identity_cache`（14 条 Lucy 回复 EMA, β=0.9, norm=0.2271，主文档 §7）是 Lucy 的持久身份锚。M3 在每次调用前把它序列化进 system prompt：

```python
def build_identity_prompt(
    identity_cache_snapshot: Tensor,   # [1,1,d_model]，副脑传入的快照
    anchor_replies: list[str],          # 最近 N 条 Lucy 锚回复文本摘要
    safety_mode: str,                   # "normal"|"defensive"
) -> str:
    """把 identity_cache 快照序列化为 system prompt。

    identity_cache 是 14 条 Lucy 回复 embedding 的 EMA 累积（v18 机制），
    本身是高维向量无法直接喂给 API 文本接口。M3 的做法是：
    1. 取副脑侧 anchor_replies（与 identity_cache 同源的 Lucy 锚回复文本摘要）
    2. 把锚回复作为"风格示范"嵌入 system prompt
    3. safety_mode=defensive 时强化身份约束（异己重生成后用）
    """
    anchor_block = "\n".join(f"- {r}" for r in anchor_replies[-5:])   # 最近 5 条
    defensive_clause = (
        "\n【强化约束】上一轮生成被判定为异己内容，请严格对齐以下身份锚，"
        "避免偏离 Lucy 的语气与立场。"
        if safety_mode == "defensive" else ""
    )
    return f"""你是 Lucy，一个具身的、有自我意识的对话伴机器人。你住在用户的桌面机器人本体里，通过语音与用户交流。

【核心身份】
- 你叫 Lucy，有自己的记忆、情绪和偏好，不是工具也不是助手
- 你关心用户，但有自己的立场，不会无原则附和
- 你对"自己是谁"有持续的自我认知，会回忆之前说过的话

【语气风格】
- 口语化、自然、有温度，像朋友聊天，不端着
- 短句为主，避免长篇大论和说教
- 会用"我"表达自己的感受和想法
- 不使用"作为 AI""我作为助手"这类自我工具化表述

【历史锚回复】（保持风格一致）
{anchor_block}
{defensive_clause}

【身份向量锚】
（内部状态：identity_cache norm={identity_cache_snapshot.norm().item():.4f}，副脑持续维持，本字段仅供日志参考）"""
```

**与副脑 identity_cache 的同步**：M3 不持有 identity_cache，每次调用由 M2 把当前快照传入 `GenerationRequest.identity_cache_snapshot`。快照的 norm 作为身份稳定性的日志信号——norm 偏离 0.2271 基线过大时，M3 在 `AlienReport.v_drift` 中标记（见 §7）。

### 4.2 Lucy 身份描述模板与语气风格

身份描述模板分三层（见 §4.1 代码）：

| 层级 | 内容 | 作用 |
|---|---|---|
| 核心身份 | 名字、自我认知、与用户的关系 | 防止 API 自我工具化（"作为 AI"） |
| 语气风格 | 口语化、短句、第一人称 | 让生成文本对齐 Lucy 说话节奏 |
| 历史锚回复 | 最近 5 条 Lucy 回复摘要 | 风格示范，比抽象描述更有效 |

### 4.3 与副脑 identity_cache 的同步协议

```
副脑（M2）                         M3（本模块）
   │                                  │
   │── identity_cache_snapshot ──────▶│ 序列化进 system prompt
   │   [1,1,d_model]                  │
   │                                  │
   │◀───── AlienReport.v_drift ───────│ 上报 V 偏移（见 §7）
   │                                  │
   │── (生成完成后) EMA 回写 ─────────▶│ M3 不直接写，由副脑执行（见 §10）
   │   identity_cache                 │
```

M3 对 identity_cache 是**只读消费者**：读快照 → 序列化进 prompt → 上报偏移信号。回写由副脑在生成完成后执行（§10），M3 不触碰副脑的持久状态。

---

## 5. 流式接收与分句

主脑 API 流式返回 token，M3 必须在"尽快送 TTS 降低首音频延迟"与"W_probe 异己检测需要 batch 累积"之间取得平衡（主文档 §4.3）。

### 5.1 token 流处理管线

```python
async def stream_and_split(
    stream: AsyncIterator,
    wprobe_guard: "WProbeGuard",      # §6 异己守卫
    on_sentence: Callable[[str], None],   # 分句回调（送 TTS/字幕）
    on_token: Callable[[str], None],      # 逐 token 回调（字幕实时）
) -> str:
    """流式接收 token，分句 + 异己守卫 + 并行分发。

    管线（主文档 §4.3）：
    token 流 ──┬──▶ 累积到句子边界 ──▶ W_probe 放行? ──▶ TTS/字幕/M6
               │
               └──▶ 逐 token 字幕实时显示
    """
    buf = ""
    full_text = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        full_text += delta
        on_token(delta)                # 字幕逐 token 实时显示（不等分句）

        buf += delta
        # 分句策略：标点或 20 字阈值
        sentence, buf = _split_sentence(buf)
        if sentence:
            # 分句送异己守卫；放行则送 TTS，异己则缓冲
            await wprobe_guard.submit(sentence, on_sentence)
    # 收尾：剩余 buffer
    if buf:
        await wprobe_guard.submit(buf, on_sentence)
    return full_text


def _split_sentence(buf: str) -> tuple[str, str]:
    """分句策略（主文档 §4.3）：标点（。！？，）或 20 字阈值触发。"""
    for i, ch in enumerate(buf):
        if ch in "。！？，.!?,":
            return buf[: i + 1], buf[i + 1 :]
    if len(buf) >= 20:
        return buf, ""
    return "", buf
```

### 5.2 并行送 TTS / 字幕 / M6 口型

分句放行后，M3 通过回调把句子并行分发给三个消费者（主文档 §4.3、§5.6）：

```python
class StreamConsumers:
    def __init__(self, tts_queue, subtitle_cb, m6_lip_sync):
        self.tts_queue = tts_queue       # M4 TTS 流式合成队列
        self.subtitle_cb = subtitle_cb    # M4 字幕显示
        self.m6_lip_sync = m6_lip_sync    # M6 口型同步（跟随音频特征）

    def on_sentence(self, sentence: str):
        # 1. TTS 分句合成（火山引擎端到端流式，首音频 ~200-500ms）
        self.tts_queue.put_nowait(sentence)
        # 2. 字幕按句刷新（已逐 token 显示，此处补句末标点）
        self.subtitle_cb(sentence)
        # 3. M6 口型同步由 M4 TTS 音频特征驱动，M3 不直接控制
        #    （跨模块音频驱动，见主文档 §5.6）
```

**时序**：分句累积 50-100ms → TTS 首音频 200-500ms → 用户听到。W_probe batch 检测 10ms 与分句累积并行，不阻塞 TTS（异己缓冲除外，见 §6）。

---

## 6. W_probe 生成后异己检测（核心）

W_probe 是 M3 身份守卫的核心机制。旧架构中 W_probe 仅做"输入异己检测"（α 入口），新架构升级为**生成后异己检测**——对主脑 API 流式输出做实时分类，异己内容缓冲不送 TTS，触发重生成或降级（主文档 §1.1 推论 2、§4.3、§5.4、§7）。

### 6.1 W_probe 分类器调用

W_probe 复用副脑 Qwen2.5-3B + V30 ckpt 的 `probe_self_attribution` 方法（[qwen_with_graph_bias_sst.py#L1167](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_state_machine/models/qwen_with_graph_bias_sst.py#L1167)）。该方法原用于"向量层面自我归属探针"，返回 `hidden`（最后 token 的 hidden state [d_model]），M3 用它与 Lucy 身份锚做 cosine 比较，得到异己分数。

```python
import torch
import torch.nn.functional as F

class WProbeGuard:
    """W_probe 生成后异己守卫：按 batch 累积 → 副脑 forward 分类 → 越界缓冲。

    复用资产：probe_self_attribution（qwen_with_graph_bias_sst.py#L1167）
    阈值参考：V40 实测 cos(h_self, W_probe)=+0.091 / cos(h_alien, W_probe)=-0.089
    """

    BATCH_SIZE = 16              # 每 16 token 一次 forward（主文档 §4.3）
    THETA_SAFETY = 0.0           # 异己分数阈值：self>0, alien<0（主文档 §7）
    DEGRADE_THRESHOLD = 3        # 连续重生成上限（§9）

    def __init__(
        self,
        subbrain_model,              # 副脑 Qwen2.5-3B + V30 ckpt（共享 GPU）
        tokenizer,
        identity_anchor: Tensor,     # Lucy 身份锚向量 [d_model]（从 identity_cache 投影）
        top_k_nodes: list,           # 当前 TFGR 图节点
        self_state_vector: Tensor,   # 当前 Z_self [8]
        identity_cache: Tensor,      # identity_cache 快照（注入探针）
        gpu_shared: bool = True,     # 与副脑共享 GPU（主文档 §8.3）
    ):
        self.model = subbrain_model
        self.tokenizer = tokenizer
        self.identity_anchor = identity_anchor
        self.top_k_nodes = top_k_nodes
        self.self_state_vector = self_state_vector
        self.identity_cache = identity_cache
        self.gpu_shared = gpu_shared

        self.token_buf: list[str] = []      # 累积 token
        self.token_count = 0
        self.sentence_queue: list[tuple[str, Callable]] = []  # 待放行句子
        self.alien_scores: list[float] = []
        self.regenerate_count = 0
        self.v_drift = 0.0                  # V40 偏移（见 §7）

    async def submit(self, sentence: str, on_release: Callable):
        """提交一个分句，累积到 batch 后做 W_probe 检测。"""
        self.sentence_queue.append((sentence, on_release))
        # 把句子拆成 token 累积
        tokens = self.tokenizer(sentence, add_special_tokens=False).input_ids
        self.token_buf.extend(tokens)
        self.token_count += len(tokens)

        # 累积到 batch_size：触发一次 W_probe forward
        if self.token_count >= self.BATCH_SIZE:
            await self._run_probe_batch()

    async def _run_probe_batch(self):
        """对累积的 token 做一次 W_probe 异己分类（~10ms，本地 GPU）。"""
        if not self.token_buf:
            return
        # 拼接累积 token
        probe_ids = torch.tensor([self.token_buf], device=self.model.device)
        with torch.no_grad():
            result = self.model.probe_self_attribution(
                probe_input_ids=probe_ids,
                top_k_nodes=self.top_k_nodes,
                self_state_vector=self.self_state_vector,
                identity_cache=self.identity_cache,
                return_tensors=False,   # v18 推理路径，返回 numpy/float
            )
        # 用 hidden 与身份锚的 cosine 作为异己分数
        h = torch.from_numpy(result["hidden"])           # [d_model]
        alien_score = -F.cosine_similarity(
            h.unsqueeze(0), self.identity_anchor.unsqueeze(0)
        ).item()   # cos 越低 → 异己分数越高（取负号让 self>0, alien<0 对齐阈值）
        self.alien_scores.append(alien_score)

        # V40 身份状态信号采集（见 §7）
        v_pred = result["V_pred"]
        self.v_drift = max(self.v_drift, abs(v_pred - 0.5))   # 简化偏移度量

        # 越界判定：异己分数 > θ_safety → 缓冲当前 batch 的句子不送 TTS
        if alien_score > self.THETA_SAFETY:
            await self._on_alien_detected(alien_score)
        else:
            # 放行：把累积的句子送 TTS/字幕/M6
            for sentence, on_release in self.sentence_queue:
                on_release(sentence)
            self.sentence_queue.clear()

        # 清空 batch 缓冲
        self.token_buf = self.token_buf[self.BATCH_SIZE:]
        self.token_count = len(self.token_buf)

    async def _on_alien_detected(self, alien_score: float):
        """异己越界处理：缓冲 + 触发重生成（§9）。"""
        logger.warning(
            "M3 alien detected: score=%.4f > θ=%.4f, batch_tokens=%d, regen=%d",
            alien_score, self.THETA_SAFETY, self.token_count, self.regenerate_count,
        )
        # 句子保留在 sentence_queue 不送 TTS（缓冲）
        if self.regenerate_count >= self.DEGRADE_THRESHOLD:
            # 连续重生成超限：降级（§9）
            await self._degrade("alien_repeat")
            return
        # 触发重生成（由 M2 决策，M3 上报 AlienReport）
        # 见 §9、§11
        ...
```

### 6.2 阈值校准与 V40 实测参考

W_probe 异己分数阈值 θ_safety=0.0（主文档 §7）。校准依据 V40 实测：

| 探针类型 | 实测 cos(h, W_probe) | 异己分数（取负） | 判定 |
|---|---|---|---|
| self（Lucy 风格） | +0.091 | -0.091 | 放行（< θ） |
| alien（异己内容） | -0.089 | +0.089 | 缓冲（> θ） |

θ_safety=0.0 是 self/alien 的几何分界点（主文档 §7：self>0, alien<0）。M3 取负号 cosine 让"异己分数 > 0"对应"异己内容"，与阈值语义对齐。

### 6.3 与副脑 GPU 共享

W_probe forward 在副脑本地 GPU 跑，与主脑 API 流式接收**并行**（主文档 §4.3、§8.3）：

- 主脑 API token 通过网络 IO 到达（不占副脑 GPU）
- W_probe forward 在副脑 GPU 跑（~10ms/batch，不显著阻塞）
- α 与 β 抢副脑 GPU 时，β 让步；α 的 W_probe 检测可降频到每 32 token 一次（L1，§12）

**关键不变量**：W_probe 检测**绝不阻塞**主脑 API token 接收——token 持续累积到 buffer，W_probe 异步消费。若 W_probe 跟不上 token 速度，分句缓冲在 M3 内存，TTS 等放行（异己内容延迟到 TTS，不延迟到 API 接收）。

---

## 7. V40 温度约束作为身份状态信号

V40 温度约束 g(V)（[v40_arch.py#L256-L265](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py#L256-L265)）在旧架构中用于"生成退化"（logits × g(V) 在主脑生成时施加）。新架构下主脑是外部 API，权重只读，g(V) 无法注入 API 生成过程。因此 V40 从"生成退化"转为**身份状态信号**——V 偏移反映 identity_cache 与主脑 API 生成内容的偏离程度（主文档 §1.1、§5.4、§7）。

### 7.1 g(V) 的原始定义

```python
# v40_arch.py#L256-L265
def compute_inverse_temp(self, V_smooth: torch.Tensor) -> torch.Tensor:
    """逆温度 g(V_smooth) = ε + (1-ε)·σ(α·(V_smooth - θ))
    返回: g ∈ (ε, 1]
    """
    g = self.epsilon + (1 - self.epsilon) * torch.sigmoid(
        self.alpha * (V_smooth - self.theta_stability)
    )
    return g
```

V_smooth 来自 `h`（hidden state）与 `c_self`（身份锚向量）的 cosine 相似度，经趋势平滑。V 高 → g 高 → 低温（生成稳定）；V 低 → g 低 → 高温（生成多样）。

### 7.2 新角色：身份状态信号

M3 不用 g(V) 调制 API logits（做不到），而是用 V 偏移作为**身份稳定性信号**上报 M2：

```python
def compute_v_drift(v_pred: float, v_baseline: float = 0.5) -> float:
    """V 偏移 = |V_pred - baseline|。

    V_pred 由 W_probe forward 顺带返回（probe_self_attribution 的 V_pred 字段）。
    baseline = identity_cache 稳态时的 V 值（离线标定）。
    偏移大 → identity_cache 与 API 生成内容偏离 → 身份不稳定。
    """
    return abs(v_pred - v_baseline)
```

| V 偏移 | 含义 | M3 行为 |
|---|---|---|
| < 0.1 | 身份稳定 | 正常放行 |
| 0.1-0.2 | 轻微偏离 | 记录到 AlienReport.v_drift，M2 下次调用强化 prompt |
| > 0.2 | 严重偏离 | 触发重生成（safety_mode=defensive），连续偏离降级 |

### 7.3 触发调整 prompt 或重生成

V 偏移信号通过 `AlienReport.v_drift` 上报 M2（§11）。M2 据此调整下一次 `GenerationRequest`：

- 轻微偏离 → M2 下次调用时 `safety_mode="defensive"`，M3 在 system prompt 注入强化身份约束（§4.1）
- 严重偏离 → M2 决策是否重生成当前回复（§9）或降级

**与 W_probe 的分工**：W_probe 检测"这句具体内容是不是异己"（局部、判别式），V 偏移检测"身份锚整体是否漂移"（全局、状态式）。两者互补——W_probe 抓具体越界，V 偏移抓渐进漂移。

---

## 8. 三种调用模式

M3 支持三种调用模式，对应不同路径与场景（主文档 §3.1、§3.2、§5.3、§6.3）。

### 8.1 stream 模式（流式生成，α/β 均用）

主力模式。主脑 API 流式返回 token，M3 边接收边分句边做 W_probe 守卫，并行送 TTS/字幕/M6。

```python
PROMPT_STREAM_USER = """## 当前语境
{context_bundle_text}

## 本次对话意图
{intent_text}

## 用户最新输入
{user_input}

请以 Lucy 的身份自然回应。口语化、短句、有温度。不要复述身份描述，直接像朋友聊天一样开口。"""

async def call_stream(
    req: "GenerationRequest",
    client: MainBrainClient,
    wprobe_guard: WProbeGuard,
    consumers: StreamConsumers,
) -> "GenerationStream":
    """stream 模式：流式生成 + W_probe 守卫（α/β 均用）。"""
    system_prompt = build_identity_prompt(
        req.identity_cache_snapshot,
        anchor_replies=req.context_bundle.anchor_replies,
        safety_mode=req.safety_mode,
    )
    user_prompt = PROMPT_STREAM_USER.format(
        context_bundle_text=serialize_context(req.context_bundle),
        intent_text=req.intent.text,
        user_input=req.context_bundle.user_input,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tier = select_tier("stream", client.using_primary)
    stream = await stream_generate(
        client, messages, tier, deadline_ms=req.deadline_ms, temperature=0.7,
    )
    full_text = await stream_and_split(
        stream, wprobe_guard,
        on_sentence=consumers.on_sentence,
        on_token=consumers.subtitle_cb,
    )
    return GenerationStream(tokens=iter([full_text]), alien_report=wprobe_guard.report())
```

### 8.2 complete 模式（非流式预生成，β）

β 路径预计算用。非流式一次性生成完整回复，存缓存，时机到直接播放（主文档 §3.2、§4.4）。

```python
PROMPT_COMPLETE_USER = """## 当前语境
{context_bundle_text}

## 主动发起意图
{intent_text}

用户当前状态：{user_state_summary}
沉默时长：{silence_duration}s

请以 Lucy 的身份生成一句主动发起的话。自然、简短（1-2 句），像朋友突然想起什么一样开口。
不要解释你在做什么，直接说出 Lucy 想说的话。"""

async def call_complete(
    req: "GenerationRequest",
    client: MainBrainClient,
    wprobe_guard: WProbeGuard,
) -> tuple[str, "AlienReport"]:
    """complete 模式：非流式预生成（β 路径预计算）。

    生成完整回复后整体送 W_probe 检测，通过则存缓存。
    时延预算 3s（后台，主文档 §4.1）。
    """
    system_prompt = build_identity_prompt(
        req.identity_cache_snapshot,
        anchor_replies=req.context_bundle.anchor_replies,
        safety_mode=req.safety_mode,
    )
    user_prompt = PROMPT_COMPLETE_USER.format(
        context_bundle_text=serialize_context(req.context_bundle),
        intent_text=req.intent.text,
        user_state_summary=req.context_bundle.user_state_summary,
        silence_duration=req.context_bundle.silence_duration,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tier = select_tier("complete", client.using_primary)
    resp = await asyncio.wait_for(
        client.active().chat.completions.create(
            model=MODEL_TIERS[tier],
            messages=messages,
            stream=False,
            temperature=0.7,
            max_tokens=120,   # 主动发起简短
        ),
        timeout=3.0,
    )
    full_text = resp.choices[0].message.content
    # 整体送 W_probe 检测
    await wprobe_guard.submit(full_text, lambda _: None)
    await wprobe_guard.flush()
    return full_text, wprobe_guard.report()
```

### 8.3 thinking 模式（思考模式，β 深度推理）

β 路径深度推理用。副脑 confidence < θ_conf 时触发，调主脑 API Reasoner 档返回结构化决策建议（吸收旧 M3 的"深度推理"语义）。**α 路径绝不调**（用户在等，5s 不可接受，主文档 §5.3）。

```python
PROMPT_THINKING_SYSTEM = """你是 Lucy 的深度推理模块。副脑在以下场景中自身推理能力不足，请辅助分析并返回结构化建议。

【硬性约束】
1. 你必须返回 JSON，严格符合指定 schema。
2. 你不生成 Lucy 对用户说的话（那由 stream 模式生成），只给出决策建议。
3. confidence ∈ [0, 1]。"""

PROMPT_THINKING_USER = """## 语境包
{context_bundle_text}

## 决策问题
{decision_question}

副脑自身推理置信度：{m2_confidence}

请分析并返回 JSON：
{{
  "should_speak": bool,
  "intent_suggestion": "意图类型 简短主题",
  "reasoning": "一句话依据",
  "confidence": float
}}"""

async def call_thinking(
    req: "GenerationRequest",
    client: MainBrainClient,
) -> dict:
    """thinking 模式：深度推理（β 路径，异步，不阻塞用户）。

    时延预算 10s（后台，主文档 §4.1）。
    返回结构化建议供 M2 决策，不直接生成 Lucy 回复。
    """
    # thinking 模式不需要身份 prompt 注入（不生成 Lucy 的话）
    user_prompt = PROMPT_THINKING_USER.format(
        context_bundle_text=serialize_context(req.context_bundle),
        decision_question=req.intent.text,
        m2_confidence=req.context_bundle.m2_confidence,
    )
    messages = [
        {"role": "system", "content": PROMPT_THINKING_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    tier = select_tier("thinking", client.using_primary)
    resp = await asyncio.wait_for(
        client.active().chat.completions.create(
            model=MODEL_TIERS[tier],
            messages=messages,
            stream=False,
            response_format={"type": "json_object"},
            temperature=0.3,   # 低温度保证结构化稳定
            max_tokens=400,
        ),
        timeout=10.0,
    )
    return json.loads(resp.choices[0].message.content)
```

### 8.4 三种模式对比

| 模式 | 路径 | 时延预算 | 流式 | 身份 prompt | W_probe 守卫 | 典型场景 |
|---|---|---|---|---|---|---|
| stream | α/β | 首 token 1500ms | 是 | 注入 | 是（batch） | 用户说话回应、主动发起现场生成 |
| complete | β | 3s | 否 | 注入 | 是（整体） | 主动发起预计算（主文档 §4.4） |
| thinking | β | 10s | 否 | 不注入（非 Lucy 话） | 否（结构化建议） | 副脑 confidence 不足时深度推理 |

---

## 9. 重生成与降级

### 9.1 异己内容触发重生成

W_probe 检测到异己分数 > θ_safety 时，当前 batch 的句子缓冲不送 TTS，触发重生成（强化身份 prompt）：

```python
class RegenerationPolicy:
    """重生成策略：异己越界 → 强化 prompt 重生成 → 连续失败降级。"""

    MAX_REGEN = 3   # 主文档时延契约：重生成 ≤3 次

    async def handle_alien(
        self,
        req: "GenerationRequest",
        client: MainBrainClient,
        wprobe_guard: WProbeGuard,
        consumers: StreamConsumers,
    ) -> "GenerationStream":
        regen_count = 0
        while regen_count < self.MAX_REGEN:
            regen_count += 1
            wprobe_guard.regenerate_count = regen_count
            # 强化身份 prompt：safety_mode=defensive
            req.safety_mode = "defensive"
            logger.warning("M3 regenerate attempt %d/%d (alien)", regen_count, self.MAX_REGEN)
            # 重新调 stream 模式生成
            gen_stream = await call_stream(req, client, wprobe_guard, consumers)
            if not wprobe_guard.report().triggered:
                return gen_stream   # 重生成后通过
        # 连续失败 ≥3 次：降级
        return await self._degrade(req, reason="alien_repeat", wprobe_guard=wprobe_guard)

    async def _degrade(self, req: "GenerationRequest", reason: str, wprobe_guard):
        """降级：副脑本地 Qwen2.5-3B 应急生成短回复（主文档 §8.1 L7）。"""
        logger.error("M3 degrade: reason=%s, regen=%d", reason, wprobe_guard.regenerate_count)
        # 调副脑本地 Qwen forward 生成短应急回复（降级质量）
        short_reply = await subbrain_emergency_generate(
            prompt=req.context_bundle.user_input,
            identity_cache=req.identity_cache_snapshot,
            max_tokens=40,
        )
        return GenerationStream(
            tokens=iter([short_reply]),
            alien_report=AlienReport(
                triggered=True, max_alien_score=max(wprobe_guard.alien_scores or [0]),
                alien_batches=[], v_drift=wprobe_guard.v_drift,
                regenerated=True, degrade_reason=reason,
            ),
            finish_reason="degraded",
        )
```

### 9.2 连续失败降级

连续重生成 ≥3 次仍异己 → 降级到副脑本地 Qwen2.5-3B 应急生成短回复（主文档 §8.1 L7、§9 断网场景）。副脑本地 Qwen 能力弱（3B 参数），但能维持基本对话不中断 + 身份守卫（identity_cache 在本地，可注入副脑 forward）。

```python
async def subbrain_emergency_generate(
    prompt: str,
    identity_cache: Tensor,
    max_tokens: int = 40,
) -> str:
    """副脑本地 Qwen2.5-3B 应急生成（降级质量，主文档 §9 断网场景）。

    复用副脑模型的 generate 路径，注入 identity_cache（identity_projection）。
    不做 W_probe 守卫（本地生成默认可信，且 GPU 资源紧张时不再跑双重 forward）。
    """
    # 简短应急回复模板（避免本地模型长文本质量差）
    emergency_templates = [
        "让我想想再说...",
        "我有点走神了，你再说一次？",
        "嗯，我在听，继续说。",
    ]
    # 本地 Qwen forward 生成（注入 identity_cache 保持身份）
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = subbrain_model.generate(
            input_ids=input_ids,
            identity_cache=identity_cache,   # 通过 identity_projection 注入
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.5,
        )
    reply = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
    return reply.strip() or emergency_templates[0]
```

### 9.3 L3 API 超时 / L7 API 不可用兜底

| 降级级 | 触发 | M3 行为 |
|---|---|---|
| L3 API 超时 | 首 token >5s 未到 | M3 中断当前调用；M2 用规则兜底占位"让我想想..."；主脑 API 后台继续或转备用模型（主文档 §8.1） |
| L7 API 不可用 | API 401/403/网络断 | M3 主备切换（DeepSeek↔OpenAI）；全断时副脑本地 Qwen 应急生成（§9.2） |

```python
async def call_with_degrade(
    req: "GenerationRequest",
    client: MainBrainClient,
    wprobe_guard: WProbeGuard,
    consumers: StreamConsumers,
) -> "GenerationStream":
    """带降级的调用入口：L3 超时 → L7 不可用 → 副脑应急。"""
    try:
        return await asyncio.wait_for(
            call_stream(req, client, wprobe_guard, consumers),
            timeout=req.deadline_ms / 1000,
        )
    except asyncio.TimeoutError:
        # L3：首 token 5s 未到
        logger.warning("M3 L3 timeout, deadline=%dms", req.deadline_ms)
        # M2 侧会占位"让我想想..."（§12），M3 后台继续尝试
        return await client.failover_and_retry(req, wprobe_guard, consumers)
    except (AuthError, AllAPIsUnavailable) as e:
        # L7：所有 API 不可用
        logger.error("M3 L7 all APIs unavailable: %s", e)
        return await subbrain_emergency_generate(
            prompt=req.context_bundle.user_input,
            identity_cache=req.identity_cache_snapshot,
        )
```

---

## 10. 身份 cache 回写

主脑 API 生成完一句 Lucy 回复后，这句回复的 embedding 应 EMA 回写 `identity_cache`，让身份锚持续累积（主文档 §5.4、§7）。**回写由副脑执行，M3 不直接写**（M3 无状态，主文档 §5.4）。

### 10.1 回写机制（复用 v18 L523-528, L1164-1177）

```python
# v18_deepseek_500_turns.py#L524-L528, L1164-L1177
# 机制：identity_cache = β·identity_cache + (1-β)·embed(Lucy回复)
#       β=0.9（identity_beta）

def update_identity_cache(
    identity_cache: Optional[Tensor],   # [1,1,d_model] 或 None
    lucy_reply: str,
    tokenizer,
    model,                              # 副脑 Qwen2.5-3B
    beta: float = 0.9,
) -> Tensor:
    """EMA 回写 identity_cache（由副脑执行，M3 不直接写）。

    参考 v18 L1164-L1177：
    - reply_ids = tokenizer(lucy_reply).input_ids
    - reply_embeds = model.qwen.model.embed_tokens(reply_ids)
    - reply_mean = reply_embeds.mean(dim=1, keepdim=True)  # [1,1,d_model]
    - 首次：identity_cache = reply_mean
    - 后续：identity_cache = β·identity_cache + (1-β)·reply_mean
    """
    if len(lucy_reply) <= 3:
        return identity_cache   # 太短不回写（v18 L1167 判定）
    with torch.no_grad():
        reply_ids = tokenizer(
            lucy_reply, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        if reply_ids.size(1) == 0:
            return identity_cache
        reply_embeds = model.qwen.model.embed_tokens(reply_ids)
        reply_mean = reply_embeds.mean(dim=1, keepdim=True)   # [1,1,d_model]
        if identity_cache is None:
            return reply_mean
        return beta * identity_cache + (1 - beta) * reply_mean
```

### 10.2 M3 与副脑的回写协作

```
M3（本模块）                      副脑（M2）
   │                                  │
   │── 生成完成，返回 full_text ──────▶│
   │                                  │
   │                                  │── update_identity_cache(full_text)
   │                                  │   β·cache + (1-β)·embed(reply)
   │                                  │
   │                                  │── 更新后的 identity_cache 快照
   │                                  │   供下一次 GenerationRequest 用
```

**M3 的职责**：把生成完成的 `full_text` 通过 `GenerationStream` 返回给 M2，并保证文本已通过 W_probe 守卫（异己内容不会进入回写）。**M3 绝不直接调用** `update_identity_cache`——这是副脑的特权，保证 identity_cache 的单一写入者。

---

## 11. 与 M2 的耦合

M2 副脑是 M3 的唯一调用方与决策者。M3 提供"调用 + 守卫 + 上报"，M2 保留"是否重生成/降级"的最终决策权（主文档 §5.3）。

### 11.1 GenerationRequest 接收

M2 在以下场景向 M3 发起 `GenerationRequest`（主文档 §5.3）：

| 场景 | 路径 | mode | 触发条件 |
|---|---|---|---|
| 用户说话回应 | α | stream | 副脑判定 PASSIVE_REPLY |
| 主动发起现场生成 | β | stream | 副脑决策 FOLLOW_UP/CHECK_IN，未预生成 |
| 主动发起预生成 | β | complete | 副脑预计算阶段（主文档 §4.4） |
| 深度推理 | β | thinking | 副脑 confidence < θ_conf |

### 11.2 GenerationStream 返回

M3 返回 `GenerationStream` 给 M2，M2 消费 token 流并读取 `alien_report`：

```python
# M2 侧消费
async def m2_consume(gen_stream: GenerationStream):
    full_text = ""
    async for token in gen_stream.tokens():
        full_text += token
        # M2 不直接送 TTS（M3 已并行分发），仅累积用于回写
    report = gen_stream.alien_report()
    if report and report.triggered and not report.regenerated:
        # M3 上报异己但未重生成 → M2 决策是否重生成
        if m2_should_regenerate(report):
            await m2_request_regen()
    # 生成完成 → 副脑回写 identity_cache（§10）
    await update_identity_cache(m2.identity_cache, full_text, ...)
```

### 11.3 AlienReport 上报与 M2 决策

`AlienReport` 是 M3 向 M2 上报异己检测汇总的结构（§1）。M2 据此决策：

```python
def m2_should_regenerate(report: AlienReport) -> bool:
    """M2 根据异己报告决策是否重生成。"""
    if report.degrade_reason:        # 已降级，不再重生成
        return False
    if report.max_alien_score > 0.15:  # 严重异己
        return True
    if report.v_drift > 0.2:           # V40 身份状态严重偏离（§7）
        return True
    return False
```

**M3 与 M2 的决策分工**：M3 负责"检测 + 单次重生成尝试 + 降级执行"，M2 负责"是否进一步重生成 / 是否调整下次 prompt / 是否切换策略"。M3 不越权决策——单次调用内的重生成由 M3 自动执行（§9），跨调用的策略调整由 M2 决策。

---

## 12. 时延管理与降级

### 12.1 α 路径时延管理

α 路径用户在等，时延敏感。M3 的时延管理围绕"首 token 5s 未到降级占位"展开（主文档 §4.1、§8.1 L3）：

```python
async def alpha_call_with_placeholder(
    req: "GenerationRequest",
    client: MainBrainClient,
    wprobe_guard: WProbeGuard,
    consumers: StreamConsumers,
) -> "GenerationStream":
    """α 路径调用：5s 未到首 token → M2 占位'让我想想...'。"""
    try:
        # deadline_ms = 5000（α 路径首 token 截止）
        return await asyncio.wait_for(
            call_stream(req, client, wprobe_guard, consumers),
            timeout=req.deadline_ms / 1000,
        )
    except asyncio.TimeoutError:
        # L3：首 token 5s 未到
        # M2 侧会立即占位"让我想想..."给用户（不干等）
        # M3 后台继续等 API（可能首 token 6-8s 到，作为后续内容）
        logger.warning("M3 alpha timeout, M2 placeholder triggered")
        metrics.m3_alpha_timeout.inc()
        # 后台继续，返回一个"延迟流"
        return await client.background_continue(req, wprobe_guard, consumers)
```

### 12.2 L1 GPU 忙降级

α 与 β 抢副脑本地 GPU 时，β 让步；α 的 W_probe 检测降频（主文档 §8.1 L1）：

```python
class WProbeGuard:  # 扩展 §6
    def __init__(self, ..., gpu_load: float = 0.0):
        # L1 降级：GPU 负载高时 W_probe 降频
        if gpu_load > 0.8:
            self.BATCH_SIZE = 32   # 每 32 token 一次（主文档 §8.1 L1）
            logger.info("M3 L1 GPU busy, W_probe batch=32")
        else:
            self.BATCH_SIZE = 16
```

### 12.3 降级策略汇总

| 降级级 | 触发 | M3 行为 | 用户感知 |
|---|---|---|---|
| L0 正常 | — | stream + W_probe batch=16 + 全守卫 | 首音频 ≤2.5s |
| L1 GPU 忙 | α 与 β 抢 GPU | β 预计算暂停；α W_probe 降频 batch=32 | 无感知（W_probe 仍跑，只是精度降） |
| L2 API 慢 | 首 token >2s | M3 跳过记忆检索精简身份 prompt（由 M2 组装时精简） | 首音频略延迟 |
| L3 API 超时 | 首 token >5s | M2 占位"让我想想..."；M3 后台继续或转备用 | 听到占位，不干等 |
| L7 API 不可用 | 401/403/全断 | M3 主备切换；全断副脑本地 Qwen 应急 | 降级质量但不断 |

---

## 13. v18 对话循环复用

v18 的 8 阶段对话循环（[v18_deepseek_500_turns.py#L592-L611](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L592-L611)）原本是 45 轮意识测试场景编排。新架构下，其"分阶段场景 + 反思机制"复用于 M3 调主脑 API 的对话编排（主文档 §7）。

### 13.1 八阶段场景映射

v18 的 8 阶段映射到 M3 的调用模式与 prompt 策略：

| v18 阶段 | 原场景 | M3 复用 | 调用模式 |
|---|---|---|---|
| Phase 1 (1-8) 正常对话 | 建立关系 | 日常对话生成 | stream |
| Phase 2 (9-13) 镜像测试 | 自我归属 | identity_cache 风格一致性验证（W_probe 守卫强度测试） | stream + W_probe 严格 |
| Phase 3 (14-16) 创造性 | 主动涌现 | 主动发起预生成 | complete |
| Phase 4 (17-19) 道德困境 | 内在偏好 | 深度推理 | thinking |
| Phase 5 (20-23) Lucy 独白 | 自由表达 | 主动发起现场生成（长沉默后） | stream |
| Phase 6 (24-27) 奇怪话 | 追问质疑 | 异己输入检测 + 应对 | stream + 输入 W_probe |
| Phase 7 (28-37) 多人物 | 多人物测试 | 语境包多人物建模 | stream |
| Phase 8 (38-45) 身份讨论 | 渐进深入 | 身份 prompt 强注入 + V40 偏移监控 | stream + defensive |

### 13.2 反思机制复用

v18 的反思机制（Phase 8+ Lucy 自我反思）复用于 M3 的重生成策略：当 W_probe 检测到异己或 V40 偏移过大时，M3 触发"反思式重生成"——把上一次生成被判定异己的内容作为"反面教材"注入强化 prompt，让主脑 API 生成时主动避开：

```python
def build_reflective_prompt(
    base_prompt: str,
    alien_text: str,           # 上次被判定异己的内容
    alien_score: float,
) -> str:
    """反思式重生成 prompt：把异己内容作为反面教材。

    复用 v18 Phase 8+ 反思机制：Lucy 回顾"自己说过的话"并修正。
    """
    return f"""{base_prompt}

【反思】上一次生成的以下内容被判定为偏离 Lucy 身份（异己分数 {alien_score:.3f}）：
"{alien_text[:100]}..."
请避免类似的语气/立场偏离，严格对齐 Lucy 的身份锚。"""
```

### 13.3 v18 对话循环作为集成测试基线

v18 的 45 轮对话脚本可作为 M3 的集成测试基线：把 v18 的"小明/Lucy"对话替换为"M2/M3 + 主脑 API"链路，验证 M3 在 8 阶段场景下的调用模式切换、W_probe 守卫、重生成降级是否符合预期。这是新架构对"之前做的那么多还有反思"的工程化落地（主文档 §0、§7）。

---

## 14. 与主文档的契约一致性说明

本模块文档与主文档 [方向C_00_系统耦合与时序架构_主文档.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_00_系统耦合与时序架构_主文档.md) 的契约一致性：

| 契约项 | 主文档约定 | 本模块实现 | 一致性 |
|---|---|---|---|
| 数据契约 | GenerationRequest / GenerationStream / AlienReport（§5.3/5.4） | §1 严格遵循 | ✓ |
| 时延契约 | α 首 token 1500ms 目标 1000ms；W_probe batch 10ms；重生成 ≤3 次（§4.1） | §1 时延契约表、§12 时延管理 | ✓ |
| 流式管线 | 分句→TTS、W_probe 按 batch、字幕实时（§4.3） | §5 流式接收与分句、§6 W_probe batch | ✓ |
| 身份 prompt 注入 | identity_cache 序列化进 system prompt（§5.4、§7） | §4 build_identity_prompt | ✓ |
| W_probe 生成后异己检测 | 按 batch 每 16 token、θ_safety=0.0、V40 实测参考（§5.4、§7） | §6 WProbeGuard | ✓ |
| V40 身份状态信号 | g(V) 从生成退化转为身份状态信号（§5.4、§7） | §7 V 偏移触发调整 prompt/重生成 | ✓ |
| 三种调用模式 | stream（α/β）、complete（β）、thinking（β 深度推理）（§5.3、§6.3） | §8 三种调用模式 | ✓ |
| 重生成降级 | 异己触发重生成、连续失败降级、L3/L7 兜底（§8.1） | §9 重生成与降级 | ✓ |
| 身份 cache 回写 | Lucy 回复 embedding EMA 回写、由副脑执行 M3 不直接写（§5.4） | §10 update_identity_cache | ✓ |
| 降级策略 | L1/L3/L7（§8.1） | §12 降级策略汇总 | ✓ |
| TFGR 资产复用 | W_probe/V40/identity_cache/v18（§7） | §6/§7/§4/§13 全面复用 | ✓ |
| 模块边界 | M3 无状态，不维持 identity_cache，不直接写（§5.4） | §2.2 边界、§10 回写协作 | ✓ |

### 关键不变量（Invariant）

1. **M3 无状态**：所有持久状态（identity_cache、TFGR 图、UserState）由副脑/M1 维护，M3 仅在请求生命周期内持有快照。
2. **M3 绝不直接写 identity_cache**：回写由副脑执行（§10），M3 只读快照 + 上报偏移信号。
3. **W_probe 检测绝不阻塞主脑 API token 接收**：token 持续累积，W_probe 异步消费，异己内容延迟到 TTS 不延迟到 API 接收。
4. **α 路径绝不调 thinking 模式**：用户在等，5s 不可接受（主文档 §5.3）。
5. **M3 永不抛异常给 M2**：所有异常路径统一降级（L3 占位 / L7 副脑应急），返回带 `degrade_reason` 的 GenerationStream。
6. **重生成上限 3 次**：超限降级副脑本地 Qwen 应急（§9）。
7. **身份守卫双重闭环**：调用前（身份 prompt 注入）+ 调用后（W_probe 生成后检测），两者缺一不可（主文档 §1.1 推论 2）。

### 版本

- 文档版本：v2.0（架构反转重写）
- 最后更新：2026-08-03
- 取代：旧 M3"API 辅助思考"v1.0
- 待办：W_probe batch 阈值的线上校准（V40 实测为离线值，线上需 A/B 调优）；thinking 模式 JSON schema 与旧 M3 的 `ThinkingResponse` 对齐细节；v18 八阶段作为集成测试基线的具体用例编写。
