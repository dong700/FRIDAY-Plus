# 方向C · 模块5：MCP 工具生态（M5）

> 主文档：[方向C_00_系统耦合与时序架构_主文档.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_00_系统耦合与时序架构_主文档.md)
>
> 关联模块：[M1 动态用户建模](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_01_动态用户建模.md) / [M2 副脑主动触发系统](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_02_副脑主动触发系统.md) / [M3 主脑API调用与身份守卫](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_03_API辅助思考.md) / [M4 全模态交互](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_04_全模态交互.md) / [M6 物理身体控制](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_06_物理身体控制.md)
>
> 本模块定位：**横切工具生态**——为 M2（副脑）、M3（主脑 API 调用与身份守卫）、主脑 API（经 LangChain Agent）提供统一的工具注册、调度、组合能力。所有工具调用经 M5 统一调度（注册 / 鉴权 / 超时 / 重试），调用方不直接连接工具后端，从而保证可观测、可降级、可热插拔。
>
> **架构反转要点**：旧版"主脑=Qwen2.5-3B 本地、副脑=规则"已反转为"主脑=外部 LLM API、副脑=Qwen2.5-3B+V30"。M5 工具生态本身职责基本不变（注册/调度/组合），主要调整在调用方语义与内部工具清单——副脑侧 TFGR 资产（W_probe / identity_cache / self_state_predictor / TFGR 时间流图 / proactive_traverser）统一暴露为 M5 内部工具，供 M2 / M3 通过同一调度通道调用，避免"绕过 M5 直连共享态"破坏横切一致性。

---

## 1. 摘要

Lucy 的"活着感"由副脑（M2，Qwen2.5-3B + V30）维持。副脑在决策循环中需要大量外部能力支撑：查日历以判断是否该主动提醒、查天气以决定是否建议带伞、查 TFGR 时间流图以选取候选事件、调智能家居以营造氛围、发消息以替用户完成事务。主脑 API（DeepSeek/OpenAI 兼容，经 LangChain Agent）在回答实时信息问题时也可能需要外部工具（网络搜索、知识检索、翻译）。M3 在辅助主脑 API 做 β 深度推理时同样需要外部知识注入。这些能力的提供者就是 M5——**MCP 工具生态**。

M5 基于 **MCP（Model Context Protocol）** 协议标准化工具的描述、注册、发现与调用。M5 的核心职责有四：

1. **注册中心**：维护工具元数据（名称 / 描述 / 参数 schema / 超时 / 重试 / 权限 / 所属路径白名单），支持动态注册与热插拔。
2. **调度引擎**：根据工具名路由到对应执行器，统一施加超时、重试、并发控制、鉴权与限流。
3. **工具组合**：基于 LangChain 的工具链与 Agent 模式，支持多工具编排与条件调用（如：天气 + 日历 → 出行建议）。
4. **时延管理**：严格区分 α / β 路径——α 路径仅允许本地快工具同步执行（<100ms），云端工具走 β 路径异步执行或 α 路径先回占位、结果后补。

M5 是横切模块，本身无业务状态，所有状态由 M1（UserState）与副脑侧 TFGR 共享态（identity_cache / W_probe / self_state_predictor / TFGR 时间流图）维护。M5 的设计目标是：**即使某个工具后端宕机，调用方仍能拿到明确的失败信号并优雅降级，绝不阻塞主对话管线。**

### 核心数据契约

```python
# 调用方 → M5
@dataclass
class ToolCallRequest:
    tool_name: str                       # 如 "calendar.query"
    arguments: dict                      # 符合该工具的参数 schema
    caller: Literal["M2", "M3", "brain"] # 调用方标识：M2=副脑、M3=主脑API调用与身份守卫、brain=主脑API（经LangChain Agent）
    path: Literal["alpha", "beta"]       # 所在路径
    timeout_ms: Optional[int] = None     # 可选，覆盖默认超时（不可超过注册默认值）
    call_id: str = field(default_factory=lambda: uuid4().hex)  # 链路追踪

# M5 → 调用方
@dataclass
class ToolCallResult:
    call_id: str
    status: Literal["success", "timeout", "error", "rejected"]
    output: Optional[dict]               # 工具返回的结构化结果
    error: Optional[str]                 # 失败原因
    latency_ms: int                      # 实际耗时
    from_cache: bool = False             # 是否命中缓存
    fallback_used: bool = False          # 是否走了降级兜底
```

### 时延契约

| 工具类别 | 路径约束 | 超时预算 | 失败处理 |
|---|---|---|---|
| 本地工具 | α/β 均可 | <100ms | 失败立即返回，不重试（α 路径） |
| 云端工具 | α 路径仅"先回占位 + 后补"；β 路径可同步等 | 2-5s | 指数退避重试 1 次，超时降级 |
| 内部工具 | α/β 均可 | 50-200ms（记忆/画像 50ms；TFGR 图/self_state 100-200ms；W_probe/identity 10-30ms） | 失败返回空结果，调用方自降级 |
| 任意工具 | α 路径 | 工具调用总和不使 α 首 token 超 2.0s | 见 §8 |

---

## 2. 模块定位（横切工具生态）

### 2.1 在系统中的位置

```
┌──────────────────────────────────────────────────────────────────┐
│  调用方：                                                          │
│   • M2 副脑（Qwen2.5-3B+V30）— 主动触发依据 + 替用户执行事务        │
│   • M3 主脑API调用与身份守卫 — 辅助主脑API做β深度推理               │
│   • 主脑API（经LangChain Agent）— 实时信息问题（α路径）             │
│   • M1 用户建模 — 仅调内部工具读状态                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │ ToolCallRequest（统一契约）
                           ▼
              ┌────────────────────────────┐
              │     M5 MCP 调度引擎         │
              │  注册中心 / 鉴权 / 超时      │
              │  重试 / 并发 / 缓存 / 限流   │
              └────────────┬───────────────┘
                           │ 按工具名路由
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌───────────┐     ┌─────────────┐    ┌──────────────┐
  │ 本地执行器 │     │ 云端执行器   │    │ 内部执行器    │
  │ <100ms     │     │ 2-5s / 异步  │    │ 50-200ms     │
  │ 日历/文件/ │     │ 天气/搜索/   │    │ TFGR图/记忆/ │
  │ 智能家居   │     │ 知识库/翻译   │    │ 画像/W_probe │
  │           │     │             │    │ identity/    │
  │           │     │             │    │ self_state/  │
  │           │     │             │    │ traverser    │
  └───────────┘     └─────────────┘    └──────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │ ToolCallResult
                           ▼
                       返回调用方
```

### 2.2 边界与不做的事

| M5 做 | M5 不做 |
|---|---|
| 工具注册、元数据维护、动态发现 | 不实现具体业务逻辑（业务逻辑在工具后端） |
| 统一调度、路由、超时、重试 | 不决定"该不该调工具"（由调用方决策） |
| 鉴权、权限校验、调用配额 | 不修改 UserState（M1 职责） |
| 工具结果缓存 | 不生成自然语言（主脑 API 职责） |
| LangChain 工具链编排 | 不直接对用户输出 |
| 调用埋点与 metrics 上报 | 不持久化对话历史 |
| 内部工具执行器代理访问副脑共享态 | 不直接训练或更新 W_probe / identity_cache（副脑侧职责） |

### 2.3 横切特性

M5 是**横切关注点**（主文档 §5.7），与垂直模块的关系：

- **M2 副脑调 M5**：主动触发依据（日历/天气/TFGR 图候选事件）、替用户执行事务（发消息/控家居/读文件）、读自身状态（self_state_predictor / identity_cache）
- **M3 主脑 API 调用与身份守卫调 M5**：辅助主脑 API 做 β 深度推理（检索外部知识 search.web / knowledge.retrieve）；调 W_probe 异己检测工具做生成后异己分类；调 identity.get 工具取身份锚快照序列化进 system prompt
- **主脑 API（经 LangChain Agent）调 M5**：回答实时信息问题（网络搜索/翻译/股票），α 路径下需"先回占位 + 后补"或仅用本地快工具
- **M1 调 M5**：仅限内部工具（user_profile.update / memory.store）做状态写入；读 UserState 走共享内存不经 M5
- **M4 / M6 不直接调 M5**：M4 多模态管线与 M6 身体控制环有独立的数据通路，不经过工具调度

这种"所有外部能力与内部状态读取都收敛到 M5"的设计带来三个收益：**可观测**（所有工具调用统一埋点）、**可降级**（统一超时与兜底）、**可热插拔**（新增工具不影响调用方代码）。

> **架构反转后的关键变化**：副脑侧 TFGR 资产（W_probe / identity_cache / self_state_predictor / TFGR 时间流图 / proactive_traverser）原本散落在副脑内部直接调用，现统一暴露为 M5 内部工具。这避免了 M3 在做主脑 API 身份守卫时绕过 M5 直接访问副脑共享态（破坏横切一致性），也让 M2 在主动触发决策时用同一调度通道拉取候选事件与自身状态（统一埋点）。

---

## 3. MCP 协议基础

### 3.1 MCP 协议简介（Model Context Protocol）

MCP（Model Context Protocol）是由 Anthropic 于 2024 年提出的**开放工具调用协议**，旨在标准化 LLM 与外部工具/数据源之间的交互。其核心思想借鉴了 LSP（Language Server Protocol）：定义一个统一的 JSON-RPC 消息格式，让任意 LLM 客户端能调用任意符合协议的工具服务器，解耦"模型"与"工具"。

**MCP 的三个核心原语**：

| 原语 | 方向 | 用途 |
|---|---|---|
| `tools/list` | Client → Server | 列出服务器提供的工具及其 schema |
| `tools/call` | Client → Server | 调用某个工具，传入参数，获取结果 |
| `resources/read` | Client → Server | 读取资源（文件/URL 等，本系统较少用） |

**MCP 的传输层**支持 stdio（本地进程通信）与 SSE/HTTP（远程通信）。M5 同时支持两种：本地工具用 stdio 子进程（低延迟），云端工具用 HTTP/SSE。内部工具不真正起 MCP server，而是在进程内直接调用副脑共享态 API，但对外仍按 MCP 描述格式暴露（统一接口）。

**为什么选 MCP 而非自定义协议**：

1. **标准化**：工具描述用 JSON Schema，与 LangChain/OpenAI function calling 原生兼容，无需适配层——主脑 API（DeepSeek/OpenAI 兼容）的 function calling 能力可直接消费 M5 工具
2. **生态**：可直接复用社区已有的 MCP server（如 filesystem、git、slack 等官方实现）
3. **解耦**：工具后端可独立部署/升级，M5 只需重新 `tools/list` 即可发现变更
4. **可观测**：统一的调用格式便于埋点与链路追踪

### 3.2 工具描述格式（JSON Schema）

每个 MCP 工具通过 JSON Schema 描述其名称、用途与参数。M5 在注册中心存储的元数据即以此格式为基础扩展：

```json
{
  "name": "calendar.query",
  "description": "查询用户日历中指定时间范围内的事件。用于副脑判断是否有即将到来的事件作为主动触发依据。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_time": {
        "type": "string",
        "format": "date-time",
        "description": "查询起始时间（ISO 8601）"
      },
      "end_time": {
        "type": "string",
        "format": "date-time",
        "description": "查询结束时间（ISO 8601）"
      },
      "calendar_id": {
        "type": "string",
        "description": "日历 ID，默认 primary",
        "default": "primary"
      }
    },
    "required": ["start_time", "end_time"]
  }
}
```

M5 在标准 MCP 描述上**扩展**了调度所需的元数据（标准 MCP 不包含这些，由 M5 注册中心维护）：

```json
{
  "name": "calendar.query",
  "description": "...",
  "inputSchema": { ... },
  "_m5_meta": {
    "category": "local",
    "default_timeout_ms": 100,
    "retry": { "max_attempts": 0, "backoff_ms": 0 },
    "permissions": ["M2", "M3", "brain"],
    "path_whitelist": ["alpha", "beta"],
    "cache_ttl_ms": 30000,
    "rate_limit": { "per_minute": 60, "per_hour": 2000 },
    "fallback": "return_empty"
  }
}
```

### 3.3 工具调用标准流程

一次完整的 MCP 工具调用经 M5 调度，流程如下：

```
调用方                M5调度引擎              工具执行器
  │                      │                       │
  │ ToolCallRequest      │                       │
  ├─────────────────────▶│                       │
  │                      │ 1. 鉴权（caller是否可调）
  │                      │ 2. 路径校验（path是否在白名单）
  │                      │ 3. 限流检查            │
  │                      │ 4. 缓存命中？          │
  │                      │    命中→直接返回       │
  │                      │ 5. 超时/重试策略装载   │
  │                      │   tools/call (JSON-RPC)│
  │                      ├──────────────────────▶│
  │                      │                       │ 执行
  │                      │◀──────────────────────┤
  │                      │   result / error      │
  │                      │ 6. 超时？→ 重试或降级 │
  │                      │ 7. 写缓存             │
  │                      │ 8. 埋点 metrics       │
  │ ToolCallResult       │                       │
  │◀─────────────────────┤                       │
```

**JSON-RPC 消息示例**（M5 → 工具执行器）：

```json
{
  "jsonrpc": "2.0",
  "id": "call-7f3a",
  "method": "tools/call",
  "params": {
    "name": "calendar.query",
    "arguments": {
      "start_time": "2026-08-03T09:00:00+08:00",
      "end_time": "2026-08-03T18:00:00+08:00"
    }
  }
}
```

工具执行器返回：

```json
{
  "jsonrpc": "2.0",
  "id": "call-7f3a",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"events\":[{\"title\":\"团队周会\",\"start\":\"2026-08-03T10:00:00+08:00\",\"end\":\"2026-08-03T11:00:00+08:00\"}]}"
      }
    ],
    "isError": false
  }
}
```

M5 将 `content[0].text` 解析为 dict，封装进 `ToolCallResult.output` 返回调用方。

---

## 4. 工具注册中心

### 4.1 注册协议

工具在可用前必须向 M5 注册中心登记。注册支持两种模式：

- **静态注册**：系统启动时从配置文件（`tools.yaml`）加载内置工具清单
- **动态注册**：运行时通过 `register` API 热登记新工具（见 §4.3）

注册时的元数据结构：

```python
@dataclass
class ToolMeta:
    name: str                           # 全局唯一，如 "calendar.query"
    description: str                    # 给 LLM/调用方看的说明
    input_schema: dict                  # JSON Schema，校验参数
    category: Literal["local", "cloud", "internal"]
    executor: str                       # 执行器标识，如 "local.calendar" / "http://weather-svc" / "internal.w_probe"
    default_timeout_ms: int
    retry_policy: RetryPolicy
    permissions: list[str]              # 允许的 caller，如 ["M2","M3","brain"]
    path_whitelist: list[str]           # 允许的路径，如 ["alpha","beta"]
    cache_ttl_ms: int                   # 0 表示不缓存
    rate_limit: Optional[RateLimit]
    fallback: Literal["return_empty", "return_none", "raise"]
    enabled: bool = True
    version: str = "1.0"

@dataclass
class RetryPolicy:
    max_attempts: int                   # 含首次，0 表示不重试
    backoff_ms: int                     # 指数退避基数
    backoff_max_ms: int
    retry_on: list[str]                 # 可重试的错误码/类型

@dataclass
class RateLimit:
    per_minute: int
    per_hour: int
```

### 4.2 工具元数据详解

每个元数据字段的语义与约束：

| 字段 | 语义 | 约束 |
|---|---|---|
| `name` | 全局唯一标识，点分命名（`域.动作`） | 必须匹配 `^[a-z]+\.[a-z_]+$` |
| `input_schema` | JSON Schema，用于参数校验 | 必须是合法 JSON Schema draft-07 |
| `category` | 决定默认超时与执行器类型 | local/cloud/internal 三选一 |
| `default_timeout_ms` | 单次调用超时 | local≤100，internal 10-200，cloud≤5000 |
| `retry_policy` | 重试策略 | α 路径工具建议 max_attempts=0 |
| `permissions` | 允许的调用方 | 未列出的 caller 调用被拒（rejected） |
| `path_whitelist` | 允许的路径 | α 路径调用云工具会被拒（除非标记 `alpha_placeholder`） |
| `cache_ttl_ms` | 结果缓存有效期 | 0 表示不缓存；实时性工具（股票）应为 0；身份/状态类工具（identity.get / self_state.read）应短 TTL（5-10s）或不缓存 |
| `rate_limit` | 调用频率上限 | 防止 LLM 失控刷爆外部 API；内部工具如 W_probe 应设较高 per_minute（1000+） |
| `fallback` | 失败兜底策略 | return_empty 返回空结构；return_none 返回 None；raise 抛错 |

### 4.3 动态注册与热插拔

M5 支持运行时动态注册新工具，无需重启：

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolMeta] = {}
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[str, str], None]] = []  # (event, tool_name)

    def register(self, meta: ToolMeta, overwrite: bool = False) -> bool:
        with self._lock:
            if meta.name in self._tools and not overwrite:
                logger.warning("Tool already registered: %s", meta.name)
                return False
            # 校验 schema 合法性
            try:
                jsonschema.Draft7Validator.check_schema(meta.input_schema)
            except jsonschema.SchemaError as e:
                logger.error("Invalid schema for %s: %s", meta.name, e)
                return False
            self._tools[meta.name] = meta
            self._notify("register", meta.name)
            logger.info("Tool registered: %s (category=%s)", meta.name, meta.category)
            return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._tools:
                return False
            del self._tools[name]
            self._notify("unregister", name)
            logger.info("Tool unregistered: %s", name)
            return True

    def disable(self, name: str):
        """热禁用：保留注册但拒绝调用，用于故障隔离"""
        with self._lock:
            if name in self._tools:
                self._tools[name].enabled = False

    def get(self, name: str) -> Optional[ToolMeta]:
        with self._lock:
            return self._tools.get(name)

    def list_all(self) -> list[ToolMeta]:
        with self._lock:
            return list(self._tools.values())
```

**热插拔场景**：智能家居网关重启时，`smarthome.*` 工具自动 disable，期间所有调用返回 `status=rejected`，调用方走兜底；网关恢复后，工具自动 re-enable。内部工具（如 `w_probe.detect`）依赖副脑 GPU 显存，副脑进程重启后自动 re-register。

### 4.4 工具发现机制

调用方（尤其是主脑 API 的 LangChain Agent）需要知道当前有哪些工具可用。M5 提供 `tools/list` 接口：

```python
def list_tools_for_caller(
    caller: str,
    path: str,
    category: Optional[str] = None,
) -> list[dict]:
    """返回调用方在指定路径下可用的工具描述（LangChain 兼容格式）"""
    result = []
    for meta in self.list_all():
        if not meta.enabled:
            continue
        if caller not in meta.permissions:
            continue
        if path not in meta.path_whitelist:
            continue
        if category and meta.category != category:
            continue
        result.append({
            "name": meta.name,
            "description": meta.description,
            "input_schema": meta.input_schema,
        })
    return result
```

主脑 API 在进入 LangChain Agent 模式时，调用 `list_tools_for_caller(caller="brain", path="alpha")` 获取当前可用工具集，动态构造 Agent 的 `tools` 参数。M3 在做 β 深度推理前调用 `list_tools_for_caller(caller="M3", path="beta")` 取可用知识源工具。

---

## 5. 工具分类与清单

M5 工具按延迟特性分为三类：**本地工具**（边缘设备本机执行，<100ms）、**云端工具**（HTTP/SSE 远程调用，2-5s）、**内部工具**（系统自身能力暴露为工具，50-200ms）。

> **调用方标识约定**（贯穿本文档）：
> - `M2` = 副脑（Qwen2.5-3B + V30，本地 GPU）
> - `M3` = 主脑 API 调用与身份守卫
> - `brain` = 主脑 API（外部 LLM，经 LangChain Agent 调用）
> - `M1` = 动态用户建模（仅可调内部工具的状态写入类）

### 5.1 本地工具（<100ms，α/β 均可，权限 M2/M3/brain）

本地工具在边缘设备本机执行，无网络往返，延迟 <100ms。典型如日历、文件系统、智能家居控制、系统控制、联系人查询。这些工具可安全进入 α 路径（受 §8.2 的 α 预算 100ms 约束）。

**调用方语义**：
- M2 调本地工具：主动触发依据（`calendar.query` 查即将事件）、替用户执行事务（`smarthome.*` 调氛围、`message.send` 替用户发消息——注：`message.send` 实为云端工具，见 §5.2）
- M3 调本地工具：β 深度推理时辅助读取本地文件（`filesystem.read`）做上下文增强
- 主脑 API（brain）调本地工具：LangChain Agent 模式回答实时问题（如 `calendar.query` 答用户"我下午几点有事"）

### 5.2 云端工具（2-5s，α 仅先回占位后补 / β 可同步，权限 M2/M3/brain）

云端工具通过 HTTP/SSE 调用远程服务，典型延迟 2-5s。受网络抖动影响可能超时。**α 路径原则上不直接同步等待云工具结果**——采用"先回占位 + 结果后补"策略（§8.1）。β 路径可同步等待或异步。

**调用方语义**：
- M2 调云端工具：主动触发依据（`weather.query` 判带伞、`search.news` 早晨播报）、替用户执行事务（`email.send` / `message.send`，需用户授权）
- M3 调云端工具：辅助主脑 API 做 β 深度推理时检索外部知识（`search.web` / `knowledge.retrieve`），结果注入主脑 API 的思考模式调用 prompt
- 主脑 API（brain）调云端工具：LangChain Agent 回答实时信息问题（`search.web` / `weather.query` / `stock.quote`），α 路径下需先回占位"让我查一下..."

### 5.3 内部工具（50-200ms，权限 M2/M3，少数 brain 可读）

内部工具把系统自身的能力以 MCP 工具形式暴露，统一调度。**架构反转后这部分是 M5 的重点改造**——副脑侧 TFGR 资产（W_probe / identity_cache / self_state_predictor / TFGR 时间流图 / proactive_traverser）原本散落直接调用，现统一暴露为 M5 内部工具，供 M2 / M3 通过同一调度通道访问。延迟介于本地与云端之间（10-200ms），可进 α 路径但需控制调用次数。

**内部工具分两组**：

1. **状态读取类**（M2/M3 频繁调用，部分 brain 可读）：
   - `tfgr.query` — TFGR 时间流图查询（节点/边/候选事件，主文档 §7 复用映射）
   - `memory.retrieve` — 长期记忆向量检索（Chroma top_k）
   - `user_profile.query` — 用户画像字段查询（偏好/关系/已知事实）
   - `w_probe.detect` — **W_probe 异己检测**（对主脑 API 生成内容做异己分类，主文档 §5.4 M3 生成后异己检测的核心实现）
   - `identity.get` — **identity_cache 查询**（读取 Lucy 身份锚快照摘要，供 M3 序列化进主脑 API system prompt）
   - `self_state.read` — **self_state_predictor 读取**（副脑自身 V/PE/energy/confidence 状态，主文档 §7 资产复用映射）
   - `proactive_traverser.decide` — **proactive_traverser 决策**（5 阶段推理链姿态选择 FOLLOW_UP/CHECK_IN/SILENCE/PASSIVE_REPLY，主文档 §7）
   - `context.bundle` — 语境包组装（UserState + TFGR + 对话历史摘要）
   - `emotion.analyze` — 文本情绪分析（valence/arousal）

2. **状态写入类**（仅 M2 / M1 可调）：
   - `tfgr.add_node` — 向 TFGR 图添加新节点（M2 在对话事件发生后调用）
   - `memory.store` — 存储新记忆条目（M2 / M1 调用）
   - `user_profile.update` — 更新用户画像字段（M1 调用）

**关键约束**：
- `w_probe.detect` 是 M3 生成后异己检测的工具化入口——M3 在流式接收主脑 API token 时按 batch 调用此工具，异己分数 > θ_safety 触发缓冲/重生成。这与主文档 §5.4 描述的 W_probe 守卫机制完全一致，只是经 M5 调度统一埋点。
- `identity.get` 返回的是 identity_cache 的**快照摘要**（不可变视图），M3 据此构造 system prompt。M3 不直接读副脑显存。
- `self_state.read` 与 `proactive_traverser.decide` 是副脑自身能力的对外暴露——主脑 API（brain）一般不调（无业务必要）；M2 在主动触发循环中调（统一埋点）；M3 在判定主脑 API 是否需重生成时可读 self_state 判身份稳定性。
- 内部工具的执行器是 `InternalExecutor`，直接访问副脑共享态 API（进程内调用，无网络），不真正起 MCP server。

### 5.4 具体工具清单

下表列出 M5 注册中心维护的核心工具清单（按类别分组）。每个工具均符合 §3.2 的 JSON Schema 描述格式，此处为紧凑表达。

#### 5.4.1 本地工具

| 工具名 | 描述 | 默认超时 | 路径白名单 | 权限 |
|---|---|---|---|---|
| `calendar.query` | 查询指定时间范围内的日历事件 | 80ms | α,β | M2,M3,brain |
| `calendar.create` | 创建新的日历事件 | 100ms | β | M2,brain |
| `calendar.update` | 修改/取消已有事件 | 100ms | β | M2,brain |
| `filesystem.read` | 读取本地文件内容（受沙箱限制） | 50ms | α,β | M2,M3,brain |
| `filesystem.write` | 写入本地文件 | 50ms | β | M2,brain |
| `filesystem.list` | 列出目录下文件 | 30ms | α,β | M2,M3,brain |
| `smarthome.light` | 控制灯光（开关/亮度/色温/场景） | 80ms | α,β | M2,brain |
| `smarthome.music` | 控制音乐播放（播放/暂停/切歌/音量） | 80ms | α,β | M2,brain |
| `smarthome.thermostat` | 控制温控设备 | 80ms | α,β | M2,brain |
| `smarthome.curtain` | 控制窗帘开合 | 80ms | α,β | M2,brain |
| `smarthome.scene` | 触发预设家居场景（如"观影模式"） | 100ms | α,β | M2,brain |
| `system.clock` | 获取当前时间/时区/主观时间 R(e) | 5ms | α,β | M2,M3,brain |
| `system.notify` | 在机器人屏幕显示本地通知 | 50ms | α,β | M2,brain |
| `contacts.lookup` | 查询联系人信息 | 50ms | α,β | M2,brain |

#### 5.4.2 云端工具

| 工具名 | 描述 | 默认超时 | 路径白名单 | 权限 |
|---|---|---|---|---|
| `weather.query` | 查询当前天气（温度/天气/风力/湿度） | 3000ms | α(占位),β | M2,M3,brain |
| `weather.forecast` | 查询未来 N 天天气预报 | 4000ms | β | M2,M3,brain |
| `search.web` | 通用网络搜索（返回摘要+链接） | 4000ms | α(占位),β | M3,brain |
| `search.news` | 新闻搜索（按关键词+时间） | 4000ms | β | M2,M3,brain |
| `knowledge.retrieve` | 知识库语义检索（RAG，top_k 结果） | 3000ms | α(占位),β | M3,brain |
| `map.route` | 路线规划（起终点+交通方式） | 3000ms | β | M2,brain |
| `map.geocode` | 地址↔经纬度转换 | 2000ms | α,β | M2,brain |
| `translate.text` | 文本翻译（多语言） | 3000ms | α,β | brain |
| `email.send` | 发送邮件（需用户授权） | 4000ms | β | M2,brain |
| `message.send` | 发送即时消息（微信/短信，需授权） | 4000ms | β | M2,brain |
| `stock.quote` | 查询股票实时行情 | 2000ms | α(占位),β | brain |
| `flight.query` | 航班查询 | 4000ms | β | M2,brain |
| `currency.convert` | 汇率换算 | 2000ms | α,β | brain |
| `media.search` | 音乐/影视资源搜索 | 3000ms | β | M2,brain |

#### 5.4.3 内部工具

| 工具名 | 描述 | 默认超时 | 路径白名单 | 权限 |
|---|---|---|---|---|
| `tfgr.query` | 查询 TFGR 时间流图节点/边/候选事件 | 100ms | α,β | M2,M3 |
| `tfgr.add_node` | 向 TFGR 图添加新节点（事件落盘） | 50ms | β | M2 |
| `memory.retrieve` | 长期记忆向量检索（top_k） | 80ms | α,β | M2,M3,brain |
| `memory.store` | 存储新记忆条目 | 50ms | β | M2,M1 |
| `user_profile.query` | 查询用户画像字段（偏好/关系/已知事实） | 30ms | α,β | M2,M3,brain |
| `user_profile.update` | 更新用户画像字段 | 30ms | β | M1 |
| `w_probe.detect` | **W_probe 异己检测**：对主脑 API 生成内容做异己分类，返回分数 | 10ms | α,β | M2,M3 |
| `identity.get` | **identity_cache 查询**：读取 Lucy 身份锚快照摘要 | 20ms | α,β | M2,M3 |
| `self_state.read` | **self_state_predictor 读取**：副脑自身 V/PE/energy/confidence | 50ms | α,β | M2,M3 |
| `proactive_traverser.decide` | **proactive_traverser 决策**：5阶段推理链姿态选择 | 200ms | β | M2 |
| `emotion.analyze` | 对文本做情绪分析（valence/arousal） | 100ms | α,β | M2,M3 |
| `context.bundle` | 组装当前语境包（UserState+TFGR+历史） | 50ms | α,β | M2,M3 |

**清单规模说明**：上述合计 **40 个工具**（本地 14 + 云端 14 + 内部 12）。其中本地与内部工具共 26 个可在 α 路径同步调用（受 §8.2 α 预算约束），云端工具 14 个走 β 路径或 α 占位策略。工具清单可随业务扩展持续动态注册。

---

## 6. 调度引擎

调度引擎是 M5 的核心，所有 `ToolCallRequest` 都经它处理后才能到达工具执行器。

### 6.1 调用路由（根据工具名 → 执行器）

```python
class Scheduler:
    def __init__(self, registry: ToolRegistry, executors: dict[str, Executor]):
        self.registry = registry
        self.executors = executors   # {"local": LocalExecutor, "cloud": CloudExecutor, "internal": InternalExecutor}

    async def invoke(self, req: ToolCallRequest) -> ToolCallResult:
        t0 = time.perf_counter()
        meta = self.registry.get(req.tool_name)
        if meta is None or not meta.enabled:
            return ToolCallResult(call_id=req.call_id, status="rejected",
                                  output=None, error="tool_not_found_or_disabled",
                                  latency_ms=0)

        # 1. 鉴权
        if req.caller not in meta.permissions:
            metrics.m5_auth_reject.inc(req.tool_name)
            return ToolCallResult(call_id=req.call_id, status="rejected",
                                  output=None, error="permission_denied",
                                  latency_ms=int((time.perf_counter()-t0)*1000))

        # 2. 路径校验
        if req.path not in meta.path_whitelist:
            metrics.m5_path_reject.inc(req.tool_name)
            return ToolCallResult(call_id=req.call_id, status="rejected",
                                  output=None, error=f"path_{req.path}_not_allowed",
                                  latency_ms=int((time.perf_counter()-t0)*1000))

        # 3. 限流
        if not self._rate_limit_allow(meta, req.caller):
            return ToolCallResult(call_id=req.call_id, status="rejected",
                                  output=None, error="rate_limited",
                                  latency_ms=int((time.perf_counter()-t0)*1000))

        # 4. 缓存
        if meta.cache_ttl_ms > 0:
            cached = self._cache_get(req, meta)
            if cached is not None:
                metrics.m5_cache_hit.inc(req.tool_name)
                return ToolCallResult(call_id=req.call_id, status="success",
                                      output=cached, error=None,
                                      latency_ms=int((time.perf_counter()-t0)*1000),
                                      from_cache=True)

        # 5. 路由到执行器
        executor = self.executors[meta.category]
        timeout_ms = req.timeout_ms or meta.default_timeout_ms
        # 调用方不可超过注册默认值（防止 LLM 失控拖死 α 路径）
        timeout_ms = min(timeout_ms, meta.default_timeout_ms)

        # 6. 执行（带超时+重试）
        result = await self._invoke_with_retry(executor, meta, req, timeout_ms, t0)

        # 7. 写缓存
        if result.status == "success" and meta.cache_ttl_ms > 0 and not result.from_cache:
            self._cache_set(req, meta, result.output)

        # 8. metrics
        metrics.m5_invoke_total.inc(req.tool_name, req.caller, req.path)
        metrics.m5_latency.observe(req.tool_name, result.latency_ms)
        return result
```

### 6.2 超时管理（本地 <100ms / 云端 2-5s / 内部 50-200ms）

超时是 α 路径时延预算的关键护栏。M5 强制执行工具注册时声明的 `default_timeout_ms`，调用方可缩短但不可延长（防止 LLM 失控把 α 路径拖死）：

```python
async def _invoke_with_retry(self, executor, meta, req, timeout_ms, t0):
    attempt = 0
    max_attempts = meta.retry_policy.max_attempts
    backoff = meta.retry_policy.backoff_ms

    while attempt <= max_attempts:
        try:
            output = await asyncio.wait_for(
                executor.call(meta, req.arguments),
                timeout=timeout_ms / 1000,
            )
            return ToolCallResult(
                call_id=req.call_id, status="success",
                output=output, error=None,
                latency_ms=int((time.perf_counter()-t0)*1000),
            )
        except asyncio.TimeoutError:
            attempt += 1
            metrics.m5_timeout.inc(req.tool_name, attempt)
            if attempt > max_attempts:
                return self._fallback(meta, req, "timeout", t0)
            await asyncio.sleep(backoff * (2 ** (attempt-1)) / 1000)
        except Exception as e:
            attempt += 1
            metrics.m5_error.inc(req.tool_name, type(e).__name__)
            if not self._is_retryable(e, meta) or attempt > max_attempts:
                return self._fallback(meta, req, f"error:{e}", t0)
            await asyncio.sleep(backoff * (2 ** (attempt-1)) / 1000)
```

**超时预算与 α 路径首 token 预算的关系**：α 路径首 token 端到端预算 2.0s（主文档 §4.1）。若 α 路径调用了本地/内部工具，所有工具调用总耗时不得超过 100ms（否则挤压主脑 API 推理时间）。M5 在 α 路径对工具调用做"预算检查"：

```python
ALPHA_TOOL_BUDGET_MS = 100  # α 路径工具调用总预算

def check_alpha_budget(self, path: str, estimated_ms: int) -> bool:
    if path != "alpha":
        return True
    used = self._alpha_tool_used_ms  # 本次 α 请求累计工具耗时
    if used + estimated_ms > ALPHA_TOOL_BUDGET_MS:
        metrics.m5_alpha_budget_exceeded.inc()
        return False
    return True
```

### 6.3 重试策略（指数退避）

```python
def _is_retryable(self, error, meta: ToolMeta) -> bool:
    # 网络错误、5xx、429 可重试；4xx（除429）、参数错误不重试
    if isinstance(error, (httpx.ConnectError, httpx.ReadTimeout)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {429, 500, 502, 503, 504}
    return False
```

**重试策略矩阵**：

| 工具类别 | max_attempts | backoff_ms | 说明 |
|---|---|---|---|
| 本地工具（α 路径） | 0 | — | 不重试，失败立即返回，保护 α 时延 |
| 本地工具（β 路径） | 1 | 50 | 允许 1 次快速重试 |
| 云端工具（β 路径） | 1 | 500 | 指数退避 1 次 |
| 内部工具 | 1 | 50 | 允许 1 次重试；W_probe / identity 类不重试（直接降级） |

### 6.4 并发控制（同时调用多个工具）

M2 副脑在一个决策 tick 中可能需要同时查日历、查天气、查 TFGR 图、读 self_state。M5 提供并发调用接口：

```python
async def invoke_parallel(self, requests: list[ToolCallRequest]) -> list[ToolCallResult]:
    """并发调用多个工具，结果按输入顺序返回"""
    tasks = [self.invoke(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if isinstance(r, ToolCallResult) else
        ToolCallResult(call_id=req.call_id, status="error",
                       output=None, error=str(r), latency_ms=0)
        for req, r in zip(requests, results)
    ]
```

**并发示例**（M2 主动触发决策时并行查四类信息）：

```python
results = await m5.invoke_parallel([
    ToolCallRequest(tool_name="calendar.query", arguments={"start_time": now, "end_time": now+3600},
                    caller="M2", path="beta"),
    ToolCallRequest(tool_name="weather.query", arguments={"location": user_home},
                    caller="M2", path="beta"),
    ToolCallRequest(tool_name="tfgr.query", arguments={"mode": "candidates", "top_k": 5},
                    caller="M2", path="beta"),
    ToolCallRequest(tool_name="self_state.read", arguments={},
                    caller="M2", path="beta"),
])
# 四个工具并发，总耗时≈最慢的那个（天气~3s），而非串行累加
```

### 6.5 鉴权与权限

权限分两层：

1. **caller 权限**：`permissions` 字段声明哪些模块可调（如 `w_probe.detect` 仅 M2/M3 可调，brain 不可调——主脑 API 不能自检异己）
2. **路径权限**：`path_whitelist` 声明可在哪些路径调（如云端工具不在 α 路径白名单，除非标记 `alpha(占位)`）

**敏感工具额外授权**：`email.send` / `message.send` / `calendar.create` 这类有副作用的工具，首次调用时需用户确认（M2 通过 M4 弹出确认 UI）。M5 记录授权状态：

```python
class ConsentStore:
    """用户对敏感工具的授权记录"""
    def __init__(self):
        self._granted: dict[str, float] = {}  # tool_name -> granted_ts
        self._consent_ttl_s = 3600  # 授权 1 小时内有效

    def is_granted(self, tool_name: str, now: float) -> bool:
        ts = self._granted.get(tool_name)
        return ts is not None and now - ts < self._consent_ttl_s

    def grant(self, tool_name: str, now: float):
        self._granted[tool_name] = now

SENSITIVE_TOOLS = {"email.send", "message.send", "calendar.create",
                   "calendar.update", "filesystem.write"}

# 调度时检查
if req.tool_name in SENSITIVE_TOOLS and not consent_store.is_granted(req.tool_name, now):
    return ToolCallResult(call_id=req.call_id, status="rejected",
                          output=None, error="consent_required", latency_ms=0)
```

---

## 7. 工具组合与编排

### 7.1 LangChain 工具链

M5 把每个注册工具适配为 LangChain 的 `BaseTool`，使其可被 LangChain Agent 直接调用——主脑 API（DeepSeek/OpenAI 兼容）的 function calling 能力原生支持：

```python
from langchain_core.tools import BaseTool, ToolException

class M5ToolAdapter(BaseTool):
    name: str
    description: str
    args_schema: type
    m5_scheduler: Scheduler
    caller: str
    path: str

    def _run(self, **kwargs) -> dict:
        # 同步路径（仅本地/内部工具）
        req = ToolCallRequest(
            tool_name=self.name, arguments=kwargs,
            caller=self.caller, path=self.path,
        )
        result = asyncio.run(self.m5_scheduler.invoke(req))
        if result.status != "success":
            raise ToolException(f"{self.name} failed: {result.error}")
        return result.output

    async def _arun(self, **kwargs) -> dict:
        req = ToolCallRequest(
            tool_name=self.name, arguments=kwargs,
            caller=self.caller, path=self.path,
        )
        result = await self.m5_scheduler.invoke(req)
        if result.status != "success":
            raise ToolException(f"{self.name} failed: {result.error}")
        return result.output


def build_langchain_tools(
    scheduler: Scheduler, caller: str, path: str
) -> list[BaseTool]:
    """根据当前 caller/path 动态构造 LangChain 工具集"""
    metas = scheduler.list_tools_for_caller(caller, path)
    tools = []
    for meta in metas:
        args_schema = jsonschema_to_pydantic(meta.input_schema, meta.name)
        tools.append(M5ToolAdapter(
            name=meta.name, description=meta.description,
            args_schema=args_schema, m5_scheduler=scheduler,
            caller=caller, path=path,
        ))
    return tools
```

### 7.2 工具组合模板

某些场景需要多个工具协作。M5 预置工具组合模板，封装常用编排逻辑：

```python
class ToolComposition:
    """预置工具组合模板"""

    async def travel_advice(self, scheduler, user_id: str, when: str) -> dict:
        """天气 + 日历 → 出行建议"""
        # 并发调天气与日历
        results = await scheduler.invoke_parallel([
            ToolCallRequest("weather.forecast", {"location": user_home, "days": 1},
                            "M2", "beta"),
            ToolCallRequest("calendar.query", {"start_time": when, "end_time": end_of_day},
                            "M2", "beta"),
        ])
        weather = results[0].output or {}
        events = (results[1].output or {}).get("events", [])

        # 组合建议
        advice = []
        if weather.get("rain_probability", 0) > 0.5:
            advice.append("今天可能下雨，记得带伞")
        if any("meeting" in e.get("title","").lower() for e in events):
            advice.append("今天有会议，注意时间")
        return {"weather": weather, "events": events, "advice": advice}

    async def proactive_morning_brief(self, scheduler) -> dict:
        """早晨主动播报组合：天气 + 日历 + 新闻摘要"""
        results = await scheduler.invoke_parallel([
            ToolCallRequest("weather.query", {"location": user_home}, "M2", "beta"),
            ToolCallRequest("calendar.query", {"start_time": today, "end_time": today_end},
                            "M2", "beta"),
            ToolCallRequest("search.news", {"query": user_interests, "top_k": 3},
                            "M2", "beta"),
        ])
        return {"weather": results[0].output, "calendar": results[1].output,
                "news": results[2].output

    async def identity_guard_check(self, scheduler, generated_text: str) -> dict:
        """身份守卫组合：W_probe 异己检测 + identity_cache 比对 + self_state 读取"""
        results = await scheduler.invoke_parallel([
            ToolCallRequest("w_probe.detect", {"text": generated_text}, "M3", "alpha"),
            ToolCallRequest("identity.get", {}, "M3", "alpha"),
            ToolCallRequest("self_state.read", {}, "M3", "alpha"),
        ])
        alien_score = (results[0].output or {}).get("score", 0.0)
        identity_snapshot = results[1].output or {}
        self_state = results[2].output or {}
        # 判定是否需重生成：异己分数高 OR V 值偏移大
        need_regen = alien_score > THETA_SAFETY or self_state.get("v_drift", 0) > V_DRIFT_THRESHOLD
        return {
            "alien_score": alien_score,
            "identity_snapshot": identity_snapshot,
            "self_state": self_state,
            "need_regen": need_regen,
        }
```

### 7.3 条件调用（工具 A 结果决定是否调工具 B）

```python
async def maybe_remind_umbrella(scheduler, user_state) -> Optional[str]:
    """条件调用：查天气→若有雨→查日历判断是否出门→决定是否提醒"""
    weather = await scheduler.invoke(ToolCallRequest(
        "weather.query", {"location": user_state.home}, "M2", "beta"))
    if weather.status != "success":
        return None
    if weather.output.get("rain_probability", 0) < 0.4:
        return None  # 不下雨，不提醒

    # 有雨，进一步查日历判断是否出门
    cal = await scheduler.invoke(ToolCallRequest(
        "calendar.query", {"start_time": now, "end_time": end_of_day},
        "M2", "beta"))
    if cal.status == "success":
        events = cal.output.get("events", [])
        if any(is_outdoor(e) for e in events):
            return "今天有雨且你有外出安排，记得带伞"
    return None
```

---

## 8. 时延管理与路径分离

时延管理是 M5 的核心设计约束，严格遵循主文档 §4 的三路径分离原则。

### 8.1 α 路径工具调用策略（仅本地快工具同步，云端走 β 或先回占位后补）

α 路径用户在等，工具调用策略：

1. **白名单**：α 路径仅允许本地工具与内部工具同步调用，云端工具默认拒绝（`path_whitelist` 不含 alpha，除非标记 `alpha(占位)`）
2. **预算检查**：α 路径所有工具调用总耗时不得超过 100ms（§6.2）
3. **占位策略**：若用户问题需要云端工具结果，主脑 API 先回一句占位（如"让我查一下..."），然后切到 β 路径异步获取结果，再补一条消息

```python
async def alpha_tool_strategy(user_query: str) -> str:
    # 判断是否需要云端工具
    if needs_cloud_tool(user_query):
        # 先回占位（α 路径，<2s 完成）
        placeholder = await brain_api_generate_placeholder(user_query)
        # 异步触发云端工具（β 路径，不阻塞 α）
        asyncio.create_task(beta_followup_with_tools(user_query))
        return placeholder
    else:
        # 仅用本地/内部工具，走 Agent
        return await brain_answer_with_tools(user_query, path="alpha")
```

**占位策略的例外**：某些云端工具配置了 `path_whitelist: ["alpha(占位)","beta"]`，表示 α 路径允许"触发但不等待"——M5 立即返回一个 `call_id` 与 pending 状态，结果通过回调补送。

### 8.2 α 路径工具调用预算与首 token 关系

主文档 §4.1 给出 α 路径首 token 端到端预算 2.0s（目标 1.5s）。工具调用是其中一环：

```
α 路径首 token 预算分解（2.0s 上限）：
  ASR 首 chunk          ~300-500ms
  副脑快速判定           ~30-50ms（含 W_probe 输入检测）
  记忆检索              ~50-80ms（memory.retrieve 内部工具）
  语境包+身份prompt组装  ~50-80ms（含 identity.get）
  ────────────────────────────
  工具调用合计预算        ≤100ms（本地+内部工具）
  ────────────────────────────
  主脑 API 首 token      ~500-1500ms（网络+推理）
  TTS 首音频             ~300-400ms
```

**关键约束**：α 路径下 M5 的工具调用总耗时不得超过 100ms。M5 在每次 α 路径工具调用前做预算检查（§6.2 `check_alpha_budget`），超预算则拒绝并返回 `status=rejected`，调用方降级（不调工具，直接进主脑 API）。

### 8.3 β 路径工具调用（可异步）

β 路径用户没在等，工具调用可同步等待或异步：

```python
async def beta_tool_invoke(req: ToolCallRequest) -> ToolCallResult:
    # β 路径允许云端工具同步等待
    return await m5_scheduler.invoke(req)

async def beta_tool_fire_and_forget(req: ToolCallRequest) -> str:
    """β 路径异步触发，不等待结果（如发消息）"""
    asyncio.create_task(m5_scheduler.invoke(req))
    return req.call_id  # 仅返回 call_id 供追踪
```

### 8.4 工具结果缓存

缓存是降低延迟与外部 API 成本的关键手段：

```python
class ToolResultCache:
    def __init__(self, max_size: int = 1000):
        self.store: dict[str, tuple[float, dict]] = {}
        self.max_size = max_size
        self._lock = threading.Lock()

    def _key(self, tool_name: str, arguments: dict) -> str:
        sig = json.dumps({"t": tool_name, "a": arguments},
                         sort_keys=True, ensure_ascii=False)
        return hashlib.md5(sig.encode()).hexdigest()

    def get(self, tool_name: str, arguments: dict, ttl_ms: int, now: float) -> Optional[dict]:
        if ttl_ms <= 0:
            return None
        k = self._key(tool_name, arguments)
        with self._lock:
            if k in self.store:
                ts, val = self.store[k]
                if now - ts < ttl_ms / 1000:
                    return val
                del self.store[k]
        return None

    def set(self, tool_name: str, arguments: dict, value: dict, now: float):
        with self._lock:
            if len(self.store) >= self.max_size:
                self._evict_oldest()
            self.store[self._key(tool_name, arguments)] = (now, value)
```

**各工具缓存 TTL 建议**：

| 工具 | 缓存 TTL | 理由 |
|---|---|---|
| `weather.query` | 600s | 天气变化慢，10 分钟缓存 |
| `weather.forecast` | 3600s | 预报 1 小时缓存 |
| `calendar.query` | 30s | 日历可能随时新增事件 |
| `search.web` | 300s | 搜索结果 5 分钟缓存 |
| `knowledge.retrieve` | 1800s | 知识库更新慢 |
| `stock.quote` | 0 | 实时行情不缓存 |
| `tfgr.query` | 5s | TFGR 图本 tick 内可缓存 |
| `user_profile.query` | 60s | 画像 1 分钟缓存 |
| `w_probe.detect` | 0 | 不缓存——异己检测每次结果独立 |
| `identity.get` | 5s | identity_cache 短 TTL 缓存，避免频繁读显存 |
| `self_state.read` | 1s | 副脑状态变化快，1s TTL |

---

## 9. 与各模块的耦合

本节描述 M5 与各垂直模块的调用关系，遵守主文档 §5.7 M5 横切的约束。

### 9.1 M2 副脑调 M5（主动触发依据 + 替用户执行事务）

M2 副脑是 M5 的最大调用方。在主动触发决策循环中，M2 通过 M5 获取主动触发依据、读自身状态、替用户执行事务：

```python
# M2 副脑主动循环 tick
async def m2_proactive_tick(state: M2State):
    # 并发拉取主动触发依据 + 自身状态
    results = await m5.invoke_parallel([
        ToolCallRequest("calendar.query", {"start_time": now, "end_time": now+7200},
                        "M2", "beta"),
        ToolCallRequest("weather.query", {"location": state.user_home}, "M2", "beta"),
        ToolCallRequest("tfgr.query", {"mode": "candidates", "top_k": 5}, "M2", "beta"),
        ToolCallRequest("user_profile.query", {"fields": ["preferences","relationship"]},
                        "M2", "beta"),
        ToolCallRequest("self_state.read", {}, "M2", "beta"),
    ])
    calendar_events = results[0].output or {}
    weather = results[1].output or {}
    candidates = results[2].output or {}
    profile = results[3].output or {}
    self_state = results[4].output or {}

    # 基于工具结果生成主动触发候选
    if weather.get("rain_probability",0) > 0.5:
        state.candidate_events.append({
            "type": "weather_reminder", "summary": "提醒带伞", "weight": 0.7,
        })
    for ev in calendar_events.get("events", []):
        if is_upcoming(ev, within_minutes=30):
            state.candidate_events.append({
                "type": "calendar_event", "summary": ev["title"], "weight": 0.8,
            })

    # confidence 不足时调 proactive_traverser 决策工具做姿态选择
    if state.candidate_confidence < THETA_CONF:
        decision = await m5.invoke(ToolCallRequest(
            "proactive_traverser.decide",
            {"candidates": state.candidate_events, "self_state": self_state,
             "user_state": state.user_state},
            "M2", "beta"))
        if decision.status == "success":
            state.posture = decision.output["posture"]  # FOLLOW_UP/CHECK_IN/SILENCE/PASSIVE_REPLY
    # TFGR 候选 + profile 加权后进入规则决策
    ...
```

**M2 替用户执行事务**：当副脑决定"替用户发消息提醒小明"时，通过 `message.send` 工具完成（需用户授权，§6.5）。

### 9.2 M3 主脑 API 调用与身份守卫调 M5（辅助主脑 API 做 β 深度推理 + 身份守卫）

M3 在两类场景下调 M5：

**场景 A — β 深度推理辅助**：M3 调主脑 API 思考模式前，先检索外部知识注入 prompt：

```python
# M3 内部
async def m3_think_with_tools(req: ThinkingRequest) -> Optional[ThinkingResponse]:
    if needs_external_knowledge(req.decision_question):
        tool_results = await m5.invoke(ToolCallRequest(
            "knowledge.retrieve",
            {"query": extract_query(req), "top_k": 3},
            "M3", "beta",
        ))
        if tool_results.status == "success":
            enriched_context = inject_tool_results(req.context, tool_results.output)
            prompt = build_prompt(enriched_context, req.decision_question, with_tools=True)
        else:
            prompt = build_prompt(req.context, req.decision_question)
    else:
        prompt = build_prompt(req.context, req.decision_question)
    # 调主脑 API 思考模式（不同 prompt，非流式或流式）
    raw = await call_brain_api(prompt, mode="thinking")
    return parse_response(raw)
```

工具调用增加的延迟从 M3 思考模式调用的预算中扣除（β 路径不阻塞用户，但有 5-10s 上限）。

**场景 B — 身份守卫辅助**：M3 流式接收主脑 API token 时按 batch 调用 `w_probe.detect` 做生成后异己检测；调用 `identity.get` 取身份锚快照构造 system prompt；调用 `self_state.read` 判 V 值偏移决定是否重生成：

```python
# M3 内部 - 流式异己守卫
async def m3_stream_with_identity_guard(req: GenerationRequest):
    # 调用前：取 identity 快照构造 system prompt
    identity_result = await m5.invoke(ToolCallRequest(
        "identity.get", {}, "M3", "alpha"))
    identity_snapshot = identity_result.output or {}
    system_prompt = build_identity_prompt(identity_snapshot, req.identity_cache_snapshot)

    # 流式调用主脑 API
    token_buffer = []
    async for token in call_brain_api_stream(req, system_prompt):
        token_buffer.append(token)
        if len(token_buffer) >= 16:  # 每 16 token 做一次异己检测
            text = "".join(token_buffer)
            alien_result = await m5.invoke(ToolCallRequest(
                "w_probe.detect", {"text": text}, "M3", "alpha"))
            alien_score = (alien_result.output or {}).get("score", 0.0)
            if alien_score > THETA_SAFETY:
                # 异己触发：缓冲当前句不送 TTS，触发重生成或降级
                await self._handle_alien(text, alien_score)
                token_buffer.clear()
                continue
            yield "".join(token_buffer)  # 放行干净 token
            token_buffer.clear()
```

> **关键**：M3 不直接访问副脑显存读 W_probe / identity_cache——必须经 M5 的 `w_probe.detect` / `identity.get` / `self_state.read` 工具。这是横切一致性的硬约束（主文档 §5.7），保证所有内部状态读取统一埋点、统一鉴权、统一可降级。

### 9.3 主脑 API（经 LangChain Agent）调 M5（实时信息问题）

主脑 API（DeepSeek/OpenAI 兼容）在回答需要实时信息的问题时，通过 LangChain Agent 模式调用 M5 工具。这是"主脑 API 也可调用工具"的实现，由 M2 在编排主脑 API 调用时配置 Agent：

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

async def brain_answer_with_tools(user_query: str, brain_llm) -> str:
    # 动态获取当前可用工具（α 路径仅本地+内部，β 路径全部）
    tools = build_langchain_tools(m5_scheduler, caller="brain", path="alpha")
    # 如果用户问题明确需要云端工具，先回占位，切到 β 路径
    if needs_cloud_tool(user_query) and not is_alpha_path_safe(user_query):
        await send_placeholder("让我查一下...")  # α 路径先回占位
        tools = build_langchain_tools(m5_scheduler, caller="brain", path="beta")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是 Lucy，可以调用工具回答问题。调用工具后基于结果回答。"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(brain_llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3,
                             verbose=False, handle_parsing_errors=True)
    result = await executor.ainvoke({"input": user_query})
    return result["output"]
```

**迭代次数限制 `max_iterations=3`**：防止 Agent 陷入工具调用死循环。每次工具调用都经 M5 调度，受超时与限流保护。

> **brain 调用方权限约束**：brain 不能调内部工具中的状态写入类（`tfgr.add_node` / `memory.store` / `user_profile.update`），也不能调 `w_probe.detect` / `identity.get` / `self_state.read` / `proactive_traverser.decide`（这些是副脑自身能力，主脑 API 无业务必要访问）。brain 可调内部工具中的 `memory.retrieve` / `user_profile.query` / `context.bundle` / `emotion.analyze`（只读且有助于回答用户）。

### 9.4 M1 用户建模调 M5（仅内部工具写状态）

M1 在更新长期状态时通过 M5 调内部工具：

```python
# M1 内部
async def m1_commit_profile_update(updates: dict):
    await m5.invoke(ToolCallRequest(
        "user_profile.update", {"fields": updates}, "M1", "beta"))

async def m1_store_memory(entry: MemoryEntry):
    await m5.invoke(ToolCallRequest(
        "memory.store", {"entry": entry.dict()}, "M1", "beta"))
```

> M1 读 UserState 走共享内存（无锁单写多读，主文档 §5.2），不经 M5；只有持久化到长期画像/记忆库时才经 M5。

### 9.5 M4 / M6 不直接调 M5

M4 多模态管线与 M6 身体控制环有独立的数据通路，不经过工具调度：

- **M4**：视觉/语音 I/O 走自己的流式管线；M4 检测到的事件推送给 M2，由 M2 决定是否调 M5 工具
- **M6**：身体控制环 30-60Hz 硬实时，绝不经过 M5 调度（M5 的最小超时 5ms 对 γ 路径仍不可接受）；M2 发运动意图到 M6 意图队列，M6 自主规划执行

---

## 10. 降级策略

工具不可用时，M5 根据 `fallback` 字段返回兜底结果，调用方据此继续。本节降级策略与主文档 §8.1 降级矩阵对齐。

### 10.1 工具级 fallback

| fallback 策略 | 行为 | 适用工具 |
|---|---|---|
| `return_empty` | 返回空结构（如 `{"events":[]}`） | calendar/filesystem/smarthome |
| `return_none` | 返回 None | weather/search/knowledge |
| `raise` | 抛异常给调用方 | 仅用于调试，生产环境慎用 |

**工具熔断**：连续失败的工具有独立熔断器，避免雪崩：

```python
class ToolCircuitBreaker:
    def __init__(self):
        self.failures: dict[str, int] = defaultdict(int)  # tool_name -> count
        self.open_until: dict[str, float] = {}            # tool_name -> reopen_ts
        self.threshold = 5
        self.recovery_s = 60

    def allow(self, tool_name: str, now: float) -> bool:
        reopen = self.open_until.get(tool_name)
        if reopen and now < reopen:
            return False
        if reopen and now >= reopen:
            # 半开：放行一次试探
            self.open_until.pop(tool_name, None)
            self.failures[tool_name] = 0
        return True

    def record(self, tool_name: str, success: bool, now: float):
        if success:
            self.failures[tool_name] = 0
        else:
            self.failures[tool_name] += 1
            if self.failures[tool_name] >= self.threshold:
                self.open_until[tool_name] = now + self.recovery_s
                logger.error("Tool circuit open: %s", tool_name)
                metrics.m5_circuit_open.inc(tool_name)
```

### 10.2 工具级降级场景示例

| 场景 | 降级行为 |
|---|---|
| 智能家居网关断连 | `smarthome.*` 全部熔断，M2 跳过家居控制，对话正常 |
| 天气 API 超时 | `weather.query` 返回 None，M2 不做带伞提醒 |
| 网络搜索失败 | 主脑 API Agent 收到 ToolException，回"我暂时查不到，稍后再试" |
| TFGR 图查询失败 | M2 用最近缓存的候选事件兜底 |
| 日历服务不可用 | 返回空事件列表，M2 不做基于日历的主动触发 |
| W_probe 检测工具失败 | M3 降级为"放行所有 token + 上报风险"（异己检测失效，对话不中断但身份守卫弱化） |
| identity.get 失败 | M3 用最近缓存的 identity 快照构造 system prompt（陈旧但可用） |
| self_state.read 失败 | M3 跳过 V 值偏移判定，仅靠 W_probe 做异己检测 |
| proactive_traverser.decide 失败 | M2 降级为纯规则决策（旧 proactive_traverser 的简化版） |

### 10.3 与主文档 §8.1 系统级降级的对齐

M5 工具失效会触发或配合系统级降级：

| 主文档降级级 | 触发 | M5 的配合行为 |
|---|---|---|
| L0 正常 | — | 全功能工具调度 |
| L1 GPU 忙 | α 与 β 抢副脑本地 GPU | β 路径的内部工具（`w_probe.detect` / `self_state.read`）调用让步；α 路径的 W_probe 检测降频（M3 改为每 32 token 一次而非 16） |
| L2 主脑 API 慢 | 首 token >2s | α 路径跳过 `memory.retrieve` 等内部工具，用最近 3 轮对话兜底；身份 prompt 精简（`identity.get` 仅取摘要） |
| L3 API 超时 | M3 >5s 无首 token | M2 用规则兜底占位"让我想想..."，主脑 API 后台继续；M5 工具调用全部切 β |
| **L4 多模态失效** | ASR/Vision 挂 | 降级为纯文本交互——M5 的 `emotion.analyze` 改为仅基于文本（视觉情绪输入失效）；`smarthome.*` / `system.notify` 等仍可用；M4 推送的事件流中断，M2 改为定时轮询 `tfgr.query` / `calendar.query` 维持主动触发 |
| L5 身体失效 | M6 通讯断 | 身体冻结，对话继续——M5 工具调度不受影响；M2 不再发 BodyIntent（与 M5 无关） |
| L6 副脑失效 | M2 崩溃 | 纯被动模式——M5 仍可被主脑 API（brain）经 LangChain Agent 调用（本地/云端工具可用），但内部工具（依赖副脑共享态）全部熔断 |
| **L7 主脑 API 不可用** | API 401/403/网络断 | M3 切备用 API（OpenAI↔DeepSeek 互备）；全断时副脑本地 Qwen2.5-3B 应急生成（降级质量）——此时 M5 内部工具（`w_probe.detect` / `identity.get` / `self_state.read`）仍可用（不依赖主脑 API），副脑用它们对自身应急生成内容做异己守卫，维持最低身份一致性 |

> **L7 是 M5 内部工具的关键韧性场景**：主脑 API 全断时，副脑本地 Qwen2.5-3B 应急生成的短回复仍需经 `w_probe.detect` 做异己检测（副脑对自己生成的文本也要查异己——防本地模型退化），`identity.get` 取身份锚做对齐。M5 内部工具的执行器访问的是副脑本地显存，与主脑 API 网络通路完全解耦，所以 L7 下仍可用。

---

## 11. TFGR 资产暴露为内部工具

本节专门阐述副脑侧 TFGR 资产如何经 M5 暴露为内部工具，统一供 M2 / M3 调度。这是主文档 §7 TFGR 资产复用映射在 M5 模块的具体落地。

### 11.1 暴露原则

主文档 §7 列出的副脑侧 TFGR 资产中，以下五类需要暴露为 M5 内部工具：

| TFGR 资产 | 主文档 §7 角色 | M5 内部工具名 | 暴露理由 |
|---|---|---|---|
| W_probe（holdout acc=1.0） | 生成后异己检测分类器 | `w_probe.detect` | M3 流式守卫需调，统一埋点；避免 M3 直接访问副脑显存 |
| identity_cache（14条EMA, β=0.9） | Lucy 持久身份锚 | `identity.get` | M3 序列化 system prompt 需取快照；M2 决策时读身份稳定性 |
| self_state_predictor（V/PE/energy/conf） | 副脑自身状态信号 | `self_state.read` | M3 判 V 偏移决定重生成；M2 主动触发读自身能量 |
| TFGR 时间流图 | M2 主动触发的事件源 + 长期记忆 | `tfgr.query` / `tfgr.add_node` | M2 候选事件遍历；M3 上下文增强 |
| proactive_traverser | M2 主动决策的规则基线 | `proactive_traverser.decide` | M2 confidence 不足时调姿态决策；统一埋点决策日志 |

**暴露原则**：
- **只读类工具优先暴露**：`w_probe.detect` / `identity.get` / `self_state.read` / `tfgr.query` 是只读，安全暴露给 M2/M3
- **写入类工具严格限权**：`tfgr.add_node` 仅 M2 可调（事件落盘由副脑负责）；`memory.store` / `user_profile.update` 仅 M2/M1 可调
- **决策类工具按需暴露**：`proactive_traverser.decide` 仅 M2 可调（副脑自身决策工具，M3/brain 无业务必要）
- **brain 一般不调内部工具的副脑专属类**：brain 可调 `memory.retrieve` / `user_profile.query` / `context.bundle` / `emotion.analyze`（有助于回答用户），不可调 `w_probe.detect` / `identity.get` / `self_state.read` / `proactive_traverser.decide` / `tfgr.add_node`

### 11.2 内部执行器实现

内部工具不真正起 MCP server，由 `InternalExecutor` 在进程内直接调用副脑共享态 API：

```python
class InternalExecutor(Executor):
    """内部工具执行器：直接访问副脑共享态（进程内调用，无网络）"""

    def __init__(self, shared_state):
        # shared_state 是副脑侧共享态的引用，包括：
        # - tfgraph: TFGR 时间流图实例
        # - memory: Chroma 向量库
        # - user_profile: 用户画像存储
        # - w_probe: W_probe 异己检测分类器（副脑 GPU 显存）
        # - identity_cache: Lucy 身份锚 EMA 缓存（副脑 GPU 显存）
        # - self_state_predictor: 副脑自身状态预测器（副脑 GPU 显存）
        # - proactive_traverser: 5阶段推理链决策器
        self.state = shared_state

    async def call(self, meta, arguments):
        if meta.name == "tfgr.query":
            return self.state.tfgraph.query(arguments)
        elif meta.name == "tfgr.add_node":
            return self.state.tfgraph.add_node(arguments)
        elif meta.name == "memory.retrieve":
            return await self.state.memory.retrieve(arguments)
        elif meta.name == "memory.store":
            return await self.state.memory.store(arguments)
        elif meta.name == "user_profile.query":
            return self.state.user_profile.query(arguments)
        elif meta.name == "user_profile.update":
            return self.state.user_profile.update(arguments)
        elif meta.name == "w_probe.detect":
            # W_probe 异己检测：对文本做分类，返回异己分数
            # 主文档 §5.4 — cos(h_self, W_probe)=+0.091 / cos(h_alien, W_probe)=-0.089
            score = await self.state.w_probe.classify(arguments["text"])
            return {"score": score, "is_alien": score > THETA_SAFETY}
        elif meta.name == "identity.get":
            # identity_cache 查询：返回 Lucy 身份锚快照摘要（不可变视图）
            snapshot = self.state.identity_cache.get_snapshot()
            return {
                "snapshot_id": snapshot.id,
                "anchor_count": snapshot.count,  # 14
                "norm": snapshot.norm,           # 0.2271
                "summary": snapshot.summary,     # 文本摘要供 M3 构造 system prompt
                "timestamp": snapshot.timestamp,
            }
        elif meta.name == "self_state.read":
            # self_state_predictor 读取：返回副脑自身状态
            # 主文档 §7 — V/PE/energy/confidence
            state = self.state.self_state_predictor.read()
            return {
                "v_value": state.v,
                "pe_value": state.pe,
                "energy": state.energy,
                "confidence": state.confidence,
                "v_drift": state.v_drift,  # V 偏移量，M3 据此判身份稳定性
            }
        elif meta.name == "proactive_traverser.decide":
            # proactive_traverser 决策：5阶段推理链姿态选择
            # 主文档 §7 — FOLLOW_UP/CHECK_IN/SILENCE/PASSIVE_REPLY
            decision = self.state.proactive_traverser.decide(
                candidates=arguments["candidates"],
                self_state=arguments.get("self_state"),
                user_state=arguments.get("user_state"),
            )
            return {
                "posture": decision.posture,
                "confidence": decision.confidence,
                "reasoning_chain": decision.reasoning_chain,  # 5阶段推理链摘要
            }
        elif meta.name == "emotion.analyze":
            return await self.state.emotion_analyzer.analyze(arguments["text"])
        elif meta.name == "context.bundle":
            return await self.state.context_assembler.bundle(arguments)
        else:
            raise ValueError(f"no handler for internal tool {meta.name}")
```

### 11.3 W_probe 异己检测工具的时序约束

`w_probe.detect` 是 M5 内部工具中时序最敏感的——M3 流式接收主脑 API token 时按 batch 调用，必须不阻塞流式管线：

```
主脑 API token 流 ──┬──▶ 累积到 16 token ──▶ M5.invoke(w_probe.detect, α)
                    │                         │ (~10ms 本地 forward)
                    │                         ├─ score < θ_safety → 放行
                    │                         └─ score ≥ θ_safety → 缓冲+重生成
                    │
                    └──▶ 字幕实时显示（不阻塞）
```

- **超时**：10ms（严格）。超时则降级为"放行 + 上报风险"（不阻塞流式管线）
- **不缓存**：每次检测结果独立
- **重试**：max_attempts=0（失败立即降级，不重试）
- **并发**：M3 可串行调用（每 batch 一次），不需并发

### 11.4 identity.get 与 M3 system prompt 构造

`identity.get` 返回 identity_cache 的快照摘要，M3 据此构造主脑 API 的 system prompt：

```python
# M3 内部
async def build_system_prompt(req: GenerationRequest) -> str:
    identity_result = await m5.invoke(ToolCallRequest(
        "identity.get", {}, "M3", "alpha"))
    if identity_result.status != "success":
        # 降级：用 req 自带的 identity_cache_snapshot（M2 传入的）
        return _build_prompt_from_snapshot(req.identity_cache_snapshot)
    identity = identity_result.output
    return f"""你是 Lucy，一个具身的对话伴机器人。
身份锚（14条 EMA 回复摘要，norm={identity['norm']:.4f}）：
{identity['summary']}
语气风格：温暖、关切、有主动关怀意识、不啰嗦。
当前身份状态置信度：{identity.get('confidence', 'unknown')}
"""
```

### 11.5 self_state.read 与 V 偏移判定

`self_state.read` 返回副脑自身状态，M3 据此判 V 偏移决定是否触发重生成：

```python
# M3 内部
async def check_identity_stability() -> tuple[bool, dict]:
    """返回 (need_regen, state)"""
    result = await m5.invoke(ToolCallRequest(
        "self_state.read", {}, "M3", "alpha"))
    if result.status != "success":
        return False, {}  # 失败时不触发重生成（避免误判）
    state = result.output
    # V 偏移大 → identity_cache 与主脑 API 生成内容偏离 → 身份不稳定
    need_regen = state.get("v_drift", 0) > V_DRIFT_THRESHOLD
    return need_regen, state
```

> **V 偏移的语义**（主文档 §5.4）：V40 温度约束 g(V) 从旧的"生成退化"转为副脑的身份状态信号——V 值偏移反映 identity_cache 与主脑 API 生成内容的偏离程度。V 偏移大 → 副脑判定身份不稳定 → 调整 prompt（强化身份锚）/ 触发重生成 / 上报 M2 决策。

### 11.6 proactive_traverser.decide 与 M2 主动触发

`proactive_traverser.decide` 是 M2 在 confidence 不足时调用的姿态决策工具，复用 TFGR 项目的 5 阶段推理链：

```python
# M2 内部
async def m2_decide_posture(candidates: list, self_state: dict, user_state: dict) -> str:
    """姿态决策：FOLLOW_UP / CHECK_IN / SILENCE / PASSIVE_REPLY"""
    result = await m5.invoke(ToolCallRequest(
        "proactive_traverser.decide",
        {"candidates": candidates, "self_state": self_state, "user_state": user_state},
        "M2", "beta"))
    if result.status != "success":
        # 降级：纯规则决策（旧 proactive_traverser 简化版）
        return _rule_based_decide(candidates, self_state, user_state)
    return result.output["posture"]
```

**5 阶段推理链**（主文档 §7 复用映射）：
1. 候选事件势能评估
2. 用户状态匹配度判定
3. 副脑自身能量/觉醒检查
4. 时机合理性判定（基于 R(e) 主观时间）
5. 姿态选择与 confidence 输出

---

## 12. 评估指标

### 12.1 核心指标

| 指标 | 定义 | 目标 |
|---|---|---|
| 工具调用成功率 | status=success 的调用数 / 总调用数 | > 95% |
| 平均延迟（本地） | 本地工具调用的平均耗时 | < 60ms |
| 平均延迟（内部） | 内部工具调用的平均耗时 | < 100ms |
| 平均延迟（云端，β） | 云端工具 β 路径平均耗时 | < 3000ms |
| α 路径工具耗时占比 | α 路径工具调用总耗时 / α 首 token 预算 | < 5%（< 100ms） |
| 缓存命中率 | 缓存命中 / (命中+未命中) | > 25% |
| 降级率 | 走 fallback 的调用 / 总调用 | < 5% |
| 熔断触发次数 | 工具熔断打开的次数 | < 1 次/小时/工具 |
| 工具可用率 | enabled 工具数 / 注册工具数 | > 95% |
| W_probe 检测延迟 | `w_probe.detect` 工具调用耗时 | < 10ms（P99） |
| 异己拦截率 | W_probe 触发重生成次数 / 主脑 API 调用次数 | < 2%（过高说明 system prompt 失效） |

### 12.2 可观测埋点

```python
class M5Metrics:
    m5_invoke_total = Counter("m5_invoke_total", "tool_name", "caller", "path")
    m5_latency = Histogram("m5_latency_ms", "tool_name",
                           buckets=[5, 20, 50, 100, 500, 1000, 3000, 5000])
    m5_timeout = Counter("m5_timeout_total", "tool_name", "attempt")
    m5_error = Counter("m5_error_total", "tool_name", "error_type")
    m5_cache_hit = Counter("m5_cache_hit_total", "tool_name")
    m5_cache_miss = Counter("m5_cache_miss_total", "tool_name")
    m5_auth_reject = Counter("m5_auth_reject_total", "tool_name")
    m5_path_reject = Counter("m5_path_reject_total", "tool_name")
    m5_fallback = Counter("m5_fallback_total", "tool_name", "reason")
    m5_circuit_open = Counter("m5_circuit_open_total", "tool_name")
    m5_alpha_budget_exceeded = Counter("m5_alpha_budget_exceeded_total")
    m5_concurrent_inflight = Gauge("m5_concurrent_inflight", "tool_name")
    # 内部工具专属
    m5_w_probe_alien_detected = Counter("m5_w_probe_alien_detected_total")
    m5_identity_get_failure = Counter("m5_identity_get_failure_total")
    m5_self_state_v_drift_high = Counter("m5_self_state_v_drift_high_total")
```

每次工具调用必须记录：工具名、调用方、路径、状态、延迟、是否缓存命中、是否降级，用于离线分析与策略调优。内部工具（W_probe / identity / self_state）的失败与异己检测次数单独计数，用于身份守卫健康度监控。

---

## 13. 关键代码结构

### 13.1 模块整体伪代码

```python
# m5/__init__.py
class M5Module:
    """模块5：MCP 工具生态"""

    def __init__(self, settings, brain_shared_state):
        self.registry = ToolRegistry()
        self.cache = ToolResultCache(max_size=1000)
        self.breakers = ToolCircuitBreaker()
        self.executors = {
            "local": LocalExecutor(settings.LOCAL_TOOL_CONFIG),
            "cloud": CloudExecutor(http_client=httpx.AsyncClient(timeout=10)),
            "internal": InternalExecutor(shared_state=brain_shared_state),
        }
        self.scheduler = Scheduler(self.registry, self.executors, self.cache, self.breakers)
        self.composition = ToolComposition()
        self._load_static_tools(settings.TOOLS_CONFIG_PATH)

    def _load_static_tools(self, path: str):
        """启动时从 tools.yaml 加载内置工具"""
        config = yaml.safe_load(open(path, encoding="utf-8"))
        for tool_cfg in config["tools"]:
            meta = ToolMeta(**tool_cfg)
            self.registry.register(meta)

    async def invoke(self, req: ToolCallRequest) -> ToolCallResult:
        return await self.scheduler.invoke(req)

    async def invoke_parallel(self, reqs: list[ToolCallRequest]) -> list[ToolCallResult]:
        return await self.scheduler.invoke_parallel(reqs)

    def list_tools(self, caller: str, path: str) -> list[dict]:
        return self.registry.list_tools_for_caller(caller, path)

    def register_tool(self, meta: ToolMeta) -> bool:
        return self.registry.register(meta)
```

### 13.2 工具执行器基类与三类实现

```python
# m5/executors/base.py
class Executor(ABC):
    """工具执行器基类"""

    @abstractmethod
    async def call(self, meta: ToolMeta, arguments: dict) -> dict:
        """执行工具调用，返回结构化结果。抛异常表示失败。"""
        ...

class LocalExecutor(Executor):
    """本地工具执行器：stdio 子进程或进程内函数"""

    def __init__(self, config):
        self.handlers = {
            "calendar.query": self._calendar_query,
            "calendar.create": self._calendar_create,
            "filesystem.read": self._fs_read,
            "smarthome.light": self._smarthome_light,
            # ... 其他本地工具
        }

    async def call(self, meta, arguments):
        handler = self.handlers.get(meta.name)
        if handler is None:
            raise ValueError(f"no handler for {meta.name}")
        # 本地工具用 run_in_executor 避免阻塞事件循环
        return await asyncio.get_event_loop().run_in_executor(
            None, handler, arguments)

    def _calendar_query(self, args) -> dict:
        return {"events": [...]}

class CloudExecutor(Executor):
    """云端工具执行器：HTTP/SSE"""

    def __init__(self, http_client):
        self.client = http_client
        self.endpoints = {
            "weather.query": "https://weather-svc/api/query",
            "search.web": "https://search-svc/api/search",
            # ...
        }

    async def call(self, meta, arguments):
        url = self.endpoints.get(meta.name)
        if url is None:
            raise ValueError(f"no endpoint for {meta.name}")
        resp = await self.client.post(url, json=arguments)
        resp.raise_for_status()
        return resp.json()

class InternalExecutor(Executor):
    """内部工具执行器：直接访问副脑共享态（实现见 §11.2）"""
    # 详见 §11.2 完整实现
```

### 13.3 LangChain 集成

```python
# m5/langchain_adapter.py
def jsonschema_to_pydantic(schema: dict, name: str) -> type:
    """把 JSON Schema 转为 pydantic 模型，供 LangChain 工具用"""
    fields = {}
    for prop, prop_schema in schema.get("properties", {}).items():
        py_type = _json_type_to_py(prop_schema.get("type", "string"))
        default = ... if prop in schema.get("required", []) else None
        fields[prop] = (py_type, Field(default=default, description=prop_schema.get("description","")))
    return type(name + "Args", (BaseModel,), fields)

def build_agent_with_m5_tools(
    m5: M5Module, brain_llm, caller: str, path: str
) -> AgentExecutor:
    tools = build_langchain_tools(m5.scheduler, caller, path)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(brain_llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=3,
                         handle_parsing_errors=True)
```

### 13.4 FastAPI 路由（可选，供调试）

```python
# m5/api.py
router = APIRouter()

@router.post("/m5/invoke", response_model=ToolCallResult)
async def invoke(req: ToolCallRequest):
    return await m5.invoke(req)

@router.get("/m5/tools", response_model=list[dict])
async def list_tools(caller: str, path: str):
    return m5.list_tools(caller, path)

@router.post("/m5/register")
async def register(meta: ToolMeta):
    return {"ok": m5.register_tool(meta)}
```

注意：M2/M3 应通过进程内调用（直接 import）使用 M5，避免额外网络开销。HTTP 路由仅供调试与独立部署。

---

## 14. 与主文档的契约一致性说明

本模块文档与主文档 [方向C_00_系统耦合与时序架构_主文档.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/方向C_00_系统耦合与时序架构_主文档.md) 的契约一致性：

| 契约项 | 主文档约定 | 本模块实现 | 一致性 |
|---|---|---|---|
| 横切定位（§5.7） | M5 是横切工具生态，M2/M3/主脑 API（经 LangChain Agent）可调；所有工具调用经 M5 统一调度 | §2 明确横切定位与调用方；§6 调度引擎统一收口 | ✓ |
| 调度契约（§5.7） | 调用方不直接连工具后端 | §4 注册中心 + §6 调度引擎；内部工具也不直接访问副脑显存，经 InternalExecutor | ✓ |
| 典型场景（§5.7） | M2 调日历/天气/TFGR图；M3 调知识检索辅助主脑 API 生成；主脑 API 调网络搜索 | §9 各模块集成完整覆盖；M3 调用语义更新为"辅助主脑 API 做 β 深度推理 + 身份守卫" | ✓ |
| TFGR 资产复用（§7） | W_probe/identity_cache/self_state_predictor/TFGR图/proactive_traverser 都可暴露为内部工具 | §5.3 + §5.4.3 + §11 完整落地 | ✓ |
| 时延预算（§4.1） | α 路径首 token <2.0s；本地工具 <100ms；云端 2-5s 或异步；内部 50-200ms | §1 时延契约表 + §6.2 超时管理 + §8 时延管理严格遵循 | ✓ |
| 三路径分离（§4.2） | α/β/γ 调度独立、资源隔离；α 与 β 抢资源时 β 让步 | §8.1 α 路径白名单+预算检查；§8.3 β 可异步 | ✓ |
| 降级策略（§8.1） | L0-L7 分级降级；工具失效有兜底；L4 多模态失效降纯文本；L7 API 不可用副脑本地应急 | §10 fallback 策略 + 工具熔断 + §10.3 系统级降级对齐（含 L4/L7 专项说明） | ✓ |
| 数据契约（§5.7） | 主文档列出典型场景，未定义 ToolCallRequest/Result | 本模块 §1 补充定义（不与主文档冲突，为主文档细化） | ✓（细化） |
| 与 M3 集成（§5.4） | M3 调主脑 API 流式接收 + W_probe 生成后异己检测 | §9.2 完整实现，M3 经 M5 调 w_probe.detect / identity.get / self_state.read | ✓ |
| 可观测性 | 关键路径埋点 | §12.2 metrics 完整覆盖（含内部工具专属指标） | ✓ |

### 关键不变量（Invariant）

1. **调用方不直接连工具后端**：所有工具调用必须经 M5 调度引擎——包括内部工具（M3 不直接访问副脑显存读 W_probe / identity_cache，必须经 `w_probe.detect` / `identity.get` 工具）。违反时无法获得超时/重试/鉴权/埋点保护。
2. **α 路径不阻塞**：α 路径仅允许本地+内部工具同步调用，且总耗时 ≤100ms；云端工具在 α 路径只能"触发不等待"+占位。
3. **工具调用受四重保护**：鉴权 + 限流 + 超时 + 熔断，任一触发即降级，绝不拖垮主对话管线。
4. **M5 无业务状态**：所有状态由 M1（UserState）与副脑侧 TFGR 共享态（identity_cache / W_probe / self_state_predictor / TFGR 时间流图）维护，M5 仅维护工具元数据与缓存（缓存可丢失不影响正确性）。
5. **工具失败永远返回明确信号**：ToolCallResult.status 明确区分 success/timeout/error/rejected，调用方据此降级，永不收到含糊结果。
6. **敏感工具需用户授权**：email.send/message.send/calendar.create 等有副作用的工具首次调用需用户确认（§6.5）。
7. **brain 调用方权限受限**：主脑 API（经 LangChain Agent）不能调内部工具中的副脑专属类（w_probe.detect / identity.get / self_state.read / proactive_traverser.decide / tfgr.add_node），只能调只读且有助于回答用户的内部工具（memory.retrieve / user_profile.query / context.bundle / emotion.analyze）。
8. **L7 降级下内部工具仍可用**：主脑 API 全断时，副脑本地 Qwen2.5-3B 应急生成，M5 内部工具（依赖副脑本地显存）仍可对应急生成内容做异己守卫，维持最低身份一致性。

### 版本

- 文档版本：v2.0（架构反转版）
- 最后更新：2026-08-03
- 主要变更：调用方语义更新（M2=副脑、M3=主脑API调用与身份守卫、brain=主脑API经LangChain Agent）；内部工具清单强化（W_probe异己检测/identity_cache查询/self_state_predictor读取/TFGR图查询/proactive_traverser决策统一暴露）；新增 §11 TFGR资产暴露为内部工具；§10 降级策略对齐主文档 L4/L7 场景
- 待办：工具清单需随业务持续扩展（目标 50+ 工具）；LangChain Agent 的迭代次数与 token 预算需线上数据校准；与 M2/M3 的具体调用契约需在联调阶段对齐埋点字段；W_probe 异己检测的 θ_safety 阈值需线上 A/B 校准
