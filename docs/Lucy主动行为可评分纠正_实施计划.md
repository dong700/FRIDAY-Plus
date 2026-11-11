# Lucy 主动行为可评分纠正 — 实施计划

## Context（为什么做）

用户前台与 Lucy 正常交流，但看不到 Lucy 后台 64 维状态核心 / 主动行为「学了什么」，也不清楚它有什么用；尤其点名「主动行为可评分纠正」和「64 维 / 打分纠错」还没做。

本次聚焦「**主动行为可评分纠正**」：对每条 Lucy 主动话（`role=proactive`）提供 5 类打分（👍好 / 🔁又重复 / ✏️记错了 / 📏太啰嗦 / 🎲不相关），分数反过来纠正她的信源话题与后续行为。

顺带修复一个**硬缺口**：正反馈奖励 `_positive_reward` 从未进入 64 维吸引子形成（后台只调 `feed(advance)→gen.update(reward=0.0)`，从不调 `bridge.step()`），导致用户点赞对 64 维学习毫无作用。

现状链路（已核实）：
- 主动话由 `_push_proactive(text, ts, pre_sb)`（web_api.py:1331）经 WebSocket `type:"proactive"` 唯一漏斗下发（CoT→CausalPlan 与 background_loop 两条路径都汇到这里）。
- 前端 index.html:237 addMsg("proactive",…)，:405 WS 收，:476 sendFeedback → /api/feedback（仅 positive/suppress）。
- 可复用能力：`sb._record_suppressed_text`（subbrain.py:1798）、`_supersede_known_facts`（web_api:787）、`remove_known_fact`（user_modeling.py:665）、`ValueFeedbackBridge.positive_feedback/suppress/step`。想法.txt 里的 `suppress_topic/hard_penalize/boost_active_nodes` 是伪代码未实现，不用。
- 用户已确认：5 类打分；啰嗦要**真正**作用于后续主动发言更简短（新增轻量 style_hint）。

---

## 1. turn_id/sid — 内存 `proactive_log`，不碰 history

- `state` 全局新增 `"proactive_log": OrderedDict()`（上限 200，`popitem(last=False)` 淘汰），用 `session.lock` 保护并发（后台 `_push_proactive` vs 主 `api_feedback`）。
- 在 `_push_proactive`（web_api.py:1331）生成 `sid`：
  ```python
  _proactive_seq = itertools.count()
  sid = f"pr{next(_proactive_seq)}_{int(time.time()*1000)}"
  with session.lock:
      state["proactive_log"][sid] = {"text": text, "ts": timestamp, "rated": False}
  msg["sid"] = sid   # 追加进 WS payload
  ```
- 前端 index.html:408 收到 `m.type==="proactive"` 时 `addMsg("proactive", m.text, {ts:m.timestamp, sid:m.sid})`。
- **不改** `history`（`{role,content}`，会被滚动裁剪、混对话回复，不适合当锚）。

## 2. `/api/feedback` 扩展协议（web_api.py:1952）

`FeedbackPayload` 新增可选字段（旧客户端不传仍可用）：
```python
class FeedbackPayload(BaseModel):
    type: str = "positive"  # positive|suppress|like|repeat|memory_error|verbose|irrelevant
    text: str = ""
    reward: float = 0.5
    amount: float = 0.4
    sid: str = ""        # 新增：主动话唯一 id
    free_text: str = ""  # 新增：memory_error 用，纠正后的正确事实
```
定位：有 `sid` → 用 `proactive_log[sid]["text"]` 原文（**修复前端 24 字符截断丢细节问题**）；无 `sid` → 回退 `p.text`（旧兼容）。命中后标 `rated=True`。

## 3. 纠错动作映射（全部复用，无新后端能力）

| type | 触发动作 |
|---|---|
| **like** | `bridge.positive_feedback(txt, reward)` |
| **repeat** | `bridge.suppress(txt, 0.5)` + `sb._record_suppressed_text(txt)`（话题疲劳重设） |
| **memory_error** | `_supersede_known_facts(free_text or txt)` 盖旧 + 再 `_supersede_known_facts(txt)` 兜底删错误陈述重叠旧事实 + `updater.add_known_fact("纠正:…", free_text)` 落正确事实 + `bridge.positive_feedback(free_text)` + `bridge.suppress(txt)` + `sb._record_suppressed_text(txt)` |
| **verbose** | `bridge.suppress(txt, 0.2)` + `sb._record_suppressed_text(txt)` + **写 style_hint**（见 §4.2） |
| **irrelevant** | `bridge.suppress(txt, 0.6)` + `sb._record_suppressed_text(txt)` |

所有动作 `try/except` 静默失败；`updater=state["updater"]`、`sb=state["subbrain_agent"]`、`owner=_cur_owner()`，访问方式与现有 positive/suppress 分支一致。

## 4. 两个关键点

### 4.1 正反馈注入缺口修复
新增 helper，在 background_loop 已存在 `BRIDGE_FEED_INTERVAL_S`（每 5s）块末尾调用：
```python
def _flush_bridge_positive() -> None:
    bridge = state.get("value_bridge")
    if bridge is None: return
    try:
        if getattr(bridge, "_positive_reward", 0.0) > 0.0:
            bridge.step()   # 把挤的正反馈并入 gen.update → 候选跨价值门控
    except Exception as e:
        print(f"[Web] 价值桥正反馈注入失败: {e}")
```
只有 `_positive_reward>0` 才走 step，正常无正反馈零开销、不与 feed 的 `_periodic` 双推进。→ like/记错的正反馈最迟 5s 内进入 64 维吸引子候选形成。

### 4.2 啰嗦真的变简短（style_hint）
- `self_model.py` 新增字段 `style_hint: str = ""`；`on_proactive_initiate` **不清除它**。
- verbose 时后端写 `sb.self_model.style_hint = "用户觉得你略啰嗦，主动发言请更简短、一句点到。"`
- 在 background_loop 构造主动意图 `intent_desc` 处（web_api.py ~1650）追加 `(self_model.style_hint or '')`，让 call_m3 主动生成带上该提示。可选：`verbose` 打分触发时也即时生成一次更简短版本。

## 5. 前端 UI（index.html）

- 仅 `role==="proactive"` 渲染**一排 5 个短评按钮**（Lucy 对话回复 `lucy` 保留现有 👍/🚫 两按钮不变）：👍好 / 🔁重复 / ✏️记错 / 📏啰嗦 / 🎲无关。
- 记错：点击"记错"在该消息下方**内联插入输入框 + "提交纠正"按钮**（复用 `.fb-row`/`.fb-btn` CSS），提交 `sendFeedback("memory_error", text, sid, freeText)`。
- 单选制：任一打分后整行 `disabled` 置灰、文案变"已反馈"；加 `done-rate` 类。
- `sendFeedback(type, text, sid, freeText)`：`payload={type, sid||""}`；`freeText` 有则 `payload.free_text`；无 sid 时 `payload.text=text`（回退）。
- addMsg proactive 分支把 `extra.sid` 绑进每个按钮闭包。

## 6. 验证

1. 重启 `python -m subbrain_demo.run_web`（后台 job），确认日志"价值奖励桥已加载"、新代码加载。
2. 静默 ≥30s 触发主动话 → WS 消息多出 `sid`，`state["proactive_log"]` 有条目。
3. 逐个打分：
   - 好 → 日志正反馈，`/api/state` `value_learning.positive_scored` 新增；5s 内 `_flush_bridge_positive` 执行。
   - 重复 → `sb._suppressed_topics` 该话题过期≈now+1500s，随后同话题不再主动召回。
   - 记错+填事实 → `known_facts` 删旧增"纠正:…"；`positive_scored` 加纠正话题、`suppressed_scored` 加错误话题。
   - 啰嗦 → `_topic_is_suppressed` True + style_hint 写入；后续主动话带"更简短"提示。
   - 无关 → 同啰嗦（话题退场）。
4. 回归：老客户端无 sid 发 `suppress` 仍生效；`history` 结构不变。

## 关键文件
- `subbrain_demo/web_api.py`（sid 生成/proactive_log、/api/feedback 扩展、`_flush_bridge_positive`、style_hint 注入）
- `subbrain_demo/web/index.html`（proactive 5 按钮、记错内联输入、sendFeedback 签名、WS extra.sid）
- `subbrain_demo/self_model.py`（`style_hint` 字段）
- `state_core/value_feedback_bridge.py`（复用，无改动或仅消费 `_positive_reward`）

## 风险/兼容
- 并发：proactive_log 统一 `session.lock`。
- 内存：OrderedDict 上限 200。
- 向后兼容：FeedbackPayload 仅加可选字段；lucy 对话回复前端不动。
- 诚实标注：`_topic_bigrams` 弱语义时"啰嗦/无关"话题级惩罚精度有限；啰嗦的**风格**约束靠 style_hint（已在 4.2）。repeat 用"重置 25min"而非"累加延长"（如需指数延长另加 `_prolong_suppressed_text`，默认不做）。