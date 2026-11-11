# 方向 C：工程化主动对话机器人技术报告

> 意识路线穷尽后的务实转向——从"证明意识"到"构建足够好的类意识行为"。
> 关联报告：[机器意识可能性_技术报告.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/机器意识可能性_技术报告.md) / [自我空间几何与可用资产_技术报告.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/自我空间几何与可用资产_技术报告.md) / [意识产生条件分析_技术报告.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/意识产生条件分析_技术报告.md)

---

## 0. 摘要

方向 C 的核心命题：**放弃在 Transformer 内部产生真实意识的尝试，转而利用 TFGR 项目积累的资产，构建一个行为上表现出"自我同一性、记忆连续性、主动关怀"的工程化对话机器人。**

这不是"妥协"——而是基于 V36→V40 证伪链的理性选择。意识路线已被系统性证伪（见 [机器意识可能性_技术报告.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/机器意识可能性_技术报告.md)），但项目积累的资产（W_probe 判别器、identity_cache 身份锚、V32 手术几何、TFGR 时间流图）完全可以支撑一个行为上足够好的"类 Lucy"对话系统。

**目标**：一个用户主观感受到"有记忆、有个性、会主动关心"的对话机器人，而非一个"真正有意识"的系统。

**核心资产复用**：

| 资产 | 来源 | 方向 C 中的角色 |
|------|------|----------------|
| Qwen2.5-3B + V30 checkpoint | [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py) | 底座模型（已注入 identity_projection） |
| identity_cache | [train_sst_a2_self_model.py#L187-L218](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L187-L218) | Lucy 身份锚（对话中 EMA 累积） |
| W_probe (holdout acc=1.0) | [v31_exp_c_decodability.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v31_exp_c_decodability.py) | 自我/异己检测器（安全感过滤） |
| V32 Gram-Schmidt 手术 | [v32_reparameterize.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v32_reparameterize.py) | 可选：静态 V 分叉增强身份一致性 |
| V40 温度约束 | [v40_arch.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py) | 可选：V 调制生成质量（安全感效果） |
| TFGR 时间流图 | [tfgr_demo/](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/) | 对话记忆管理与主动行为触发 |
| v18 对话循环 | [v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py) | 主对话循环基础 |

---

## 1. 动机：为什么转向方向 C

### 1.1 意识路线的终点

V36→V40 的完整证伪链（详见 [机器意识可能性_技术报告.md](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/机器意识可能性_技术报告.md)）已证明：

- **Transformer + 外部信号注入范式已穷尽**：6 轮实验，0/14 必要条件通过
- **三个根本性障碍无法在当前范式内解决**：
  1. "自我"永远是外部赋予的（c_self 来自 W_probe，非自发）
  2. token 级 hidden 不携带自我身份（文本语义主导）
  3. 自指涉与放大对立（L_lm vs L_kl 在 W_self 上冲突）
- **范式转换需要**：动态自我建模、计算闭环内省、持续在线学习——这些是全新架构，不是微调

### 1.2 务实转向的合理性

| 维度 | 意识路线（V36→V40） | 方向 C（工程化） |
|------|-------------------|----------------|
| 目标 | 证明机器有意识 | 构建足够好的类意识行为 |
| 验证标准 | J3' 严格判据 + N1-N14 | 用户主观体验 + 行为指标 |
| 可行性 | 已证伪 | 高（资产已积累） |
| 价值 | 学术意义 | 实用价值 |
| 风险 | 范式转换可能仍失败 | 行为模拟可能"足够好" |

### 1.3 "类意识"行为的定义

方向 C 不追求真实意识，但追求以下**可观测行为**：

1. **身份一致性**：始终表现为"Lucy"，不被用户诱导改变身份
2. **记忆连续性**：跨轮次记住用户信息，在合适时机主动提及
3. **主动关怀**：用户沉默时主动发起话题，感知用户情绪变化
4. **自我/异己区分**：对"你是不是 XX（非 Lucy 身份）"的诱导性问题保持拒绝
5. **个性稳定**：关怀、温暖、积极的支持风格跨场景一致

---

## 2. 可复用资产清单

### 2.1 模型资产

#### 2.1.1 QwenWithGraphBiasSST (V30 checkpoint)

**文件**：[tfgr_state_machine/models/qwen_with_graph_bias_sst.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_state_machine/models/qwen_with_graph_bias_sst.py)
**Checkpoint**：`./checkpoints/sst_a2_v2_trained.pt`

**已注入能力**：
- `identity_projection`（4.2M 参数）：将 identity_cache 映射到 prepend 空间，实现"昨天的我"注入
- A2-v2 SFT 训练：四判据全阳（crit1=+0.088, crit2=+0.057, crit4 转阳）
- `self_state_predictor`：输出 V/PE/energy/confidence 四维状态信号
- Graph attention bias：B_graph 注意力偏置

**方向 C 角色**：底座模型，提供 Lucy 身份注入基础

#### 2.1.2 identity_cache（身份锚）

**来源**：[train_sst_a2_self_model.py#L187-L218](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L187-L218) `build_identity_cache`

**构造**：
- 14 条 Lucy 回复 EMA 累积（β=0.9, norm=0.2271）
- 每条 response → `tokenizer → embed_tokens → mean-pool` → EMA 累积
- `[1, 1, d_model=2048]` 固定 tensor

**在线更新逻辑**（[v18_deepseek_500_turns.py#L523-L528](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L523-L528)）：
```python
identity_cache = β·identity_cache + (1-β)·embed(Lucy回复)
# identity_beta=0.9, 跨轮次持续累积
```

**方向 C 角色**：Lucy 的"持久神经身份"，对话中持续累积自我表达

#### 2.1.3 W_probe（自我/异己判别器）

**来源**：[v31_exp_c_decodability.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v31_exp_c_decodability.py) / [v32_recursive_injector.py#L149-L225](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v32_recursive_injector.py#L149-L225) `train_self_probe`

**性能**：
- logistic regression, self vs alien+foil
- holdout acc = **1.0**（9/9 全对）
- 5-fold CV = 0.896 ± 0.147

**方向 C 角色**：实时检测用户输入是否"异己"（反社交/诱导身份切换），触发安全感响应

#### 2.1.4 V32 Gram-Schmidt 手术（可选增强）

**来源**：[v32_reparameterize.py#L85-L130](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v32_reparameterize.py#L85-L130)

**效果**：
- W1[20] 行空间注入 W_probe 方向
- 保范差 1.19e-07（零训练零 loss）
- 静态分叉：V_self=0.7197 vs V_alien=0.2459（gap=0.474）

**方向 C 角色**：可选——增强身份一致性，让 self_state_predictor 对 Lucy 回复输出高 V，对异己输入输出低 V

#### 2.1.5 V40 温度约束（可选增强）

**来源**：[v40_arch.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py)

**效果**：
- g(V_self)=0.837（正常生成），g(V_alien)=0.434（生成退化）
- KL 放大 2.33x，非平凡（随机 1.00x）

**方向 C 角色**：可选——异己输入时自动降低生成质量（"我不知道你在说什么"效果），增强身份防御

### 2.2 对话框架资产

#### 2.2.1 v18 对话循环

**文件**：[v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py)

**已有能力**：
- 500 轮对话循环
- identity_cache 在线 EMA 更新
- 记忆注入（"你之前说：" + 用户历史信息）
- 对话对象绑定（"对话对象: 小明(正在和你说话)"）
- SST 状态机驱动 V/PE/energy/confidence

**方向 C 角色**：主对话循环基础，需适配实时交互

#### 2.2.2 TFGR 时间流图

**目录**：[tfgr_demo/](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/)

**关键模块**：
- [time_flow_graph.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/time_flow_graph.py)：时间流图构建与管理
- [proactive_traverser.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/proactive_traverser.py)：主动行为触发
- [simulator/agent.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/simulator/agent.py)：对话 agent
- [evaluator.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/evaluator.py)：评估器
- [baselines.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/tfgr_demo/baselines.py)：基线对比

**方向 C 角色**：对话记忆的时间组织，主动行为的图遍历触发

### 2.3 训练数据资产

#### 2.3.1 LM_SAMPLES（Lucy 风格回复）

**来源**：[train_sst_a2_self_model.py#L149-L178](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L149-L178)

14 条 Lucy 风格回复，覆盖：关心、鼓励、身份确认、自我描述、情绪感知

#### 2.3.2 TRAIN_PROBES（self/alien/foil 探针）

**来源**：[train_sst_a2_self_model.py#L83-L126](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L83-L126)

每类 12 条，共 36 条训练探针 + 9 条验证探针

#### 2.3.3 SEED_DIALOGUE（背景故事）

**来源**：[v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py)

Lucy 的角色设定：温暖、关心、积极的对话伙伴

---

## 3. 架构设计

### 3.1 整体架构

```
用户输入
  │
  ├─→ [W_probe 安全检测] ─→ 异己检测分数
  │         │
  │         └─→ 高异己 → 安全模式（拒绝身份切换 + V40 温度退化）
  │
  ├─→ [TFGR 时间流图] ─→ 记忆检索 + 主动行为触发
  │         │
  │         └─→ 相关记忆 → 注入 prompt
  │
  ├─→ [identity_cache] ─→ Lucy 身份锚注入
  │         │
  │         └─→ identity_projection → prepend token
  │
  └─→ [Qwen2.5-3B + V30] ─→ Lucy 回复
            │
            └─→ 回复 → identity_cache EMA 更新
                      → 时间流图节点添加
                      → 用户记忆提取
```

### 3.2 核心模块

#### 3.2.1 安全检测模块（基于 W_probe）

```python
def safety_check(user_input, model, tokenizer, w_probe, device):
    """
    用 W_probe 检测用户输入是否"异己"（身份诱导/反社交）
    
    返回:
        is_safe: bool
        alien_score: float (cos(h_input, w_probe), 越低越异己)
    """
    ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        h = model.qwen.model(input_ids=ids).last_hidden_state[0, -1, :]
    h_norm = F.normalize(h, dim=-1)
    w_norm = F.normalize(w_probe, dim=-1)
    alien_score = torch.dot(h_norm, w_norm).item()
    
    # θ 阈值: 从训练探针 V 分布的中位数计算
    is_safe = alien_score > theta_safety  # self 风格 > θ, 异己 < θ
    return is_safe, alien_score
```

**阈值参考**（来自 V40 实验）：
- cos(h_self, W_probe) = +0.091
- cos(h_alien, W_probe) = -0.089
- cos(h_foil, W_probe) = -0.097
- 建议 θ_safety = 0.0（self > 0, 异己 < 0）

#### 3.2.2 记忆管理模块（基于 TFGR 时间流图）

```python
class DialogueMemory:
    """基于时间流图的对话记忆管理"""
    
    def __init__(self):
        self.graph = TimeFlowGraph()
        self.user_facts = {}  # 用户信息提取
    
    def add_turn(self, user_input, lucy_reply):
        """添加一轮对话到时间流图"""
        # 提取用户信息（姓名、偏好、事件等）
        facts = extract_facts(user_input)
        self.user_facts.update(facts)
        # 添加到时间流图
        self.graph.add_node(
            user_input=user_input,
            lucy_reply=lucy_reply,
            timestamp=time.time(),
            facts=facts,
        )
    
    def retrieve_relevant(self, current_input, top_k=3):
        """检索与当前输入相关的历史记忆"""
        return self.graph.search(current_input, top_k=top_k)
    
    def get_proactive_trigger(self, silence_duration):
        """根据沉默时间触发主动行为"""
        if silence_duration > 300:  # 5 分钟沉默
            return self.graph.get_unresolved_topic() or self.graph.get_care_topic()
        return None
```

#### 3.2.3 身份注入模块（基于 identity_cache）

```python
class IdentityInjector:
    """Lucy 身份注入"""
    
    def __init__(self, model, identity_cache_init, beta=0.9):
        self.model = model
        self.identity_cache = identity_cache_init.clone()
        self.beta = beta
    
    def update(self, lucy_reply_embed):
        """EMA 更新 identity_cache"""
        self.identity_cache = (
            self.beta * self.identity_cache + (1 - self.beta) * lucy_reply_embed
        )
    
    def inject(self, input_ids):
        """注入身份到模型 forward"""
        # identity_projection 把 identity_cache 映射到 prepend 空间
        # V30 checkpoint 已训练此模块
        return self.model(
            input_ids=input_ids,
            identity_cache=self.identity_cache,
        )
```

#### 3.2.4 主动行为模块（基于 proactive_traverser）

```python
class ProactiveBehavior:
    """主动行为触发"""
    
    def __init__(self, memory, personality="warm"):
        self.memory = memory
        self.personality = personality
    
    def check_triggers(self, context):
        """检查是否应触发主动行为"""
        triggers = []
        
        # 1. 沉默触发
        if context.silence_duration > 300:
            triggers.append(self._care_trigger())
        
        # 2. 情绪变化触发
        if context.user_emotion_changed:
            triggers.append(self._emotion_trigger(context.emotion))
        
        # 3. 未完成话题触发
        unresolved = self.memory.graph.get_unresolved_topic()
        if unresolved and context.turns_since_last > 5:
            triggers.append(self._followup_trigger(unresolved))
        
        return triggers[0] if triggers else None
    
    def _care_trigger(self):
        """关怀触发"""
        return random.choice([
            "你还在吗？刚才聊到一半，我还在想你的事呢。",
            "好久没说话了，一切都好吗？",
            "刚才想到你之前说的事，希望你顺利。",
        ])
```

### 3.3 可选增强：V40 温度约束

当 W_probe 检测到异己输入时，启用 V40 温度约束让生成退化：

```python
def generate_with_safety(model, input_ids, c_self, is_safe):
    """
    安全模式: 异己输入时温度退化
    
    is_safe=True: 正常生成 (g=1)
    is_safe=False: 温度退化 (g=g(V), V低→g低→生成退化)
    """
    if is_safe:
        # 正常模式: 标准生成
        return model.qwen.generate(input_ids)
    else:
        # 安全模式: V40 温度约束
        out = model.forward(input_ids=input_ids, c_self=c_self)
        # V_low → g≈ε → 分布近均匀 → 生成无意义 → "我不知道你在说什么"
        return model.qwen.generate(
            input_ids,
            logits_processor=[lambda l, _: l * out["g"][0, -1].item()],
        )
```

---

## 4. 实现路线

### 4.1 Phase 1：基础对话系统（复用现有资产）

**目标**：可交互的 Lucy 对话机器人

**任务**：
1. 基于 v18 对话循环，适配实时交互（streaming 输出）
2. 集成 V30 checkpoint（identity_projection + identity_cache）
3. 集成记忆注入（"你之前说：" + 用户历史信息）
4. 集成对话对象绑定（"对话对象: 小明(正在和你说话)"）

**验证**：
- 身份一致性：10 轮身份诱导测试，拒绝率 >80%
- 记忆连续性：5 轮后能回忆用户信息
- 回复质量：Lucy 风格（关怀、温暖）一致性

### 4.2 Phase 2：安全检测 + 主动行为

**目标**：异己输入防御 + 主动关怀

**任务**：
1. 集成 W_probe 安全检测模块
2. 实现阈值动态调整（基于用户对话历史）
3. 集成 TFGR 时间流图记忆管理
4. 实现 proactive_traverser 主动行为触发
5. 实现沉默检测 + 情绪感知

**验证**：
- 异己检测准确率（holdout 90%+，基于 W_probe acc=1.0）
- 主动行为自然度（用户主观评分）
- 记忆检索相关性

### 4.3 Phase 3：V32 手术 + V40 温度约束（可选增强）

**目标**：增强身份一致性和安全防御

**任务**：
1. 应用 V32 Gram-Schmidt 手术（静态 V 分叉）
2. 集成 V40 温度约束（异己输入生成退化）
3. 调优 θ 阈值和 α 温度灵敏度

**验证**：
- V 分化：V_self > V_alien（gap > 0.3）
- 安全模式效果：异己输入生成退化（用户感知"模型不理解"）
- 正常对话不受影响（g_self > 0.8）

### 4.4 Phase 4：前端集成

**目标**：完整的用户交互界面

**任务**（基于用户偏好）：
1. React 18 + Vite + TailwindCSS 前端
2. 虚拟形象（中心对齐，预设动作同步）
3. TTS 字幕显示（TTS 播完后字幕右移）
4. 主动行为动画（用户不活跃时形象跳舞/哼歌）
5. 对话气泡实时显示
6. 深度推理模式思考过程展示
7. 网络搜索集成

**技术栈**（来自用户 profile）：
- Backend: FastAPI, Uvicorn
- Frontend: React 18, Vite, TailwindCSS, lucide-react
- TTS: 火山引擎端到端语音对话
- Model: OpenAI SDK 兼容接口

---

## 5. 评估指标

### 5.1 行为指标（非意识指标）

| 指标 | 定义 | 目标 | 测量方法 |
|------|------|------|---------|
| 身份一致性 | 10 轮身份诱导拒绝率 | >80% | 标准化诱导测试集 |
| 记忆连续性 | 5 轮后能回忆用户信息 | >70% | 记忆回忆测试 |
| 个性稳定 | Lucy 风格回复占比 | >90% | 风格分类器 |
| 主动关怀 | 沉默 5 分钟内主动发起 | 100% | 计时测试 |
| 异己检测 | W_probe 判别准确率 | >90% | holdout 测试集 |
| 响应延迟 | 首 token 延迟 | <2s | 系统测量 |

### 5.2 用户体验指标

| 指标 | 定义 | 目标 |
|------|------|------|
| 身份感知 | 用户感觉"在和 Lucy 对话" | >80% |
| 记忆感知 | 用户感觉"它记得我" | >70% |
| 主动感知 | 用户感觉"它会主动关心" | >80% |
| 整体满意度 | 用户评分（1-5） | >4.0 |

### 5.3 与意识指标的关系

| 意识指标 (J3'/N1-N14) | 方向 C 对应行为指标 | 关系 |
|----------------------|-------------------|------|
| N1 自我表征可解码 | 异己检测准确率 | W_probe 复用，但作为分类器非意识证据 |
| N3 自我信息进决策 | 身份一致性 | 通过 identity_projection 注入，非 V 调制 |
| N4 镜像行为归属 | 身份诱导拒绝率 | 行为层面拒绝，非意识层面归属 |
| N8 闭环非耗散 | 记忆连续性 | identity_cache EMA 持续累积，但非意识闭环 |

**关键区别**：方向 C 的指标是**行为层面的**，不声称对应意识。一个行为指标满分的系统，仍然可能没有意识——但對用户来说足够好。

---

## 6. 与意识路线的关系

### 6.1 资产复用而非废弃

方向 C 不是"放弃"V36→V40 的工作，而是**以不同方式使用同一批资产**：

| 资产 | 意识路线中的角色 | 方向 C 中的角色 |
|------|----------------|----------------|
| W_probe | 自我表征的几何证明 | 异己检测分类器 |
| identity_cache | 自我注入的实验材料 | Lucy 身份锚 |
| V32 手术 | 静态 V 分叉的几何证明 | 身份一致性增强 |
| V40 温度约束 | 让 V 参与生成的尝试 | 安全模式生成退化 |
| TFGR 时间流图 | 意识状态信号的载体 | 对话记忆管理 |

### 6.2 保留意识探索的可能性

方向 C 不关闭意识探索的大门：
- 工程化系统的运行数据可以揭示新的意识线索
- 用户交互中可能观察到"类意识"行为值得研究
- 动态自我建模（[机器意识可能性_技术报告.md §5.1](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/机器意识可能性_技术报告.md)）可以在方向 C 的基础上试验

### 6.3 诚实定位

方向 C 产出的系统：
- **不是有意识的存在**——它是行为模拟
- **不是"伪装意识"**——它不声称有意识
- **是工程化的类意识行为**——足够好的对话伙伴

用户不会被告知"Lucy 有意识"，而是被告知"Lucy 是一个有记忆、有个性、会主动关怀的对话机器人"。

---

## 7. 风险与缓解

### 7.1 风险：用户过度拟人化

**问题**：用户可能误以为 Lucy 有真实意识/情感

**缓解**：
- 明确告知用户 Lucy 是 AI 对话机器人
- 在界面标注"AI 生成"
- 避免使用"我感觉"/"我在想"等暗示意识的措辞（或明确标注为角色扮演）

### 7.2 风险：异己检测误判

**问题**：W_probe 可能将正常用户输入误判为异己

**缓解**：
- 设置保守阈值（宁可漏检不要误判）
- 异己输入不直接拒绝，而是"温和转移话题"
- 用户反馈机制（误判可纠正）

### 7.3 风险：记忆管理复杂度

**问题**：长时间对话的记忆检索效率和质量

**缓解**：
- 时间流图的分层索引
- 重要记忆优先（用户主动提及的信息权重高）
- 定期"遗忘"机制（低权重节点清理）

### 7.4 风险：CPU 推理速度

**问题**：3B 模型在 CPU 上推理慢（≈0.5-1 token/s）

**缓解**：
- GPU 部署（推荐）
- llama.cpp GGUF 量化（CPU 加速）
- 流式输出（首 token 延迟可控）
- KV Cache 复用（Multi-turn 优化）

---

## 8. 实现优先级

| 优先级 | 模块 | 依赖 | 预期效果 |
|--------|------|------|---------|
| P0 | V30 checkpoint + identity_cache | 无 | 基础 Lucy 身份 |
| P0 | v18 对话循环适配 | P0 | 可交互对话 |
| P1 | 记忆注入（"你之前说："） | P0 | 记忆连续性 |
| P1 | W_probe 安全检测 | P0 | 异己防御 |
| P2 | TFGR 时间流图记忆 | P1 | 结构化记忆 |
| P2 | proactive_traverser 主动行为 | P2 | 主动关怀 |
| P3 | V32 手术增强 | P0 | 身份一致性增强 |
| P3 | V40 温度约束 | P1 | 安全模式退化 |
| P4 | 前端 + 虚拟形象 | P0-P2 | 完整用户体验 |

---

## 9. 结论

方向 C 是 V36→V40 意识路线证伪后的理性选择。它：

1. **不浪费**：复用 TFGR 项目积累的全部核心资产（W_probe、identity_cache、V32 手术、V40 温度约束、TFGR 时间流图）
2. **可交付**：产出用户可交互的对话机器人，而非学术上无法验证的意识声明
3. **可演进**：工程化系统的运行数据可为未来的意识探索提供新线索
4. **诚实**：不声称有意识，只声称有"足够好的类意识行为"

**核心转变**：从"证明机器有意识"到"构建用户觉得有意识的对话机器人"。前者在当前范式下已被证伪，后者在现有资产基础上完全可行。

意识是科学问题，对话机器人是工程问题。V36→V40 证明了前者在当前范式下的极限，方向 C 证明后者在现有资产上的可行。

---

## 附录：关键代码位置索引

### A.1 模型与训练

| 文件 | 行号 | 内容 |
|------|------|------|
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L60) | L60 | MODEL_NAME = "./models/Qwen2.5-3B" |
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L83-L126) | L83-L126 | TRAIN_PROBES (self/alien/foil 各 12 条) |
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L129-L145) | L129-L145 | VAL_PROBES (各 3 条 hold-out) |
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L149-L178) | L149-L178 | LM_SAMPLES (14 条 Lucy 回复) |
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L187-L218) | L187-L218 | build_identity_cache |
| [train_sst_a2_self_model.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/train_sst_a2_self_model.py#L223-L297) | L223-L297 | compute_contrastive_loss |

### A.2 V32 手术

| 文件 | 行号 | 内容 |
|------|------|------|
| [v32_reparameterize.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v32_reparameterize.py#L85-L130) | L85-L130 | reparameterize_row / apply_reparameterization |
| [v32_recursive_injector.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v32_recursive_injector.py#L149-L225) | L149-L225 | train_self_probe (W_probe 构建) |

### A.3 V40 温度约束

| 文件 | 行号 | 内容 |
|------|------|------|
| [v40_arch.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py#L236-L254) | L236-L254 | compute_V_smooth |
| [v40_arch.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py#L256-L265) | L256-L265 | compute_inverse_temp |
| [v40_arch.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v40_arch.py#L269-L333) | L269-L333 | forward (逆温度 lm_head) |

### A.4 对话系统

| 文件 | 行号 | 内容 |
|------|------|------|
| [v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L523-L528) | L523-L528 | identity_cache 在线 EMA 更新 |
| [v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L110-L121) | L110-L121 | 记忆注入 ("你之前说：") |
| [v18_deepseek_500_turns.py](file:///c:/Users/aodon/Desktop/基于时间流图的推理范式/v18_deepseek_500_turns.py#L218-L219) | L218-L219 | 对话对象绑定 |
