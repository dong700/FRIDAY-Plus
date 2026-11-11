# 脑学调研：长时序理解 · 记忆即智能 · TFGR 主导化路线

> 调研时间：2026-09-14。动机：比赛后深度复盘得出的判断——"智能并不在计算，而是由记忆里产生；TFGR 要逐步成为主导，解决涌现"。本文用脑科学严格检验这个命题，收罗长时序理解的神经机制，并给出 TFGR 主导化的分阶段路线。
> 原则：诚实。支持、修正、反对三面证据都摆出来。所有条目经搜索核实，未能确凿核实的标"待核实"。

---

## 第一部分：命题检验——"智能不在计算而在记忆"？

### 1.1 检验结论（先说结论）

**作为纲领成立，作为完备理论不成立。** 最站得住的表述是：

> **"智能是记忆的组织形式问题——当存储的不是事件快照而是可复用的预测性结构（后继关系/生成模型），且系统在行动闭环中用检索代替计算时，智能表现为涌现属性。"**

三面证据：

| 立场 | 证据 | 强度 |
|---|---|---|
| **支持** | Hopfield (1982)：吸引子动力学下"计算"与"联想检索"是同一物理过程——能量面下降既叫检索也叫计算，本体不可分；Ramsauer et al. (ICLR 2020, "Hopfield Networks is All You Need")：Transformer 注意力本质是现代联想记忆的一步检索——当代最强智能系统的心脏确实是记忆检索；海马索引理论（Teyler & DiScenna 1986）：海马只存索引，皮层存特征内容——**"记忆图主导、生成模型执行"正是脑的架构** | 强 |
| **修正** | Hawkins (2004/2021)、Stachenfeld et al. (2017)、Whittington et al. (2020)、Clark (2013) 一致表明：脑存的不是静态记录，而是**预测性结构**（时间序列模型、折扣未来占据、结构基底）；记忆必须以"生成预测—误差修正"的方式被使用才产生智能 | **最强** |
| **反对/限定** | Fodor & Pylyshyn (1988) 系统性论证：纯联想检索无组合性保证；Lake et al. (2017)：智能还需因果模型、直觉理论；Cowan (2001) 工作记忆约 4 组块硬瓶颈——**智能不是"存得多"，是"选择性调用"**；Buzsáki (2019) 本人强调意义来自**行动**而非存储；Poeppel & Adolfi (2020, eNeuro) 反驳由硬件单方面推出认知 | 不可绕过 |

### 1.2 关键机制逐条

**1) Hopfield 联想记忆即计算（Hopfield 1982）→ 现代 Hopfield = 注意力（Ramsauer et al., ICLR 2020）**
- 机制：能量面吸引子即存储即计算；现代 Hopfield 网络（连续态、指数容量）与 Transformer 注意力数学等价。
- 对 TFGR 启示：检索算子可形式化为能量下降，获得收敛性与容量保证——TFGR 的激活扩散不是"搜索技巧"，它和 LLM 的注意力是同构的两种记忆检索。

**2) 继任表征 / 预测地图（Dayan 1993；Stachenfeld et al. 2017, "The hippocampus as a predictive map"）**
- 机制：海马存储的不是当前状态，而是"预测的折扣和"（未来状态占据期望）——**记忆与预测在数学上合一**。这是"记忆=智能"命题最严格的形式化。
- 对 TFGR 启示：节点不能只存"发生了什么"，要携带"接下来通常会怎样"的转移信息（边=转移模式，不是快照）。对照 Hawkins 记忆-预测框架（2004/2021）：边应存**转移模式**，预测精度才是记忆质量指标。

**3) Tolman-Eichenbaum Machine（Whittington et al., Cell 2020；与 Transformer 数学同构，Whittington et al., ICLR 2022）**
- 机制：内嗅皮层学**结构基底**，海马绑定感觉因子——统一空间与关系记忆；且与 Transformer 同构。
- 对 TFGR 启示：需要"结构层+绑定层"双模块；泛化来自结构复用（同一图式套到新事件上）。

**4) 预测加工 / 自由能（Friston；Clark 2013 综述）**
- 机制：先验即存储的规律，感知=先验约束下的推断，行动=主动采样降低预测误差。
- 修正含义：记忆是推断的**先验**而非智能本体，且必须闭环到行动。对 TFGR：内建误差信号驱动的写入与精度加权（我们的 PE/PE_V 已是这个方向的雏形）。

**5) Buzsáki《The Brain from Inside Out》(2019)**
- 机制：脑先有自组织动力学，"垃圾模式"经行动结果获得意义，认知=内部"如果-则"预演。**这是"记忆/内部动力学主导"立场的最强系统论述。**
- 对 TFGR 启示：TFGR 的主导地位必须靠**自身动力学证据**（自发回放、自发声荡）确立，不能只靠外部注入（用户消息触发）。我们 background_loop 的自发反思正是这个方向。

**6) 反方限定（TFGR 必须正面回应的三条）**
- **组合泛化**：联想检索无组合性保证 → TFGR 必须证明其结构可分离重组，而非整体检索，否则退化为"高级查找表"。
- **选择性调用**：工作记忆 4 组块瓶颈 → 智能不在库存量，在调用机制。我们的 top-k + 势能竞争 + 激活后压制正是"选择性调用"的工程实现。
- **行动闭环**：无感知-运动闭环则语义无锚 → TFGR 的价值必须落在"改变下一步行动"上（价值桥/打分纠错闭环已具备雏形）。

---

## 第二部分：长时序理解的脑学机制（核心）

### 2.1 大脑怎么表征时间——六个机制

**1) 海马时间细胞（Pastalkova et al. 2008 Science；MacDonald/Lepage/Eichenbaum 2011 Neuron 首次命名；Eichenbaum 2013 TiCS 综述 "Memory on time"）**
- 机制：海马 CA1 神经元各自在特定时间段发放，感受野按序铺满整个延迟期，把"何时"编进情节记忆；**晚期时间感受野比早期更宽**。
- 证据：海马失活只破坏"分钟级经过时间"的判别（Jacobs et al. 2013 J Neurosci）——海马负责分钟级而非秒级时间记忆；啮齿类海马内部时间被证实为对数压缩。
- 对 TFGR：事件节点不必存绝对时钟，"激活顺序段"即时间戳。

**2) 时间上下文模型 TCM（Howard & Kahana 2002 J Math Psychol；神经映射版 2005 Psychol Rev）**
- 机制：维护一个**缓慢漂移的"时间上下文"向量**；每条项目与编码时上下文绑定；检索时项目恢复旧上下文（"跳回过去"）。近因效应=当前上下文与近期重叠更高；连续性效应=时间上相邻→联想更强，**自动涌现，无需显式规则**。
- 对 TFGR：给每个节点维护缓慢漂移的 EMA 上下文向量，激活扩散边权=时间相似性×上下文重叠度。漂移向量本身就是天然时间戳。

**3) 尺度不变时间表征 SITH（Shankar & Howard 2012 Neural Computation；Howard 2018 TiCS "Memory as perception of the past"）★ 长时序关键**
- 机制：一层指数衰减的漏积分器对刺激历史做 Laplace 变换，另一层线性算子近似逆变换，得到**对数压缩的过去表征**——固定数量单元铺开从毫秒到天的过去，越远精度越低但**永不丢失**（无特征尺度的幂律遗忘）。Howard 的类比：时间是记忆的"脚手架"，记忆=对过去的知觉，可被"扫描"。
- 证据：时间细胞感受野宽度随峰值时间增长（符合对数压缩预测）；猴背外侧前额叶单细胞呈压缩时间线（Tiganj et al.，正式发表待核实）；内嗅皮层用时间常数谱构建过去的记录（Bright et al. 2020 PNAS）。
- **对 TFGR：这是"长时序"的正确形态——固定容量、对数铺开，不是无限精确日志。**

**4) 层级时间感受野（Hasson et al. 2008 J Neurosci 提出 TRW；2015 TiCS 过程记忆层级框架，完整篇名待核实；Honey et al. 2012 Neuron；Murray et al. 2014 Nat Neurosci）**
- 机制：皮层时间感受窗从早期感觉区（毫秒-秒）沿层级向高级区（秒-分钟）逐级变长；打乱电影时间结构后高级区不再响应，证明其只对长时程结构敏感。Murray 2014：各皮层区自发放电的内在时间尺度呈层级排序，与解剖层级对应。
- **关键立场（Hasson 2015）："记忆不是某个仓库的功能，而是一切信息处理的内在属性。"** —— 这句话直接支撑"TFGR 主导化"。
- 对 TFGR：图不应均匀采样时间——按层级设多档时间窗，慢层节点做事件的抽象与合并（我们的两级感知栈秒级 ST-GCN++ + 分钟级序列 Transformer 正是这个架构，LSTR 是其 ML 版本）。

**5) 脑内无统一时钟（Buzsáki 2019, "The Brain from Inside Out" 第10/12章）**
- 机制：脑不是外部时间的记录仪；时间由网络自身动力学（振荡、细胞组装轨迹）从内部丈量——"脑外无时钟"，时间是生成动作与预测的副产品；脑是对数尺度的非均衡网络。
- 对 TFGR：节点"时间"可内源化——用系统自身的漂移量/事件步数丈量，而非依赖外部时钟。

**6) ML 领域的移植先例（方向已被验证可复用）**
- Kabir, Mochizuki-Freeman & Tiganj (2024, arXiv:2412.15292)：把尺度不变记忆网络嵌入深度 RL，智能体跨时间尺度稳健学习并**自发涌现时间细胞样单元**。
- Chien, Zhang & Honey (ICLR 2021)：语言模型 LSTM 单元自发形成时间尺度组织。
- Yamashita & Tani (2008 PLoS Comput Biol, MTRNN)：多时间尺度 RNN 自组织出功能层级。
- 结论：log 压缩与多时间尺度在 ML 两端均有可复用先例，TFGR 方向与神经科学及 ML 皆对得上。

### 2.2 事件边界由什么信号决定（ATG 切段可直接用）

**事件分割理论 EST（Zacks et al. 2007 Psychological Bulletin）的神经实现：**
- Kumar, Zacks, Hasson, Norman 等（2023 Cognitive Science）：用 GPT-2 逐词计算预测分布，发现事件边界对应**"贝叶斯惊异"（belief update）瞬升**而非普通 surprisal——边界是"预测模型被更新"的时刻，不是"预测错了"的时刻。
- Nguyen, Etzel, Bezdek & Zacks（2026 eLife）：边界前皮层模式剧变、边界后跨脑区模式同步稳定；预测误差与预测不确定性两类信号**部分分离、各自解释边界**。
- Baldassano et al. 2017 Neuron：HMM 从连续 fMRI 中无标注检出**嵌套层级**的稳定事件状态——感觉区事件短、高阶区事件长；高阶事件边界耦合海马激活升高，且**边界处海马激活越强，日后回忆越好**——边界是记忆的"取数点"。
- 多巴胺信号：Zacks et al. 2011 中脑/纹状体在不可预测点瞬态激活；Clewett 团队发现事件边界处 VTA 激活预测记忆时间膨胀（正式卷期待核实）——多巴胺系统对事件结构敏感，推动相邻事件在记忆中分离。
- 非预测误差边界：可预测转换也能成边界（Shim, Davachi & Baldassano）——工作记忆"停更/清空"同样触发；稳定上下文特征（地点/时间/目标/人物，Event Indexing Model 五维）任一跳变即候选边界。

**对 ATG 的直接映射——三信号投票切段**：
```
切段信号 = 预测误差瞬升（事件模型更新）+ 上下文特征跳变（地点/人物/目标）+ 新颖性/不确定性升高
任一过自适应阈值 → 边界：封存当前事件节点、提高边界处编码权重、拉大相邻事件时间距离特征
替换现行"同源画面15s固定去抖窗口"——去抖窗口只是三信号里"上下文跳变"的最粗糙近似
```

### 2.3 固化五步（episodic→semantic 在图上的操作序列）

| 步 | 脑机制 | 出处 | TFGR 对应操作 |
|---|---|---|---|
| ① 边界封存 | 事件终止处写入海马式稀疏事件节点（索引：指针+时间/地点锚） | 海马索引理论 | 事件节点只存"索引+边"，特征细节由生成模型重建 |
| ② 压缩重放 | SPW-R 以 10-20 倍压缩重放经验序列传向皮层 | Buzsáki 两阶段模型 (1989)；Wilson & McNaughton 1994 | 离线期按衰减权重回放路径，用回放样本更新生成模型/统计 |
| ③ 交错统计抽取 | 多次重放让重叠表征提取共享结构 | McClelland, McNaughton & O'Reilly 1995 (CLS) | 图上建立跨事件联想边、合并重复节点 |
| ④ 图式化 | 与图式一致的新信息单次学习即固化、脱离海马依赖 | Tse et al. 2007 Science | 高连通子网=图式；对齐图式的新边直接转永久 |
| ⑤ 语义化稀释 | 情境细节遗忘、概念保留（巩固增强语义、稀释情境） | Yassa & Reagh 2013 竞争痕迹理论 | 势能遗忘削减细节边、保留主干边 |

**CLS 的 2020s 修订（重要）**：Kumaran, Hassabis & McClelland (2016 TiCS)——重放是**按目标加权的经验统计**，不是均匀回放；McClelland, McNaughton & Lampinen (2020 Phil Trans R Soc B)——新信息落在已知维度上可**快速整合**，新增维度才需慢速交错学习。对 TFGR：离线演化时"与既有边一致的合并"走快路径，新增维度走慢路径分批回放。

**海马索引理论的直接印证（Tanaka & McHugh et al. 2018 Science）**：CA1 一群 c-Fos+ 锥体细胞以"索引式"表征情境身份，与携带稳定空间信息的神经元分工——**印迹即索引**。事件节点存"索引/指针+边"，细节存外部生成模型；检索=索引激活→生成模型重建。这正是"记忆图主导、生成模型执行"的神经对应。

---

## 第三部分：涌现的机制链与工程判据

### 3.1 脑里最被接受的涌现机制链

```
边界分割（造出离散元件）→ 海马索引+CLS（快慢双库）→ 离线 SPW-R 重放（时间压缩+新旧组合）
→ 图式快速吸收（Tse 2007）→ 建构性重组模拟（Schacter & Addis 2007：想象/洞见即重组）
```

**直接证据——重放产生新内容**：
- Gupta et al. (2010)：大鼠 T 迷宫实验，重放中出现动物**从未走过的"捷径"序列**——旧经验被离线重组为新轨迹。Buzsáki (2015 Hippocampus 综述) 明确：SPW-R 重放中"新近信息与既有信息被组合，支持决策、规划，并可能允许创造性思维"。
- Wagner et al. (2004 Nature "Sleep inspires insight")：数字简化任务中隐藏捷径规则，睡 8 小时组约 60% 发现，清醒对照约 23%——**睡眠重放产生质变而非量变**。Lerner "时间脚手架"模型：重放的时间压缩打通了清醒时 Hebbian 学习窗口外的跨时距关联。
- Schacter & Addis (2007) 建构性情节模拟：情节记忆的核心功能不是存档，而是提供**可拆卸的元件**（人物/地点/物体/情绪）重组成从未发生的模拟；海马损伤者既难回忆也难想象连贯场景——回忆与创造是同一机制。
- 建构性记忆（Schacter & Thakral 2024 J Cogn Neurosci）：提取不是读盘，是用线索驱动模式补全再"重写"一次；重组合在提供创造力的同时必然产生记忆错误——**遗忘细节与语义化是同一过程的两面**。

### 3.2 机器系统普遍缺的两步（TFGR 的机会）

1. **带目标加权的选择性重放**：脑不是均匀采样经验重放，而是按任务相关性重放并改写统计分布（Kumaran 2016 明确此功能）。多数机器系统的"离线反思"是均匀遍历或随机采样。
2. **受约束的重组评估回路**：重组出的新路径要有"预测有效性检验"+新颖性/价值门控（VTA 式）来决定去留。多数系统只有"随机采样+梯度"，**没有"重放→重组→检验→固化"这个闭环**。

→ 这两步就是 TFGR 离线演化要长出来的东西。我们的价值桥（正反馈/压制加权）恰好是"目标加权"的雏形——但只用于检索趋利避害，还没用于**重放采样加权**和**重组评估**。

### 3.3 涌现的工程化判据（什么才算涌现发生）

参照 Anderson (1972 "More is Different")、Wei et al. (2022 能力跃升)、Krakauer, Krakauer & Mitchell (2025 的警告)——**三条件同时满足才算涌现，只满足第一条只能叫"能力出现"**：
1. **相变信号**：某能力指标随 TFGR 记忆规模增长呈"近随机→过阈值跃升"；
2. **不可外推**：跃升点无法由小规模曲线外推预测；
3. **表征粗粒化**：出现可检测的新内部结构（新聚类/新图模式/压缩率跳变），而非仅指标假象。

---

## 第四部分：TFGR 主导化路线图（脑机制 → 工程分阶段）

### 4.0 现状定位（诚实盘点）

| 脑机制 | 我们已有的 | 差距 |
|---|---|---|
| 海马索引（图主导） | TFGR 图+激活扩散已是检索主通道 | 但 LLM 上下文里仍直接塞原始对话历史，图不是唯一入口 |
| 时间细胞/对数压缩 | 势能时间遗忘（指数半衰 48h/28d） | 衰减曲线是单指数，非幂律；无对数槽位；远端无粗粒化合并 |
| 层级时间感受野 | 两级感知栈（秒级 ST-GCN++ + 分钟级序列） | 图内无层级时间窗；ATG 活动区间未挂到更长时段节点下 |
| 三信号事件边界 | 同源画面 15s 去抖（工程补丁） | 无预测误差切段、无上下文跳变检测、无边界编码加权 |
| TCM 上下文漂移 | 无 | 节点无漂移上下文向量，边权仅时间距离+相似度 |
| CLS 快慢双库 | LLM(慢/云端)+TFGR(快/本地) 结构天然对应 | 固化流程无序：无"重放→抽取→图式→稀释"五步序列 |
| 目标加权重放 | 价值桥偏好加权（仅检索层） | 重放/反思采样未按价值加权；无"重组→检验→固化"闭环 |
| 组合泛化 | 未验证 | TFGR 结构能否分离重组未做实验 |
| 涌现判据 | 无 | 无相变检测、无表征粗粒化度量 |

### 4.1 阶段一：索引主导（TFGR 成为唯一记忆入口）

**脑依据**：海马索引理论——海马不存内容只存索引，皮层负责特征重建；Tanaka & McHugh 2018 证实印迹即索引。

**工程动作**：
1. M3/DeepSeek 上下文里**移除原始对话历史的直接注入**，一律改为：图激活检索 → 拿回"事件索引+相对时间+激活路径" → 生成模型重建细节。（对应脑：线索激活索引→反向投射模式补全）
2. 节点补**四时间戳**（对齐上一份调研的 Zep 方案，也与索引锚点一致）：valid_at / invalid_at / created_at / superseded_at。
3. 对话历史原文只归档不进上下文，成为图的"episode 溯源层"（对齐 HippoRAG 2 的段落节点）。

**验收信号**：同一对话任务，纯图检索上下文 vs 原始历史注入的效果对比（LongMemEval-S 口径）。

### 4.2 阶段二：时间表征升级（SITH/层级 TRF 落图）

**脑依据**：SITH 对数压缩（Shankar & Howard 2012）；层级 TRW（Hasson 2008/2015）；TCM 上下文漂移（Howard & Kahana 2002）。

**工程动作**：
1. **势能衰减曲线换幂律**：`p(t) ∝ 1/(1+t/τ)^α`（α≈1 量级），替代现行单指数半衰。幂律无特征尺度——远处节点按对数邻域**合并粗粒化**而非删掉（对应时间细胞"远端感受野变宽"），彻底改变"3天前 vs 30天前"的相对新旧保持。
2. **两级时间窗落图**：快窗层（秒-分钟：逐事件精细保留，指数衰减）+ 慢窗层（分钟-天：对数压缩、抽象合并、幂律衰减）；层间尺度比取 3-10 倍（MTRNN 的 τ 比）。ATG 活动区间挂在慢窗节点下。
3. **节点加漂移上下文向量**：每节点维护 EMA 上下文向量（可用现有 embedding 或轻量 bigram 向量），激活扩散边权 = 时间相似性 × 上下文重叠度——近因与连续性效应自动涌现，不再手写规则。
4. **事件边界三信号投票**：预测误差瞬升（用现有 PE/PE_V）+ 上下文特征跳变（地点/人物/目标字段变化）+ 新颖性，任一过自适应阈值即切段；边界处提高编码权重（importance 加成），相邻事件时间距离特征拉大（对应记忆时间膨胀）。15s 去抖窗口降级为"上下文跳变"信号的兜底。

### 4.3 阶段三：固化与重放闭环（涌现的生产线）

**脑依据**：CLS + SPW-R 重放（Buzsáki 两阶段）；Gupta 2010 重放产生捷径；Wagner 2004 睡眠洞见；Kumaran 2016 目标加权重放；Schacter 建构性重组。

**工程动作**（全部挂在现有 background_loop 上）：
1. **离线重放周期**（对应睡眠）：图空闲期不是"随机反思"，而是按**价值桥权重采样**旧路径重放（目标加权），重放中允许旧节点联想边被随机/目标驱动重新串接。
2. **重组→检验→固化闭环**（机器普遍缺的一步）：重组产生的新路径（新联想边/合并候选/shortcut 规则）不直接入图，先做**预测有效性检验**——用它对未来 N 个 tick 的预测误差与不重组基线比较；显著更好才固化（对应多巴胺新颖性/价值门控）。
3. **图式节点**：高连通子网自动升格为"图式/虚拟节点"（现有 virtual node 机制可承接）；与图式一致的新事件单次即固化（新边直接永久），不再走慢速路径——固化的速度由**图结构**决定，不由时间决定（Tse 2007 的核心）。
4. **语义化稀释**：固化完成后，细节边势能加速衰减、主干边保留——把"遗忘细节"与"保留语义"做成同一势能阀的两端（Yassa & Reagh 2013）。

### 4.4 阶段四：涌现的观测与判据

1. 记录能力指标随图规模的增长曲线，检测**相变点**（近随机→跃升）。
2. 检测**表征粗粒化**：定期对图做聚类/模式挖掘，出现新图式节点、新稳定聚类、算法压缩率跳变即证据。
3. 只有"相变+不可外推+粗粒化"三者齐备才宣称涌现；只出现前者如实标注为"能力出现"。

### 4.5 与 ML 路线的关系（两份调研的分工）

- 上一份 ML 调研（ESTimator/LSTR/PPR/Memory-R1）给的是**工程手段**：事件边界的现成算法、双窗模板、可微检索与训练信号。
- 本份脑学调研给的是**组织原则**：什么该主导（记忆结构）、时间该怎么表征（对数压缩+层级）、涌现该怎么生产（目标加权重放+受约束重组）。
- 两者不冲突且互相锁定：例如事件边界——ML 侧 ESTimator 给自适应阈值算法，脑学侧给"贝叶斯惊异+VTA 门控+边界编码加权"的机制完整性；可微分检索——ML 侧给 PPR/EMDR2，脑学侧给"记忆=预测性结构"的形式化方向（继任表征）。

---

## 第五部分：对命题的最终立场（写给后续的自己）

1. "智能在记忆"的正确工程化不是"把一切塞进图"，而是**改变分工**：LLM 从"持有上下文的计算者"降为"被索引激活的重建器"，TFGR 从"检索附件"升为"唯一记忆入口+自演化主体"。脑里海马与皮层就是这么分工的，且有 2018 Science 的索引印迹证据。
2. 但三条底线必须守住，否则 TFGR 退化为高级查找表：**组合泛化可验证**（结构可分离重组）、**选择性调用**（top-k 竞争+压制，已具备）、**行动闭环**（每次检索最终要落到改变行为上，价值桥是接口）。
3. 涌现不是玄学：分割造元件 → 快慢双库 → 目标加权重放 → 重组过检验固化 → 图式加速吸收。Wagner 2004 证明这条链能产生质变。机器系统普遍缺"目标加权重放"和"重组检验闭环"两步——这就是 TFGR 的主攻方向。
4. 诚实提醒：Krakauer et al. 2025 警告"指标跃升≠涌现"；Boden 创造力三型中记忆重组只覆盖**组合型创造**——TFGR 解决的是涌现的一条腿（重组），变革型创造需要修改概念空间本身，那是更远的事，不应现在宣称。

---

## 附：来源

**时间表征**
- [Eichenbaum, Memory on time (TiCS 2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3558533/) · [Eichenbaum, On the integration of space, time, and memory (Neuron 2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5662113/) · [Jacobs et al., Critical Role of the Hippocampus in Memory for Elapsed Time (J Neurosci 2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6618651/)
- [Sederberg, Howard & Kahana, A context-based theory of recency and contiguity (Psychol Rev 2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2585999/) · [Howard et al., TCM (Psychol Rev 2005)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1421376/)
- [Shankar & Howard, A Scale-Invariant Internal Representation of Time (Neural Computation 2012)](https://direct.mit.edu/neco/article-abstract/24/1/134/7733) · [Howard, Memory as perception of the past (TiCS 2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5881576/) · [Bright et al., Temporal record of the past in entorhinal cortex (PNAS 2020)](https://www.semanticscholar.org/paper/11c56c29aa7a51728a2c3cf36bdd31167f253bf4)
- [Hasson lab](https://hassonlab.princeton.edu/publications) · [Honey et al., Slow Cortical Dynamics (Neuron 2012)](https://collaborate.princeton.edu/en/publications/slow-cortical-dynamics-and-the-accumulation-of-information-over-l-2) · [Murray et al., A hierarchy of intrinsic timescales (Nat Neurosci 2014)](https://profiles.wustl.edu/en/publications/a-hierarchy-of-intrinsic-timescales-across-primate-cortex/)
- [Buzsáki, The Brain from Inside Out (OUP 2019)](https://buzsakilab.com/wp/2019/02/06/the-brain-from-inside-out-by-buzsaki-g/)
- [Kabir et al., Deep RL with time-scale invariant memory (arXiv:2412.15292)](https://arxiv.org/html/2412.15292) · [Chien, Zhang & Honey (ICLR 2021)](https://arxiv.org/pdf/2012.06717v2) · [Yamashita & Tani, MTRNN (PLoS Comput Biol 2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000220)

**事件结构与固化**
- [Kumar et al. 2023, Bayesian Surprise Predicts Human Event Segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11654724/) · [Nguyen et al. 2026, Multiple event segmentation mechanisms (eLife)](https://elifesciences.org/articles/107955/peer-reviews)
- [Baldassano et al. 2017, Discovering event structure (Neuron)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558154/) · [Zacks et al. 2011, Prediction error and dopamine in segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8653780/)
- [Teyler & Rudy 2007, Hippocampal Indexing Theory updated](http://people.whitman.edu/~herbrawt/hippocampus.pdf) · [Tanaka & McHugh 2018, The Hippocampal Engram as a Memory Index (Science)](http://gooa.las.ac.cn/apipaperc/api/paper/pdfdown?wid=JA201904027475933ZK&roid=RO201904027475933ZK)
- [O'Reilly et al. 2014, CLS review](https://cbmm-new.mit.edu/sites/default/files/documents/Week3_O-Reilly2014.pdf) · [Kumaran, Hassabis & McClelland 2016, CLS Updated (TiCS)](https://psychologie.uzh.ch/dam/jcr:83bd242c-50a5-45df-919e-3c134ff6979e/Kumaran.TiCS.2016.pdf) · [McClelland, McNaughton & Lampinen 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7209926/)
- [Buzsáki 2015, Hippocampal sharp wave-ripple review (Hippocampus)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4648295/) · [Pfeiffer 2020, The content of hippocampal replay](https://pmc.ncbi.nlm.nih.gov/articles/PMC7027863/)
- [Wagner et al. 2004, Sleep inspires insight (Nature) 数据讨论](https://www.learningscientists.org/blog/2017/1/12-1) · [Lerner, Unsupervised Temporal Learning During Sleep Supports Insight](https://lerner.sleepmemory.lab.utsa.edu/wp-content/uploads/2023/02/PDF-11.pdf)
- [Tse et al. 2007, Schemas and Memory Consolidation (Science)](https://www.science.org/doi/abs/10.1126/science.1135935) · [Schacter & Thakral 2024, Constructive Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223725/) · [Shim, Davachi & Baldassano, Predictable boundaries](https://davachilab.psychology.columbia.edu/sites/default/files/content/Predictable%20Boundaries.pdf)

**记忆即智能与涌现**
- Ramsauer et al., Hopfield Networks is All You Need (ICLR 2020, arXiv:2008.02217) · Hopfield 1982 (PNAS) · Dayan 1993 (Neural Computation)
- [Stachenfeld et al. 2017, The hippocampus as a predictive map (Nat Neurosci)](https://www.nature.com/articles/nn.4650) · Whittington et al. 2020 (Cell, Tolman-Eichenbaum Machine) · Whittington et al. 2022 (ICLR, TEM 与 Transformer 同构)
- Clark 2013, Whatever next? Predictive brains (BBS) · Fodor & Pylyshyn 1988 (Cognition) · Lake et al. 2017 (Behavioral and Brain Sciences) · Cowan 2001 (BBS) · Poeppel & Adolfi 2020 (eNeuro)
- Anderson 1972, More is Different (Science) · Wei et al. 2022 (arXiv:2206.07682) · Krakauer, Krakauer & Mitchell 2025（涌现判据批评，出处待核实）· Boden, The Creative Mind

**待核实清单**：Eichenbaum 2013 Neuron 具体篇名；"SITH"缩写正式出处；Hasson 2015 TiCS 完整篇名；Tiganj 猴前额叶论文正式发表刊物；Clewett VTA 研究正式卷期；preplay 原始论文出处；Krakauer et al. 2025 出处。
