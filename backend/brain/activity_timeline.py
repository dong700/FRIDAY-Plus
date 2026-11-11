"""
活动时序图 (Activity Timeline Graph, ATG)

设计来源: 用户核心需求 + 论文调研
    - 用户: "正常一个人他应该是什么样子的...不能分类,不能分类"
    - 用户: "看到用户坐在桌子前面...主动关心一下,顺便看一下用户具体在干什么"
    - Salvucci (CHI 2009): 多任务连续体理论 (中断/恢复 + 目标记忆)
    - FOCAL (arXiv:2604.19541): 任务层稳定 + 动作层可碎片化
    - Koppula & Saxena: 物体-活动关联 (entities)
    - Activity Graph Transformer (arXiv:2101.08540): 重叠/再发生/非序列依赖
    - Semi-Markov: 活动时长期望 (异常检测)

与 TFGR 的关系 (互补, 不替代):
    - TFGR 存离散事件 (点): "9:20 用户说: 我在写报告"
    - ATG 存持续活动 (区间): "9:20-12:00 在写报告 (中间被倒水打断2min)"
    - 结合点: 事件触发活动创建 / 活动产出事件 / 联合查询
    - 时间结构化: ATG + TFGR 共同构成"时间的结构化表示",
      让快照式 LLM 通过序列化文本"看到"时间流逝

核心创新:
    1. 具体描述不分类: "在写代码" / "在看文档" / "在打电话", 保留语义
    2. 分层: task_context (稳定任务层) + description (可变动作层)
    3. 中断/恢复链: interrupt_by / resume_of / context_snapshot (目标记忆)
    4. 持续区间: t_end=None 表示进行中, 支持活动持续演化
    5. 多维度查询: 当前活动/历史时间线/中断链/待办任务/活动持续时长

使用:
    atg = ActivityTimeline()
    atg.observe_activity(description="在写代码", task_context="实现登录功能",
                         source="vision", now=time.time())
    current = atg.get_current_activity(now=time.time())
    pending = atg.get_pending_tasks()
"""
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx


# 视觉噪音过滤阈值 (2026-08-10): 视觉活动持续超过此时长 (秒) 且无画面变化,
# 视为"静止画面/已知背景"而非新事件, 序列化时标注以免 Lucy 反复注意/提问.
# 基于时间判定, 不绑定任何具体画面特征 (非硬触发).
_STATIC_VISION_THRESHOLD_S = 600.0   # 10 分钟


# ==================== 边类型 ====================

class ActivityEdgeType(str, Enum):
    """活动图边类型 (借鉴 Allen 区间代数 + Salvucci 中断恢复理论)"""
    # 时序边 (Allen 关系子集)
    BEFORE = "before"              # A 在 B 之前
    OVERLAPS = "overlaps"          # A B 重叠 (并发多任务, 如边写代码边喝咖啡)
    CONTAINS = "contains"          # A 包含 B (写报告期间喝了咖啡)
    # 中断/恢复边 (Salvucci 多任务连续体理论)
    INTERRUPT = "interrupt"        # A 打断 B
    RESUME = "resume"              # A 恢复自 B
    # 语义/因果边
    ENABLE = "enable"              # 事实/物体 使能 活动 (deadline周三 → 在写报告)
    PRODUCE = "produce"            # 活动 产出 事实 (写报告 → 报告进度50%)
    INSTANCE_OF = "instance_of"    # 同类再发生 (今天的"写报告" ~ 昨天的"写报告")
    CAUSAL = "causal"              # 因果 (累了 → 喝咖啡)


class ActivityStatus(str, Enum):
    """活动状态机 (借鉴 Salvucci + MARAuder's Map 渐变思想)"""
    ONGOING = "ongoing"            # 进行中
    PAUSED = "paused"              # 被中断, 等待恢复
    RESUMED = "resumed"            # 已恢复 (从 paused 回来)
    COMPLETED = "completed"        # 已完成


# ==================== 活动节点 ====================

@dataclass
class ActivityNode:
    """活动节点 — 持续性、可重叠、有状态、有具体描述 (不分类)

    设计要点 (论文调研凝结):
        - description: 具体描述 ("在写代码"), 不是分类标签 ("work")
        - task_context: 任务层稳定 ("实现登录功能"), 动作层 description 可变
          (FOCAL: 任务层稳定, 动作层可碎片化)
        - t_end=None: 支持持续活动 (进行中)
        - interrupt_by/resume_of: 中断/恢复链 (Salvucci: 多任务连续体)
        - context_snapshot: 中断时上下文快照 (Salvucci: 目标记忆)
        - entities: 关联物体/概念 (Koppula: 物体-活动关联, 如杯子→喝咖啡)
        - expected_duration: 期望时长 (Semi-Markov: 异常检测)
        - confidence: 置信度 (Koppula: 图结构多假设, 这里简化为单一置信度)
    """
    id: str
    # 具体描述 (不分类)
    description: str                          # "在写代码" / "在看文档" / "在打电话"
    task_context: str = ""                    # 任务层 "实现登录功能" (动作层可变, 任务层稳定)
    # v20 归属身份 (2026-09-02): 该活动属于哪个服务对象. 默认 "user" (主对话者);
    # 多服务对象 (张三/李四) 传具体身份 id, 家人是 user 名下的关联实体不单独隔离.
    owner: str = "user"
    # 时序
    t_start: float = 0.0                      # 开始时间 (unix 秒)
    t_end: Optional[float] = None             # 结束时间 (None = 进行中)
    status: ActivityStatus = ActivityStatus.ONGOING
    # 中断/恢复链 (Salvucci 目标记忆理论)
    interrupt_by: Optional[str] = None        # 被哪个活动打断 (activity_id)
    resume_of: Optional[str] = None           # 恢复自哪个活动 (activity_id)
    context_snapshot: str = ""                # 中断时上下文 ("写到登录校验逻辑, 卡在token验证")
    # 关联 (Koppula 物体-活动关联)
    entities: List[str] = field(default_factory=list)        # 关联物体/概念 ["屏幕", "代码编辑器", "咖啡杯"]
    related_facts: List[str] = field(default_factory=list)   # 关联事实 ["deadline周三", "项目报告"]
    # 元数据
    source: str = "unknown"                   # vision / text / topic / video
    confidence: float = 0.5                   # 置信度 0..1
    raw_signal: str = ""                      # 原始信号 (scene_description / 用户原话)
    # 时长期望 (Semi-Markov: 异常检测)
    expected_duration_s: float = 1800.0       # 期望时长 (秒), 默认30min
    # 记录时间
    t_record: float = 0.0                     # 系统记录时间
    access_count: int = 0                     # 访问次数

    @property
    def duration_s(self) -> float:
        """已持续时间 (秒). 进行中的活动计算到 now, 但这里用 t_end 或 t_start 之差"""
        end = self.t_end if self.t_end is not None else self.t_start
        return max(0.0, end - self.t_start)

    @property
    def is_ongoing(self) -> bool:
        """进行中 (ONGOING 或 RESUMED, 且未结束)"""
        return self.status in (ActivityStatus.ONGOING, ActivityStatus.RESUMED) and self.t_end is None

    @property
    def is_paused(self) -> bool:
        return self.status == ActivityStatus.PAUSED

    def duration_until(self, now: float) -> float:
        """到 now 时刻的持续时间 (进行中活动)"""
        end = self.t_end if self.t_end is not None else now
        return max(0.0, end - self.t_start)


# ==================== 边数据 ====================

@dataclass
class ActivityEdgeData:
    edge_type: ActivityEdgeType
    strength: float = 1.0          # 因果强度 / 关联强度
    note: str = ""


# ==================== 活动时序图 ====================

class ActivityTimeline:
    """活动时序图 (Activity Timeline Graph)

    存储持续活动节点 + 多类型边, 支持增量式图构建和多维度查询.

    线程安全: 所有写操作加锁, 读操作返回快照.

    增量式构建逻辑 (新信号到达时):
        1. 新信号 → 候选 ActivityNode
        2. 与现有 ongoing 活动比较:
           - 描述相似 + 时间连续 → 合并 (延长 t_end)
           - 描述不同 + 时间重叠 → overlaps 边 (并发)
           - 描述不同 + 看起来打断 → 现有活动 paused, 新活动 interrupt 边
        3. 检查 paused 活动:
           - 信号暗示恢复 → resume 边
        4. 关联实体/事实 → enable/produce 边

    设计原则:
        - 具体描述, 不分类 (用户强调)
        - 任务层稳定, 动作层可碎片化 (FOCAL)
        - 中断/恢复链 (Salvucci)
        - 单一假设 + 置信度 (demo 阶段简化, 未来可扩展多假设)
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self._lock = threading.Lock()
        self._next_id = 0
        # 活动描述相似度阈值 (合并/中断判断用)
        self.merge_similarity_threshold = 0.6      # 描述相似度 > 0.6 → 合并
        self.interrupt_gap_threshold = 30.0        # 时间间隔 > 30s 且描述不同 → 可能中断
        # 活动自动完成: 长时间无更新视为完成
        self.auto_complete_timeout_s = 120.0       # ongoing 活动 120s 无更新 → 自动完成
        # 视觉画面去抖窗口 (秒): 同一摄像头画面两次采样间隔在此窗口内视为同源画面延续 → 合并.
        # (2026-09-02 补充: 此前 observe_activity 已引用但 __init__ 漏定义, 导致每次视觉
        #   场景都报 "'ActivityTimeline' object has no attribute 'vision_debounce_window_s'",
        #   ATG 去抖从未真正生效 → 静态画面被切成碎片活动链 -> 幻觉. 与 graph 层 _VISUAL_DEBOUNCE_S 对齐.)
        self.vision_debounce_window_s = 15.0
        # 最近完成的活动缓冲 (活动结束 → 沉淀到 TFGR 的桥接)
        # 设计 (用户核心诉求): 事件先在 ATG 建模, 结束后带完整时间区间写入 TFGR.
        # 否则 TFGR 里只有离散事件点, 没有"5:30-6:10 在吃饭"这种完整区间记录,
        # Lucy 思考时无法从 TFGR 归纳出"吃饭都在 5-6 点"的时间规律.
        self._completed_buffer: List[Dict] = []
        self._DRAIN_DEBOUNCE_S = 5.0   # 同一活动完成后的去抖窗口 (避免重复沉淀)
        self._last_drain_time = 0.0

        # 记忆纠错 (2026-08-09): 用户语言纠正对感知层认知的 top-down 修正.
        # 大脑机制: 用户澄清 (语言/社会信息) 具有 top-down 权威, 修正底层视觉感知
        # 形成的错误认知 (如"在自拍"). 记录纠正后, 抑制期内同名感知被重命名,
        # 序列化时对被纠正的活动附加澄清说明.
        # corrections: [{term, corrected_to, until, note}]
        self.corrections: List[Dict] = []

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"A{self._next_id:06d}"

    # -------- 增删 --------

    def _add_edge(self, a_id: str, b_id: str, edge_type: ActivityEdgeType,
                  strength: float = 1.0, note: str = "") -> None:
        self.G.add_edge(a_id, b_id,
                        key=f"{edge_type.value}:{a_id}->{b_id}",
                        data=ActivityEdgeData(edge_type=edge_type,
                                              strength=strength, note=note))

    def get_activity(self, activity_id: str) -> Optional[ActivityNode]:
        """读单个活动节点 (无锁, 假设图结构稳定)"""
        if activity_id not in self.G:
            return None
        return self.G.nodes[activity_id].get("data")

    # -------- 增量式观察 (核心入口) --------

    def apply_correction(
        self,
        term: str,
        corrected_to: str = "",
        now: float = 0.0,
        suppress_s: float = 600.0,
        note: str = "",
    ) -> bool:
        """用户纠正感知认知 (top-down 知觉修正).

        用户澄清 (如"那不是自拍") 修正 Lucy 对近期活动的错误认知.
        记录纠正后, 抑制期内同名感知输入会被重命名/抑制, 防止错误认知继续扩散.

        Args:
            term: 被纠正的认知关键词 (如 "自拍")
            corrected_to: 纠正后的认知 (如 "在玩手机"); 为空表示"该认知不成立", 抑制其冒头
            now: 当前时间戳
            suppress_s: 抑制时长 (秒), 默认 10 分钟
            note: 用户原话 (供序列化时澄清说明)

        Returns:
            bool: 是否成功记录纠正
        """
        if not term or not term.strip():
            return False
        now = now if now > 0 else time.time()
        with self._lock:
            # 去重: 同 term 未过期则更新 (延长抑制期)
            for c in self.corrections:
                if c["term"] == term and c["until"] > now:
                    c["corrected_to"] = corrected_to
                    c["until"] = now + suppress_s
                    c["note"] = note or c["note"]
                    c["t_start"] = now
                    return True
            self.corrections.append({
                "term": term,
                "corrected_to": corrected_to,
                "until": now + suppress_s,
                "note": note,
                "t_start": now,
            })
            # 清理过期纠正
            self.corrections = [c for c in self.corrections if c["until"] > now]
            return True

    def _find_correction(self, description: str, now: float) -> Optional[Dict]:
        """查找命中描述的活动纠正 (仅在抑制期内生效)"""
        for c in self.corrections:
            if c["until"] > now and c["term"] and c["term"] in description:
                return c
        return None

    def _correct_description(self, description: str, now: float) -> str:
        """对感知输入应用纠正: 命中生效纠正 → 重命名为纠正后认知 (否则原样)"""
        c = self._find_correction(description, now)
        if c is None:
            return description
        if c.get("corrected_to"):
            return c["corrected_to"]
        # corrected_to 为空 → 该认知不成立, 抑制为中性描述 (不冒头)
        return "在电脑前"

    def observe_activity(
        self,
        description: str,
        task_context: str = "",
        now: float = 0.0,
        source: str = "unknown",
        raw_signal: str = "",
        entities: Optional[List[str]] = None,
        related_facts: Optional[List[str]] = None,
        confidence: float = 0.5,
        expected_duration_s: float = 1800.0,
        owner: str = "user",
    ) -> str:
        """观察到一个活动信号 → 增量式更新图

        返回: 活动 id (新建或合并后的)

        决策逻辑:
            1. 先检查是否需要自动完成过期的 ongoing 活动
            2. 找当前 ongoing 活动, 判断是合并/重叠/中断
            3. 检查是否是恢复 paused 活动
            4. 创建新节点 + 添加边
        """
        with self._lock:
            # 记忆纠错: 用户澄清后, 抑制期内把被纠正的感知重命名 (top-down 知觉修正)
            description = self._correct_description(description, now)

            # 1. 自动完成过期活动
            self._auto_complete_stale(now)

            # 2. 查找当前 ongoing 活动
            ongoing = self._find_ongoing_activities(now, owner=owner)

            if ongoing:
                # 取最近的 ongoing 活动 (t_start 最大)
                current = max(ongoing, key=lambda a: a.t_start)
                sim = _description_similarity(description, current.description)

                # v22 同源视觉画面去抖合并 (2026-09-02 根治"AT建模碎片化幻觉"):
                # 同一摄像头画面持续存在时, _extract_simple_activity 只按关键词把同一画面
                # 抖动成"在吃东西/在休息/在吃东西", 字符级 Jaccard 不合并, 导致同一坐姿被
                # 切成 ABABAB 多段 → CoT 误读成"吃吃停停七八轮"幻觉.
                # 规则: 新信号与当前活动都来自 vision, 且距上次刷新在视觉去抖窗口内,
                #   一律按"同源画面延续"合并 (更新描述/刷新 t_record), 不切成新活动.
                #   只有画面真的长时间无信号 (超 auto_complete_timeout) 才判结束.
                # 非硬触发: 仅针对"同源传感器持续报告"这一事实, 不绑定任何画面特征.
                last_refresh_gap = now - (current.t_record or current.t_start)
                if (source == "vision" and current.source == "vision"
                        and sim < self.merge_similarity_threshold
                        and last_refresh_gap < self.vision_debounce_window_s):
                    # 更新当前活动描述为最新观察 (更贴近此刻画面), 延长 t_record 防误完成
                    self._merge_into(current, description, now, source, raw_signal,
                                     entities, confidence)
                    # 描述更新优先级: 视觉碎片场景里不断合并会盖掉真实活动语义, 但此处
                    # 保留最新视觉观察作为活动描述 (同一画面下措辞趋同), 语义损失可忽略
                    current.description = description
                    return current.id

                if sim >= self.merge_similarity_threshold:
                    # 描述相似 → 合并 (延长 t_end, 更新 confidence/entities)
                    self._merge_into(current, description, now, source, raw_signal,
                                     entities, confidence)
                    return current.id

                # 描述不同 → 先检查是否是恢复某个 paused 活动
                # (场景: 倒水打断写代码 → 倒水 ongoing → 新信号"写代码" = 恢复原任务)
                paused_list = self._find_paused_activities(owner=owner)
                if paused_list:
                    # 找最匹配的 paused 活动
                    best_paused = None
                    best_sim = 0.0
                    for p in paused_list:
                        p_sim = _description_similarity(description, p.description)
                        if p_sim > best_sim:
                            best_sim = p_sim
                            best_paused = p
                    if best_paused and best_sim >= self.merge_similarity_threshold * 0.8:
                        # 恢复 paused 活动
                        resumed_id = self._create_node(
                            description=description,
                            task_context=best_paused.task_context,
                            t_start=now, t_end=None,
                            status=ActivityStatus.RESUMED,
                            source=source, raw_signal=raw_signal,
                            entities=entities or best_paused.entities,
                            related_facts=related_facts or best_paused.related_facts,
                            confidence=confidence,
                            expected_duration_s=best_paused.expected_duration_s,
                            resume_of=best_paused.id,
                            owner=owner,
                        )
                        self._add_edge(resumed_id, best_paused.id, ActivityEdgeType.RESUME,
                                       note=f"恢复 {best_paused.description}")
                        return resumed_id

                # 不是恢复 → 可能是新活动 (重叠或打断)
                # 简化判断: 如果新活动看起来是"短事件" (如倒水/喝咖啡/接电话),
                # 视为打断; 否则视为重叠 (并发多任务)
                # (demo 阶段用 expected_duration 近似, 未来可用分类器)
                is_short_interruption = expected_duration_s < 600.0  # < 10min 视为短中断

                if is_short_interruption:
                    # 当前活动暂停, 新活动打断
                    current.status = ActivityStatus.PAUSED
                    current.t_end = now  # 暂停时间点
                    current.context_snapshot = f"暂停于: {current.description} (t={now:.0f})"

                    new_id = self._create_node(
                        description=description,
                        task_context=task_context or current.task_context,
                        t_start=now, t_end=None,
                        status=ActivityStatus.ONGOING,
                        source=source, raw_signal=raw_signal,
                        entities=entities, related_facts=related_facts,
                        confidence=confidence,
                        expected_duration_s=expected_duration_s,
                        owner=owner,
                    )
                    # 设置被打断节点的 interrupt_by (被新活动打断)
                    current.interrupt_by = new_id
                    # interrupt 边: new → current (new 打断了 current)
                    self._add_edge(new_id, current.id, ActivityEdgeType.INTERRUPT,
                                   note=f"{description} 打断了 {current.description}")
                    return new_id
                else:
                    # 重叠 (并发多任务)
                    new_id = self._create_node(
                        description=description,
                        task_context=task_context,
                        t_start=now, t_end=None,
                        status=ActivityStatus.ONGOING,
                        source=source, raw_signal=raw_signal,
                        entities=entities, related_facts=related_facts,
                        confidence=confidence,
                        expected_duration_s=expected_duration_s,
                        owner=owner,
                    )
                    # overlaps 边 (双向)
                    self._add_edge(new_id, current.id, ActivityEdgeType.OVERLAPS,
                                   note=f"{description} 与 {current.description} 并发")
                    self._add_edge(current.id, new_id, ActivityEdgeType.OVERLAPS,
                                   note=f"{current.description} 与 {description} 并发")
                    return new_id

            # 3. 没有 ongoing, 检查是否是恢复 paused 活动
            paused_list = self._find_paused_activities(owner=owner)
            if paused_list:
                # 取最近暂停的活动
                paused = max(paused_list, key=lambda a: a.t_end or a.t_start)
                sim = _description_similarity(description, paused.description)
                if sim >= self.merge_similarity_threshold * 0.8:
                    # 描述相似 → 恢复
                    resumed_id = self._create_node(
                        description=description,
                        task_context=paused.task_context,
                        t_start=now, t_end=None,
                        status=ActivityStatus.RESUMED,
                        source=source, raw_signal=raw_signal,
                        entities=entities or paused.entities,
                        related_facts=related_facts or paused.related_facts,
                        confidence=confidence,
                        expected_duration_s=paused.expected_duration_s,
                        resume_of=paused.id,  # 标记: 我恢复了 paused
                        owner=owner,
                    )
                    # resume 边: resumed → paused
                    self._add_edge(resumed_id, paused.id, ActivityEdgeType.RESUME,
                                   note=f"恢复 {paused.description}")
                    return resumed_id

            # 4. 全新活动
            new_id = self._create_node(
                description=description,
                task_context=task_context,
                t_start=now, t_end=None,
                status=ActivityStatus.ONGOING,
                source=source, raw_signal=raw_signal,
                entities=entities, related_facts=related_facts,
                confidence=confidence,
                expected_duration_s=expected_duration_s,
                owner=owner,
            )
            # 添加 before 边 (与最近的所有活动建立时序关系)
            recent = self._find_recent_activities(now, window_s=3600.0, owner=owner)
            for r in recent:
                if r.id != new_id:
                    self._add_edge(r.id, new_id, ActivityEdgeType.BEFORE,
                                   note=f"{r.description} → {description}")
            return new_id

    def complete_activity(self, activity_id: str, now: float, note: str = "") -> None:
        """手动完成一个活动"""
        with self._lock:
            node = self.get_activity(activity_id)
            if node and node.status != ActivityStatus.COMPLETED:
                node.status = ActivityStatus.COMPLETED
                node.t_end = now
                self._record_completed(node)

    def _create_node(self, **kwargs) -> str:
        """创建节点 (调用方持锁)"""
        node_id = self._gen_id()
        kwargs["id"] = node_id
        if "t_record" not in kwargs:
            kwargs["t_record"] = kwargs.get("t_start", 0.0)
        node = ActivityNode(**kwargs)
        self.G.add_node(node_id, data=node)
        return node_id

    def _merge_into(self, node: ActivityNode, new_desc: str, now: float,
                    source: str, raw_signal: str,
                    entities: Optional[List[str]], confidence: float) -> None:
        """合并新信号到现有节点 (调用方持锁)"""
        # 进行中活动: 不更新 t_end (保持 None), 只更新元数据
        if source != "unknown":
            node.source = source
        if raw_signal:
            node.raw_signal = raw_signal
        if entities:
            for e in entities:
                if e not in node.entities:
                    node.entities.append(e)
        # 置信度: 取较高值
        node.confidence = max(node.confidence, confidence)
        node.access_count += 1
        # 关键: 记录本次刷新时间. auto_complete 判断"活动是否还在进行"靠它 —
        # 只要视觉信号持续刷新 (每15s), t_record 就一直更新, 长活动不会被误切成段.
        node.t_record = now

    def _auto_complete_stale(self, now: float) -> int:
        """自动完成过期/停滞的 ongoing 活动 (调用方持锁)

        规则 (修复: 长活动不再被"持续时间超期望3倍"误切段):
            - 短中断活动 (expected < 600s): 超过期望时长 → 完成 (倒水早该结束了)
            - 长活动 (expected >= 600s): 距上次刷新超过 auto_complete_timeout → 完成.
              关键: "写报告"这类可长期持续的活动, 只要视觉信号持续刷新 (每15s),
              就一直 ongoing, 不会因"写太久"被切成多个短段. 只有用户真的长时间
              离开/无信号 (如 120s 无更新) 才判定活动结束.

        返回: 自动完成的活动数
        """
        count = 0
        for node_id in list(self.G.nodes):
            node = self.get_activity(node_id)
            if node and node.is_ongoing:
                duration = node.duration_until(now)
                if node.expected_duration_s < 600.0:
                    # 短中断活动: 超过期望时长即完成
                    if duration > node.expected_duration_s:
                        node.status = ActivityStatus.COMPLETED
                        node.t_end = now
                        self._record_completed(node)
                        count += 1
                else:
                    # 长活动: 距上次刷新超时 (无新信号) 才完成
                    if now - node.t_record > self.auto_complete_timeout_s:
                        node.status = ActivityStatus.COMPLETED
                        node.t_end = now
                        self._record_completed(node)
                        count += 1
        return count

    def _record_completed(self, node: ActivityNode) -> None:
        """记录一个刚结束的活动 (供 web_api 沉淀到 TFGR). 调用方持锁."""
        if node.t_end is None:
            return
        self._completed_buffer.append({
            "activity_id": node.id,
            "description": node.description,
            "task_context": node.task_context,
            "t_start": node.t_start,
            "t_end": node.t_end,
            "source": node.source,
        })

    def drain_completed_activities(self, now: float = 0.0,
                                   max_count: int = 20) -> List[Dict]:
        """取出最近完成的活动 (清空缓冲), 供外部沉淀到 TFGR.

        返回: 完成活动记录列表 [{activity_id, description, task_context,
               t_start, t_end, source}]
        """
        with self._lock:
            self._last_drain_time = now if now else time.time()
            items = list(self._completed_buffer[:max_count])
            del self._completed_buffer[:max_count]
            return items

    # -------- 内部查找 --------

    def _find_ongoing_activities(self, now: float, owner: Optional[str] = None) -> List[ActivityNode]:
        """找所有 ongoing 活动 (调用方持锁). owner=None 不过滤"""
        result = []
        for node_id in self.G.nodes:
            node = self.get_activity(node_id)
            if node and node.is_ongoing:
                if owner is not None and getattr(node, 'owner', 'user') != owner:
                    continue
                result.append(node)
        return result

    def _find_paused_activities(self, owner: Optional[str] = None) -> List[ActivityNode]:
        """找所有 paused 活动 (调用方持锁). owner=None 不过滤"""
        result = []
        for node_id in self.G.nodes:
            node = self.get_activity(node_id)
            if node and node.is_paused:
                if owner is not None and getattr(node, 'owner', 'user') != owner:
                    continue
                result.append(node)
        return result

    def _find_recent_activities(self, now: float, window_s: float,
                                include_ongoing: bool = True, owner: Optional[str] = None) -> List[ActivityNode]:
        """找最近 window_s 秒内的活动 (调用方持锁)

        Args:
            window_s: 回看窗口 (秒)
            include_ongoing: 是否强制包含进行中的活动.
                关键: 进行中活动若 t_start 很早 (如用户 7 点开始工作, 现在 14 点),
                `now - t_start` 远超 window_s, 会被普通时间过滤排除 —
                导致"用户从 7 点就一直坐着"这种长期持续信息丢失.
                设为 True 时, 进行中活动不受窗口限制, 始终返回 (让动态建模喂足).
            owner: 归属身份过滤 (v20, None=不过滤)
        """
        result = []
        for node_id in self.G.nodes:
            node = self.get_activity(node_id)
            if not node:
                continue
            if owner is not None and getattr(node, 'owner', 'user') != owner:
                continue
            if include_ongoing and node.is_ongoing:
                result.append(node)
            elif (now - node.t_start) <= window_s:
                result.append(node)
        return result

    # -------- 查询 API (供副脑 M2 / M3 使用) --------

    def get_current_activity(self, now: float, owner: Optional[str] = None) -> Optional[ActivityNode]:
        """现在在干什么 (最新的 ongoing 活动)

        如果有多个 ongoing (并发), 返回 t_start 最大的那个.
        用户场景: "Lucy 看到用户坐在桌子前, 顺便看用户在干什么"
        Args:
            owner: 归属身份过滤 (v20, None=不过滤)
        """
        with self._lock:
            return self._get_current_activity_unlocked(now, owner)

    def _get_current_activity_unlocked(self, now: float, owner: Optional[str] = None) -> Optional[ActivityNode]:
        """无锁版本 (调用方持锁)"""
        ongoing = self._find_ongoing_activities(now, owner=owner)
        if not ongoing:
            return None
        return max(ongoing, key=lambda a: a.t_start)

    def get_current_activities(self, now: float) -> List[ActivityNode]:
        """现在在干什么 (所有 ongoing 活动, 包括并发的)

        场景: 用户边写代码边喝咖啡 → 返回两个活动
        """
        with self._lock:
            return self._find_ongoing_activities(now)

    def get_recent_timeline(self, now: float, lookback_s: float = 3600.0) -> List[ActivityNode]:
        """之前在干什么 (最近 lookback_s 秒的活动序列, 按时间排序)

        副脑用: 了解用户活动历史, 推断活动模式
        """
        with self._lock:
            recent = self._find_recent_activities(now, lookback_s)
            return sorted(recent, key=lambda a: a.t_start)

    def get_interrupt_chain(self, activity_id: str) -> List[ActivityNode]:
        """获取与这个任务相关的中断/恢复链

        场景: A(写代码) 被B(倒水)打断 → B结束 → A恢复 → A又被C(电话)打断
        从当前活动出发, 沿 resume_of / interrupt_by 字段回溯所有相关活动.
        返回按时间排序的活动列表: [A, B, A_resumed, C, ...]

        Salvucci 多任务连续体理论: 中断后恢复质量取决于上下文保存
        """
        with self._lock:
            chain = []
            current_id = activity_id
            visited = set()
            while current_id and current_id not in visited:
                visited.add(current_id)
                node = self.get_activity(current_id)
                if not node:
                    break
                chain.append(node)
                # 优先沿 resume_of 跳到原被中断的活动
                if node.resume_of:
                    current_id = node.resume_of
                # 否则沿 interrupt_by 跳到打断活动
                elif node.interrupt_by:
                    current_id = node.interrupt_by
                else:
                    break
            return sorted(chain, key=lambda a: a.t_start)

    def get_pending_tasks(self) -> List[ActivityNode]:
        """有什么没做完 (paused 状态的活动)

        用户场景: "Lucy 主动关心, 顺便看有没有可以帮忙做的"
        副脑用: 生成"需要帮忙"的提议
        """
        with self._lock:
            return self._find_paused_activities()

    def get_activity_duration(self, activity_id: str, now: float) -> float:
        """活动已持续时间 (秒)

        Semi-Markov 异常检测: 持续时间 > expected_duration × 2 → 可能卡住
        """
        with self._lock:
            node = self.get_activity(activity_id)
            if not node:
                return 0.0
            return node.duration_until(now)

    def is_activity_stale(self, activity_id: str, now: float) -> bool:
        """活动是否持续过久 (异常检测)

        场景: 写代码写了 4 小时 (期望 30min) → 可能卡住, 该 CHECK_IN
        """
        with self._lock:
            return self._is_activity_stale_unlocked(activity_id, now)

    def _is_activity_stale_unlocked(self, activity_id: str, now: float) -> bool:
        """无锁版本 (调用方持锁)"""
        node = self.get_activity(activity_id)
        if not node or not node.is_ongoing:
            return False
        duration = node.duration_until(now)
        return duration > node.expected_duration_s * 2.0

    def get_activity_hypotheses(self, now: float) -> List[Tuple[ActivityNode, float]]:
        """当前活动假设 (带置信度)

        demo 阶段: 返回 ongoing 活动列表 + 置信度
        未来: 支持多假设图结构 (Koppula 启发)
        """
        with self._lock:
            ongoing = self._find_ongoing_activities(now)
            return [(a, a.confidence) for a in ongoing]

    # -------- 时序化文本 (喂给 M3/LLM, 让快照推理"看到"时间流逝) --------

    @staticmethod
    def _conf_annotation(node) -> tuple:
        """视觉观察置信度 → (描述前缀, 不确定注记).

        不确定度先验 (PE 式, 非硬规则): 视觉观察可能看错, 低置信度时直接写成
        "疑似/可能不准", 让模型天生看到不确定性, 从而在"活动时序(视觉) vs 用户语言"
        冲突时主动疑惑/询问, 而不是把观察当事实固守. (修复吃东西冲突不反问的根因)
        仅对视觉来源生效; 用户/语言来源不标注.

        Returns:
            (desc_prefix, note): 前缀如 "疑似", 注记如 " (视觉置信度低, 可能看错)".
        """
        src = str(getattr(node, 'source', '') or '')
        if 'vision' not in src:
            return ("", "")
        conf = float(getattr(node, 'confidence', 0.5) or 0.5)
        if conf < 0.4:
            return ("疑似", " (视觉置信度低, 可能看错)")
        if conf < 0.7:
            return ("", " (视觉置信度中, 可能不准)")
        return ("", "")

    # token 减负上限 (v21): 时间线默认最多展示最近多少个活动. 时间感知最关键的是
    # "最近活动 + 当前状态 + 待办", 较早期细节可查 TFGR 记忆, 故折叠省略不损失感知.
    _DEFAULT_TIMELINE_MAX_ITEMS = 14

    def serialize_timeline_text(self, now: float, lookback_s: float = 3600.0, owner: Optional[str] = None,
                                max_items: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        """把时间线序列化为文本 (供 M3 ContextBundle 使用)

        Args:
            owner: 归属身份过滤 (v20, None=不过滤; 传 owner 则只序列化该服务对象的活动)

        解决: LLM 快照式推理不理解时间流逝
        方法: 把 ActivityTimeline 序列化为时间线文本, 让 LLM 通过文本"看到"时间

        输出示例:
            "[用户活动时间线]
             09:20 开始: 在写代码 (任务: 实现登录功能) [vision]
             09:45 中断: 在倒水 (打断: 在写代码)
             09:47 恢复: 在写代码 (任务: 实现登录功能)
             10:30 中断: 同事问问题 (打断: 在写代码)
             10:35 恢复: 在写代码 (任务: 实现登录功能)
             当前: 在写代码 (已持续 2h15min)"
        """
        with self._lock:
            recent = self._find_recent_activities(now, lookback_s, owner=owner)
            if not recent:
                return "[用户活动时间线]\n(无近期活动记录)"

            recent.sort(key=lambda a: a.t_start)
            lines = ["[用户活动时间线]"]
            # v21 token 减负: 按条数折叠较早期活动 (默认保留最近 14 条), 尾部 [当前]/[待办] 不受影响
            _max_items = max_items if max_items is not None else self._DEFAULT_TIMELINE_MAX_ITEMS
            _show = recent
            if _max_items and len(recent) > _max_items:
                _omit = len(recent) - _max_items
                _show = recent[-_max_items:]
                lines.append(f"(省略 {_omit} 个较早活动, 保留最近 {_max_items} 个)")
            for node in _show:
                t_str = _format_time(node.t_start)
                # 记忆纠错: 若该活动命中生效纠正, 附加澄清说明 (让 LLM 区分"感知"与"用户澄清")
                corr = self._find_correction(node.description, now)
                corr_note = f" (用户澄清: {corr['note']})" if (corr and corr.get("note")) else ""
                # 不确定度先验: 视觉观察按置信度标注"疑似/可能不准", 让模型看到视觉可能看错
                # (而非把观察当事实 — 修复"视觉与用户语言冲突却不疑惑/不问"的根因)
                pre, conf_note = self._conf_annotation(node)
                desc_display = f"{pre}{node.description}" if pre else node.description
                if node.status == ActivityStatus.ONGOING:
                    dur = node.duration_until(now)
                    # 视觉噪音过滤 (2026-08-10 用户核心诉求): 静态画面 (如一直穿红衣服坐着) 
                    # 长时间无变化, 不应每次都被当成新鲜信号让 Lucy 反复注意/提问. 
                    # 标注"静止画面(无新变化)"让模型感知这是已知背景而非新事件.
                    # 基于"持续时长"判定, 不绑定任何具体画面特征 (非硬触发).
                    static_hint = ""
                    if node.source == "vision" and dur >= _STATIC_VISION_THRESHOLD_S:
                        static_hint = " (静止画面, 长时间无画面变化, 属已知背景, 非新事件)"
                    # 进行中活动显示开始时刻 + 已持续时长:
                    # 例 "07:00 进行中: 在写代码 ... 已持续 7h (从 07:00 一直到现在)"
                    # 让 LLM 感知"用户从早上一直持续到现在", 而非只知道"当前在写代码"
                    lines.append(
                        f"{t_str} 进行中: {desc_display} "
                        f"(任务: {node.task_context or '未指定'}) "
                        f"[{node.source}] 已持续 {_format_duration(dur)} (从 {t_str} 起)"
                        f"{static_hint}{conf_note}{corr_note}"
                    )
                elif node.status == ActivityStatus.PAUSED:
                    lines.append(f"{t_str} 已暂停: {desc_display} "
                                 f"(任务: {node.task_context or '未指定'}) "
                                 f"[被 {node.interrupt_by or '?'} 打断]{conf_note}{corr_note}")
                elif node.status == ActivityStatus.RESUMED:
                    lines.append(f"{t_str} 已恢复: {desc_display} "
                                 f"(任务: {node.task_context or '未指定'}) "
                                 f"[恢复自 {node.resume_of or '?'}]{conf_note}{corr_note}")
                elif node.status == ActivityStatus.COMPLETED:
                    dur = node.duration_s
                    lines.append(f"{t_str} 已完成: {desc_display} "
                                 f"(任务: {node.task_context or '未指定'}) "
                                 f"耗时 {_format_duration(dur)}{conf_note}{corr_note}")

            # 当前状态摘要
            current = self._get_current_activity_unlocked(now, owner)
            if current:
                dur = current.duration_until(now)
                stale = self._is_activity_stale_unlocked(current.id, now)
                stale_hint = " [持续过久, 可能需要帮助]" if stale else ""
                cur_start = _format_time(current.t_start)
                # 静止画面标注 (与上文 ongoing 分支一致): 视觉静态长时间无变化 → 已知背景
                cur_static = ""
                if current.source == "vision" and dur >= _STATIC_VISION_THRESHOLD_S:
                    cur_static = " [静止画面, 无新变化, 已知背景]"
                lines.append(f"\n[当前] {current.description} (任务: {current.task_context or '未指定'}) "
                             f"从 {cur_start} 起已持续 {_format_duration(dur)}"
                             f"{cur_static}{stale_hint}")

            # 待办任务
            pending = self._find_paused_activities()
            if pending:
                lines.append(f"\n[待办] {len(pending)} 个未完成活动:")
                for p in pending:
                    lines.append(f"  - {p.description} (任务: {p.task_context or '未指定'})")

            return "\n".join(lines)

    def get_activity_summary(self, now: float) -> Dict:
        """活动摘要 (供副脑 M2 时机决策用)

        返回结构化摘要:
            current_activity: 当前活动描述
            current_duration_s: 当前活动持续时间
            is_stale: 是否持续过久
            pending_count: 待办任务数
            interrupt_count_today: 今日中断次数
            activity_switch_count: 今日活动切换次数
        """
        with self._lock:
            current = self._get_current_activity_unlocked(now)
            pending = self._find_paused_activities()
            recent = self._find_recent_activities(now, 86400.0)  # 24h

            # 统计中断次数
            interrupt_count = 0
            for _, _, data in self.G.edges(data=True):
                edge = data.get("data")
                if edge and edge.edge_type == ActivityEdgeType.INTERRUPT:
                    interrupt_count += 1

            return {
                "current_activity": current.description if current else None,
                "current_task_context": current.task_context if current else None,
                "current_duration_s": current.duration_until(now) if current else 0.0,
                "is_stale": self._is_activity_stale_unlocked(current.id, now) if current else False,
                "pending_count": len(pending),
                "pending_descriptions": [p.description for p in pending],
                "interrupt_count_24h": interrupt_count,
                "activity_switch_count_24h": len(recent),
                "confidence": current.confidence if current else 0.0,
            }


# ==================== 工具函数 ====================

def _description_similarity(a: str, b: str) -> float:
    """活动描述相似度 (简单 Jaccard 词重叠, demo 用)

    未来可换句向量 (sentence-transformers).
    "在写代码" vs "在写代码" → 1.0
    "在写代码" vs "在调试bug" → 0.2 (词重叠低)
    "在写代码" vs "在倒水" → 0.0
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # 简单分词 (中文按字, 英文按空格)
    set_a = set(_tokenize(a))
    set_b = set(_tokenize(b))
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def _tokenize(text: str) -> List[str]:
    """简单分词: 中文按字 + 英文按空格"""
    tokens = []
    # 英文部分按空格分
    for part in text.split():
        if part.isascii():
            tokens.append(part.lower())
        else:
            # 中文按字
            tokens.extend(list(part))
    return tokens


def _format_time(ts: float) -> str:
    """格式化时间戳为 HH:MM"""
    import time as _time
    return _time.strftime("%H:%M", _time.localtime(ts))


def _format_duration(s: float) -> str:
    """格式化时长为人类可读"""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.0f}min"
    h = s / 3600
    m = (s % 3600) / 60
    return f"{h:.0f}h{m:.0f}min"
