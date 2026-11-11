"""
内在奖励机制 - 激发模型主动学习和自我更新

基于以下论文实现：
1. Curiosity-driven Exploration (Pathak et al., 2017)
2. Intrinsic Curiosity Module (ICM)
3. Random Network Distillation (RND)

核心功能：
1. 好奇心驱动探索 - 基于预测误差计算内在奖励
2. 新颖性检测 - 检测新状态并给予奖励
3. 学习进度追踪 - 追踪学习效率
4. 自我监督信号 - 生成自我监督信号
5. 知识缺口检测 - 检测需要学习的领域
6. 主动学习触发 - 决定何时进行主动学习
"""

import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("brain.intrinsic_reward")


class StateEncoder:
    """
    状态编码器 - 将复杂状态转换为可比较的表示
    
    功能：
    1. 状态哈希化
    2. 状态特征提取
    3. 状态相似度计算
    """
    
    def encode(self, state: Dict[str, Any]) -> str:
        """
        将状态编码为字符串哈希
        
        Args:
            state: 状态字典
        
        Returns:
            状态哈希字符串
        """
        try:
            state_str = json.dumps(state, sort_keys=True, default=str)
            return hashlib.md5(state_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"State encoding error: {e}")
            return str(hash(str(state)))[:16]
    
    def extract_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        提取状态特征向量
        
        Args:
            state: 状态字典
        
        Returns:
            特征向量
        """
        features = []
        
        if "dialog_status" in state:
            status_map = {"idle": 0, "active": 1, "paused": 2}
            features.append(status_map.get(state["dialog_status"], 0))
        
        if "is_user_idle" in state:
            features.append(1 if state["is_user_idle"] else 0)
        
        if "idle_time" in state:
            features.append(min(state["idle_time"] / 3600, 1.0))
        
        if "recent_messages_count" in state:
            features.append(min(state["recent_messages_count"] / 100, 1.0))
        
        if "memory_count" in state:
            features.append(min(state["memory_count"] / 1000, 1.0))
        
        if "pending_tasks" in state:
            features.append(min(len(state["pending_tasks"]) / 10, 1.0))
        
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def compute_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """
        计算两个状态的相似度
        
        Args:
            state1: 状态1
            state2: 状态2
        
        Returns:
            相似度 [0, 1]
        """
        features1 = self.extract_features(state1)
        features2 = self.extract_features(state2)
        
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(np.clip(similarity, 0, 1))


class IntrinsicRewardModule:
    """
    内在奖励模块
    
    功能：
    1. 好奇心驱动探索 - 基于预测误差
    2. 新颖性检测 - 基于访问计数
    3. 学习进度追踪 - 基于损失改进
    4. 探索策略建议 - 决定何时探索新领域
    
    整合到大脑决策：
    - 在 decider.py 的 think_and_act() 中调用
    - 影响 LLM 的决策权重
    """
    
    _instance: Optional['IntrinsicRewardModule'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._encoder = StateEncoder()
        
        self._visited_states: Dict[str, int] = {}
        self._state_history: deque = deque(maxlen=1000)
        self._action_history: deque = deque(maxlen=500)
        self._prediction_errors: deque = deque(maxlen=100)
        self._learning_progress: deque = deque(maxlen=100)
        self._reward_history: deque = deque(maxlen=200)
        
        self._novelty_threshold = 0.3
        self._exploration_rate = 0.2
        self._curiosity_weight = 0.5
        
        self._total_interactions = 0
        self._exploration_count = 0
        
        logger.info("IntrinsicRewardModule initialized")
    
    def compute_curiosity_reward(
        self,
        state: Dict[str, Any],
        prediction_error: float = 0.5
    ) -> float:
        """
        计算好奇心奖励
        
        基于预测误差计算内在奖励：
        - 预测误差大 → 状态新颖 → 高奖励
        - 预测误差小 → 状态熟悉 → 低奖励
        
        公式: r_curiosity = η * prediction_error * (1 + novelty_bonus)
        
        Args:
            state: 当前状态
            prediction_error: 预测误差 [0, 1]
        
        Returns:
            内在奖励值 [0, 1]
        """
        state_key = self._encoder.encode(state)
        
        novelty_bonus = self._compute_novelty_bonus(state_key)
        
        curiosity_reward = self._curiosity_weight * prediction_error * (1 + novelty_bonus)
        
        curiosity_reward = float(np.clip(curiosity_reward, 0, 1))
        
        self._update_state_count(state_key)
        self._prediction_errors.append(prediction_error)
        self._state_history.append({
            "state_key": state_key,
            "timestamp": datetime.now().isoformat(),
            "prediction_error": prediction_error
        })
        
        logger.debug(
            f"Curiosity reward: {curiosity_reward:.3f} "
            f"(error={prediction_error:.3f}, novelty={novelty_bonus:.3f})"
        )
        
        return curiosity_reward
    
    def compute_exploration_bonus(
        self,
        state: Dict[str, Any],
        action: str
    ) -> float:
        """
        计算探索奖励
        
        鼓励探索新的状态-动作对：
        - 访问次数少 → 高奖励
        - 访问次数多 → 低奖励
        
        公式: r_explore = 1 / (1 + count(s, a))
        
        Args:
            state: 当前状态
            action: 执行的动作
        
        Returns:
            探索奖励 [0, 1]
        """
        state_key = self._encoder.encode(state)
        state_action_key = f"{state_key}_{action}"
        
        visit_count = self._visited_states.get(state_action_key, 0)
        
        exploration_bonus = 1.0 / (1.0 + visit_count)
        
        self._visited_states[state_action_key] = visit_count + 1
        self._action_history.append({
            "state_key": state_key,
            "action": action,
            "bonus": exploration_bonus,
            "timestamp": datetime.now().isoformat()
        })
        
        self._exploration_count += 1
        
        logger.debug(f"Exploration bonus: {exploration_bonus:.3f} for action '{action}'")
        
        return float(exploration_bonus)
    
    def compute_learning_progress_reward(
        self,
        current_loss: float,
        previous_loss: float
    ) -> float:
        """
        计算学习进度奖励
        
        基于损失函数的改进计算奖励：
        - 损失下降 → 学习进步 → 正奖励
        - 损失上升 → 学习退步 → 负奖励
        
        公式: r_progress = (prev_loss - curr_loss) / prev_loss
        
        Args:
            current_loss: 当前损失
            previous_loss: 之前损失
        
        Returns:
            学习进度奖励 [-1, 1]
        """
        if previous_loss == 0:
            return 0.0
        
        progress = (previous_loss - current_loss) / previous_loss
        
        progress = float(np.clip(progress, -1, 1))
        
        self._learning_progress.append(progress)
        
        logger.debug(f"Learning progress reward: {progress:.3f}")
        
        return progress
    
    def should_explore(self, state: Dict[str, Any]) -> Tuple[bool, float]:
        """
        判断是否应该探索
        
        基于新颖性和探索率决定是否应该探索新领域
        
        Args:
            state: 当前状态
        
        Returns:
            (是否应该探索, 探索动机强度)
        """
        state_key = self._encoder.encode(state)
        novelty = self._compute_novelty_bonus(state_key)
        
        recent_exploration_rate = self._get_recent_exploration_rate()
        
        exploration_motivation = novelty * (1 - recent_exploration_rate)
        
        should = novelty > self._novelty_threshold or exploration_motivation > 0.5
        
        logger.debug(
            f"Should explore: {should} "
            f"(novelty={novelty:.3f}, motivation={exploration_motivation:.3f})"
        )
        
        return should, exploration_motivation
    
    def get_intrinsic_motivation_summary(self) -> Dict[str, Any]:
        """
        获取内在动机总结
        
        Returns:
            内在动机状态摘要
        """
        avg_prediction_error = (
            float(np.mean(list(self._prediction_errors)))
            if self._prediction_errors else 0.0
        )
        avg_learning_progress = (
            float(np.mean(list(self._learning_progress)))
            if self._learning_progress else 0.0
        )
        avg_reward = (
            float(np.mean(list(self._reward_history)))
            if self._reward_history else 0.0
        )
        
        return {
            "total_interactions": self._total_interactions,
            "exploration_count": self._exploration_count,
            "unique_states_visited": len(self._visited_states),
            "average_prediction_error": round(avg_prediction_error, 4),
            "average_learning_progress": round(avg_learning_progress, 4),
            "average_intrinsic_reward": round(avg_reward, 4),
            "exploration_rate": round(
                self._exploration_count / max(self._total_interactions, 1), 4
            ),
            "recent_exploration_rate": round(self._get_recent_exploration_rate(), 4)
        }
    
    def get_exploration_suggestion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取探索建议
        
        基于当前状态和历史，建议下一步探索方向
        
        Args:
            state: 当前状态
        
        Returns:
            探索建议
        """
        should, motivation = self.should_explore(state)
        
        recent_actions = [a["action"] for a in list(self._action_history)[-20:]]
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        least_explored = None
        if action_counts:
            least_explored = min(action_counts, key=action_counts.get)
        
        suggestion = {
            "should_explore": should,
            "motivation": round(motivation, 3),
            "novelty_score": round(self._compute_novelty_bonus(
                self._encoder.encode(state)
            ), 3),
            "suggested_action": least_explored,
            "reason": self._generate_exploration_reason(should, motivation)
        }
        
        return suggestion
    
    def record_interaction(
        self,
        state: Dict[str, Any],
        action: str,
        result: Dict[str, Any],
        external_reward: float = 0.0
    ) -> float:
        """
        记录一次交互并计算总奖励
        
        Args:
            state: 状态
            action: 动作
            result: 结果
            external_reward: 外部奖励
        
        Returns:
            总奖励
        """
        self._total_interactions += 1
        
        # ★ 动态 prediction_error：基于 state 的新颖度
        # 新颖度高 → 预测误差大 → 好奇心强
        # 新颖度低 → 预测误差小 → 已熟悉 → 好奇心低
        state_key = self._encoder.encode(state)
        novelty = self._compute_novelty_bonus(state_key)
        prediction_error = 0.3 + 0.7 * novelty  # 范围 [0.3, 1.0]
        
        curiosity = self.compute_curiosity_reward(state, prediction_error=prediction_error)
        exploration = self.compute_exploration_bonus(state, action)
        
        total_reward = 0.4 * curiosity + 0.3 * exploration + 0.3 * external_reward
        
        self._reward_history.append(total_reward)
        
        logger.info(
            f"Interaction recorded: action='{action}', "
            f"curiosity={curiosity:.3f}, exploration={exploration:.3f}, "
            f"total={total_reward:.3f}"
        )
        
        return total_reward
    
    def _compute_novelty_bonus(self, state_key: str) -> float:
        """计算新颖性奖励"""
        visit_count = self._visited_states.get(state_key, 0)
        novelty = 1.0 / (1.0 + visit_count)
        return float(novelty)
    
    def _update_state_count(self, state_key: str) -> None:
        """更新状态计数"""
        self._visited_states[state_key] = self._visited_states.get(state_key, 0) + 1
    
    def _get_recent_exploration_rate(self) -> float:
        """获取最近的探索率"""
        if len(self._action_history) < 10:
            return 0.0
        
        recent_actions = list(self._action_history)[-50:]
        unique_actions = len(set(a["action"] for a in recent_actions))
        
        return unique_actions / max(len(recent_actions), 1)
    
    def _generate_exploration_reason(self, should: bool, motivation: float) -> str:
        """生成探索原因说明"""
        if not should:
            return "当前状态足够熟悉，无需探索"
        
        if motivation > 0.7:
            return "发现高度新颖的状态，强烈建议探索"
        elif motivation > 0.5:
            return "检测到新颖状态，建议进行探索"
        else:
            return "存在一定的探索空间"


class SelfUpdateModule:
    """
    自我更新模块
    
    功能：
    1. 检测知识缺口 - 识别需要学习的领域
    2. 主动学习触发 - 决定何时进行主动学习
    3. 模型自我改进 - 执行自我改进流程
    
    整合到大脑决策：
    - 在 decider.py 中检测知识缺口
    - 触发主动学习任务
    - 更新大脑状态
    """
    
    _instance: Optional['SelfUpdateModule'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        self._knowledge_gaps: List[Dict[str, Any]] = []
        self._update_history: List[Dict[str, Any]] = []
        self._confidence_history: deque = deque(maxlen=100)
        
        self._confidence_threshold = 0.6
        self._gap_threshold = 3
        self._update_interval = 3600
        
        self._last_update_time: Optional[datetime] = None
        self._total_updates = 0
        
        logger.info("SelfUpdateModule initialized")
    
    def detect_knowledge_gap(
        self,
        query: str,
        confidence: float,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        检测知识缺口
        
        当置信度低于阈值时，记录知识缺口（自动去重）
        
        Args:
            query: 查询内容
            confidence: 置信度 [0, 1]
            context: 上下文
        
        Returns:
            知识缺口信息（如果存在且非重复）
        """
        self._confidence_history.append(confidence)
        
        # 跳过无意义的固定循环动作，避免误报
        # 知识缺口应由 LLM 动机分析提炼出精确 topic，不应从循环动作名生成
        skip_actions = {"llm_analysis", "idle_check", "periodic_analysis", "user_dialog", "对话"}
        if query in skip_actions or query.startswith("idle") or len(query) < 4:
            return None
        
        if confidence >= self._confidence_threshold:
            return None
        
        # 去重：基于 query 内容做去重，避免同一问题重复生成缺口
        query_key = hashlib.md5(query.encode()).hexdigest()[:8]
        for existing_gap in self._knowledge_gaps:
            if (not existing_gap["resolved"] 
                and hashlib.md5(existing_gap["query"].encode()).hexdigest()[:8] == query_key):
                logger.debug(f"Knowledge gap already exists for: '{query[:30]}...'")
                return None
        
        gap = {
            "id": query_key,
            "query": query,
            "confidence": confidence,
            "context": context,
            "detected_at": datetime.now().isoformat(),
            "resolved": False,
            "priority": self._calculate_gap_priority(confidence)
        }
        
        self._knowledge_gaps.append(gap)
        
        # 限制未解决缺口数量，避免无限累积
        unresolved = [g for g in self._knowledge_gaps if not g["resolved"]]
        if len(unresolved) > 20:
            # 保留优先级最高的 20 个
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            unresolved.sort(key=lambda x: priority_order.get(x["priority"], 4))
            to_remove_ids = {g["id"] for g in unresolved[20:]}
            self._knowledge_gaps = [
                g for g in self._knowledge_gaps 
                if g["resolved"] or g["id"] not in to_remove_ids
            ]
        
        logger.info(
            f"Knowledge gap detected: '{query[:30]}...' "
            f"(confidence={confidence:.2f}, priority={gap['priority']})"
        )
        
        return gap
    
    def should_self_update(self) -> Tuple[bool, str]:
        """
        判断是否应该自我更新
        
        基于以下条件判断：
        1. 未解决的知识缺口数量
        2. 距离上次更新的时间
        3. 平均置信度趋势
        
        Returns:
            (是否应该更新, 原因)
        """
        unresolved_gaps = [g for g in self._knowledge_gaps if not g["resolved"]]
        
        if len(unresolved_gaps) >= self._gap_threshold:
            return True, f"存在{len(unresolved_gaps)}个未解决的知识缺口"
        
        if self._last_update_time:
            elapsed = (datetime.now() - self._last_update_time).total_seconds()
            if elapsed < self._update_interval:
                return False, f"距离上次更新仅{elapsed:.0f}秒"
        
        if len(self._confidence_history) >= 20:
            recent_avg = np.mean(list(self._confidence_history)[-20:])
            older_avg = np.mean(list(self._confidence_history)[:-20]) if len(self._confidence_history) > 20 else recent_avg
            
            if recent_avg < older_avg * 0.9:
                return True, f"置信度趋势下降 ({recent_avg:.2f} < {older_avg:.2f})"
        
        return False, "暂无自我更新需求"
    
    async def trigger_self_update(self) -> Dict[str, Any]:
        """
        触发自我更新
        
        执行自我更新流程：
        1. 分析知识缺口
        2. 生成学习任务
        3. 更新知识库
        
        Returns:
            更新结果
        """
        logger.info("Starting self-update process...")
        
        unresolved_gaps = [g for g in self._knowledge_gaps if not g["resolved"]]
        
        update_result = {
            "started_at": datetime.now().isoformat(),
            "gaps_addressed": len(unresolved_gaps),
            "actions": [],
            "status": "completed"
        }
        
        for gap in unresolved_gaps:
            action = await self._address_gap(gap)
            update_result["actions"].append(action)
            gap["resolved"] = True
            gap["resolved_at"] = datetime.now().isoformat()
        
        self._last_update_time = datetime.now()
        self._total_updates += 1
        
        self._update_history.append(update_result)
        
        logger.info(
            f"Self-update completed: {len(unresolved_gaps)} gaps addressed, "
            f"total updates: {self._total_updates}"
        )
        
        return update_result
    
    async def _address_gap(self, gap: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理知识缺口
        
        Args:
            gap: 知识缺口
        
        Returns:
            处理动作
        """
        action = {
            "gap_id": gap["id"],
            "query": gap["query"],
            "action_type": "knowledge_acquisition",
            "status": "addressed",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            from .llm import get_llm
            llm = get_llm()
            
            prompt = f"""我需要学习关于以下主题的知识：

查询：{gap['query']}
上下文：{gap.get('context', {})}

请生成：
1. 这个主题的核心知识点（3-5个）
2. 学习这个主题的建议方法
3. 相关的扩展学习方向

以JSON格式返回。"""
            
            learning_content = await llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True
            )
            
            action["learning_content"] = learning_content
            
            try:
                from .memory import get_memory_manager
                memory_manager = get_memory_manager()
                await memory_manager.save_to_viking([{
                    "text": f"学习笔记：{gap['query']}",
                    "content": learning_content,
                    "role": "assistant",  # 大脑自省，不是用户说的
                    "type": "self_learning",
                    "priority": gap["priority"]
                }])
                action["saved_to_memory"] = True
            except Exception as e:
                logger.warning(f"Failed to save learning to memory: {e}")
                action["saved_to_memory"] = False
            
        except Exception as e:
            logger.error(f"Failed to address gap: {e}")
            action["status"] = "failed"
            action["error"] = str(e)
        
        return action
    
    def _calculate_gap_priority(self, confidence: float) -> str:
        """计算知识缺口优先级"""
        if confidence < 0.3:
            return "critical"
        elif confidence < 0.5:
            return "high"
        elif confidence < 0.6:
            return "medium"
        else:
            return "low"
    
    def get_update_summary(self) -> Dict[str, Any]:
        """
        获取更新总结
        
        Returns:
            更新状态摘要
        """
        unresolved_gaps = [g for g in self._knowledge_gaps if not g["resolved"]]
        resolved_gaps = [g for g in self._knowledge_gaps if g["resolved"]]
        
        avg_confidence = (
            float(np.mean(list(self._confidence_history)))
            if self._confidence_history else 0.0
        )
        
        return {
            "total_gaps_detected": len(self._knowledge_gaps),
            "unresolved_gaps": len(unresolved_gaps),
            "resolved_gaps": len(resolved_gaps),
            "total_updates": self._total_updates,
            "average_confidence": round(avg_confidence, 4),
            "last_update": (
                self._last_update_time.isoformat()
                if self._last_update_time else None
            ),
            "recent_gaps": [
                {
                    "query": g["query"][:50],
                    "priority": g["priority"],
                    "resolved": g["resolved"]
                }
                for g in self._knowledge_gaps[-5:]
            ]
        }
    
    def get_learning_tasks(self) -> List[Dict[str, Any]]:
        """
        获取学习任务列表
        
        Returns:
            待学习任务
        """
        unresolved_gaps = [g for g in self._knowledge_gaps if not g["resolved"]]
        
        tasks = []
        for gap in sorted(unresolved_gaps, key=lambda x: {
            "critical": 0, "high": 1, "medium": 2, "low": 3
        }.get(x["priority"], 4)):
            tasks.append({
                "task_id": gap["id"],
                "type": "knowledge_acquisition",
                "description": f"学习：{gap['query']}",
                "priority": gap["priority"],
                "detected_at": gap["detected_at"]
            })
        
        return tasks


class IntrinsicMotivationSystem:
    """
    内在动机系统 - 整合内在奖励和自我更新
    
    这是大脑的"自我驱动"系统，让AI能够：
    1. 自主探索新领域
    2. 检测并填补知识缺口
    3. 持续自我改进
    """
    
    _instance: Optional['IntrinsicMotivationSystem'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._reward_module = IntrinsicRewardModule()
        self._update_module = SelfUpdateModule()
        
        logger.info("IntrinsicMotivationSystem initialized")
    
    async def evaluate_and_act(
        self,
        state: Dict[str, Any],
        action: str,
        result: Dict[str, Any],
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        评估并执行内在动机驱动的行为
        
        Args:
            state: 当前状态
            action: 执行的动作
            result: 动作结果
            confidence: 结果置信度
        
        Returns:
            内在动机评估结果
        """
        total_reward = self._reward_module.record_interaction(
            state, action, result
        )
        
        gap = self._update_module.detect_knowledge_gap(
            action, confidence, state
        )
        
        should_update, reason = self._update_module.should_self_update()
        
        update_result = None
        if should_update:
            update_result = await self._update_module.trigger_self_update()
        
        return {
            "intrinsic_reward": total_reward,
            "knowledge_gap": gap,
            "self_update_triggered": should_update,
            "self_update_reason": reason,
            "update_result": update_result,
            "motivation_summary": self._reward_module.get_intrinsic_motivation_summary()
        }
    
    def get_exploration_guidance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取探索指导
        
        Args:
            state: 当前状态
        
        Returns:
            探索指导信息
        """
        suggestion = self._reward_module.get_exploration_suggestion(state)
        learning_tasks = self._update_module.get_learning_tasks()
        
        return {
            "exploration_suggestion": suggestion,
            "learning_tasks": learning_tasks,
            "should_explore": suggestion["should_explore"],
            "motivation": suggestion["motivation"]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态摘要
        """
        return {
            "reward_module": self._reward_module.get_intrinsic_motivation_summary(),
            "update_module": self._update_module.get_update_summary(),
            "system_health": self._evaluate_system_health()
        }
    
    def _evaluate_system_health(self) -> Dict[str, Any]:
        """评估系统健康度"""
        reward_summary = self._reward_module.get_intrinsic_motivation_summary()
        update_summary = self._update_module.get_update_summary()
        
        exploration_health = "good" if reward_summary["exploration_rate"] > 0.1 else "low"
        knowledge_health = "good" if update_summary["unresolved_gaps"] < 5 else "needs_attention"
        confidence_health = "good" if update_summary["average_confidence"] > 0.6 else "low"
        
        overall = "healthy"
        if exploration_health != "good" or knowledge_health != "good":
            overall = "needs_attention"
        if confidence_health == "low":
            overall = "needs_improvement"
        
        return {
            "overall": overall,
            "exploration": exploration_health,
            "knowledge": knowledge_health,
            "confidence": confidence_health
        }


def get_intrinsic_reward() -> IntrinsicRewardModule:
    """获取内在奖励模块单例"""
    return IntrinsicRewardModule()


def get_self_update() -> SelfUpdateModule:
    """获取自我更新模块单例"""
    return SelfUpdateModule()


def get_intrinsic_motivation_system() -> IntrinsicMotivationSystem:
    """获取内在动机系统单例"""
    return IntrinsicMotivationSystem()
