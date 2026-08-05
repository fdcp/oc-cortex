"""
Phase 6b: 技术决策溯源 + 主题总结骨架

在 Phase 6 基础上增加两项能力：

1. 技术决策溯源 (DecisionTracer)
   - 从 SQLite 知识图谱中按关系类别追踪决策链
   - 支持多跳追溯（BFS + visited 防环）
   - 可回答"为什么选 Qdrant""A 和 B 的选型对比"等因果关系

2. 主题总结骨架 (SummarySkeleton)
   - 从图谱中定位主题节点 → BFS 扩散获取相关任务
   - 按实体分组构建结构化大纲
   - 注入决策链上下文 → LLM 生成带因果关系的结构化总结

决策关系分类（宽泛匹配）:
  - 选用类: 使用, 集成了, 实现, 实现了, 支持, 通过兼容层加载
  - 选型类: 对比, 对比了, 区别于, 适用于, 适合
  - 排除类: 禁用, 缺少, 阻塞于
  - 替换类: 替代了, 阉割了, 新增了, 补充了
  - 依赖类: 属于, 包含, 对应

示例:
  from code_p6b_skeleton import DecisionTracer, SummarySkeleton
  from code_p5e_db import KGDatabase

  # 决策溯源
  db = KGDatabase("output/triple/knowledge_graph.db")
  tracer = DecisionTracer(db)
  chain = tracer.trace_decision("Qdrant")
  print(tracer.format_context(chain))

  # 骨架总结
  from code_p6_summarizer import SessionSummarizer
  summarizer = SessionSummarizer("code_p3_config.yaml")
  skeleton = SummarySkeleton(db, summarizer, tracer)
  result = skeleton.generate_summary("FlashAttention 的实现和优化")
  print(result.summary)
"""
import json
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from code_update_prompt_utils import load_prompt

from code_p1_utils import count_tokens, safe_truncate
from code_p6_summarizer import SessionSummarizer, _get_client, DEFAULT_MODEL


# ============================================================
# 决策关系分类（宽泛匹配，覆盖 91 种关系中的决策相关信息）
# ============================================================

DECISION_CATEGORIES = {
    "选用": {"使用", "集成了", "实现", "实现了", "支持", "通过兼容层加载",
             "保留fallback模型", "保留"},
    "选型": {"对比", "对比了", "区别于", "适用于", "适合", "支持精度"},
    "排除": {"禁用", "缺少", "阻塞于", "阉割了"},
    "替换": {"替代了", "新增了", "补充了"},
    "依赖": {"属于", "包含", "对应"},
}

# 反向映射: relation → category
_RELATION_TO_CATEGORY = {}
for _cat, _rels in DECISION_CATEGORIES.items():
    for _rel in _rels:
        _RELATION_TO_CATEGORY[_rel] = _cat


# ============================================================
# 数据模型
# ============================================================

@dataclass
class DecisionStep:
    """决策链中的单步"""
    entity: str              # 当前实体
    related_entity: str      # 关联实体
    relation: str            # 关系标签
    category: str            # 决策类别: 选用/选型/排除/替换/依赖
    hop: int                 # 跳数（从起点算起）
    direction: str           # "out" (head→tail) 或 "in" (tail→head)
    source_task: str         # 来源 task_id
    weight: float            # 边权重


@dataclass
class DecisionChain:
    """完整决策链"""
    root_entity: str
    steps: list[DecisionStep]
    visited_entities: list[str]  # BFS 访问到的所有实体（含非决策边）

    @property
    def hop_count(self) -> int:
        """最大跳数"""
        return max((s.hop for s in self.steps), default=0)


@dataclass
class SkeletonSection:
    """结构化总结的单个章节"""
    entity: str
    entity_type: str
    tasks: list[dict]          # [{task_id, task_label, task_summary, created_at}]
    decision_steps: list[DecisionStep]  # 该实体相关的决策步骤


@dataclass
class SkeletonResult:
    """骨架总结结果"""
    topic: str
    summary: str                # LLM 生成的结构化总结
    skeleton: str               # 结构化大纲文本
    decision_chain: DecisionChain
    task_summaries: list[dict]  # 参与总结的 task 列表
    elapsed_ms: int
    debug: dict = field(default_factory=dict)


# ============================================================
# 决策溯源器
# ============================================================

class DecisionTracer:
    """技术决策溯源：从知识图谱中追踪实体间的决策关系链"""

    def __init__(
        self,
        kg_db,
        decision_categories: Optional[dict] = None,
        max_hops: int = 3,
        max_visited: int = 50,
    ):
        """
        Args:
            kg_db: KGDatabase 实例
            decision_categories: 自定义决策关系分类（默认使用全局 DECISION_CATEGORIES）
            max_hops: 最大追溯跳数
            max_visited: BFS 最大访问节点数（防爆炸）
        """
        self.db = kg_db
        self.categories = decision_categories or DECISION_CATEGORIES
        self.max_hops = max_hops
        self.max_visited = max_visited

        # 构建反向映射
        self._relation_to_cat = {}
        for cat, rels in self.categories.items():
            for rel in rels:
                self._relation_to_cat[rel] = cat

    def trace_decision(self, entity: str, max_hops: Optional[int] = None) -> DecisionChain:
        """从指定实体出发，多跳追溯决策链。

        使用 BFS 遍历，只保留匹配决策类别的边，同时记录所有访问到的实体。

        Args:
            entity: 起始实体名
            max_hops: 覆盖默认最大跳数

        Returns:
            DecisionChain 包含决策步骤和访问到的实体
        """
        hops = max_hops if max_hops is not None else self.max_hops

        # 解析实体（精确 → 模糊）
        resolved = self._resolve_entity(entity)
        if not resolved:
            logger.warning(f"未找到实体: {entity}")
            return DecisionChain(root_entity=entity, steps=[], visited_entities=[])

        root = resolved
        visited = {root}
        queue = deque([(root, 0)])
        steps = []

        while queue and len(visited) < self.max_visited:
            current, hop = queue.popleft()
            if hop >= hops:
                continue

            # 查询出边和入边
            edges = self.db.get_edges(current, direction="both")

            for edge in edges:
                is_out = edge["head"] == current
                neighbor = edge["tail"] if is_out else edge["head"]
                relation = edge["relation"]

                # 判断是否为决策关系
                category = self._relation_to_cat.get(relation)
                if category:
                    steps.append(DecisionStep(
                        entity=current,
                        related_entity=neighbor,
                        relation=relation,
                        category=category,
                        hop=hop + 1,
                        direction="out" if is_out else "in",
                        source_task=edge["source_task"],
                        weight=edge["weight"],
                    ))

                # BFS 继续扩展（无论是否决策边）
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))

        # 去重：同 (entity, related_entity, relation, direction) 只保留权重最高的一条
        seen = {}
        for step in steps:
            key = (step.entity, step.related_entity, step.relation, step.direction)
            if key not in seen or step.weight > seen[key].weight:
                seen[key] = step
        steps = sorted(seen.values(), key=lambda s: (s.hop, s.category))

        logger.info(
            f"决策溯源 [{root}]: {len(steps)} 个决策步骤, "
            f"{len(visited)} 个访问实体, {max((s.hop for s in steps), default=0)} 跳"
        )
        return DecisionChain(
            root_entity=root,
            steps=steps,
            visited_entities=list(visited),
        )

    def get_related_tasks(self, chain: DecisionChain) -> list[str]:
        """从决策链中收集所有关联的 task_id（去重）"""
        task_ids = set()
        for step in chain.steps:
            if step.source_task:
                task_ids.add(step.source_task)

        # 也从访问到的实体中收集
        for entity_name in chain.visited_entities:
            node = self.db.get_node(entity_name)
            if node:
                for tid in node["source_tasks"]:
                    task_ids.add(tid)

        return sorted(task_ids)

    def format_context(
        self,
        chain: DecisionChain,
        max_steps: int = 20,
        group_by_category: bool = True,
    ) -> str:
        """将决策链格式化为可注入 Prompt 的上下文文本。

        Args:
            chain: 决策链
            max_steps: 最多显示的步骤数
            group_by_category: 是否按类别分组（默认 True）
        """
        if not chain.steps:
            return ""

        lines = [f"## 决策链 [{chain.root_entity}] ({len(chain.steps)} 步, {chain.hop_count} 跳)\n"]

        if group_by_category:
            grouped: dict[str, list[DecisionStep]] = {}
            for step in chain.steps[:max_steps]:
                grouped.setdefault(step.category, []).append(step)

            for category, cat_steps in grouped.items():
                lines.append(f"### {category}类")
                for step in cat_steps:
                    arrow = "→" if step.direction == "out" else "←"
                    task_info = f" [{step.source_task[:20]}]" if step.source_task else ""
                    lines.append(
                        f"  - {step.entity} {arrow} {step.related_entity} "
                        f"({step.relation}, {step.hop}跳){task_info}"
                    )
                lines.append("")
        else:
            for i, step in enumerate(chain.steps[:max_steps], 1):
                arrow = "→" if step.direction == "out" else "←"
                task_info = f" [{step.source_task[:20]}]" if step.source_task else ""
                lines.append(
                    f"{i}. [{step.category}] {step.entity} {arrow} {step.related_entity} "
                    f"({step.relation}, {step.hop}跳){task_info}"
                )

        return "\n".join(lines)

    def _resolve_entity(self, entity: str) -> Optional[str]:
        """解析实体名称：精确匹配 → 模糊匹配"""
        node = self.db.get_node(entity)
        if node:
            return entity

        results = self.db.search_entities(entity, limit=3)
        if results:
            return results[0]["name"]

        return None


# ============================================================
# 主题总结骨架
# ============================================================

class SummarySkeleton:
    """基于图谱的主题总结骨架构建器

    流程: 定位主题节点 → BFS 扩散 → 收集关联 Task → 构建结构化大纲
         → 注入决策链上下文 → LLM 生成结构化总结
    """

    def __init__(
        self,
        kg_db,
        summarizer: SessionSummarizer,
        tracer: Optional[DecisionTracer] = None,
        max_bfs_depth: int = 2,
        max_bfs_nodes: int = 30,
    ):
        """
        Args:
            kg_db: KGDatabase 实例
            summarizer: Phase 6 SessionSummarizer 实例
            tracer: DecisionTracer 实例（可选，不传则自动创建）
            max_bfs_depth: BFS 扩散深度
            max_bfs_nodes: BFS 最大节点数
        """
        self.db = kg_db
        self.summarizer = summarizer
        self.tracer = tracer or DecisionTracer(kg_db)
        self.max_bfs_depth = max_bfs_depth
        self.max_bfs_nodes = max_bfs_nodes

    def generate_summary(
        self,
        topic: str,
        use_decision_trace: bool = True,
        max_tasks: int = 10,
    ) -> SkeletonResult:
        """生成带图谱骨架的主题总结。

        Args:
            topic: 主题关键词（如 "FlashAttention"、"序列并行"）
            use_decision_trace: 是否注入决策链上下文
            max_tasks: 最多纳入的 Task 数

        Returns:
            SkeletonResult 包含结构化总结和决策链
        """
        t0 = time.time()
        debug = {}

        # ---- Stage 1: 定位主题实体 ----
        topic_entity = self._resolve_topic(topic)
        debug["topic_entity"] = topic_entity

        if not topic_entity:
            # 降级：直接用 Phase 6 总结器
            logger.warning(f"图谱中未找到主题 [{topic}]，降级到纯向量总结")
            p6_result = self.summarizer.summarize(topic, top_k=max_tasks)
            empty_chain = DecisionChain(root_entity=topic, steps=[], visited_entities=[])
            return SkeletonResult(
                topic=topic,
                summary=p6_result.summary,
                skeleton="(图谱中未找到相关实体，使用纯向量检索)",
                decision_chain=empty_chain,
                task_summaries=[
                    {"task_id": s.task_id, "task_label": s.task_label,
                     "task_summary": s.task_summary, "created_at": s.created_at}
                    for s in p6_result.sources
                ],
                elapsed_ms=p6_result.elapsed_ms,
                debug={"fallback": "vector_only", **p6_result.debug},
            )

        # ---- Stage 2: BFS 扩散获取相关实体 ----
        related_entities = self.db.bfs_expand(
            [topic_entity], depth=self.max_bfs_depth, max_nodes=self.max_bfs_nodes,
        )
        debug["related_entities"] = len(related_entities)

        # ---- Stage 3: 收集关联 Task 并按时间排序 ----
        task_map: dict[str, dict] = {}
        entity_task_map: dict[str, list[str]] = {}

        for entity_name in related_entities:
            node = self.db.get_node(entity_name)
            if not node:
                continue
            entity_task_ids = node["source_tasks"]
            entity_task_map[entity_name] = entity_task_ids

            for tid in entity_task_ids:
                if tid not in task_map:
                    task_obj = self.summarizer.searcher.task_map.get(tid)
                    if task_obj:
                        task_map[tid] = {
                            "task_id": tid,
                            "task_label": task_obj.task_label,
                            "task_summary": task_obj.task_summary,
                            "created_at": task_obj.created_at or "",
                        }

        # 按时间排序
        sorted_tasks = sorted(task_map.values(), key=lambda t: t["created_at"])
        sorted_tasks = sorted_tasks[:max_tasks]
        debug["total_tasks"] = len(sorted_tasks)

        # ---- Stage 4: 决策链 ----
        chain = DecisionChain(root_entity=topic_entity, steps=[], visited_entities=list(related_entities))
        if use_decision_trace:
            chain = self.tracer.trace_decision(topic_entity)
            debug["decision_steps"] = len(chain.steps)

        # ---- Stage 5: 构建结构化大纲 ----
        sections = self._build_sections(
            topic_entity, related_entities, entity_task_map, task_map, chain,
        )
        skeleton_text = self._format_skeleton(sections)
        debug["sections"] = len(sections)

        # ---- Stage 6: 构建 Prompt + LLM 生成 ----
        decision_context = ""
        if chain.steps:
            decision_context = f"\n【决策链】\n{self.tracer.format_context(chain)}"

        task_summaries_text = "\n\n".join(
            f"- [{t['task_label']}] (task_id: {t['task_id']}, {t['created_at'][:10]}): {t['task_summary']}"
            for t in sorted_tasks
        )

        user_prompt = load_prompt("SKELETON_USER_PROMPT").format(
            topic=topic,
            skeleton=skeleton_text,
            decision_context=decision_context,
            task_summaries=task_summaries_text if task_summaries_text else "(无关联任务记录)",
        )

        summary_text = self._call_llm(user_prompt)
        elapsed = int((time.time() - t0) * 1000)
        debug["total_time_ms"] = elapsed
        debug["input_tokens"] = count_tokens(user_prompt)
        debug["output_tokens"] = count_tokens(summary_text)

        return SkeletonResult(
            topic=topic,
            summary=summary_text,
            skeleton=skeleton_text,
            decision_chain=chain,
            task_summaries=sorted_tasks,
            elapsed_ms=elapsed,
            debug=debug,
        )

    # ---- 内部方法 ----

    def _resolve_topic(self, topic: str) -> Optional[str]:
        """解析主题关键词到图谱实体"""
        node = self.db.get_node(topic)
        if node:
            return topic

        results = self.db.search_entities(topic, limit=5)
        if results:
            return results[0]["name"]

        # 尝试分词搜索
        words = topic.split()
        for word in words:
            results = self.db.search_entities(word, limit=3)
            if results:
                return results[0]["name"]

        return None

    def _build_sections(
        self,
        topic_entity: str,
        related_entities: set[str],
        entity_task_map: dict[str, list[str]],
        task_map: dict[str, dict],
        chain: DecisionChain,
    ) -> list[SkeletonSection]:
        """为每个有 task 的实体构建章节"""
        # 收集决策步骤按实体分组
        entity_decisions: dict[str, list[DecisionStep]] = {}
        for step in chain.steps:
            entity_decisions.setdefault(step.entity, []).append(step)
            entity_decisions.setdefault(step.related_entity, []).append(step)

        sections = []
        task_ids_used = set()

        # 主题实体优先
        if topic_entity in entity_task_map:
            tasks = [
                task_map[tid]
                for tid in entity_task_map[topic_entity]
                if tid in task_map
            ]
            if tasks:
                for t in tasks:
                    task_ids_used.add(t["task_id"])
                sections.append(SkeletonSection(
                    entity=topic_entity,
                    entity_type=self.db.get_node(topic_entity).get("entity_type", ""),
                    tasks=tasks,
                    decision_steps=entity_decisions.get(topic_entity, []),
                ))

        # 其他实体
        for entity_name in sorted(related_entities - {topic_entity}):
            if entity_name not in entity_task_map:
                continue
            tasks = [
                task_map[tid]
                for tid in entity_task_map[entity_name]
                if tid in task_map and tid not in task_ids_used
            ]
            if tasks:
                for t in tasks:
                    task_ids_used.add(t["task_id"])
                node = self.db.get_node(entity_name)
                sections.append(SkeletonSection(
                    entity=entity_name,
                    entity_type=node.get("entity_type", "") if node else "",
                    tasks=tasks,
                    decision_steps=entity_decisions.get(entity_name, []),
                ))

        return sections

    def _format_skeleton(self, sections: list[SkeletonSection]) -> str:
        """将章节格式化为结构化文本"""
        if not sections:
            return "(无相关结构化信息)"

        parts = []
        for section in sections:
            header = f"### {section.entity}"
            if section.entity_type:
                header += f" ({section.entity_type})"

            decision_hint = ""
            if section.decision_steps:
                decisions = []
                for step in section.decision_steps[:5]:
                    arrow = "→" if step.direction == "out" else "←"
                    decisions.append(f"{arrow} {step.related_entity} ({step.relation})")
                decision_hint = f"\n  决策关系: {'; '.join(decisions)}"

            task_lines = []
            for t in section.tasks:
                task_lines.append(f"  - [{t['task_label']}] ({t['created_at'][:10]}): {t['task_summary'][:100]}")

            parts.append(f"{header}\n{decision_hint}\n" + "\n".join(task_lines))

        return "\n\n".join(parts)

    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM 生成结构化总结"""
        client = _get_client()
        try:
            output = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": load_prompt("SKELETON_SYSTEM_PROMPT")},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"[LLM 调用失败: {e}]"

        raw = output.choices[0].message.content
        if not raw:
            return "[LLM 返回空内容]"

        import re
        content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return content
