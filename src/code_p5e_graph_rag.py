"""
Phase 5 Extension: Graph-RAG 搜索模块
在 Phase 4 搜索器基础上集成知识图谱扩散，实现 Graph-RAG 增强检索。

流程:
  Query → 实体抽取 → 图谱 BFS 扩散 → 扩散 Task 加入候选池 → 与向量检索合并 → Reranker → Top-K
"""
import json
import os
import re
import time

from loguru import logger
from openai import OpenAI

from code_p4_searcher import SessionSearcher, SessionSearchResult
from code_p5e_db import KGDatabase
from code_update_prompt_utils import load_prompt


# ============================================================
# 查询实体抽取（轻量 LLM 调用，供 Graph-RAG 与 MCP 共用）
# ============================================================

# 默认模型 (entity 模式使用 hy3 code_p5_config.yaml 一致)
DEFAULT_MODEL = "hy3"


def _get_client() -> OpenAI:
    """创建 OpenAI 客户端（复用 code_p5_config.yaml 配置）"""
    api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    return OpenAI(
        api_key=api_key,
        base_url="https://opencode.ai/zen/go/v1",
        timeout=30,
    )


def extract_query_entities(
    query: str,
    max_retries: int = 2,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """从用户查询中提取关键实体名称，用于图谱扩散。

    Args:
        query: 用户搜索查询
        max_retries: 解析失败时重试次数
        model: LLM 模型名

    Returns:
        实体名称列表（失败时返回空列表）
    """
    prompt_template = load_prompt("QUERY_ENTITY_PROMPT")
    client = _get_client()

    for attempt in range(max_retries + 1):
        try:
            output = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_template.format(query=query)}],
                temperature=0.1,
                max_tokens=2000,  # reasoning 模型 (如 hy3) 需要预算放 thinking
            )
        except Exception as e:
            logger.warning(f"查询实体抽取 LLM 调用失败 (attempt {attempt + 1}): {e}")
            continue  # 网络错误可重试

        # 以下情况不重试（模型本身返回异常，重试无意义）
        if not output.choices:
            logger.warning("查询实体抽取 LLM 返回空 choices, 跳过图谱扩散")
            break

        message = output.choices[0].message
        if message is None:
            logger.warning("查询实体抽取 LLM 返回空 message, 跳过图谱扩散")
            break

        content = message.content

        # Reasoning 模型 (如 hy3) 可能把 token 消耗在 thinking 上,
        # 导致 content 为空;此时检查 reasoning_content 作为 fallback
        if not content or not content.strip():
            reasoning = None
            if hasattr(message, "model_extra") and message.model_extra:
                reasoning = message.model_extra.get("reasoning_content", "")
            if reasoning and reasoning.strip():
                logger.warning("查询实体抽取 LLM content 为空, 从 reasoning_content 提取")
                content = reasoning
            else:
                logger.warning("查询实体抽取 LLM 返回空内容 (content 和 reasoning 均为空), 跳过图谱扩散")
                break

        content = content.strip()
        # 去除 <think>...</think> 标签块 (部分模型会输出思考过程)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # 去除可能包裹的 markdown 代码块标记
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()

        try:
            data = json.loads(content)
            entities = data.get("entities", [])
            if isinstance(entities, list):
                return [e.strip() for e in entities if isinstance(e, str) and len(e.strip()) >= 2]
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"查询实体抽取解析失败: {content[:100]}")
            break  # 解析失败不重试

    logger.warning("查询实体抽取未成功, 返回空列表 (退化为纯向量搜索)")
    return []


# ============================================================
# Graph-RAG 搜索器
# ============================================================

class GraphRAGSearcher:
    """Graph-RAG 增强搜索器：向量检索 + 图谱扩散 + Reranker"""

    def __init__(
        self,
        searcher: SessionSearcher,
        kg_db: KGDatabase,
        graph_weight: float = 0.3,
        bfs_depth: int = 1,
        max_expand_nodes: int = 30,
        max_graph_tasks: int = 50,
        rerank_multiplier: float = 2.0,
    ):
        """
        Args:
            searcher: Phase 4 SessionSearcher 实例
            kg_db: KGDatabase 实例
            graph_weight: 图谱扩散结果的权重加成
            bfs_depth: BFS 扩散深度
            max_expand_nodes: BFS 最大扩散节点数
            max_graph_tasks: 图谱扩散最多引入的 task 数
            rerank_multiplier: rerank 输入候选倍数 (>= 1.0)。
                控制 vector 粗排召回的 task 数 = top_k * rerank_multiplier,
                rerank 阶段再从这批候选里精排 top_k。
                - 1.0: 最少, 仅 top_k, 几乎完全依赖 rerank 重排
                - 2.0: 默认, 平衡 (历史行为)
                - 5.0: 跟 CLI 一致, 召回更全但 rerank 更慢
        """
        if rerank_multiplier < 1.0:
            raise ValueError(
                f"rerank_multiplier 必须 >= 1.0, 当前: {rerank_multiplier}"
            )
        self.searcher = searcher
        self.db = kg_db
        self.graph_weight = graph_weight
        self.bfs_depth = bfs_depth
        self.max_expand_nodes = max_expand_nodes
        self.max_graph_tasks = max_graph_tasks
        self.rerank_multiplier = rerank_multiplier

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
        use_graph_rag: bool = True,
    ) -> tuple[list[SessionSearchResult], dict]:
        """Graph-RAG 增强搜索。

        Args:
            query: 搜索查询
            top_k: 返回结果数
            use_reranker: 是否用 reranker
            use_graph_rag: 是否启用图谱扩散

        Returns:
            (results, debug_info)
        """
        t0 = time.time()
        debug = {}

        # Stage 1: 向量检索
        t_vec = time.time()
        # 按 rerank_multiplier 多取, 至少 top_k (防止 < 1.0 截到 0)
        vector_top_k = max(int(top_k * self.rerank_multiplier), top_k)
        vector_results = self.searcher.search(
            query=query,
            top_k=vector_top_k,     # 给后续合并留 buffer
            skip_rerank=True,       # 延迟 rerank, 合并后统一做
        )
        debug["vector_results"] = len(vector_results)
        debug["vector_time_ms"] = int((time.time() - t_vec) * 1000)

        if not use_graph_rag:
            # 不启用图谱扩散，直接 rerank 返回
            if use_reranker and vector_results:
                vector_results = self._rerank(vector_results, query, top_k)
            debug["total_time_ms"] = int((time.time() - t0) * 1000)
            return vector_results[:top_k], debug

        # Stage 2: 图谱扩散
        t_graph = time.time()
        query_entities = extract_query_entities(query)
        debug["query_entities"] = query_entities

        # 模糊匹配: 将抽取的实体映射到图谱中的实际节点
        seed_entities = []
        for e in query_entities:
            # 先尝试精确匹配
            if self.db.get_node(e):
                seed_entities.append(e)
            else:
                # 模糊搜索取 top-3
                matches = self.db.search_entities(e, limit=3)
                seed_entities.extend(m["name"] for m in matches)
        seed_entities = list(set(seed_entities))
        debug["seed_entities"] = seed_entities

        graph_results = []
        if seed_entities:
            expanded = self.db.bfs_expand(
                seed_entities,
                depth=self.bfs_depth,
                max_nodes=self.max_expand_nodes,
            )
            debug["expanded_entities"] = len(expanded)

            # 收集扩散实体关联的 task
            task_scores: dict[str, dict] = {}  # task_id -> {score, entities}
            for entity_name in expanded:
                node = self.db.get_node(entity_name)
                if not node:
                    continue
                for tid in node["source_tasks"]:
                    if tid not in task_scores:
                        task_scores[tid] = {"score": 0.0, "entities": []}
                    task_scores[tid]["score"] += self.graph_weight * (node["task_count"] / 10.0)
                    task_scores[tid]["entities"].append(entity_name)

            # 构建图谱候选 Task
            all_tasks = {t.task_id: t for t in self.searcher.tasks}
            for tid, info in sorted(task_scores.items(), key=lambda x: -x[1]["score"]):
                if tid not in all_tasks:
                    continue
                if len(graph_results) >= self.max_graph_tasks:
                    break
                task = all_tasks[tid]
                graph_results.append(SessionSearchResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    task_label=task.task_label,
                    task_summary=task.task_summary,
                    rerank_score=0.0,
                    hybrid_score=info["score"],
                    chunks=[],
                ))
            debug["graph_candidates"] = len(graph_results)

        debug["graph_time_ms"] = int((time.time() - t_graph) * 1000)

        # Stage 3: 合并候选池 (去重，保留最高分)
        merged = self._merge_candidates(vector_results, graph_results)
        debug["merged_candidates"] = len(merged)

        # Stage 4: Reranker
        t_rerank = time.time()
        if use_reranker and merged:
            merged = self._rerank(merged, query, top_k)
        debug["rerank_time_ms"] = int((time.time() - t_rerank) * 1000)

        results = merged[:top_k]

        # 统计来源分布
        vector_ids = {r.task_id for r in vector_results}
        graph_ids = {r.task_id for r in graph_results}
        source_dist = {"vector": 0, "graph": 0, "both": 0}
        for r in results:
            if r.task_id in vector_ids and r.task_id in graph_ids:
                source_dist["both"] += 1
            elif r.task_id in graph_ids:
                source_dist["graph"] += 1
            else:
                source_dist["vector"] += 1
        debug["source_distribution"] = source_dist
        debug["total_time_ms"] = int((time.time() - t0) * 1000)
        debug["final_count"] = len(results)

        return results, debug

    def _merge_candidates(
        self,
        vector_results: list[SessionSearchResult],
        graph_results: list[SessionSearchResult],
    ) -> list[SessionSearchResult]:
        """合并向量和图谱候选，去重保留高分。"""
        seen: dict[str, SessionSearchResult] = {}

        # 先放向量结果
        for r in vector_results:
            if r.task_id not in seen or r.hybrid_score > seen[r.task_id].hybrid_score:
                seen[r.task_id] = r

        # 再放图谱结果 (加成后的分数可能更高)
        for r in graph_results:
            if r.task_id not in seen or r.hybrid_score > seen[r.task_id].hybrid_score:
                seen[r.task_id] = r
            elif r.task_id in seen:
                # 两个来源都有 → 加分
                seen[r.task_id].hybrid_score += r.hybrid_score * 0.5

        return sorted(seen.values(), key=lambda x: -x.hybrid_score)

    def _rerank(
        self,
        candidates: list[SessionSearchResult],
        query: str,
        top_k: int,
    ) -> list[SessionSearchResult]:
        """复用 SessionSearcher 的 reranker"""
        if not self.searcher.reranker:
            return candidates

        texts = [f"{c.task_label}: {c.task_summary}" for c in candidates]
        rerank_results = self.searcher.reranker.rank(query, texts, top_k=top_k * 2)

        reranked = []
        for rr in rerank_results:
            orig = candidates[rr.index]
            reranked.append(SessionSearchResult(
                task_id=orig.task_id,
                session_id=orig.session_id,
                task_label=orig.task_label,
                task_summary=orig.task_summary,
                rerank_score=rr.score,
                hybrid_score=orig.hybrid_score,
                chunks=orig.chunks,
            ))
        return reranked
