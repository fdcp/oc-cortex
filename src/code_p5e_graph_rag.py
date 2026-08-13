"""
Phase 5 Extension: Graph-RAG 搜索模块
在 Phase 4 搜索器基础上集成知识图谱扩散，实现 Graph-RAG 增强检索。

流程 (v2):
  Query → 实体抽取 → 图谱 BFS 扩散 → 反 IDF 累加 → graph_results
       → 向量检索 (内部两级 RRF) → vector_results
       → 外层加权 RRF: hybrid = (1 - gw) / (k + rank_v) + gw / (k + rank_g)
       → Reranker → Top-K

设计要点:
  - graph_channel_weight (默认 0.3): 图谱通道在外层 RRF 的权重,
    向量通道权重 = 1 - graph_channel_weight
  - 反 IDF: 每个 entity 贡献 = log(1 + N / task_count), N = 全局 task 数,
    稀有 (task_count 小) 的实体 > 热门实体
  - 外层 RRF 纯 rank-based, 不依赖绝对分数量级, 自然消解 vector/graph 量纲差
  - Rerank 阶段输入为 SessionSearchResult 列表 (已含 chunks, 不需要二次展开)
"""
import json
import math
import os
import re
import time

from loguru import logger
from openai import OpenAI

from code_p1_utils import build_retrieval_query
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
    """Graph-RAG 增强搜索器：向量检索 + 图谱扩散 + 外层加权 RRF + Reranker"""

    # 外层 RRF 常数, 跟内层 RRF (k=60) 保持一致
    _OUTER_RRF_K = 30

    def __init__(
        self,
        searcher: SessionSearcher,
        kg_db: KGDatabase,
        graph_channel_weight: float = 0.3,
        bfs_depth: int = 1,
        max_expand_nodes: int = 30,
        max_graph_tasks: int = 50,
        rerank_multiplier: float = 2.0,
        query_instruction: str = "",
    ):
        """
        Args:
            searcher: Phase 4 SessionSearcher 实例
            kg_db: KGDatabase 实例
            graph_channel_weight: 图谱通道在外层 RRF 的权重, 取值 [0, 1]。
                决定 graph_results / vector_results 在融合时的相对重要性。
                - 0.0: 完全忽略图谱 (退化为纯向量)
                - 0.3 (默认): 历史行为, 偏向量
                - 0.5: 两路平衡
                - 1.0: 完全信任图谱
            bfs_depth: BFS 扩散深度
            max_expand_nodes: BFS 最大扩散节点数
            max_graph_tasks: 图谱扩散最多引入的 task 数
            rerank_multiplier: rerank 输入候选倍数 (>= 1.0)。
                控制 vector 粗排召回的 task 数 = top_k * rerank_multiplier,
                rerank 阶段再从这批候选里精排 top_k。
            query_instruction: BGE embedding 的 query instruction 前缀。
                跟 code_p4_search_cli.py 保持一致, 推荐设为
                config.code_p3_config.yaml 的 query_instruction_for_retrieval。
                只对 vector 检索生效 (rerank 和 LLM 实体抽取用原始 query,
                避免 instruction 干扰它们各自的处理逻辑)。
        """
        if not 0.0 <= graph_channel_weight <= 1.0:
            raise ValueError(
                f"graph_channel_weight 必须在 [0, 1] 区间, 当前: {graph_channel_weight}"
            )
        if rerank_multiplier < 1.0:
            raise ValueError(
                f"rerank_multiplier 必须 >= 1.0, 当前: {rerank_multiplier}"
            )
        self.searcher = searcher
        self.db = kg_db
        self.graph_channel_weight = graph_channel_weight
        self.bfs_depth = bfs_depth
        self.max_expand_nodes = max_expand_nodes
        self.max_graph_tasks = max_graph_tasks
        self.rerank_multiplier = rerank_multiplier
        self.query_instruction = query_instruction

        # 缓存全局 task 数 (反 IDF 的分母 N)
        # 假定 searcher 生命周期内 searcher.tasks 不变;如需更新可手动重置
        self._total_tasks = len(searcher.tasks)

    # ------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------

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

        # BGE embedding 需要 query instruction 前缀 (跟 CLI 对齐);
        # rerank 和 LLM 实体抽取仍用原始 query, 不拼接 instruction
        retrieval_query = build_retrieval_query(query, self.query_instruction)

        if not use_graph_rag:
            # 不启用图谱扩散, 走纯向量路径
            # 直接调 searcher.search (skip_rerank=not use_reranker),
            # 跟 code_p4_search_cli.run_single_search 结构完全一致
            # (rerank 在 searcher 内部完成, 不走 GraphRAG 自己的 _rerank)
            results = self.searcher.search(
                query=retrieval_query,
                top_k=top_k,
                skip_rerank=not use_reranker,
            )

            debug["vector_results"] = len(results)
            debug["top_k"] = top_k
            debug["retrieval_query"] = retrieval_query
            debug["total_time_ms"] = int((time.time() - t0) * 1000)

            return results[:top_k], debug

        # Stage 1: 向量检索 (use_graph_rag=True 才走)
        t_vec = time.time()
        # 按 rerank_multiplier 多取, 至少 top_k (防止 < 1.0 截到 0)
        vector_top_k = max(int(top_k * self.rerank_multiplier), top_k)
        vector_results = self.searcher.search(
            query=retrieval_query,
            top_k=vector_top_k,     # 给后续合并留 buffer
            skip_rerank=True,       # 延迟 rerank, 合并后统一做
        )
        debug["vector_results"] = len(vector_results)
        debug["retrieval_query"] = retrieval_query
        debug["vector_time_ms"] = int((time.time() - t_vec) * 1000)

        # Stage 2: 图谱扩散 + 反 IDF 累加
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

        graph_results: list[SessionSearchResult] = []
        if seed_entities:
            expanded = self.db.bfs_expand(
                seed_entities,
                depth=self.bfs_depth,
                max_nodes=self.max_expand_nodes,
            )
            debug["expanded_entities"] = len(expanded)

            # 反 IDF 累加: log(1 + N / task_count)
            # 稀有 entity (task_count 小) 贡献大, 热门 hub 被打压
            total_tasks = max(self._total_tasks, 1)  # 防御: 空语料除 0
            task_scores: dict[str, dict] = {}  # task_id -> {score, entities}
            nodes = self.db.get_nodes_batch(list(expanded))
            for entity_name, node in nodes.items():
                tc = max(node["task_count"], 1)  # 防御
                contrib = math.log(1 + total_tasks / tc)
                for tid in node["source_tasks"]:
                    if tid not in task_scores:
                        task_scores[tid] = {"score": 0.0, "entities": []}
                    task_scores[tid]["score"] += contrib
                    task_scores[tid]["entities"].append(entity_name)

            # 构建图谱候选 Task (按累加分降序, 截到 max_graph_tasks)
            # 同时补 chunks, 让 graph-only 结果也能进 rerank
            all_tasks = {t.task_id: t for t in self.searcher.tasks}
            for tid, info in sorted(task_scores.items(), key=lambda x: -x[1]["score"]):
                if len(graph_results) >= self.max_graph_tasks:
                    break
                if tid not in all_tasks:
                    continue
                task = all_tasks[tid]
                # 补 chunks: 让图谱召回的 task 也能给用户看原文
                chunk_details = self.searcher._expand_chunks(task.chunk_ids)
                graph_results.append(SessionSearchResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    task_label=task.task_label,
                    task_summary=task.task_summary,
                    rerank_score=0.0,
                    hybrid_score=info["score"],   # 仅用于排序, 最终 hybrid 由外层 RRF 覆盖
                    chunks=chunk_details,
                ))
            debug["graph_candidates"] = len(graph_results)

        debug["graph_time_ms"] = int((time.time() - t_graph) * 1000)

        # Stage 3: 外层加权 RRF 融合
        merged = self._outer_rrf(vector_results, graph_results, k=self._OUTER_RRF_K)
        debug["merged_candidates"] = len(merged)
        debug["outer_rrf_k"] = self._OUTER_RRF_K
        debug["graph_channel_weight"] = self.graph_channel_weight

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

    # ------------------------------------------------------------
    # 外层加权 RRF (纯函数式, 易测试)
    # ------------------------------------------------------------

    def _outer_rrf(
        self,
        vector_results: list[SessionSearchResult],
        graph_results: list[SessionSearchResult],
        k: int = 60,
    ) -> list[SessionSearchResult]:
        """外层加权 RRF 融合 (vector + graph 两路, 纯 rank-based)。

        公式:
            RRF(t) = (1 - gw) / (k + rank_v(t)) + gw / (k + rank_g(t))
            其中 rank_v / rank_g 是 task 在各自通道列表中的位置 (1-based);
            不在的通道视为 rank = +∞ (贡献 0)。

        Args:
            vector_results: 向量检索结果, 已按内部 RRF 分降序
            graph_results: 图谱扩散结果, 已按反 IDF 累加分降序
            k: RRF 常数 (默认 60, 跟内层 RRF 一致)

        Returns:
            融合后的结果, 按 hybrid_score (外层 RRF 分) 降序,
            保留原 SessionSearchResult 的 chunks (graph 独有则用 graph 版本,
            共同出现则优先用 vector 版本——因为 vector 通常带更完整的 chunks)。
        """
        INF = float("inf")
        rank_v = {r.task_id: i + 1 for i, r in enumerate(vector_results)}
        rank_g = {r.task_id: i + 1 for i, r in enumerate(graph_results)}

        # 收集所有候选 (vector 优先, graph 独有作为补充)
        result_by_id: dict[str, SessionSearchResult] = {}
        for r in vector_results:
            result_by_id[r.task_id] = r
        for r in graph_results:
            if r.task_id not in result_by_id:
                result_by_id[r.task_id] = r

        gw = self.graph_channel_weight
        fused: list[SessionSearchResult] = []
        for tid, src in result_by_id.items():
            rv = rank_v.get(tid, INF)
            rg = rank_g.get(tid, INF)
            rrf = (1.0 - gw) / (k + rv) + gw / (k + rg)
            fused.append(SessionSearchResult(
                task_id=src.task_id,
                session_id=src.session_id,
                task_label=src.task_label,
                task_summary=src.task_summary,
                rerank_score=0.0,
                hybrid_score=rrf,
                chunks=src.chunks,
            ))

        fused.sort(key=lambda x: -x.hybrid_score)
        return fused

    # ------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------

    def _rerank(
        self,
        candidates: list[SessionSearchResult],
        query: str,
        top_k: int,
    ) -> list[SessionSearchResult]:
        """Graph 路径专用 rerank: 用 task_summary 作为 rerank 文本。

        设计说明 (v2):
          - 直接取 top_k (不再 *2), 因为外层 RRF 排序已经够准,
            不需要给 reranker 额外的 buffer
          - 老的 top_k*2 是为了给带 bug 的 _merge_candidates 兜底,
            现在 RRF 加性 + rank-based 已经把那个 bug 解决了
          - 只用 task_summary, 不拼 label: label 太短 (5-15 字) 信息密度低,
            rerank 容易被 "标签词命中" 这种表面信号带偏

        注意: use_graph_rag=False 路径不走这个方法, 它直接调
        self.searcher.search() 跟 code_p4_search_cli 结构完全一致。
        """
        if not self.searcher.reranker:
            return candidates

        texts = [c.task_summary for c in candidates]
        rerank_results = self.searcher.reranker.rank(query, texts, top_k=top_k)
        print(
            f"Rerank: {len(rerank_results)} candidates, top_k={top_k}, query='{query}'"
        )
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
