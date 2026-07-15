"""
Phase 4 搜索编排模块
跨 session 搜索完整流程:
  Query → 混合检索(tasks) → Reranker 精排 → 取回 chunk 原文 → 结构化输出

复用 Phase 3 的 Phase3Store + Qdrant 数据
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from code_p1_utils import Config
from code_p1_models import Chunk, CleanedToolCall
from code_p2_models import Task
from code_p3_qdrant_store import Phase3Store, HybridResult, SearchResult
from code_p4_reranker import Qwen3Reranker, RerankResult


# ============================================================
# 搜索结果
# ============================================================

@dataclass
class ChunkDetail:
    """搜索结果中关联的 chunk 详情"""
    chunk_id: str
    session_id: str
    turn_index: int
    summary: str
    cleaned_text_preview: str   # 前 200 字预览
    raw_size_tokens: int = 0
    cleaned_size_tokens: int = 0


@dataclass
class SessionSearchResult:
    """单条跨 session 搜索结果"""
    task_id: str
    session_id: str
    task_label: str
    task_summary: str
    rerank_score: float
    hybrid_score: float         # RRF score (粗排)
    chunks: list[ChunkDetail] = field(default_factory=list)


# ============================================================
# 数据加载 (复用 search_demo 的逻辑)
# ============================================================

def _load_tasks(path: str) -> list[Task]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    tasks.append(Task.from_dict(json.loads(line)))
                except Exception:
                    pass
    return tasks


def _load_chunks(path: str) -> list[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    chunks.append(Chunk(
                        chunk_id=data["chunk_id"],
                        session_id=data["session_id"],
                        turn_index=data["turn_index"],
                        user_message=data["user_message"],
                        assistant_messages=data.get("assistant_messages", []),
                        tool_calls=[CleanedToolCall(**tc) for tc in data.get("tool_calls", [])],
                        mcp_calls=[CleanedToolCall(**mc) for mc in data.get("mcp_calls", [])],
                        raw_size_tokens=data.get("raw_size_tokens", 0),
                        cleaned_size_tokens=data.get("cleaned_size_tokens", 0),
                        created_at=data.get("created_at"),
                        task_summary=data.get("task_summary"),
                    ))
                except Exception:
                    pass
    return chunks


def _load_summaries(path: str) -> dict[str, str]:
    summaries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    summaries[data["chunk_id"]] = data["summary"]
                except Exception:
                    pass
    return summaries


# ============================================================
# SessionSearcher
# ============================================================

class SessionSearcher:
    """
    跨 session 搜索引擎

    封装 Phase3Store (向量检索) + Qwen3Reranker (精排) + chunk 详情展开
    """

    def __init__(self, config_path: str = "code_p3_config.yaml"):
        # 1. 加载配置
        self.config = Config.load(config_path)
        logger.info(f"加载配置: {config_path}")

        # 2. 加载源数据到内存 map
        tasks_file = self.config.get("phase2.tasks_file", "./output/tasks.jsonl")
        chunks_file = self.config.get("phase1.chunks_file", "./output/chunks.jsonl")
        summaries_file = self.config.get(
            "phase2.chunk_summaries_file", "./output/chunks_summary_p2.jsonl"
        )

        self.tasks = _load_tasks(tasks_file)
        self.chunks = _load_chunks(chunks_file)
        self.summaries = _load_summaries(summaries_file)

        # task map: task_id -> Task
        self.task_map: dict[str, Task] = {t.task_id: t for t in self.tasks}
        # chunk map: chunk_id -> Chunk
        self.chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        logger.info(
            f"数据加载: {len(self.tasks)} tasks, "
            f"{len(self.chunks)} chunks, {len(self.summaries)} summaries"
        )

        # 3. 初始化 Phase3Store (embedding + Qdrant + BM25)
        cache_folder = self.config.get("embedding.cache_folder")
        import os
        if cache_folder is None:
            cache_folder = os.path.expanduser("~/.cache/huggingface/hub")

        self.store = Phase3Store(
            model_name=self.config.get("embedding.model", "BAAI/bge-small-zh-v1.5"),
            dim=self.config.get("embedding.dim", 512),
            batch_size=self.config.get("embedding.batch_size", 32),
            device=self.config.get("embedding.device", "cpu"),
            qdrant_path=self.config.get("qdrant.path", "./qdrant_data"),
            tasks_collection=self.config.get("qdrant.collections.tasks", "tasks"),
            chunks_collection=self.config.get("qdrant.collections.chunks", "chunks"),
            sparse_method=self.config.get("sparse.method", "bm25"),
            jieba_mode=self.config.get("sparse.jieba_mode", "search"),
            bm25_k1=self.config.get("sparse.bm25_params.k1", 1.5),
            bm25_b=self.config.get("sparse.bm25_params.b", 0.75),
            fuse_k=self.config.get("sparse.fuse_k", 60),
            bge_m3_model=(
                self.config.get("sparse.bge_m3_model", "BAAI/bge-m3")
                if self.config.get("sparse.method") == "bge_m3"
                else None
            ),
            cache_folder=cache_folder,
            offline_mode=self.config.get("embedding.offline_mode", True),
        )
        self.store.init_collections()

        # 重建 BM25 索引 (in-memory, 从已加载的数据)
        if self.store.sparse_method == "bm25":
            self._rebuild_bm25_index()

        # 4. 初始化 Reranker
        reranker_model = self.config.get(
            "reranker.model", "Qwen/Qwen3-Reranker-0.6B"
        )
        reranker_device = self.config.get("reranker.device", "cpu")
        reranker_instruction = self.config.get("reranker.instruction")
        reranker_max_length = self.config.get("reranker.max_length", 8192)
        reranker_batch_size = self.config.get("reranker.batch_size", 4)

        self.reranker = Qwen3Reranker(
            model_name=reranker_model,
            device=reranker_device,
            cache_folder=cache_folder,
            offline_mode=self.config.get("embedding.offline_mode", True),
            instruction=reranker_instruction,
            max_length=reranker_max_length,
            batch_size=reranker_batch_size,
        )

        logger.info("SessionSearcher 初始化完成")

    def _rebuild_bm25_index(self) -> None:
        """从已加载的 tasks 数据重建 BM25 索引"""
        from code_p3_qdrant_store import _stable_uuid

        texts = [t.task_summary for t in self.tasks]
        point_ids = [_stable_uuid(t.task_id) for t in self.tasks]
        payloads = [
            {
                "task_id": t.task_id,
                "session_id": t.session_id,
                "task_label": t.task_label,
                "task_summary": t.task_summary,
                "chunk_ids": t.chunk_ids,
                "created_at": t.created_at,
            }
            for t in self.tasks
        ]
        self.store.build_bm25_index(
            self.store.tasks_collection, texts, point_ids, payloads
        )
        logger.info("BM25 索引重建完成 (tasks)")

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 5,
        skip_rerank: bool = False,
    ) -> list[SessionSearchResult]:
        """
        跨 session 搜索

        Args:
            query: 自然语言查询
            top_k: 返回结果数
            candidate_multiplier: 粗排候选倍数 (top_k * N)
            skip_rerank: True 跳过 reranker，直接返回粗排结果

        Returns:
            SessionSearchResult 列表 (按 rerank_score 降序)
        """
        t_total = time.time()
        logger.info(f"搜索: '{query}' (top_k={top_k}, skip_rerank={skip_rerank})")

        # 1. 混合检索 → 候选 tasks
        n_candidates = top_k * candidate_multiplier
        candidates = self.store.search_hybrid(
            query, collection=self.store.tasks_collection, top_k=n_candidates
        )
        logger.info(f"粗排: {len(candidates)} 个候选 task")

        if not candidates:
            logger.warning("粗排无结果")
            return []

        if skip_rerank:
            # 跳过 reranker, 直接用 RRF score
            results = []
            for c in candidates[:top_k]:
                task_id = c.payload.get("task_id", "")
                task = self.task_map.get(task_id)
                if not task:
                    continue
                chunk_details = self._expand_chunks(task.chunk_ids)
                results.append(SessionSearchResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    task_label=task.task_label,
                    task_summary=task.task_summary,
                    rerank_score=0.0,
                    hybrid_score=c.rrf_score,
                    chunks=chunk_details,
                ))
            logger.info(
                f"搜索完成 (skip_rerank): {len(results)} 结果, "
                f"耗时 {time.time() - t_total:.2f}s"
            )
            return results

        # 2. Reranker 精排
        documents = [c.payload.get("task_summary", "") for c in candidates]
        reranked = self.reranker.rank(query, documents, top_k=top_k)

        # 3. 展开: 关联 chunks
        results = []
        for rr in reranked:
            candidate = candidates[rr.index]
            task_id = candidate.payload.get("task_id", "")
            task = self.task_map.get(task_id)
            if not task:
                logger.warning(f"task_map 未找到: {task_id}")
                continue

            chunk_details = self._expand_chunks(task.chunk_ids)
            results.append(SessionSearchResult(
                task_id=task.task_id,
                session_id=task.session_id,
                task_label=task.task_label,
                task_summary=task.task_summary,
                rerank_score=rr.score,
                hybrid_score=candidate.rrf_score,
                chunks=chunk_details,
            ))

        logger.info(
            f"搜索完成: {len(results)} 结果, "
            f"耗时 {time.time() - t_total:.2f}s"
        )
        return results

    def _expand_chunks(self, chunk_ids: list[str]) -> list[ChunkDetail]:
        """通过 chunk_ids 从内存 map 取 chunk 详情"""
        details = []
        for cid in chunk_ids:
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            summary = self.summaries.get(cid, "")
            cleaned = chunk.cleaned_text()
            preview = cleaned[:200] + ("..." if len(cleaned) > 200 else "")
            details.append(ChunkDetail(
                chunk_id=chunk.chunk_id,
                session_id=chunk.session_id,
                turn_index=chunk.turn_index,
                summary=summary,
                cleaned_text_preview=preview,
                raw_size_tokens=chunk.raw_size_tokens,
                cleaned_size_tokens=chunk.cleaned_size_tokens,
            ))
        return details
