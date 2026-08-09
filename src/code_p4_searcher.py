"""
Phase 4 搜索编排模块
跨 session 搜索完整流程 (两阶段查询):
  Query → 两路 chunk 检索 RRF → Dense(tasks.task_summary) RRF → Reranker → chunk 展开

Sparse 支持 BM25 (开发) 或 BGE-M3 (上线)，由 config sparse.method 决定。
复用 Phase 3 的 Phase3Store + Qdrant 数据
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from code_p1_utils import Config
from code_p1_models import Chunk, CleanedToolCall
from code_p2_models import Task
from code_p3_qdrant_store import Phase3Store, SearchResult
from code_p4_reranker import Qwen3Reranker


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
        self.chunks_summary_method = self.config.get(
            "sparse.chunks_summary_method", "sparse"
        )
        if self.chunks_summary_method not in {"sparse", "dense"}:
            raise ValueError(
                "sparse.chunks_summary_method must be 'sparse' or 'dense'"
            )

        # 2. 加载源数据到内存 map
        # 相对路径回退到仓库根（与 code_MS_inspect.py 一致, 兼容任意 cwd）
        _repo_root = Path(config_path).resolve().parent.parent
        def _resolve_path(p: str) -> str:
            pp = Path(p)
            return str(pp if pp.is_absolute() else (_repo_root / pp))

        tasks_file = _resolve_path(self.config.get("phase2.tasks_file", "./output/tasks.jsonl"))
        chunks_file = _resolve_path(self.config.get("phase1.chunks_file", "./output/chunks.jsonl"))
        summaries_file = _resolve_path(self.config.get(
            "phase2.chunk_summaries_file", "./output/chunks_summary_p2.jsonl"
        ))

        self.tasks = _load_tasks(tasks_file)
        self.chunks = _load_chunks(chunks_file)
        self.summaries = _load_summaries(summaries_file)

        # task map: task_id -> Task
        self.task_map: dict[str, Task] = {t.task_id: t for t in self.tasks}
        # chunk map: chunk_id -> Chunk
        self.chunk_map: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        # chunk -> task mapping (用于两阶段查询: Sparse 命中 chunk 后映射回 task)
        self.chunk_to_task: dict[str, str] = {}
        for t in self.tasks:
            for cid in t.chunk_ids:
                self.chunk_to_task[cid] = t.task_id
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
            qdrant_path=_resolve_path(self.config.get("qdrant.path", "./qdrant_data")),
            tasks_collection=self.config.get("qdrant.collections.tasks", "tasks"),
            chunks_summary_collection=self.config.get(
                "qdrant.collections.chunks_summary", "chunks_summary"
            ),
            chunks_cleaned_text_collection=self.config.get(
                "qdrant.collections.chunks_cleaned_text", "chunks_cleaned_text"
            ),
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
        """从已加载的 chunks 数据重建两个 chunk 集合的 BM25 索引。"""
        from code_p3_qdrant_store import _stable_uuid

        cleaned_texts = []
        cleaned_point_ids = []
        cleaned_payloads = []
        summary_texts = []
        summary_point_ids = []
        summary_payloads = []
        for c in self.chunks:
            cleaned = c.cleaned_text()
            if cleaned:
                cleaned_texts.append(cleaned)
                cleaned_point_ids.append(_stable_uuid(c.chunk_id))
                cleaned_payloads.append({
                    "chunk_id": c.chunk_id,
                    "session_id": c.session_id,
                    "turn_index": c.turn_index,
                    "task_id": self.chunk_to_task.get(c.chunk_id, ""),
                })
            summary = self.summaries.get(c.chunk_id, "")
            if summary:
                summary_texts.append(summary)
                summary_point_ids.append(_stable_uuid(c.chunk_id))
                summary_payloads.append({
                    "chunk_id": c.chunk_id,
                    "session_id": c.session_id,
                    "turn_index": c.turn_index,
                    "task_id": self.chunk_to_task.get(c.chunk_id, ""),
                    "summary": summary,
                })
        self.store.build_bm25_index(
            self.store.chunks_cleaned_text_collection,
            cleaned_texts,
            cleaned_point_ids,
            cleaned_payloads,
        )
        self.store.build_bm25_index(
            self.store.chunks_summary_collection,
            summary_texts,
            summary_point_ids,
            summary_payloads,
        )
        logger.info(
            f"BM25 索引重建完成 ({self.store.chunks_cleaned_text_collection}): "
            f"{len(cleaned_texts)} 条 cleaned_text; "
            f"({self.store.chunks_summary_collection}): {len(summary_texts)} 条 summary"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 5,
        skip_rerank: bool = False,
    ) -> list[SessionSearchResult]:
        """
        跨 session 搜索 (两级 RRF 查询)

        阶段 1: Dense 搜 tasks 集合 (task_summary 语义匹配)
        阶段 2: chunks_summary 与 chunks_cleaned_text 两路检索 → 分别映射回 task
        阶段 3: 两路 chunk task 结果先做 RRF
        阶段 4: chunk RRF 结果与 Dense task 结果再做 RRF
        阶段 5: Reranker 精排 (可选)
        阶段 6: 展开 chunk 详情

        Args:
            query: 自然语言查询
            top_k: 返回结果数
            candidate_multiplier: 粗排候选倍数 (top_k * N)
            skip_rerank: True 跳过 reranker，直接返回粗排结果

        Returns:
            SessionSearchResult 列表 (按 rerank_score 降序)
        """
        t_total = time.time()
        n_candidates = top_k * candidate_multiplier
        logger.info(
            f"搜索: '{query}' (top_k={top_k}, skip_rerank={skip_rerank})"
        )

        # 阶段 1: Dense → tasks 集合
        dense_results = self.store.search_dense(
            query, collection=self.store.tasks_collection, top_k=n_candidates
        )
        logger.info(f"  Dense[tasks]: {len(dense_results)} 结果")

        # 阶段 2: 两路 chunk 检索 → 分别映射回 task
        sparse_top_k = n_candidates * 3  # 取更多 chunk, 映射后去重
        summary_chunk_results = self._search_chunks_summary(query, sparse_top_k)
        summary_task_results = self._aggregate_chunks_to_tasks(summary_chunk_results)
        logger.info(
            f"  {self.chunks_summary_method}[chunks_summary] → task 聚合: "
            f"{len(summary_task_results)} 个 task"
        )

        if self.store.sparse_method == "bge_m3":
            sparse_chunk_results = self.store.search_sparse_bge_m3(
                query,
                collection=self.store.chunks_cleaned_text_collection,
                top_k=sparse_top_k,
            )
        else:
            sparse_chunk_results = self.store.search_sparse_bm25(
                query,
                collection=self.store.chunks_cleaned_text_collection,
                top_k=sparse_top_k,
            )
        logger.info(
            f"  Sparse[{self.store.sparse_method}][chunks_cleaned_text]: "
            f"{len(sparse_chunk_results)} chunk 命中"
        )

        # 将 chunk 级 Sparse 结果聚合为 task 级 (同一 task 取最高分)
        sparse_task_results = self._aggregate_chunks_to_tasks(sparse_chunk_results)
        logger.info(
            f"  Sparse → task 聚合: {len(sparse_task_results)} 个 task"
        )

        # 阶段 3: 先融合两个 chunk 路径
        chunk_fused = self.store._rrf_fuse(
            summary_task_results,
            sparse_task_results,
            self.store.fuse_k,
        )
        chunk_fused_as_search = [
            SearchResult(result.point_id, result.rrf_score, result.payload)
            for result in chunk_fused
        ]
        logger.info(f"  两路 chunk RRF 融合: {len(chunk_fused)} 个 task")

        # 阶段 4: chunk RRF 与 dense task 结果再次融合
        fused = self.store._rrf_fuse(
            dense_results,
            chunk_fused_as_search,
            self.store.fuse_k,
        )
        candidates = fused[:n_candidates]
        logger.info(f"  RRF 融合: {len(candidates)} 个候选 task")

        if not candidates:
            logger.warning("粗排无结果")
            return []

        if skip_rerank:
            results = []
            # 不再 candidates[:top_k] 截断, 返回 n_candidates 全量
            # 调用方负责最终 [:top_k] 截断 (跟 skip_rerank=False 路径的
            # "全部 n_candidates 喂给 rerank" 保持一致语义)
            for c in candidates:
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

        # 阶段 5: Reranker 精排
        documents = [c.payload.get("task_summary", "") for c in candidates]
        reranked = self.reranker.rank(query, documents, top_k=top_k)

        # 阶段 6: 展开 chunk 详情
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

    def _search_chunks_summary(self, query: str, top_k: int) -> list[SearchResult]:
        """按配置从 chunks_summary 集合执行 dense 或 sparse 检索。"""
        if self.chunks_summary_method == "dense":
            return self.store.search_dense(
                query,
                collection=self.store.chunks_summary_collection,
                top_k=top_k,
            )
        if self.store.sparse_method == "bm25":
            return self.store.search_sparse_bm25(
                query,
                collection=self.store.chunks_summary_collection,
                top_k=top_k,
            )
        return self.store.search_sparse_bge_m3(
            query,
            collection=self.store.chunks_summary_collection,
            top_k=top_k,
        )

    def _aggregate_chunks_to_tasks(
        self, chunk_results: list[SearchResult]
    ) -> list[SearchResult]:
        """
        将 chunk 级 Sparse 结果聚合为 task 级
        同一 task 下多个 chunk 命中时, 取最高 score
        point_id 使用 task 级的 stable UUID, 以便 RRF 融合时正确去重
        """
        from code_p3_qdrant_store import _stable_uuid

        task_best: dict[str, SearchResult] = {}
        for r in chunk_results:
            task_id = r.payload.get("task_id", "")
            if not task_id:
                continue
            if task_id not in task_best or r.score > task_best[task_id].score:
                task = self.task_map.get(task_id)
                if task:
                    task_payload = {
                        "task_id": task.task_id,
                        "session_id": task.session_id,
                        "task_label": task.task_label,
                        "task_summary": task.task_summary,
                        "chunk_ids": task.chunk_ids,
                    }
                    task_best[task_id] = SearchResult(
                        point_id=_stable_uuid(task.task_id),
                        score=r.score,
                        payload=task_payload,
                    )
        return list(task_best.values())

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
