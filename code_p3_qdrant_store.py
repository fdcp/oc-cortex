"""
Phase 3 核心模块: Qdrant 双集合向量存储
- tasks 集合: task_summary 向量化
- chunks 集合: summary + cleaned_text 向量化
"""
import platform
import hashlib
import time
from typing import Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from code_p2_models import Task
from code_p1_models import Chunk


# ============================================================
# 设备自动检测
# ============================================================

def _detect_device(configured: str) -> str:
    """
    根据配置和运行环境决定 embedding 设备
    auto -> mps (Apple Silicon) / cuda / cpu
    """
    if configured != "auto":
        return configured

    system = platform.system()
    if system == "Darwin":
        # macOS: 尝试 MPS (Apple Silicon)
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("检测到 MPS (Apple Silicon), 使用 mps 设备")
                return "mps"
        except ImportError:
            pass
        logger.info("macOS 环境, 使用 cpu 设备")
        return "cpu"

    # Linux / Windows: 尝试 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"检测到 CUDA, 使用 cuda 设备 ({torch.cuda.get_device_name(0)})")
            return "cuda"
    except ImportError:
        pass

    logger.info("未检测到 GPU, 使用 cpu 设备")
    return "cpu"


# ============================================================
# 字符串 -> 稳定 UUID
# ============================================================

def _stable_uuid(text: str) -> str:
    """
    从字符串生成稳定的 UUID (基于 MD5)
    用于 Qdrant point id, 保证相同内容映射到相同 id
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    # 格式化为 UUID 风格: 8-4-4-4-12
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ============================================================
# Phase3Store
# ============================================================

class Phase3Store:
    """
    Qdrant 双集合向量存储
    封装 QdrantClient + SentenceTransformer
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dim: int = 1024,
        batch_size: int = 16,
        device: str = "auto",
        qdrant_path: str = "./qdrant_data",
        tasks_collection: str = "tasks",
        chunks_collection: str = "chunks",
    ):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.tasks_collection = tasks_collection
        self.chunks_collection = chunks_collection

        # 1. 确定设备
        self.device = _detect_device(device)
        logger.info(f"Embedding 设备: {self.device}")

        # 2. 加载 embedding 模型
        logger.info(f"加载 embedding 模型: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        logger.info("Embedding 模型加载完成")

        # 3. 初始化 Qdrant 客户端 (文件持久化)
        logger.info(f"初始化 Qdrant 客户端: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)

    # --------------------------------------------------------
    # 集合初始化
    # --------------------------------------------------------

    def init_collections(self) -> None:
        """创建或验证 tasks / chunks 集合 (1024-dim, Cosine)"""
        for name in [self.tasks_collection, self.chunks_collection]:
            existing = [c.name for c in self.client.get_collections().collections]
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=rest.VectorParams(
                        size=self.dim,
                        distance=rest.Distance.COSINE,
                    ),
                )
                logger.info(f"创建集合: {name} (dim={self.dim}, Cosine)")
            else:
                logger.info(f"集合已存在: {name}")

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量 embed 文本
        返回 list[list[float]], 每个向量 dim 维
        """
        if not texts:
            return []

        total = len(texts)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info(f"开始 embedding: {total} 条文本, {n_batches} 个 batch (batch_size={self.batch_size})")

        all_vectors = []
        t_total = time.time()
        for batch_idx, i in enumerate(range(0, total, self.batch_size), 1):
            batch = texts[i : i + self.batch_size]
            t_batch = time.time()
            # sentence-transformers encode 返回 numpy array
            vectors = self.encoder.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_vectors.extend(vectors.tolist())
            batch_elapsed = time.time() - t_batch
            logger.info(
                f"  batch {batch_idx}/{n_batches}: "
                f"{len(batch)} 条, 耗时 {batch_elapsed:.1f}s "
                f"(累计 {len(all_vectors)}/{total})"
            )

        total_elapsed = time.time() - t_total
        logger.info(f"Embedding 完成: {len(all_vectors)} 个向量, 总耗时 {total_elapsed:.1f}s")
        return all_vectors

    # --------------------------------------------------------
    # Task upsert
    # --------------------------------------------------------

    def upsert_tasks(self, tasks: list[Task]) -> int:
        """
        将 Task 列表写入 tasks 集合
        - 向量: task_summary embedding
        - Payload: 完整 Task 字段
        返回成功 upsert 的数量
        """
        if not tasks:
            logger.warning("upsert_tasks: 无 task 可写入")
            return 0

        logger.info(f"准备 upsert {len(tasks)} 个 task ...")

        # 1. 构建 embed 文本
        texts = [t.task_summary for t in tasks]

        # 2. 批量 embed
        vectors = self.embed(texts)
        logger.info(f"Task embedding 完成: {len(vectors)} 个向量")

        # 3. 构建 Qdrant points
        points = []
        for task, vec in zip(tasks, vectors):
            point_id = _stable_uuid(task.task_id)
            payload = {
                "task_id": task.task_id,
                "session_id": task.session_id,
                "task_label": task.task_label,
                "task_summary": task.task_summary,
                "chunk_ids": task.chunk_ids,
                "created_at": task.created_at,
            }
            points.append(rest.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            ))

        # 4. 批量 upsert
        self.client.upsert(
            collection_name=self.tasks_collection,
            points=points,
        )

        logger.info(f"Task upsert 完成: {len(points)} 条写入 '{self.tasks_collection}'")
        return len(points)

    # --------------------------------------------------------
    # Chunk upsert
    # --------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        summaries: dict[str, str],
        tasks: Optional[list[Task]] = None,
    ) -> int:
        """
        将 Chunk 列表写入 chunks 集合
        - 向量: (summary + cleaned_text) embedding
        - Payload: chunk 字段 + task_id 关联
        - summaries: {chunk_id: summary} 字典 (Phase 2 输出)
        - tasks: Task 列表, 用于构建 chunk->task 反向映射
        返回成功 upsert 的数量
        """
        if not chunks:
            logger.warning("upsert_chunks: 无 chunk 可写入")
            return 0

        logger.info(f"准备 upsert {len(chunks)} 个 chunk ...")

        # 1. 构建 chunk_id -> task_id 反向映射
        chunk_to_task: dict[str, str] = {}
        if tasks:
            for t in tasks:
                for cid in t.chunk_ids:
                    chunk_to_task[cid] = t.task_id

        # 2. 构建 embed 文本: summary + cleaned_text
        texts = []
        valid_chunks = []
        skipped = 0

        for c in chunks:
            summary = summaries.get(c.chunk_id, "")
            cleaned = c.cleaned_text()

            if not summary and not cleaned:
                skipped += 1
                continue

            # 拼接: summary 在前, cleaned_text 在后
            if summary:
                embed_text = f"{summary}\n\n{cleaned}"
            else:
                embed_text = cleaned

            texts.append(embed_text)
            valid_chunks.append(c)

        if skipped > 0:
            logger.warning(f"跳过 {skipped} 个空 chunk (无 summary 且无 cleaned_text)")

        if not texts:
            logger.warning("upsert_chunks: 无有效 chunk 可写入")
            return 0

        # 3. 批量 embed
        vectors = self.embed(texts)
        logger.info(f"Chunk embedding 完成: {len(vectors)} 个向量")

        # 4. 构建 Qdrant points
        points = []
        for chunk, vec in zip(valid_chunks, vectors):
            point_id = _stable_uuid(chunk.chunk_id)
            summary = summaries.get(chunk.chunk_id, "")
            payload = {
                "chunk_id": chunk.chunk_id,
                "session_id": chunk.session_id,
                "turn_index": chunk.turn_index,
                "task_id": chunk_to_task.get(chunk.chunk_id, ""),
                "summary": summary,
                "raw_size_tokens": chunk.raw_size_tokens,
                "cleaned_size_tokens": chunk.cleaned_size_tokens,
                "created_at": chunk.created_at or "",
            }
            points.append(rest.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            ))

        # 5. 批量 upsert
        self.client.upsert(
            collection_name=self.chunks_collection,
            points=points,
        )

        logger.info(f"Chunk upsert 完成: {len(points)} 条写入 '{self.chunks_collection}'")
        return len(points)

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict:
        """返回两个集合的统计信息"""
        stats = {}
        for name in [self.tasks_collection, self.chunks_collection]:
            try:
                info = self.client.get_collection(name)
                # 兼容新旧版 qdrant-client (vectors_count 在 v1.9+ 改名)
                vectors_count = getattr(info, "vectors_count", None)
                if vectors_count is None:
                    # 新版: 从 config 或 points_count 推断
                    vectors_count = getattr(info, "points_count", "?")
                stats[name] = {
                    "vectors_count": vectors_count,
                    "points_count": getattr(info, "points_count", "?"),
                    "status": str(info.status),
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        return stats
