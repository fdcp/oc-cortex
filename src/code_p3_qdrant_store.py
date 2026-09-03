"""
Phase 3 核心模块: Qdrant 三集合向量存储 + 混合检索
- tasks 集合: task_summary 向量化 (dense only)
 - chunks_summary 集合: chunk summary 向量化 (dense + sparse)
 - chunks_cleaned_text 集合: cleaned_text 向量化 (dense + BM25)
 - 混合检索: dense (summary) + sparse (summary/cleaned_text) + RRF 融合
"""
import os
import platform
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from code_p2_models import Task
from code_p1_models import Chunk


# ============================================================
# Qdrant 客户端工厂 (embedded path vs server url)
# ============================================================

def create_qdrant_client(
    qdrant_path: str = "./qdrant_data",
    qdrant_url: Optional[str] = None,
):
    """创建 Qdrant 客户端, server 模式优先。

    - qdrant_url 非空 → server 模式 `QdrantClient(url=...)`, 支持多实例并发。
    - 否则 → embedded 模式 `QdrantClient(path=...)`, 单进程独占目录。
    """
    from qdrant_client import QdrantClient

    url = (qdrant_url or "").strip() or None
    if url:
        logger.info(f"初始化 Qdrant 客户端 (server 模式): {url}")
        return QdrantClient(url=url)
    logger.info(f"初始化 Qdrant 客户端 (embedded 模式): {qdrant_path}")
    return QdrantClient(path=qdrant_path)


# ============================================================
# 检索结果
# ============================================================

@dataclass
class SearchResult:
    """单条检索结果"""
    point_id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class HybridResult:
    """混合检索结果 (dense + sparse RRF 融合)"""
    point_id: str
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    payload: dict = field(default_factory=dict)


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
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("检测到 MPS (Apple Silicon), 使用 mps 设备")
                return "mps"
        except ImportError:
            pass
        logger.info("macOS 环境, 使用 cpu 设备")
        return "cpu"

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
    """从字符串生成稳定的 UUID (基于 MD5), 用于 Qdrant point id"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ============================================================
# Phase3Store
# ============================================================

class Phase3Store:
    """
    Qdrant 三集合向量存储 + 混合检索
    封装 QdrantClient + SentenceTransformer + BM25/BGE-M3
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 512,
        batch_size: int = 32,
        device: str = "auto",
        qdrant_path: str = "./qdrant_data",
        qdrant_url: Optional[str] = None,
        tasks_collection: str = "tasks",
        chunks_summary_collection: str = "chunks_summary",
        chunks_cleaned_text_collection: str = "chunks_cleaned_text",
        sparse_method: str = "bm25",
        jieba_mode: str = "search",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        fuse_k: int = 60,
        bge_m3_model: Optional[str] = None,
        cache_folder: Optional[str] = None,
        offline_mode: bool = True,
    ):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.tasks_collection = tasks_collection
        self.chunks_summary_collection = chunks_summary_collection
        self.chunks_cleaned_text_collection = chunks_cleaned_text_collection
        self.sparse_method = sparse_method
        self.fuse_k = fuse_k

        # 缓存目录配置
        if cache_folder is None:
            cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
        self.cache_folder = cache_folder
        self.offline_mode = offline_mode

        # 1. 确定设备
        self.device = _detect_device(device)
        logger.info(f"Embedding 设备: {self.device}")

        # 2. 加载 dense embedding 模型
        logger.info(f"加载 dense 模型: {model_name}")
        try:
            self.encoder = SentenceTransformer(
                model_name, device=self.device, cache_folder=cache_folder
            )
        except Exception as e:
            if offline_mode and ("not found" in str(e).lower() or "no such file" in str(e).lower()):
                logger.warning(f"本地缓存未找到模型 {model_name}, 切换到在线模式下载")
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self.encoder = SentenceTransformer(
                    model_name, device=self.device, cache_folder=cache_folder
                )
            else:
                raise
        logger.info("Dense 模型加载完成")

        # 3. 初始化 Qdrant 客户端 (url 优先, 为空则用本地 path)
        self.qdrant_path = qdrant_path
        self.qdrant_url = (qdrant_url or "").strip() or None
        self.is_server_mode = self.qdrant_url is not None
        self.client = create_qdrant_client(qdrant_path, qdrant_url)

        # 4. 稀疏检索初始化
        self._bm25_index: dict[str, object] = {}   # collection -> BM25Okapi
        self._bm25_docs: dict[str, list[dict]] = {} # collection -> [{point_id, payload}]
        self._sparse_encoder: Optional[SentenceTransformer] = None
        self._m3_tokenizer = None
        self._m3_auto_model = None
        self._m3_device_str: str = "cpu"

        if sparse_method == "bm25":
            self._init_bm25(jieba_mode, bm25_k1, bm25_b)
        elif sparse_method == "bge_m3":
            self._init_bge_m3(bge_m3_model, device)
        else:
            logger.warning(f"未知 sparse_method: {sparse_method}, 仅使用 dense 检索")

    # --------------------------------------------------------
    # BM25 初始化
    # --------------------------------------------------------

    @staticmethod
    def _patch_pkg_resources() -> None:
        """
        修复 jieba 与 setuptools >= 80 的兼容性问题
        setuptools 80+ 移除了 pkg_resources.resource_stream,
        jieba 依赖它加载字典, 需要手动 patch
        """
        import os
        import pkg_resources

        if hasattr(pkg_resources, "resource_stream"):
            return

        import importlib

        def _resource_stream(package, resource_name):
            try:
                mod = importlib.import_module(package)
                base = os.path.dirname(os.path.abspath(mod.__file__))
                path = os.path.join(base, resource_name)
                if os.path.exists(path):
                    return open(path, "rb")
            except Exception:
                pass
            raise FileNotFoundError(
                f"{resource_name} not found in {package}"
            )

        pkg_resources.resource_stream = _resource_stream

    def _init_bm25(self, jieba_mode: str, k1: float, b: float) -> None:
        """初始化 jieba 分词器"""
        self._patch_pkg_resources()
        import jieba
        self._jieba = jieba
        self._jieba_mode = jieba_mode
        self._bm25_k1 = k1
        self._bm25_b = b
        jieba.setLogLevel(20)  # WARNING
        logger.info(f"BM25 稀疏检索: jieba mode={jieba_mode}, k1={k1}, b={b}")

    def _tokenize(self, text: str) -> list[str]:
        """jieba 分词"""
        if self._jieba_mode == "search":
            return list(self._jieba.cut_for_search(text))
        return list(self._jieba.cut(text))

    def build_bm25_index(
        self,
        collection: str,
        texts: list[str],
        point_ids: list[str],
        payloads: list[dict],
    ) -> None:
        """
        构建 BM25 索引 (in-memory)
        - texts: 原始文本 (会被 jieba 分词)
        - point_ids: 对应的 Qdrant point id
        - payloads: 对应的 payload
        """
        from rank_bm25 import BM25Okapi

        logger.info(f"构建 BM25 索引: {collection}, {len(texts)} 条文档")
        t0 = time.time()

        tokenized = [self._tokenize(t) for t in texts]
        valid_idx = [i for i, tokens in enumerate(tokenized) if tokens]
        if len(valid_idx) < len(tokenized):
            logger.warning(f"跳过 {len(tokenized) - len(valid_idx)} 条空文档")
            tokenized = [tokenized[i] for i in valid_idx]
            point_ids = [point_ids[i] for i in valid_idx]
            payloads = [payloads[i] for i in valid_idx]

        self._bm25_index[collection] = BM25Okapi(
            tokenized, k1=self._bm25_k1, b=self._bm25_b
        )
        self._bm25_docs[collection] = [
            {"point_id": pid, "payload": pl}
            for pid, pl in zip(point_ids, payloads)
        ]
        logger.info(
            f"BM25 索引完成: {len(valid_idx)} 条文档, "
            f"耗时 {time.time() - t0:.2f}s"
        )

    # --------------------------------------------------------
    # BGE-M3 初始化
    # --------------------------------------------------------

    def _init_bge_m3(self, bge_m3_model: Optional[str], device: str) -> None:
        """加载 BGE-M3 模型 (用于 sparse vectors)

        sentence-transformers 版本的 BGE-M3 不支持 sparse_vec 输出，
        所以我们额外加载 tokenizer + auto_model，通过 token embedding
        的 L2 norm 手动计算 sparse weights。
        """
        model_name = bge_m3_model or "BAAI/bge-m3"
        logger.info(f"加载 BGE-M3 模型: {model_name}")
        m3_device = _detect_device(device)
        try:
            self._sparse_encoder = SentenceTransformer(
                model_name, device=m3_device, trust_remote_code=True,
                cache_folder=self.cache_folder
            )
            # 额外加载 tokenizer 和获取 auto_model 引用，用于 sparse 计算
            from transformers import AutoTokenizer
            self._m3_tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_folder
            )
        except Exception as e:
            if self.offline_mode and ("not found" in str(e).lower() or "no such file" in str(e).lower()):
                logger.warning(f"本地缓存未找到 BGE-M3 模型, 切换到在线模式下载")
                import os
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self._sparse_encoder = SentenceTransformer(
                    model_name, device=m3_device, trust_remote_code=True,
                    cache_folder=self.cache_folder
                )
                from transformers import AutoTokenizer
                self._m3_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=self.cache_folder
                )
            else:
                raise
        self._m3_auto_model = self._sparse_encoder._first_module().auto_model
        self._m3_device_str = m3_device
        logger.info("BGE-M3 模型加载完成 (含 tokenizer + auto_model)")

    def _encode_sparse_bge_m3(self, texts: list[str]) -> list[rest.SparseVector]:
        """使用 BGE-M3 token embeddings 计算 sparse vectors

        sentence-transformers 版本不提供 sparse_vec 输出，所以我们：
        1. 用 BGE-M3 tokenizer 分词
        2. 通过 transformer forward pass 获取 token embeddings
        3. 计算每个 token embedding 的 L2 norm 作为权重
        4. 按 token ID max-pool (同一 token 取最大权重)
        5. 过滤 special tokens
        """
        import torch

        if not self._m3_tokenizer or not self._m3_auto_model:
            raise RuntimeError("BGE-M3 tokenizer/auto_model 未加载")

        all_sparse = []
        device = self._m3_device_str

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Tokenize
            encoded = self._m3_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass to get token embeddings
            with torch.no_grad():
                outputs = self._m3_auto_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                token_embeds = outputs.last_hidden_state  # [batch, seq_len, dim]

            # Compute sparse weights per text
            for b_idx in range(len(batch)):
                ids = input_ids[b_idx]
                mask = attention_mask[b_idx]
                embeds = token_embeds[b_idx]

                sparse_weights = {}
                for t_idx in range(len(ids)):
                    if mask[t_idx] == 0:
                        continue
                    tid = ids[t_idx].item()
                    # Skip special tokens: <s>=0, <pad>=1, </s>=2, <unk>=3
                    if tid <= 3:
                        continue
                    weight = torch.norm(embeds[t_idx]).item()
                    if tid not in sparse_weights or weight > sparse_weights[tid]:
                        sparse_weights[tid] = weight

                indices = sorted(sparse_weights.keys())
                values = [float(sparse_weights[idx]) for idx in indices]
                all_sparse.append(rest.SparseVector(
                    indices=[int(x) for x in indices],
                    values=values,
                ))

        return all_sparse

    # --------------------------------------------------------
    # 集合初始化
    # --------------------------------------------------------

    def init_collections(self) -> None:
        """创建或验证 tasks / chunks_summary / chunks_cleaned_text 集合"""
        for name in [
            self.tasks_collection,
            self.chunks_summary_collection,
            self.chunks_cleaned_text_collection,
        ]:
            existing = [c.name for c in self.client.get_collections().collections]
            if name not in existing:
                if self.sparse_method == "bge_m3":
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config={
                            "dense": rest.VectorParams(
                                size=self.dim,
                                distance=rest.Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={
                            "sparse": rest.SparseVectorParams(
                                index=rest.SparseIndexParams(
                                    on_disk=False,
                                ),
                            ),
                        },
                    )
                    logger.info(f"创建集合: {name} (dense dim={self.dim} + sparse)")
                else:
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
        """批量 embed 文本, 返回 list[list[float]]"""
        if not texts:
            return []

        total = len(texts)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info(
            f"开始 embedding: {total} 条文本, "
            f"{n_batches} 个 batch (batch_size={self.batch_size})"
        )

        all_vectors = []
        t_total = time.time()
        for batch_idx, i in enumerate(range(0, total, self.batch_size), 1):
            batch = texts[i : i + self.batch_size]
            t_batch = time.time()
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
        logger.info(
            f"Embedding 完成: {len(all_vectors)} 个向量, "
            f"总耗时 {total_elapsed:.1f}s"
        )
        return all_vectors

    # --------------------------------------------------------
    # Task upsert
    # --------------------------------------------------------

    def upsert_tasks(self, tasks: list[Task]) -> int:
        """将 Task 列表写入 tasks 集合"""
        if not tasks:
            logger.warning("upsert_tasks: 无 task 可写入")
            return 0

        logger.info(f"准备 upsert {len(tasks)} 个 task ...")
        texts = [t.task_summary for t in tasks]
        vectors = self.embed(texts)
        logger.info(f"Task embedding 完成: {len(vectors)} 个向量")

        sparse_vectors = None
        if self.sparse_method == "bge_m3":
            sparse_vectors = self._encode_sparse_bge_m3(texts)
            logger.info(f"Task sparse embedding (BGE-M3) 完成: {len(sparse_vectors)}")

        points = []
        for idx, (task, vec) in enumerate(zip(tasks, vectors)):
            point_id = _stable_uuid(task.task_id)
            payload = {
                "task_id": task.task_id,
                "session_id": task.session_id,
                "task_label": task.task_label,
                "task_summary": task.task_summary,
                "chunk_ids": task.chunk_ids,
                "created_at": task.created_at,
            }

            if self.sparse_method == "bge_m3":
                points.append(rest.PointStruct(
                    id=point_id,
                    vector={"dense": vec, "sparse": sparse_vectors[idx]},
                    payload=payload,
                ))
            else:
                points.append(rest.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                ))

        self.client.upsert(
            collection_name=self.tasks_collection,
            points=points,
        )
        logger.info(f"Task upsert 完成: {len(points)} 条写入 '{self.tasks_collection}'")

        return len(points)

    # --------------------------------------------------------
    # Chunk upsert
    # --------------------------------------------------------

    def upsert_chunks_summary(
        self,
        chunks: list[Chunk],
        summaries: dict[str, str],
        tasks: Optional[list[Task]] = None,
    ) -> int:
        """将 Chunk summary 写入 chunks_summary 集合并建立 sparse 索引。"""
        if not chunks:
            logger.warning("upsert_chunks_summary: 无 chunk 可写入")
            return 0

        logger.info(f"准备 upsert {len(chunks)} 个 chunk summary ...")

        chunk_to_task: dict[str, str] = {}
        if tasks:
            for t in tasks:
                for cid in t.chunk_ids:
                    chunk_to_task[cid] = t.task_id

        texts = []
        valid_chunks = []
        skipped = 0

        for c in chunks:
            summary = summaries.get(c.chunk_id, "")
            if not summary:
                skipped += 1
                continue
            texts.append(summary)
            valid_chunks.append(c)

        if skipped > 0:
            logger.warning(
                f"跳过 {skipped} 个无 summary 的 chunk"
            )
        if not texts:
            logger.warning("upsert_chunks_summary: 无有效 summary 可写入")
            return 0

        vectors = self.embed(texts)
        logger.info(f"Chunks summary embedding 完成: {len(vectors)} 个向量")

        sparse_vectors = None
        if self.sparse_method == "bge_m3":
            sparse_vectors = self._encode_sparse_bge_m3(texts)
            logger.info(
                f"Chunks summary sparse embedding (BGE-M3) 完成: {len(sparse_vectors)}"
            )

        points = []
        for idx, (chunk, vec) in enumerate(zip(valid_chunks, vectors)):
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

            if self.sparse_method == "bge_m3":
                points.append(rest.PointStruct(
                    id=point_id,
                    vector={"dense": vec, "sparse": sparse_vectors[idx]},
                    payload=payload,
                ))
            else:
                points.append(rest.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                ))

        self.client.upsert(
            collection_name=self.chunks_summary_collection,
            points=points,
        )
        logger.info(
            f"Chunks summary upsert 完成: {len(points)} 条写入 "
            f"'{self.chunks_summary_collection}'"
        )

        if self.sparse_method == "bm25":
            self.build_bm25_index(
                self.chunks_summary_collection,
                texts,
                [_stable_uuid(c.chunk_id) for c in valid_chunks],
                [
                    {
                        "chunk_id": c.chunk_id,
                        "session_id": c.session_id,
                        "turn_index": c.turn_index,
                        "task_id": chunk_to_task.get(c.chunk_id, ""),
                        "summary": summaries.get(c.chunk_id, ""),
                    }
                    for c in valid_chunks
                ],
            )

        return len(points)

    def upsert_chunks_cleaned_text(
        self,
        chunks: list[Chunk],
        tasks: Optional[list[Task]] = None,
    ) -> int:
        """将 Chunk cleaned_text 写入 chunks_cleaned_text 集合 (dense + BM25)"""
        if not chunks:
            logger.warning("upsert_chunks_cleaned_text: 无 chunk 可写入")
            return 0

        logger.info(f"准备 upsert {len(chunks)} 个 chunk cleaned_text ...")

        chunk_to_task: dict[str, str] = {}
        if tasks:
            for t in tasks:
                for cid in t.chunk_ids:
                    chunk_to_task[cid] = t.task_id

        texts = []
        valid_chunks = []
        skipped = 0

        for c in chunks:
            cleaned = c.cleaned_text()
            if not cleaned:
                skipped += 1
                continue
            texts.append(cleaned)
            valid_chunks.append(c)

        if skipped > 0:
            logger.warning(
                f"跳过 {skipped} 个空 chunk (无 cleaned_text)"
            )
        if not texts:
            logger.warning("upsert_chunks_cleaned_text: 无有效 cleaned_text 可写入")
            return 0

        vectors = self.embed(texts)
        logger.info(f"Chunks cleaned_text embedding 完成: {len(vectors)} 个向量")

        sparse_vectors = None
        if self.sparse_method == "bge_m3":
            sparse_vectors = self._encode_sparse_bge_m3(texts)
            logger.info(
                f"Chunks cleaned_text sparse embedding (BGE-M3) 完成: "
                f"{len(sparse_vectors)}"
            )

        points = []
        payload_list = []
        for idx, (chunk, vec) in enumerate(zip(valid_chunks, vectors)):
            point_id = _stable_uuid(chunk.chunk_id)
            payload = {
                "chunk_id": chunk.chunk_id,
                "session_id": chunk.session_id,
                "turn_index": chunk.turn_index,
                "task_id": chunk_to_task.get(chunk.chunk_id, ""),
                "raw_size_tokens": chunk.raw_size_tokens,
                "cleaned_size_tokens": chunk.cleaned_size_tokens,
                "created_at": chunk.created_at or "",
            }
            payload_list.append(payload)

            if self.sparse_method == "bge_m3":
                points.append(rest.PointStruct(
                    id=point_id,
                    vector={"dense": vec, "sparse": sparse_vectors[idx]},
                    payload=payload,
                ))
            else:
                points.append(rest.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                ))

        self.client.upsert(
            collection_name=self.chunks_cleaned_text_collection,
            points=points,
        )
        logger.info(
            f"Chunks cleaned_text upsert 完成: {len(points)} 条写入 "
            f"'{self.chunks_cleaned_text_collection}'"
        )

        # BM25 索引 (cleaned_text)
        if self.sparse_method == "bm25":
            pids = [_stable_uuid(c.chunk_id) for c in valid_chunks]
            self.build_bm25_index(
                self.chunks_cleaned_text_collection, texts, pids, payload_list
            )

        return len(points)

    # --------------------------------------------------------
    # 检索: Dense
    # --------------------------------------------------------

    def search_dense(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """稠密检索 (embedding cosine similarity)"""
        collection = collection or self.chunks_summary_collection
        query_vec = self.embed([query])[0]

        using = "dense" if self.sparse_method == "bge_m3" else None
        hits = self.client.query_points(
            collection_name=collection,
            query=query_vec,
            using=using,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in hits.points:
            results.append(SearchResult(
                point_id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            ))
        return results

    # --------------------------------------------------------
    # 检索: Sparse (BM25)
    # --------------------------------------------------------

    def search_sparse_bm25(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """BM25 稀疏检索 (in-memory)"""
        collection = collection or self.chunks_cleaned_text_collection

        if collection not in self._bm25_index:
            logger.warning(f"BM25 索引不存在: {collection}")
            return []

        bm25 = self._bm25_index[collection]
        docs = self._bm25_docs[collection]

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        scored_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in scored_idx:
            if scores[i] <= 0:
                break
            doc = docs[i]
            results.append(SearchResult(
                point_id=doc["point_id"],
                score=float(scores[i]),
                payload=doc["payload"],
            ))
        return results

    def search_sparse_bm25_tokens(
        self,
        tokens: list[str],
        collection: Optional[str] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """BM25 检索, 接受预分词 tokens (跳过 jieba 重切).

        用于 P4 alias expansion: caller 已经把 query + extra terms 拼成 token 列表,
        这里直接调 bm25.get_scores(tokens), 避免 jieba 二次切词引入冗余/重复.
        """
        collection = collection or self.chunks_cleaned_text_collection

        if collection not in self._bm25_index:
            logger.warning(f"BM25 索引不存在: {collection}")
            return []

        if not tokens:
            return []

        bm25 = self._bm25_index[collection]
        docs = self._bm25_docs[collection]

        scores = bm25.get_scores(tokens)

        scored_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for i in scored_idx:
            if scores[i] <= 0:
                break
            doc = docs[i]
            results.append(SearchResult(
                point_id=doc["point_id"],
                score=float(scores[i]),
                payload=doc["payload"],
            ))
        return results

    # --------------------------------------------------------
    # 检索: Sparse (BGE-M3)
    # --------------------------------------------------------

    def search_sparse_bge_m3(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """BGE-M3 稀疏检索 (Qdrant sparse vector search)"""
        collection = collection or self.chunks_summary_collection

        sparse_vec = self._encode_sparse_bge_m3([query])[0]

        hits = self.client.query_points(
            collection_name=collection,
            query=sparse_vec,
            using="sparse",
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in hits.points:
            results.append(SearchResult(
                point_id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            ))
        return results

    # --------------------------------------------------------
    # 检索: Hybrid (RRF 融合)
    # --------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
    ) -> list[HybridResult]:
        """
        混合检索: dense + sparse + RRF 融合
        RRF_score(d) = sum(1 / (k + rank_i(d)))
        """
        collection = collection or self.chunks_summary_collection
        logger.info(
            f"混合检索 [{collection}]: query='{query[:40]}...', "
            f"dense_top={dense_top_k}, sparse_top={sparse_top_k}, "
            f"fuse_k={self.fuse_k}"
        )

        # 1. Dense search
        t0 = time.time()
        dense_results = self.search_dense(query, collection, dense_top_k)
        t_dense = time.time() - t0
        logger.info(f"  Dense: {len(dense_results)} 结果, {t_dense:.2f}s")

        # 2. Sparse search
        t1 = time.time()
        if self.sparse_method == "bm25":
            sparse_results = self.search_sparse_bm25(
                query, collection, sparse_top_k
            )
        elif self.sparse_method == "bge_m3":
            sparse_results = self.search_sparse_bge_m3(
                query, collection, sparse_top_k
            )
        else:
            sparse_results = []
        t_sparse = time.time() - t1
        logger.info(
            f"  Sparse ({self.sparse_method}): "
            f"{len(sparse_results)} 结果, {t_sparse:.2f}s"
        )

        # 3. RRF 融合
        t2 = time.time()
        fused = self._rrf_fuse(dense_results, sparse_results, self.fuse_k)
        t_fuse = time.time() - t2
        logger.info(f"  RRF 融合: {len(fused)} 结果, {t_fuse:.3f}s")

        return fused[:top_k]

    def search_hybrid_cross_collection(
        self,
        query: str,
        dense_collection: Optional[str] = None,
        sparse_collection: Optional[str] = None,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
    ) -> list[HybridResult]:
        """
        跨集合混合检索: dense 搜一个集合, sparse 搜另一个集合, RRF 融合
        典型用法: dense → chunks_summary (语义匹配), sparse → chunks_cleaned_text (关键词匹配)
        """
        dense_collection = dense_collection or self.chunks_summary_collection
        sparse_collection = sparse_collection or self.chunks_cleaned_text_collection

        logger.info(
            f"跨集合混合检索: query='{query[:40]}...', "
            f"dense[{dense_collection}], sparse[{sparse_collection}], "
            f"dense_top={dense_top_k}, sparse_top={sparse_top_k}"
        )

        # 1. Dense search
        t0 = time.time()
        dense_results = self.search_dense(query, dense_collection, dense_top_k)
        t_dense = time.time() - t0
        logger.info(f"  Dense [{dense_collection}]: {len(dense_results)} 结果, {t_dense:.2f}s")

        # 2. Sparse search
        t1 = time.time()
        if self.sparse_method == "bm25":
            sparse_results = self.search_sparse_bm25(
                query, sparse_collection, sparse_top_k
            )
        elif self.sparse_method == "bge_m3":
            sparse_results = self.search_sparse_bge_m3(
                query, sparse_collection, sparse_top_k
            )
        else:
            sparse_results = []
        t_sparse = time.time() - t1
        logger.info(
            f"  Sparse ({self.sparse_method}) [{sparse_collection}]: "
            f"{len(sparse_results)} 结果, {t_sparse:.2f}s"
        )

        # 3. RRF 融合
        t2 = time.time()
        fused = self._rrf_fuse(dense_results, sparse_results, self.fuse_k)
        t_fuse = time.time() - t2
        logger.info(f"  RRF 融合: {len(fused)} 结果, {t_fuse:.3f}s")

        return fused[:top_k]

    def _rrf_fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        k: int = 60,
    ) -> list[HybridResult]:
        """
        Reciprocal Rank Fusion
        RRF_score(d) = sum(1 / (k + rank_i(d)))
        rank 从 1 开始
        """
        scores: dict[str, dict] = {}

        for rank, r in enumerate(dense_results, 1):
            pid = r.point_id
            if pid not in scores:
                scores[pid] = {"rrf": 0.0, "payload": r.payload}
            scores[pid]["rrf"] += 1.0 / (k + rank)
            scores[pid]["dense_rank"] = rank
            scores[pid]["dense_score"] = r.score

        for rank, r in enumerate(sparse_results, 1):
            pid = r.point_id
            if pid not in scores:
                scores[pid] = {"rrf": 0.0, "payload": r.payload}
            scores[pid]["rrf"] += 1.0 / (k + rank)
            scores[pid]["sparse_rank"] = rank
            scores[pid]["sparse_score"] = r.score

        sorted_items = sorted(
            scores.items(), key=lambda x: x[1]["rrf"], reverse=True
        )

        results = []
        for pid, info in sorted_items:
            results.append(HybridResult(
                point_id=pid,
                rrf_score=info["rrf"],
                dense_rank=info.get("dense_rank"),
                dense_score=info.get("dense_score"),
                sparse_rank=info.get("sparse_rank"),
                sparse_score=info.get("sparse_score"),
                payload=info["payload"],
            ))
        return results

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def get_stats(self) -> dict:
        """返回三个集合的统计信息"""
        stats = {}
        for name in [
            self.tasks_collection,
            self.chunks_summary_collection,
            self.chunks_cleaned_text_collection,
        ]:
            try:
                info = self.client.get_collection(name)
                vectors_count = getattr(info, "vectors_count", None)
                if vectors_count is None:
                    vectors_count = getattr(info, "points_count", "?")
                stats[name] = {
                    "vectors_count": vectors_count,
                    "points_count": getattr(info, "points_count", "?"),
                    "status": str(info.status),
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        return stats
