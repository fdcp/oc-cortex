"""
Phase 3 混合检索演示脚本
用法:
  # 全量数据 bge-small-zh + BM25
  python code_p3_search_demo.py --mode all

  # 1/5 数据 Qwen3 + BGE-M3 (示例)
  python code_p3_search_demo.py --mode sample

  # 自定义查询
  python code_p3_search_demo.py --mode all --query "GPU对比分析"
"""
import argparse
import json
import sys
import time
import random
import os
from pathlib import Path

# 加载配置并设置 HuggingFace 环境 (必须在其他导入之前)
from code_p1_utils import Config
_args = argparse.ArgumentParser(add_help=False)
_args.add_argument("--config", default="code_p3_config.yaml")
_temp_args, _ = _args.parse_known_args()
_config = Config.load(_temp_args.config)

from code_p3_hf_config import setup_hf_env
_cache_folder = _config.get("embedding.cache_folder")
if _cache_folder is None:
    _cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
_offline_mode = _config.get("embedding.offline_mode", True)
setup_hf_env(offline_mode=_offline_mode, cache_folder=_cache_folder)

from loguru import logger
from code_p1_utils import setup_logger
from code_p1_models import Chunk, CleanedToolCall
from code_p2_models import Task
from code_p3_qdrant_store import Phase3Store, SearchResult, HybridResult


# ============================================================
# 数据加载 (复用 code_p3_main.py 的逻辑)
# ============================================================

def load_tasks(path: str) -> list[Task]:
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


def load_chunks(path: str) -> list[Chunk]:
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


def load_summaries(path: str) -> dict[str, str]:
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
# 结果格式化
# ============================================================

def fmt_search_results(results: list[SearchResult], title: str) -> str:
    """格式化单路检索结果"""
    lines = [f"\n{'=' * 50}", f"  {title}", f"{'=' * 50}"]
    for i, r in enumerate(results, 1):
        label = r.payload.get("task_label") or r.payload.get("summary", "")[:60]
        chunk_id = r.payload.get("chunk_id", "")
        task_id = r.payload.get("task_id", "")
        ident = chunk_id or task_id
        lines.append(
            f"  #{i:2d}  score={r.score:.4f}  "
            f"[{ident[:40]}]  {label}"
        )
    return "\n".join(lines)


def fmt_hybrid_results(results: list[HybridResult], title: str) -> str:
    """格式化混合检索结果"""
    lines = [f"\n{'=' * 50}", f"  {title}", f"{'=' * 50}"]
    for i, r in enumerate(results, 1):
        label = r.payload.get("task_label") or r.payload.get("summary", "")[:60]
        chunk_id = r.payload.get("chunk_id", "")
        task_id = r.payload.get("task_id", "")
        ident = chunk_id or task_id

        d_info = f"D#{r.dense_rank}({r.dense_score:.3f})" if r.dense_rank else "D-"
        s_info = f"S#{r.sparse_rank}({r.sparse_score:.3f})" if r.sparse_rank else "S-"

        lines.append(
            f"  #{i:2d}  RRF={r.rrf_score:.4f}  "
            f"{d_info} {s_info}  "
            f"[{ident[:40]}]  {label}"
        )
    return "\n".join(lines)


# ============================================================
# 示例查询
# ============================================================

DEMO_QUERIES = [
    "GPU对比分析 A800 H100 H800",
    "优化器学习率调度",
    "session管理 数据库查询",
    "旋转位置编码 RoPE",
    "文档升级 markdown",
]


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 混合检索演示")
    parser.add_argument("--config", default="code_p3_config.yaml")
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "sample"],
        help="all=全量 bge+bm25, sample=1/5数据 Qwen3+BGE-M3"
    )
    parser.add_argument("--query", default=None, help="自定义查询 (覆盖示例)")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    args = parser.parse_args()

    # 1. 加载配置 (已在模块级别完成)
    config = _config
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    if _offline_mode:
        logger.info(f"离线模式: 仅使用本地缓存 ({_cache_folder})")

    # 2. 确定模式参数
    if args.mode == "sample":
        # Qwen3 + BGE-M3 模式
        dense_model = "Qwen/Qwen3-Embedding-0.6B"
        dim = 1024
        sparse_method = "bge_m3"
        bge_m3_model = config.get("sparse.bge_m3_model", "BAAI/bge-m3")
        batch_size = 4
        qdrant_path = "./qdrant_data_sample"
        sample_ratio = 1 / 5
        logger.info(f"模式: sample (Qwen3 + BGE-M3, 1/5 数据)")
    else:
        # bge-small-zh + BM25 模式 (使用 config 中的配置)
        dense_model = config.get("embedding.model", "BAAI/bge-small-zh-v1.5")
        dim = config.get("embedding.dim", 512)
        sparse_method = config.get("sparse.method", "bm25")
        bge_m3_model = config.get("sparse.bge_m3_model", "BAAI/bge-m3")
        batch_size = config.get("embedding.batch_size", 32)
        qdrant_path = config.get("qdrant.path", "./qdrant_data")
        sample_ratio = 1.0
        logger.info(f"模式: all ({dense_model} + BM25, 全量数据)")

    # 3. 加载数据
    tasks_file = config.get("phase2.tasks_file", "./output/tasks.jsonl")
    chunks_file = config.get("phase1.chunks_file", "./output/chunks.jsonl")
    summaries_file = config.get("phase2.chunk_summaries_file", "./output/chunks_summary_p2.jsonl")

    tasks = load_tasks(tasks_file)
    chunks = load_chunks(chunks_file)
    summaries = load_summaries(summaries_file)
    logger.info(f"加载: {len(tasks)} tasks, {len(chunks)} chunks, {len(summaries)} summaries")

    # 4. 采样 (sample 模式)
    if sample_ratio < 1.0:
        n_tasks = max(1, int(len(tasks) * sample_ratio))
        n_chunks = max(1, int(len(chunks) * sample_ratio))
        random.seed(42)
        tasks = random.sample(tasks, n_tasks)
        chunks = random.sample(chunks, n_chunks)
        logger.info(f"采样后: {len(tasks)} tasks, {len(chunks)} chunks")

    # 5. 初始化 Phase3Store
    t0 = time.time()
    store = Phase3Store(
        model_name=dense_model,
        dim=dim,
        batch_size=batch_size,
        device=config.get("embedding.device", "cpu"),
        qdrant_path=qdrant_path,
        tasks_collection=config.get("qdrant.collections.tasks", "tasks"),
        chunks_summary_collection=config.get("qdrant.collections.chunks_summary", "chunks_summary"),
        chunks_cleaned_text_collection=config.get("qdrant.collections.chunks_cleaned_text", "chunks_cleaned_text"),
        sparse_method=sparse_method,
        jieba_mode=config.get("sparse.jieba_mode", "search"),
        bm25_k1=config.get("sparse.bm25_params.k1", 1.5),
        bm25_b=config.get("sparse.bm25_params.b", 0.75),
        fuse_k=config.get("sparse.fuse_k", 60),
        bge_m3_model=bge_m3_model if sparse_method == "bge_m3" else None,
        cache_folder=config.get("embedding.cache_folder"),
        offline_mode=config.get("embedding.offline_mode", True),
    )

    store.init_collections()

    # 6. Upsert
    logger.info("-" * 40)
    logger.info("开始 upsert ...")
    store.upsert_tasks(tasks)
    store.upsert_chunks_summary(chunks, summaries, tasks=tasks)
    store.upsert_chunks_cleaned_text(chunks, tasks=tasks)

    stats = store.get_stats()
    elapsed = time.time() - t0
    logger.info(f"Upsert 完成, 总耗时: {elapsed:.1f}s")
    for name, info in stats.items():
        if "error" not in info:
            logger.info(f"  {name}: points={info['points_count']}")

    # 7. 运行检索演示
    queries = [args.query] if args.query else DEMO_QUERIES

    logger.info("\n" + "=" * 60)
    logger.info("混合检索演示")
    logger.info("=" * 60)

    for query in queries:
        logger.info(f"\n>>> 查询: {query}")

        # Dense only (on chunks_summary)
        dense_results = store.search_dense(
            query, collection=store.chunks_summary_collection, top_k=args.top_k
        )
        print(fmt_search_results(
            dense_results,
            f"Dense 检索 ({dense_model}) → {store.chunks_summary_collection}"
        ))

        # Sparse only (on chunks_cleaned_text)
        if sparse_method == "bm25":
            sparse_results = store.search_sparse_bm25(
                query, collection=store.chunks_cleaned_text_collection, top_k=args.top_k
            )
            print(fmt_search_results(
                sparse_results,
                f"Sparse 检索 (BM25) → {store.chunks_cleaned_text_collection}"
            ))
        else:
            sparse_results = store.search_sparse_bge_m3(
                query, collection=store.chunks_cleaned_text_collection, top_k=args.top_k
            )
            print(fmt_search_results(
                sparse_results,
                f"Sparse 检索 (BGE-M3) → {store.chunks_cleaned_text_collection}"
            ))

        # Hybrid (cross-collection: dense → summary, sparse → cleaned_text)
        hybrid_results = store.search_hybrid_cross_collection(
            query,
            dense_collection=store.chunks_summary_collection,
            sparse_collection=store.chunks_cleaned_text_collection,
            top_k=args.top_k,
        )
        print(fmt_hybrid_results(
            hybrid_results,
            f"Hybrid 跨集合检索 (RRF, k={store.fuse_k}): "
            f"dense[{store.chunks_summary_collection}] + "
            f"sparse[{store.chunks_cleaned_text_collection}]"
        ))

    logger.info("\n" + "=" * 60)
    logger.info("演示完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
