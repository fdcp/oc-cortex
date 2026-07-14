"""
Phase 3 主入口
流程: 加载 Phase 1/2 输出 -> Embedding -> Qdrant 双集合 upsert

用法:
  python code_p3_main.py
  python code_p3_main.py --config code_p3_config.yaml
  python code_p3_main.py --tasks ./output/tasks.jsonl --chunks ./output/chunks.jsonl
"""
import argparse
import json
import sys
import time
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
from code_p3_qdrant_store import Phase3Store


# ============================================================
# 数据加载
# ============================================================

def load_tasks_from_jsonl(path: str) -> list[Task]:
    """从 JSONL 文件加载 Task 对象"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task = Task.from_dict(data)
                tasks.append(task)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Task 解析失败 (行 {line_num}): {e}")
    return tasks


def load_chunks_from_jsonl(path: str) -> list[Chunk]:
    """从 JSONL 文件加载 Chunk 对象"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = Chunk(
                    chunk_id=data["chunk_id"],
                    session_id=data["session_id"],
                    turn_index=data["turn_index"],
                    user_message=data["user_message"],
                    assistant_messages=data.get("assistant_messages", []),
                    tool_calls=[
                        CleanedToolCall(**tc)
                        for tc in data.get("tool_calls", [])
                    ],
                    mcp_calls=[
                        CleanedToolCall(**mc)
                        for mc in data.get("mcp_calls", [])
                    ],
                    raw_size_tokens=data.get("raw_size_tokens", 0),
                    cleaned_size_tokens=data.get("cleaned_size_tokens", 0),
                    created_at=data.get("created_at"),
                    task_summary=data.get("task_summary"),
                )
                chunks.append(chunk)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Chunk 解析失败 (行 {line_num}): {e}")
    return chunks


def load_chunk_summaries(path: str) -> dict[str, str]:
    """从 chunks_summary_p2.jsonl 加载 {chunk_id: summary} 字典"""
    summaries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk_id = data["chunk_id"]
                summary = data["summary"]
                summaries[chunk_id] = summary
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Chunk summary 解析失败 (行 {line_num}): {e}")
    return summaries


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Qdrant 向量存储")
    parser.add_argument(
        "--config", default="code_p3_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Phase 2 tasks JSONL 文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--chunks", default=None,
        help="Phase 1 chunks JSONL 文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--summaries", default=None,
        help="Phase 2 chunk summaries JSONL 文件路径 (覆盖配置文件)"
    )
    args = parser.parse_args()

    # 1. 加载配置 (已在模块级别完成)
    config = _config
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    if _offline_mode:
        logger.info(f"离线模式: 仅使用本地缓存 ({_cache_folder})")

    logger.info("=" * 60)
    logger.info("Phase 3: Qdrant 双集合向量存储")
    logger.info("=" * 60)

    # 2. 加载 Phase 2 tasks
    tasks_file = args.tasks or config.get(
        "phase2.tasks_file", "./output/tasks.jsonl"
    )
    logger.info(f"加载 tasks: {tasks_file}")

    if not Path(tasks_file).exists():
        logger.error(f"Tasks 文件不存在: {tasks_file}")
        sys.exit(1)

    tasks = load_tasks_from_jsonl(tasks_file)
    logger.info(f"加载 {len(tasks)} 个 task")

    # 3. 加载 Phase 1 chunks
    chunks_file = args.chunks or config.get(
        "phase1.chunks_file", "./output/chunks.jsonl"
    )
    logger.info(f"加载 chunks: {chunks_file}")

    if not Path(chunks_file).exists():
        logger.error(f"Chunks 文件不存在: {chunks_file}")
        sys.exit(1)

    chunks = load_chunks_from_jsonl(chunks_file)
    logger.info(f"加载 {len(chunks)} 个 chunk")

    # 4. 加载 Phase 2 chunk summaries
    summaries_file = args.summaries or config.get(
        "phase2.chunk_summaries_file", "./output/chunks_summary_p2.jsonl"
    )
    logger.info(f"加载 chunk summaries: {summaries_file}")

    if not Path(summaries_file).exists():
        logger.error(f"Chunk summaries 文件不存在: {summaries_file}")
        sys.exit(1)

    summaries = load_chunk_summaries(summaries_file)
    logger.info(f"加载 {len(summaries)} 条 chunk summary")

    # 5. 检查数据完整性
    if not tasks and not chunks:
        logger.error("没有加载到任何 task 或 chunk, 请检查输入文件")
        sys.exit(1)

    # 6. 初始化 Phase3Store
    t0 = time.time()

    store = Phase3Store(
        model_name=config.get("embedding.model", "Qwen/Qwen3-Embedding-0.6B"),
        dim=config.get("embedding.dim", 1024),
        batch_size=config.get("embedding.batch_size", 16),
        device=config.get("embedding.device", "auto"),
        qdrant_path=config.get("qdrant.path", "./qdrant_data"),
        tasks_collection=config.get("qdrant.collections.tasks", "tasks"),
        chunks_collection=config.get("qdrant.collections.chunks", "chunks"),
        sparse_method=config.get("sparse.method", "bm25"),
        jieba_mode=config.get("sparse.jieba_mode", "search"),
        bm25_k1=config.get("sparse.bm25_params.k1", 1.5),
        bm25_b=config.get("sparse.bm25_params.b", 0.75),
        fuse_k=config.get("sparse.fuse_k", 60),
        bge_m3_model=config.get("sparse.bge_m3_model", "BAAI/bge-m3"),
        cache_folder=config.get("embedding.cache_folder"),
        offline_mode=config.get("embedding.offline_mode", True),
    )

    # 7. 初始化集合
    store.init_collections()

    # 8. Upsert tasks
    tasks_upserted = 0
    if tasks:
        logger.info("-" * 40)
        logger.info("开始 upsert tasks ...")
        t1 = time.time()
        tasks_upserted = store.upsert_tasks(tasks)
        logger.info(f"Task upsert 耗时: {time.time() - t1:.1f}s")
    else:
        logger.warning("无 task, 跳过 task upsert")

    # 9. Upsert chunks
    chunks_upserted = 0
    if chunks:
        logger.info("-" * 40)
        logger.info("开始 upsert chunks ...")
        t2 = time.time()
        chunks_upserted = store.upsert_chunks(chunks, summaries, tasks=tasks)
        logger.info(f"Chunk upsert 耗时: {time.time() - t2:.1f}s")
    else:
        logger.warning("无 chunk, 跳过 chunk upsert")

    elapsed_total = time.time() - t0

    # 10. 打印统计
    logger.info("-" * 40)
    stats = store.get_stats()
    logger.info("=" * 60)
    logger.info("Phase 3 完成")
    logger.info(f"  总耗时: {elapsed_total:.1f}s")
    logger.info(f"  输入 task: {len(tasks)}, upsert: {tasks_upserted}")
    logger.info(f"  输入 chunk: {len(chunks)}, upsert: {chunks_upserted}")
    logger.info(f"  chunk summary 匹配: {len(summaries)}")

    for name, info in stats.items():
        if "error" in info:
            logger.warning(f"  集合 '{name}': 读取统计失败 - {info['error']}")
        else:
            logger.info(
                f"  集合 '{name}': "
                f"vectors={info.get('vectors_count', '?')}, "
                f"points={info.get('points_count', '?')}, "
                f"status={info.get('status', '?')}"
            )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
