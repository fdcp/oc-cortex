"""
Phase 2 主入口
流程: 加载 Phase 1 chunks -> 按 session 分组 -> LLM 提炼 task -> 输出

用法:
  export DASHSCOPE_API_KEY='your-api-key'
  python code_p2_main.py
  python code_p2_main.py --chunks ./output/chunks.jsonl
  python code_p2_main.py --config code_p2_config.yaml --output ./output/tasks.jsonl
  python code_p2_main.py --limit 3    # 只处理前 3 个 session (快速验证)
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from loguru import logger

from code_p1_utils import Config, setup_logger, count_tokens
from code_p1_models import Chunk, CleanedToolCall
from code_p2_task_extractor import TaskExtractor
from code_p2_models import Task


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
                )
                chunks.append(chunk)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Chunk 解析失败 (行 {line_num}): {e}")
    return chunks


def group_chunks_by_session(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """按 session_id 分组"""
    groups: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        groups[c.session_id].append(c)
    # 按 turn_index 排序
    for sid in groups:
        groups[sid].sort(key=lambda x: x.turn_index)
    return dict(groups)


def save_tasks(tasks: list[Task], output_path: str):
    """保存 tasks 到 JSONL"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"已写入 {len(tasks)} 个 task 到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Task 清单生成")
    parser.add_argument(
        "--config", default="code_p2_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--chunks", default=None,
        help="Phase 1 chunks JSONL 文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--output", default="./output/tasks.jsonl",
        help="task 输出路径"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 个 session (快速验证)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="并发线程数 (覆盖配置文件, 1=串行, 默认从 config 读取)"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = Config.load(args.config)
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("Phase 2: Task 清单生成")
    logger.info("=" * 60)

    # 2. 加载 Phase 1 chunks
    chunks_file = args.chunks or config.get(
        "phase1.chunks_file", "./output/chunks.jsonl"
    )
    logger.info(f"加载 chunks: {chunks_file}")

    chunks = load_chunks_from_jsonl(chunks_file)
    logger.info(f"加载 {len(chunks)} 个 chunk")

    if not chunks:
        logger.error("没有加载到任何 chunk,请检查输入文件")
        sys.exit(1)

    # 3. 按 session 分组
    sessions_chunks = group_chunks_by_session(chunks)
    logger.info(f"共 {len(sessions_chunks)} 个 session")

    # 4. 限制处理数量
    if args.limit:
        session_ids = list(sessions_chunks.keys())[:args.limit]
        sessions_chunks = {sid: sessions_chunks[sid] for sid in session_ids}
        logger.info(f"限制处理前 {args.limit} 个 session")

    # 5. 初始化 TaskExtractor
    concurrency = args.concurrency or config.get("llm.concurrency", 1)
    extractor = TaskExtractor(
        model=config.get("llm.model", "qwen-plus"),
        api_key_env=config.get("llm.api_key_env", "DASHSCOPE_API_KEY"),
        base_url=config.get(
            "llm.base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        max_retries=config.get("llm.max_retries", 3),
        timeout=config.get("llm.timeout", 120),
        max_tokens_per_chunk=config.get("llm.max_tokens_per_chunk", 2000),
        max_total_prompt_tokens=config.get(
            "llm.max_total_prompt_tokens", 30000
        ),
        concurrency=concurrency,
    )

    # 6. 执行 task 提取
    import time as _time
    t0 = _time.time()
    tasks = extractor.extract_tasks_batch(sessions_chunks)
    elapsed = _time.time() - t0
    logger.info(f"Task 提取耗时: {elapsed:.1f}s")


    # 7. 输出
    save_tasks(tasks, args.output)

    # 8. 打印摘要
    logger.info("=" * 60)
    logger.info("Phase 2 完成")
    logger.info(f"  输入 session: {len(sessions_chunks)}")
    logger.info(f"  生成 task: {len(tasks)}")
    logger.info(f"  输出文件: {args.output}")
    logger.info("=" * 60)

    # 9. 打印示例
    if tasks:
        for t in tasks[:3]:
            logger.info("-" * 40)
            logger.info(f"Task ID:    {t.task_id}")
            logger.info(f"Session:    {t.session_id}")
            logger.info(f"Label:      {t.task_label}")
            logger.info(f"Summary:    {t.task_summary}")
            logger.info(f"Chunks:     {t.chunk_ids}")

    # 10. 统计
    task_counts = defaultdict(int)
    for t in tasks:
        task_counts[t.session_id] += 1
    if task_counts:
        avg = sum(task_counts.values()) / len(task_counts)
        logger.info(f"平均每 session task 数: {avg:.1f}")
        logger.info(f"Task 数分布: {dict(task_counts)}")


if __name__ == "__main__":
    main()
