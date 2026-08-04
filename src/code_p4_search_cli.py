"""
Phase 4 跨 session 搜索 CLI

用法:
  # 单次查询
  python code_p4_search_cli.py --query "GPU对比分析"

  # 交互模式
  python code_p4_search_cli.py --interactive

  # 跳过 Reranker (对比粗排效果)
  python code_p4_search_cli.py --query "优化器学习率" --no-rerank

  # 调整返回数量
  python code_p4_search_cli.py --query "session管理" --top-k 10
"""
import argparse
import os
import sys
import time

# HuggingFace 环境设置 (必须在其他导入之前)
_args_pre = argparse.ArgumentParser(add_help=False)
_args_pre.add_argument("--config", default="config/code_p3_config.yaml")
_temp_args, _ = _args_pre.parse_known_args()

from code_p1_utils import Config
_config = Config.load(_temp_args.config)

from code_p3_hf_config import setup_hf_env
_cache_folder = _config.get("embedding.cache_folder")
if _cache_folder is None:
    _cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
_offline_mode = _config.get("embedding.offline_mode", True)
setup_hf_env(offline_mode=_offline_mode, cache_folder=_cache_folder)

from loguru import logger
from code_p1_utils import setup_logger
from code_p4_searcher import SessionSearcher, SessionSearchResult


# ============================================================
# 格式化输出
# ============================================================

def format_result(r: SessionSearchResult, idx: int, show_chunks: bool = True) -> str:
    """格式化单条搜索结果"""
    lines = []
    lines.append(f"\n{'─' * 60}")
    lines.append(f"  #{idx}  rerank={r.rerank_score:.4f}  hybrid={r.hybrid_score:.4f}")
    lines.append(f"  Task:   {r.task_label}")
    lines.append(f"  ID:     {r.task_id}")
    lines.append(f"  Session:{r.session_id}")
    lines.append(f"  摘要:   {r.task_summary}")

    if show_chunks and r.chunks:
        lines.append(f"  关联 Chunks ({len(r.chunks)} 个):")
        for ci, chunk in enumerate(r.chunks, 1):
            summary_short = (
                chunk.summary[:100] + "..."
                if len(chunk.summary) > 100
                else chunk.summary
            )
            lines.append(
                f"    [{ci}] {chunk.chunk_id} "
                f"(turn {chunk.turn_index}, "
                f"raw={chunk.raw_size_tokens}tok, "
                f"clean={chunk.cleaned_size_tokens}tok)"
            )
            if summary_short:
                lines.append(f"        摘要: {summary_short}")
            preview_short = chunk.cleaned_text_preview[:120].replace("\n", " ")
            lines.append(f"        预览: {preview_short}...")

    return "\n".join(lines)


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

def run_single_search(
    searcher: SessionSearcher,
    query: str,
    top_k: int = 5,
    skip_rerank: bool = False,
):
    """执行单次搜索并打印结果"""
    t0 = time.time()
    results = searcher.search(
        query,
        top_k=top_k,
        skip_rerank=skip_rerank,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    mode = "粗排 (Hybrid, 无 Reranker)" if skip_rerank else "精排 (Hybrid + Reranker)"
    print(f"  查询: {query}")
    print(f"  模式: {mode}")
    print(f"  结果: {len(results)} 条, 耗时 {elapsed:.2f}s")
    print(f"{'=' * 60}")

    for i, r in enumerate(results, 1):
        print(format_result(r, i))

    print(f"\n{'─' * 60}")
    return results


def run_interactive(
    searcher: SessionSearcher,
    top_k: int,
    skip_rerank: bool,
):
    """交互模式: 循环输入查询"""
    print("\n" + "=" * 60)
    print("  跨 Session 搜索 (交互模式)")
    print("  输入查询, 输入 'quit' 或 'q' 退出")
    print("  输入 'demo' 运行示例查询")
    print("=" * 60)

    while True:
        try:
            query = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "q", "exit"):
            print("再见!")
            break
        if query.lower() == "demo":
            for q in DEMO_QUERIES:
                run_single_search(searcher, q, top_k, skip_rerank)
            continue

        run_single_search(searcher, query, top_k, skip_rerank)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 跨 session 搜索")
    parser.add_argument("--config", default="config/code_p3_config.yaml")
    parser.add_argument("--query", default=None, help="查询文本")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="跳过 Reranker, 仅使用粗排 (hybrid RRF)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="交互模式: 循环输入查询"
    )
    args = parser.parse_args()

    # 配置 logger
    config = _config
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    if _offline_mode:
        logger.info(f"离线模式: 仅使用本地缓存 ({_cache_folder})")

    # 初始化搜索引擎
    logger.info("初始化搜索引擎 ...")
    t0 = time.time()
    searcher = SessionSearcher(config_path=args.config)
    logger.info(f"搜索引擎初始化完成, 耗时 {time.time() - t0:.1f}s")

    if args.interactive:
        run_interactive(
            searcher,
            args.top_k,
            args.no_rerank,
        )
    elif args.query:
        run_single_search(
            searcher,
            args.query,
            args.top_k,
            args.no_rerank,
        )
    else:
        # 无参数时运行 demo
        print("未指定 --query, 运行示例查询:")
        for q in DEMO_QUERIES:
            run_single_search(
                searcher,
                q,
                args.top_k,
                args.no_rerank,
            )


if __name__ == "__main__":
    main()
