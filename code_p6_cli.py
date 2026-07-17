"""
Phase 6 CLI: 跨 Session 总结命令行工具

用法:
  python3 code_p6_cli.py "我最近做过的 FlashAttention 相关工作"
  python3 code_p6_cli.py "性能优化" --graph-rag
  python3 code_p6_cli.py "RoPE 相关" --time-from 2026-07-01 --time-to 2026-07-15
  python3 code_p6_cli.py "序列并行" --detail full --top-k 5
"""
import argparse
import json
import sys
import time

from code_p6_summarizer import SessionSummarizer


def main():
    parser = argparse.ArgumentParser(description="跨 Session 总结工具")
    parser.add_argument("query", help="总结查询（自然语言）")
    parser.add_argument("--top-k", type=int, default=8, help="检索 Task 数 (默认 8)")
    parser.add_argument("--config", default="code_p3_config.yaml", help="配置文件路径")
    parser.add_argument("--graph-rag", action="store_true", help="启用 Graph-RAG 增强检索")
    parser.add_argument("--time-from", default=None, help="时间范围起始 (ISO: 2026-07-01)")
    parser.add_argument("--time-to", default=None, help="时间范围截止 (ISO: 2026-07-15)")
    parser.add_argument(
        "--detail", choices=["summary", "preview", "full"], default="summary",
        help="Chunk 详情级别: summary(摘要) | preview(摘要+前500字) | full(完整)"
    )
    parser.add_argument("--max-tokens", type=int, default=12000, help="LLM 输入 token 预算")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--no-chunks", action="store_true", help="不包含 chunk 原文（仅用 task 摘要）")

    args = parser.parse_args()

    # 时间范围
    time_range = None
    if args.time_from and args.time_to:
        time_range = (args.time_from, args.time_to)
    elif args.time_from or args.time_to:
        print("错误: --time-from 和 --time-to 必须同时指定", file=sys.stderr)
        sys.exit(1)

    # 初始化
    print(f"初始化 SessionSummarizer (config={args.config})...", file=sys.stderr)
    t0 = time.time()
    summarizer = SessionSummarizer(
        config_path=args.config,
        max_context_tokens=args.max_tokens,
    )
    init_time = time.time() - t0
    print(f"初始化完成 ({init_time:.1f}s)", file=sys.stderr)

    # 执行总结
    print(f"\n查询: \"{args.query}\"", file=sys.stderr)
    print(f"参数: top_k={args.top_k}, graph_rag={args.graph_rag}, detail={args.detail}", file=sys.stderr)
    if time_range:
        print(f"时间范围: {time_range[0]} ~ {time_range[1]}", file=sys.stderr)
    print("正在检索和生成总结...\n", file=sys.stderr)

    result = summarizer.summarize(
        query=args.query,
        top_k=args.top_k,
        time_range=time_range,
        use_graph_rag=args.graph_rag,
        include_chunks=not args.no_chunks,
        chunk_detail_level=args.detail,
    )

    # 输出
    if args.json:
        output = {
            "query": result.query,
            "summary": result.summary,
            "sources": [
                {
                    "task_id": s.task_id,
                    "session_id": s.session_id,
                    "task_label": s.task_label,
                    "task_summary": s.task_summary,
                    "score": round(s.rerank_score, 4),
                    "chunk_count": s.chunk_count,
                    "created_at": s.created_at,
                }
                for s in result.sources
            ],
            "debug": {
                "total_tokens": result.total_tokens,
                "tasks_used": result.debug.get("tasks_used", 0),
                "elapsed_ms": result.elapsed_ms,
            },
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"  跨 Session 总结")
        print(f"  查询: \"{result.query}\"")
        print(f"  来源: {len(result.sources)} 个任务 | {result.total_tokens} tokens | {result.elapsed_ms}ms")
        print("=" * 70)
        print()
        print(result.summary)
        print()
        print("-" * 70)
        print("参与总结的任务:")
        for i, s in enumerate(result.sources, 1):
            print(f"  {i}. [{s.task_label}] score={s.rerank_score:.3f} ({s.session_id[:12]}...)")
            print(f"     {s.task_summary[:80]}...")


if __name__ == "__main__":
    main()
