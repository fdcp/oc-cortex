"""
Phase 6b CLI: 决策溯源 + 主题总结骨架命令行工具

用法:
  # 决策溯源
  python3 code_p6b_cli.py trace "Qdrant"
  python3 code_p6b_cli.py trace "FlashAttention" --hops 3 --json

  # 骨架总结
  python3 code_p6b_cli.py summary "FlashAttention 的实现和优化"
  python3 code_p6b_cli.py summary "序列并行" --no-trace --json
"""
import argparse
import json
import os
import sys
import time

from loguru import logger


def _init_db(db_path: str = None):
    """初始化 KGDatabase"""
    from code_p5e_db import KGDatabase

    if db_path and os.path.exists(db_path):
        return KGDatabase(db_path)

    # 自动查找
    candidates = [
        "output/triple/knowledge_graph.db",
        "output/entity/knowledge_graph.db",
    ]
    for path in candidates:
        if os.path.exists(path):
            return KGDatabase(path)

    print("错误: 未找到知识图谱数据库。请先运行 Phase 5 构建图谱。", file=sys.stderr)
    sys.exit(1)


def cmd_trace(args):
    """执行决策溯源"""
    from code_p6b_skeleton import DecisionTracer

    db = _init_db(args.db)
    tracer = DecisionTracer(db, max_hops=args.hops)

    print(f"决策溯源: [{args.entity}] (max_hops={args.hops})\n", file=sys.stderr)
    chain = tracer.trace_decision(args.entity)

    if not chain.steps:
        print(f"未找到 [{args.entity}] 的决策链。", file=sys.stderr)
        print(f"提示: 尝试搜索实体: python3 code_p6b_cli.py search \"{args.entity}\"", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "root": chain.root_entity,
            "steps": [
                {
                    "entity": s.entity,
                    "related": s.related_entity,
                    "relation": s.relation,
                    "category": s.category,
                    "hop": s.hop,
                    "direction": s.direction,
                    "source_task": s.source_task,
                    "weight": s.weight,
                }
                for s in chain.steps
            ],
            "visited": len(chain.visited_entities),
            "max_hop": chain.hop_count,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(tracer.format_context(chain, max_steps=args.max_steps))
        print(f"\n共 {len(chain.steps)} 个决策步骤, "
              f"{len(chain.visited_entities)} 个关联实体, "
              f"最大 {chain.hop_count} 跳")


def cmd_summary(args):
    """执行骨架总结"""
    from code_p6_summarizer import SessionSummarizer
    from code_p6b_skeleton import DecisionTracer, SummarySkeleton

    db = _init_db(args.db)
    summarizer = SessionSummarizer(config_path=args.config)
    tracer = DecisionTracer(db, max_hops=args.hops)

    print(f"初始化完成 (db={db.db_path})", file=sys.stderr)
    print(f"主题: \"{args.topic}\"", file=sys.stderr)
    print(f"参数: trace={not args.no_trace}, max_tasks={args.max_tasks}\n", file=sys.stderr)
    print("正在构建骨架和生成总结...\n", file=sys.stderr)

    skeleton = SummarySkeleton(db, summarizer, tracer)
    result = skeleton.generate_summary(
        topic=args.topic,
        use_decision_trace=not args.no_trace,
        max_tasks=args.max_tasks,
    )

    if args.json:
        output = {
            "topic": result.topic,
            "summary": result.summary,
            "skeleton": result.skeleton,
            "decision_chain": {
                "root": result.decision_chain.root_entity,
                "steps": len(result.decision_chain.steps),
                "max_hop": result.decision_chain.hop_count,
            },
            "tasks": result.task_summaries,
            "debug": result.debug,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"  主题总结骨架")
        print(f"  主题: \"{result.topic}\"")
        steps_info = f"{len(result.decision_chain.steps)} 步" if result.decision_chain.steps else "无"
        print(f"  决策链: {steps_info} | {len(result.task_summaries)} 个任务 | {result.elapsed_ms}ms")
        print("=" * 70)

        if result.skeleton and result.skeleton != "(无相关结构化信息)":
            print(f"\n--- 知识图谱结构 ---\n{result.skeleton}\n")

        print(f"--- 主题总结 ---\n{result.summary}")

        if result.task_summaries:
            print(f"\n{'─' * 70}")
            print("参与总结的任务:")
            for i, t in enumerate(result.task_summaries, 1):
                print(f"  {i}. [{t['task_label']}] ({t['created_at'][:10]})")
                print(f"     {t['task_summary'][:80]}...")


def cmd_search(args):
    """搜索图谱实体"""
    db = _init_db(args.db)
    results = db.search_entities(args.query, limit=args.limit)

    if not results:
        print(f"未找到匹配 [{args.query}] 的实体")
        return

    for r in results:
        aliases = f" (别名: {', '.join(r['aliases'])})" if r["aliases"] else ""
        print(f"  {r['name']}{aliases} [{r['entity_type']}] task_count={r['task_count']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6b: 决策溯源 + 主题总结骨架")
    parser.add_argument("--db", default=None, help="知识图谱 SQLite 数据库路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- trace ----
    p_trace = subparsers.add_parser("trace", help="技术决策溯源")
    p_trace.add_argument("entity", help="起始实体名")
    p_trace.add_argument("--hops", type=int, default=3, help="最大追溯跳数 (默认 3)")
    p_trace.add_argument("--max-steps", type=int, default=20, help="最多显示步骤数 (默认 20)")
    p_trace.add_argument("--json", action="store_true", help="输出 JSON 格式")

    # ---- summary ----
    p_summary = subparsers.add_parser("summary", help="骨架主题总结")
    p_summary.add_argument("topic", help="主题关键词")
    p_summary.add_argument("--config", default="config/code_p3_config.yaml", help="搜索配置文件")
    p_summary.add_argument("--hops", type=int, default=3, help="决策追溯跳数 (默认 3)")
    p_summary.add_argument("--max-tasks", type=int, default=10, help="最多 Task 数 (默认 10)")
    p_summary.add_argument("--no-trace", action="store_true", help="不注入决策链（仅用骨架）")
    p_summary.add_argument("--json", action="store_true", help="输出 JSON 格式")

    # ---- search ----
    p_search = subparsers.add_parser("search", help="搜索图谱实体")
    p_search.add_argument("query", help="搜索关键词")
    p_search.add_argument("--limit", type=int, default=10, help="返回数量 (默认 10)")

    args = parser.parse_args()

    if args.command == "trace":
        cmd_trace(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
