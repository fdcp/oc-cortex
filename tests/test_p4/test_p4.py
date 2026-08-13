"""
P4 alias expansion A/B benchmark (sparse-only, token-list API).

用法:
    python3 tests/test_p4/test_p4.py [--query "..."] [--top-k 5] [--case-sensitive]

A/B 对比:
  alias_expansion enabled=false  → BM25 用原 query jieba tokens
  alias_expansion enabled=true   → BM25 用扩展后 tokens (canonical + aliases)
  Dense + Reranker 两个路径不变, 仅 sparse 受影响。
"""
import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# HuggingFace 环境设置 (必须在其他导入之前)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from code_p1_utils import Config, setup_logger
from code_p4_searcher import SessionSearcher


# Demo queries — 覆盖中文 / 英文缩写 / 短查询 / 长查询
DEFAULT_QUERIES = [
    "FA 原理",
    "SP 与 TP 的区别",
    "序列并行",
    "BF16 混合精度",
    "不同GPU的特性对比和选型",
]


def run_one(
    searcher: SessionSearcher,
    query: str,
    top_k: int,
    case_label: str,
) -> list:
    t0 = time.time()
    results = searcher.search(query, top_k=top_k)
    elapsed = time.time() - t0
    print(f"\n[{case_label}] '{query}'  耗时 {elapsed*1000:.0f}ms  {len(results)} 条")
    for i, r in enumerate(results, 1):
        print(
            f"  #{i}  rerank={r.rerank_score:.4f}  hybrid={r.hybrid_score:.4f}  "
            f"{r.task_label[:60]}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="P4 alias expansion A/B benchmark")
    parser.add_argument("--config", default="config/code_p3_config.yaml")
    parser.add_argument("--query", default=None, help="单条查询; 不传则跑 DEMO_QUERIES")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = Config.load(args.config)
    setup_logger(
        log_file=config.get("logging.file"),
        level="WARNING",
    )

    print("初始化搜索引擎 ...")
    t0 = time.time()
    searcher = SessionSearcher(config_path=args.config)
    print(f"初始化耗时 {time.time()-t0:.1f}s\n")

    queries = [args.query] if args.query else DEFAULT_QUERIES

    for q in queries:
        print("\n" + "=" * 70)
        print(f"  A/B: '{q}'")
        print("=" * 70)

        searcher.set_alias_expansion(False)
        a_results = run_one(searcher, q, args.top_k, "A: alias OFF ")

        searcher.set_alias_expansion(True)
        b_results = run_one(searcher, q, args.top_k, "B: alias ON  ")

        a_ids = [r.task_id for r in a_results]
        b_ids = [r.task_id for r in b_results]
        new_in_b = [b_id for b_id in b_ids if b_id not in a_ids]
        lost_in_b = [a_id for a_id in a_ids if a_id not in b_ids]
        print(f"  → 新召回 (B 独有): {len(new_in_b)} 条 {new_in_b[:3]}")
        print(f"  → 丢失召回 (A 独有): {len(lost_in_b)} 条 {lost_in_b[:3]}")


if __name__ == "__main__":
    main()