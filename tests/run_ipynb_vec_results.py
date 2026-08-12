"""Replicate ipynb Cell 3 `vec_results, vec_debug = rag.search(...)` standalone.

目的:
- 把 tests/sql_search.ipynb Cell 3 中 vec_results 那一行单独抽出来
- 在 SessionSearcher 内部各方法加 monkey-patch, 捕获每个子阶段的 input/output/timing
- 与 doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json 做等价性验证

用法:
    cd tests && python3 run_ipynb_vec_results.py
    # 或
    python3 tests/run_ipynb_vec_results.py   # 内部自动 chdir 到 tests/

输出:
    tests/ipynb_vec_results_trace.json  (与 trace JSON 结构对齐, 便于 diff)
"""

import json
import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# 0. 环境: 必须先于 transformers/sentence_transformers import
# -----------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
os.chdir(TESTS_DIR)  # 模拟 ipynb 的 cwd=tests/ 行为 (../config/... 路径生效)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(TESTS_DIR.parent / "src"))

from code_p1_utils import Config  # noqa: E402
from code_p3_qdrant_store import SearchResult  # noqa: E402
from code_p4_searcher import SessionSearcher  # noqa: E402
from code_p5e_db import KGDatabase  # noqa: E402
from code_p5e_graph_rag import GraphRAGSearcher, build_retrieval_query  # noqa: E402

# -----------------------------------------------------------------------------
# 1. ipynb Cell 3 中 vec_results 那行的精确复刻
# -----------------------------------------------------------------------------
CONFIG_PATH = "../config/code_p3_config.yaml"
KG_DB_PATH = "../output/triple/knowledge_graph.db"
QUERY = "GPU的对比和选型"
TOP_K = 5

cfg = Config.load(CONFIG_PATH)

searcher = SessionSearcher(CONFIG_PATH)
kg_db = KGDatabase(KG_DB_PATH)

# 跟 ipynb 完全一致: bfs_depth=1, graph_channel_weight=0.1 (注意: 0.1 不是默认 0.3)
rag = GraphRAGSearcher(
    searcher=searcher,
    kg_db=kg_db,
    bfs_depth=1,
    graph_channel_weight=0.1,  # ← ipynb Cell 3 实际值
    query_instruction=cfg.get("query_instruction_for_retrieval", ""),
    rerank_multiplier=1,
)

# -----------------------------------------------------------------------------
# 2. Monkey-patch 各子阶段, 捕获 input/output/timing
# -----------------------------------------------------------------------------
trace_log: list[dict] = []


def _attach():
    """Install monkey-patches on searcher internals."""

    # ---- 2.1 store.search_dense (Dense[tasks] 与 Dense[chunks_summary] 都会调) ----
    orig_dense = searcher.store.search_dense

    def patched_dense(query, collection, top_k, **kwargs):
        t0 = time.time()
        result = orig_dense(query, collection=collection, top_k=top_k, **kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)
        trace_log.append(
            {
                "stage": "search_dense",
                "input": {"query": query, "collection": collection, "top_k": top_k},
                "output": [
                    {
                        "rank": i + 1,
                        "point_id": r.point_id,
                        "score": round(r.score, 6),
                        "task_id": r.payload.get("task_id", ""),
                        "task_label": r.payload.get("task_label", ""),
                    }
                    for i, r in enumerate(result)
                ],
                "output_count": len(result),
                "elapsed_ms": elapsed_ms,
            }
        )
        return result

    searcher.store.search_dense = patched_dense

    # ---- 2.2 store.search_sparse_bm25 (chunks_summary/chunks_cleaned_text 都会调) ----
    orig_bm25 = searcher.store.search_sparse_bm25

    def patched_bm25(query, collection, top_k, **kwargs):
        t0 = time.time()
        result = orig_bm25(query, collection=collection, top_k=top_k, **kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)
        trace_log.append(
            {
                "stage": "search_sparse_bm25",
                "input": {"query": query, "collection": collection, "top_k": top_k},
                "output": [
                    {
                        "rank": i + 1,
                        "point_id": r.point_id,
                        "score": round(r.score, 6),
                        "chunk_id": r.payload.get("chunk_id", ""),
                        "task_id": r.payload.get("task_id", ""),
                    }
                    for i, r in enumerate(result)
                ],
                "output_count": len(result),
                "elapsed_ms": elapsed_ms,
            }
        )
        return result

    searcher.store.search_sparse_bm25 = patched_bm25

    # ---- 2.3 store._rrf_fuse (两次: chunk-level RRF + final RRF) ----
    orig_rrf = searcher.store._rrf_fuse

    def patched_rrf(dense_results, sparse_results, k=60):
        t0 = time.time()
        result = orig_rrf(dense_results, sparse_results, k)
        elapsed_ms = int((time.time() - t0) * 1000)
        inputs_data = [
            ("dense_results", dense_results),
            ("sparse_results", sparse_results),
        ]
        input_summaries = []
        for name, inp in inputs_data:
            input_summaries.append(
                {
                    "name": name,
                    "count": len(inp),
                    "top5_task_ids": [
                        r.payload.get("task_id", "") for r in inp[:5]
                    ],
                }
            )
        trace_log.append(
            {
                "stage": "_rrf_fuse",
                "input": {
                    "fuse_k": k,
                    "n_dense_results": len(dense_results),
                    "n_sparse_results": len(sparse_results),
                    "input_summaries": input_summaries,
                },
                "output": [
                    {
                        "rank": i + 1,
                        "point_id": r.point_id,
                        "rrf_score": round(r.rrf_score, 6),
                        "task_id": r.payload.get("task_id", ""),
                        "task_label": r.payload.get("task_label", ""),
                        "dense_rank": r.dense_rank,
                        "sparse_rank": r.sparse_rank,
                    }
                    for i, r in enumerate(result)
                ],
                "output_count": len(result),
                "elapsed_ms": elapsed_ms,
            }
        )
        return result

    searcher.store._rrf_fuse = patched_rrf

    # ---- 2.4 reranker.rank ----
    orig_rank = searcher.reranker.rank

    def patched_rank(query, documents, top_k, **kwargs):
        t0 = time.time()
        result = orig_rank(query, documents, top_k=top_k, **kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)
        trace_log.append(
            {
                "stage": "reranker",
                "input": {
                    "query": query,
                    "n_documents": len(documents),
                    "top_k": top_k,
                    "documents_preview": [d[:60] + ("…" if len(d) > 60 else "") for d in documents],
                },
                "output": [
                    {"index": rr.index, "score": round(rr.score, 6)} for rr in result
                ],
                "output_count": len(result),
                "elapsed_ms": elapsed_ms,
                "_saved_documents": documents,  # 隐藏字段: 给 map 用
            }
        )
        return result

    searcher.reranker.rank = patched_rank


_attach()


# -----------------------------------------------------------------------------
# 3. 实际执行 ipynb Cell 3 中 vec_results 那一行
# -----------------------------------------------------------------------------
retrieval_query = build_retrieval_query(QUERY, rag.query_instruction)
t_total = time.time()
vec_results, vec_debug = rag.search(
    QUERY, top_k=TOP_K, use_graph_rag=False, use_reranker=True
)
total_ms = int((time.time() - t_total) * 1000)


# -----------------------------------------------------------------------------
# 4. 把 rerank 的 index → task_id 映射上 (candidates 由 SessionSearcher 闭包持有)
#    reranker.patch 里保存了 documents, 而 results 是按 reranked 顺序的 SessionSearchResult,
#    所以 results[i].task_id ↔ candidates[reranked[i].index].payload.task_id
# -----------------------------------------------------------------------------
rerank_stage = next(s for s in trace_log if s["stage"] == "reranker")
documents = rerank_stage.pop("_saved_documents")
rerank_out = rerank_stage["output"]
# 找到 candidates 那一段 (final RRF 的输出)
final_rrf_stage = max(
    (s for s in trace_log if s["stage"] == "_rrf_fuse"),
    key=lambda s: s["output_count"],
)
# final_rrf 的 output 是 chunk-level RRF ∪ dense 的并集 (去重 + rrf_score 累加),
# SessionSearcher 把它转成 SearchResult 后取 [:n_candidates],
# documents = [c.payload.get("task_summary","") for c in candidates]
# 这里我们直接复用 SessionSearchResult 顺序
for i, r in enumerate(vec_results):
    if i < len(rerank_out):
        rerank_out[i]["task_id"] = r.task_id
        rerank_out[i]["task_label"] = r.task_label
        rerank_out[i]["hybrid_score"] = round(r.hybrid_score, 6)
        # documents[rerank_out[i]["index"]] 就是该 candidate 的 task_summary


# -----------------------------------------------------------------------------
# 5. 整理 final_top5 (跟 ipynb 输出一致)
# -----------------------------------------------------------------------------
final_top5 = [
    {
        "rank": i + 1,
        "task_id": r.task_id,
        "task_label": r.task_label,
        "rerank_score": round(r.rerank_score, 6),
        "hybrid_score": round(r.hybrid_score, 6),
        "task_summary_preview": r.task_summary[:120],
    }
    for i, r in enumerate(vec_results)
]


# -----------------------------------------------------------------------------
# 6. 写出 JSON (结构跟 trace_no-ground JSON 严格对齐)
# -----------------------------------------------------------------------------
output = {
    "meta": {
        "query_raw": QUERY,
        "query_instruction": rag.query_instruction,
        "retrieval_query": retrieval_query,
        "top_k": TOP_K,
        "use_graph_rag": False,
        "use_reranker": True,
        "graph_channel_weight": rag.graph_channel_weight,
        "bfs_depth": rag.bfs_depth,
        "rerank_multiplier": rag.rerank_multiplier,
        "config_path": CONFIG_PATH,
        "kg_db_path": KG_DB_PATH,
        "cwd": str(TESTS_DIR),
        "total_time_ms": total_ms,
        "vec_debug": vec_debug,
    },
    "stages": trace_log,
    "final_top5": final_top5,
}

out_path = TESTS_DIR / "ipynb_vec_results_trace.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"[vec_results] 写入: {out_path}")
print(f"  query        : {QUERY}")
print(f"  retrieval    : {retrieval_query}")
print(f"  use_graph_rag: False, use_reranker: True")
print(f"  total_time   : {total_ms}ms ({total_ms/1000:.2f}s)")
print(f"{'='*60}")
print(f"Final top 5:")
for r in final_top5:
    print(
        f"  {r['rank']}. [{r['rerank_score']:.4f}] "
        f"{r['task_id']} - {r['task_label']}"
    )