"""Graph-RAG 检索流程分阶段 trace 工具

复现 `MCP knowledge-graph` 的 `summarize(use_graph_rag=true)` 内部流程,
打印每路召回的 task_id + RRF 排名 + 来源分布, 用于排查和性能分析。

不修改任何源码, 仅做只读调用。

Usage:
    cd /Users/zhaoxiuwei/Desktop/oc_sess_graph
    export OPENCODE_ZEN_API_KEY=$(python3 -c \
        "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); \
print(d['opencode-go']['key'])")
    export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

    python3 tests/graph_rag_search/trace_rag.py
    python3 tests/graph_rag_search/trace_rag.py --query "FlashAttention" --top-k 8
    python3 tests/graph_rag_search/trace_rag.py --rerank-multiplier 1.0 --no-rerank
    python3 tests/graph_rag_search/trace_rag.py --export-json result.json
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, "src")

from code_p4_searcher import SessionSearcher, SessionSearchResult
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher, extract_query_entities


def run(query: str, top_k: int, rerank_multiplier: float, use_rerank: bool, export_json: str = None):
    print("=" * 80)
    print(f"Query: {query}  top_k={top_k}  rerank_multiplier={rerank_multiplier}  rerank={use_rerank}")
    print("=" * 80)

    searcher = SessionSearcher("config/code_p3_config.yaml")
    db = KGDatabase("output/triple/knowledge_graph.db")
    rag = GraphRAGSearcher(
        searcher, db,
        graph_channel_weight=0.3,
        rerank_multiplier=rerank_multiplier,
    )

    # ---------- Stage 1: 向量检索 (手工拆 3 路) ----------
    vector_top_k = max(int(top_k * rag.rerank_multiplier), top_k)
    n_inner = vector_top_k * 5
    sparse_top_k_inner = n_inner * 3

    print(f"\n[Stage 1] GraphRAG 传 vector_top_k={vector_top_k} -> SessionSearcher n_inner={n_inner}")
    vector_results = searcher.search(query=query, top_k=vector_top_k, skip_rerank=not use_rerank)

    # 1a. Dense[tasks]
    dense = searcher.store.search_dense(
        query, collection=searcher.store.tasks_collection, top_k=n_inner
    )
    dense_ids = [r.point_id for r in dense]
    print(f"  1a. Dense[tasks]          : limit={n_inner}, hit={len(dense_ids)}")

    # 1b. Sparse[chunks_summary]
    summary_tokens = searcher._tokenize_query(query)
    summary_chunk_results = searcher._search_chunks_summary_with_tokens(summary_tokens, sparse_top_k_inner)
    summary_task_results = searcher._aggregate_chunks_to_tasks(summary_chunk_results)
    summary_ids = [r.point_id for r in summary_task_results]
    print(f"  1b. Sparse[chunks_summary]: limit={sparse_top_k_inner}, "
          f"chunk_hit={len(summary_chunk_results)}, task_agg={len(summary_ids)}")

    # 1c. Sparse[chunks_cleaned_text]
    cleaned_tokens = searcher._tokenize_query(query)
    if searcher.store.sparse_method == "bge_m3":
        sparse_chunk_results = searcher.store.search_sparse_bge_m3(
            " ".join(cleaned_tokens),
            collection=searcher.store.chunks_cleaned_text_collection,
            top_k=sparse_top_k_inner,
        )
    else:
        sparse_chunk_results = searcher.store.search_sparse_bm25_tokens(
            cleaned_tokens,
            collection=searcher.store.chunks_cleaned_text_collection,
            top_k=sparse_top_k_inner,
        )
    sparse_task_results = searcher._aggregate_chunks_to_tasks(sparse_chunk_results)
    sparse_ids = [r.point_id for r in sparse_task_results]
    print(f"  1c. Sparse[chunks_cleaned]: limit={sparse_top_k_inner}, "
          f"chunk_hit={len(sparse_chunk_results)}, task_agg={len(sparse_ids)}")

    # ---------- Stage 2: chunk 两路 RRF ----------
    chunk_fused = searcher.store._rrf_fuse(summary_task_results, sparse_task_results, searcher.store.fuse_k)
    chunk_fused_ids = [r.point_id for r in chunk_fused]
    print(f"\n[Stage 2] chunk 两路 RRF 融合: {len(chunk_fused_ids)} task")
    print(f"   summary 独有: {len(set(summary_ids) - set(sparse_ids))}, "
          f"cleaned 独有: {len(set(sparse_ids) - set(summary_ids))}, "
          f"共有: {len(set(summary_ids) & set(sparse_ids))}")

    # ---------- Stage 3: vector RRF 融合 (SessionSearcher 内部) ----------
    print(f"\n[Stage 3] vector_results (SessionSearcher 内部 RRF): {len(vector_results)} task")
    print(f"   Dense 独有: {len(set(dense_ids) - set(chunk_fused_ids))}, "
          f"chunk 路径独有: {len(set(chunk_fused_ids) - set(dense_ids))}, "
          f"共有: {len(set(dense_ids) & set(chunk_fused_ids))}")

    # ---------- Stage 4: 图谱 BFS ----------
    t_g = time.time()
    query_entities = extract_query_entities(query)
    seed_entities = []
    for e in query_entities:
        if db.get_node(e):
            seed_entities.append(e)
        else:
            matches = db.search_entities(e, limit=3)
            seed_entities.extend(m["name"] for m in matches)
    seed_entities = list(set(seed_entities))

    graph_results = []
    expanded = []
    if seed_entities:
        expanded = db.bfs_expand(seed_entities, depth=rag.bfs_depth, max_nodes=rag.max_expand_nodes)
        total_tasks = max(rag._total_tasks, 1)
        task_scores = {}
        nodes = db.get_nodes_batch(list(expanded))
        for entity_name, node in nodes.items():
            tc = max(node["task_count"], 1)
            contrib = math.log(1 + total_tasks / tc)
            for tid in node["source_tasks"]:
                if tid not in task_scores:
                    task_scores[tid] = {"score": 0.0, "entities": []}
                task_scores[tid]["score"] += contrib
                task_scores[tid]["entities"].append(entity_name)
        all_tasks = {t.task_id: t for t in searcher.tasks}
        for tid, info in sorted(task_scores.items(), key=lambda x: -x[1]["score"]):
            if len(graph_results) >= rag.max_graph_tasks:
                break
            if tid not in all_tasks:
                continue
            task = all_tasks[tid]
            chunk_details = searcher._expand_chunks(task.chunk_ids)
            graph_results.append(SessionSearchResult(
                task_id=task.task_id,
                session_id=task.session_id,
                task_label=task.task_label,
                task_summary=task.task_summary,
                rerank_score=0.0,
                hybrid_score=info["score"],
                chunks=chunk_details,
            ))
    graph_ids = [r.task_id for r in graph_results]
    print(f"\n[Stage 4] 图谱 BFS: query={query_entities} seed={seed_entities}")
    print(f"   扩散节点: {len(expanded)} 个, graph_candidates: {len(graph_ids)}  (耗时 {time.time()-t_g:.2f}s)")

    # ---------- Stage 5: 外层 RRF ----------
    merged = rag._outer_rrf(vector_results, graph_results, k=rag._OUTER_RRF_K)
    merged_ids = [r.task_id for r in merged]
    print(f"\n[Stage 5] 外层 RRF 融合: {len(merged_ids)} task")
    vec_id_set = {r.task_id for r in vector_results}
    print(f"   vector 独有: {len(vec_id_set - set(graph_ids))}, "
          f"graph 独有: {len(set(graph_ids) - vec_id_set)}, "
          f"共有: {len(set(graph_ids) & vec_id_set)}")

    # ---------- Stage 6: Reranker ----------
    final_results = []
    t_rerank = 0.0
    if use_rerank:
        t_r = time.time()
        final = rag._rerank(merged, query, top_k)
        t_rerank = time.time() - t_r
        print(f"\n[Stage 6] Reranker: {len(merged_ids)} -> {len(final)}  耗时 {t_rerank:.2f}s")
        for i, r in enumerate(final, 1):
            src = ("both" if r.task_id in set(graph_ids) and r.task_id in vec_id_set
                   else "graph" if r.task_id in set(graph_ids) else "vector")
            print(f"  [{i}] {r.task_id}  rerank={r.rerank_score:.4f}  src={src}")
            print(f"       {r.task_label[:60]}")
            final_results.append({
                "rank": i,
                "task_id": r.task_id,
                "task_label": r.task_label,
                "rerank_score": r.rerank_score,
                "hybrid_score": r.hybrid_score,
                "src": src,
            })
    else:
        print(f"\n[Stage 6] Reranker 跳过 (--no-rerank)")
        for i, r in enumerate(merged[:top_k], 1):
            src = ("both" if r.task_id in set(graph_ids) and r.task_id in vec_id_set
                   else "graph" if r.task_id in set(graph_ids) else "vector")
            print(f"  [{i}] {r.task_id}  hybrid={r.hybrid_score:.4f}  src={src}")
            print(f"       {r.task_label[:60]}")
            final_results.append({
                "rank": i,
                "task_id": r.task_id,
                "task_label": r.task_label,
                "rerank_score": 0.0,
                "hybrid_score": r.hybrid_score,
                "src": src,
            })

    # ---------- 导出 JSON ----------
    if export_json:
        result = {
            "query": query,
            "top_k": top_k,
            "rerank_multiplier": rerank_multiplier,
            "use_rerank": use_rerank,
            "stages": {
                "dense": {
                    "limit": n_inner,
                    "hit": len(dense_ids),
                    "task_ids": dense_ids,
                },
                "sparse_summary": {
                    "limit": sparse_top_k_inner,
                    "chunk_hit": len(summary_chunk_results),
                    "task_agg": len(summary_ids),
                    "task_ids": summary_ids,
                },
                "sparse_cleaned": {
                    "limit": sparse_top_k_inner,
                    "chunk_hit": len(sparse_chunk_results),
                    "task_agg": len(sparse_ids),
                    "task_ids": sparse_ids,
                },
                "chunk_fused": {
                    "count": len(chunk_fused_ids),
                    "task_ids": chunk_fused_ids,
                    "summary_only": sorted(set(summary_ids) - set(sparse_ids)),
                    "cleaned_only": sorted(set(sparse_ids) - set(summary_ids)),
                    "shared": sorted(set(summary_ids) & set(sparse_ids)),
                },
                "vector": {
                    "count": len(vector_results),
                    "task_ids": [r.task_id for r in vector_results],
                    "dense_only": sorted(set(dense_ids) - set(chunk_fused_ids)),
                    "chunk_only": sorted(set(chunk_fused_ids) - set(dense_ids)),
                    "shared": sorted(set(dense_ids) & set(chunk_fused_ids)),
                },
                "graph": {
                    "query_entities": query_entities,
                    "seed_entities": seed_entities,
                    "expanded_count": len(expanded),
                    "candidates": graph_ids,
                },
                "merged": {
                    "count": len(merged_ids),
                    "task_ids": merged_ids,
                    "vector_only": sorted(vec_id_set - set(graph_ids)),
                    "graph_only": sorted(set(graph_ids) - vec_id_set),
                    "shared": sorted(set(graph_ids) & vec_id_set),
                },
                "final": {
                    "count": len(final_results),
                    "rerank_time_s": t_rerank,
                    "results": final_results,
                },
            },
        }
        out_path = os.path.abspath(export_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[Export] JSON 已写入: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Graph-RAG 检索流程分阶段 trace 工具, 复现 MCP summarize 内部流程",
    )
    parser.add_argument("--query", default="分布式训练", help="查询字符串 (默认: 分布式训练)")
    parser.add_argument("--top-k", type=int, default=8, help="最终返回结果数 (默认: 8)")
    parser.add_argument(
        "--rerank-multiplier", type=float, default=1.0,
        help="GraphRAGSearcher.rerank_multiplier (默认 1.0, 推荐小 corpus 用 1.0)",
    )
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Reranker 精排, 只打印粗排结果")
    parser.add_argument(
        "--export-json", metavar="PATH", default=None,
        help="导出完整结果到 JSON 文件 (含每阶段 task_id 集合 + 耗时 + 来源分布)",
    )
    args = parser.parse_args()

    run(args.query, args.top_k, args.rerank_multiplier, not args.no_rerank, args.export_json)


if __name__ == "__main__":
    main()
