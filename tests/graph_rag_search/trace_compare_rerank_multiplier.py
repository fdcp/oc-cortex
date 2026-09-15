"""rerank_multiplier 对比测试: 比较 1.0 vs 2.0 配置下 Graph-RAG 各阶段召回差异

针对小 corpus (<500 task) 场景, 验证 rerank_multiplier=1.0 是否在不影响
最终召回质量的前提下减少 Reranker 耗时。

Usage:
    cd /Users/zhaoxiuwei/Desktop/oc_sess_graph
    export OPENCODE_ZEN_API_KEY=$(python3 -c \
        "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); \
print(d['opencode-go']['key'])")
    export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

    python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py
    python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query "FlashAttention"
    python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query-list queries.txt
"""
import argparse
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


def run_one(rerank_multiplier: float, query: str, top_k: int, searcher, db):
    print("\n" + "=" * 90)
    print(f"rerank_multiplier = {rerank_multiplier}")
    print("=" * 90)

    rag = GraphRAGSearcher(
        searcher, db,
        graph_channel_weight=0.3,
        rerank_multiplier=rerank_multiplier,
    )

    vector_top_k = max(int(top_k * rag.rerank_multiplier), top_k)
    n_inner = vector_top_k * 5
    sparse_top_k_inner = n_inner * 3

    print(f"\n[Stage 1] GraphRAG 传 vector_top_k={vector_top_k} -> n_inner={n_inner}")
    vector_results = searcher.search(query=query, top_k=vector_top_k, skip_rerank=True)

    dense = searcher.store.search_dense(
        query, collection=searcher.store.tasks_collection, top_k=n_inner
    )
    dense_ids = [r.point_id for r in dense]
    print(f"  1a. Dense[tasks]          : limit={n_inner}, hit={len(dense_ids)}")

    summary_tokens = searcher._tokenize_query(query)
    summary_chunk_results = searcher._search_chunks_summary_with_tokens(summary_tokens, sparse_top_k_inner)
    summary_task_results = searcher._aggregate_chunks_to_tasks(summary_chunk_results)
    summary_ids = [r.point_id for r in summary_task_results]
    print(f"  1b. Sparse[chunks_summary]: limit={sparse_top_k_inner}, "
          f"chunk_hit={len(summary_chunk_results)}, task_agg={len(summary_ids)}")

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

    chunk_fused = searcher.store._rrf_fuse(summary_task_results, sparse_task_results, searcher.store.fuse_k)
    chunk_fused_ids = [r.point_id for r in chunk_fused]
    print(f"\n[Stage 2] chunk 两路 RRF 融合: {len(chunk_fused_ids)} task")
    print(f"   summary 独有: {len(set(summary_ids) - set(sparse_ids))}, "
          f"cleaned 独有: {len(set(sparse_ids) - set(summary_ids))}, "
          f"共有: {len(set(summary_ids) & set(sparse_ids))}")

    print(f"\n[Stage 3] vector_results (SessionSearcher 内部 RRF): {len(vector_results)} task")
    print(f"   Dense 独有: {len(set(dense_ids) - set(chunk_fused_ids))}, "
          f"chunk 路径独有: {len(set(chunk_fused_ids) - set(dense_ids))}, "
          f"共有: {len(set(dense_ids) & set(chunk_fused_ids))}")

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
    print(f"\n[Stage 4] 图谱 BFS: query={query_entities} seed={seed_entities}")

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
    print(f"   扩散节点: {len(expanded)} 个, graph_candidates: {len(graph_ids)}  (耗时 {time.time()-t_g:.2f}s)")

    merged = rag._outer_rrf(vector_results, graph_results, k=rag._OUTER_RRF_K)
    merged_ids = [r.task_id for r in merged]
    print(f"\n[Stage 5] 外层 RRF 融合: {len(merged_ids)} task")
    vec_id_set = {r.task_id for r in vector_results}
    print(f"   vector 独有: {len(vec_id_set - set(graph_ids))}, "
          f"graph 独有: {len(set(graph_ids) - vec_id_set)}, "
          f"共有: {len(set(graph_ids) & vec_id_set)}")

    t_r = time.time()
    final = rag._rerank(merged, query, top_k)
    t_rerank = time.time() - t_r
    print(f"\n[Stage 6] Reranker: {len(merged_ids)} -> {len(final)}  耗时 {t_rerank:.2f}s")
    for i, r in enumerate(final, 1):
        src = ("both" if r.task_id in set(graph_ids) and r.task_id in vec_id_set
               else "graph" if r.task_id in set(graph_ids) else "vector")
        print(f"  [{i}] {r.task_id}  rerank={r.rerank_score:.4f}  src={src}")
        print(f"       {r.task_label[:50]}")

    return {
        "rerank_multiplier": rerank_multiplier,
        "dense": dense_ids,
        "sparse_summary": summary_ids,
        "sparse_cleaned": sparse_ids,
        "chunk_fused": chunk_fused_ids,
        "vector": [r.task_id for r in vector_results],
        "graph": graph_ids,
        "merged": merged_ids,
        "final": [r.task_id for r in final],
        "t_rerank": t_rerank,
    }


def main():
    parser = argparse.ArgumentParser(
        description="rerank_multiplier 对比: 1.0 vs 2.0 配置下召回质量与耗时",
    )
    parser.add_argument("--query", default="分布式训练", help="查询字符串 (默认: 分布式训练)")
    parser.add_argument("--top-k", type=int, default=8, help="最终返回结果数 (默认: 8)")
    parser.add_argument(
        "--query-list", metavar="PATH", default=None,
        help="批量 query 模式: 从文件读取 query 列表 (一行一个), 跑完后输出召回一致率统计",
    )
    args = parser.parse_args()

    if args.query_list:
        with open(args.query_list, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"批量模式: 从 {args.query_list} 读取 {len(queries)} 个 query")
        run_batch(queries, args.top_k)
        return

    print("初始化 SessionSearcher + KGDatabase (一次, 两次 run 共享) ...")
    searcher = SessionSearcher("config/code_p3_config.yaml")
    db = KGDatabase("output/triple/knowledge_graph.db")

    r1 = run_one(1.0, args.query, args.top_k, searcher, db)
    r2 = run_one(2.0, args.query, args.top_k, searcher, db)

    print("\n" + "=" * 90)
    print("对比总结")
    print("=" * 90)

    for stage in ["dense", "sparse_summary", "sparse_cleaned", "chunk_fused",
                  "vector", "graph", "merged", "final"]:
        s1 = set(r1[stage])
        s2 = set(r2[stage])
        if stage == "final":
            same = s1 == s2
            print(f"\n[{stage}]")
            print(f"  ×1.0: {len(s1)} -> {sorted(s1)}")
            print(f"  ×2.0: {len(s2)} -> {sorted(s2)}")
            print(f"  一致: {same}")
        else:
            only_in_2 = s2 - s1
            only_in_1 = s1 - s2
            print(f"\n[{stage}]")
            print(f"  ×1.0: {len(s1)} 个, ×2.0: {len(s2)} 个, "
                  f"差: {len(only_in_2)} (×2.0 独有) / {len(only_in_1)} (×1.0 独有)")
            if only_in_2 and len(only_in_2) <= 5:
                print(f"  ×2.0 独有: {sorted(only_in_2)}")
            elif only_in_2:
                print(f"  ×2.0 独有 (前 5): {sorted(only_in_2)[:5]}")
            if only_in_1 and len(only_in_1) <= 5:
                print(f"  ×1.0 独有: {sorted(only_in_1)}")
            elif only_in_1:
                print(f"  ×1.0 独有 (前 5): {sorted(only_in_1)[:5]}")

    print(f"\n[Reranker 耗时]")
    print(f"  ×1.0: {r1['t_rerank']:.2f}s  ({len(r1['merged'])} 篇)")
    print(f"  ×2.0: {r2['t_rerank']:.2f}s  ({len(r2['merged'])} 篇)")
    if r1["t_rerank"] > 0:
        speedup = r2["t_rerank"] / r1["t_rerank"]
        print(f"  加速比: ×1.0 比 ×2.0 快 {speedup:.2f}x")


def run_batch(queries: list, top_k: int):
    print("初始化 SessionSearcher + KGDatabase (一次, 批量共享) ...")
    searcher = SessionSearcher("config/code_p3_config.yaml")
    db = KGDatabase("output/triple/knowledge_graph.db")

    n_total = len(queries)
    n_final_same = 0
    n_vector_same = 0
    sum_speedup = 0.0
    n_speedup = 0
    per_query_results = []

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{n_total}] Query: {q}")
        r1 = run_one(1.0, q, top_k, searcher, db)
        r2 = run_one(2.0, q, top_k, searcher, db)

        final_same = set(r1["final"]) == set(r2["final"])
        vector_same = set(r1["vector"]) == set(r2["vector"])
        speedup = r2["t_rerank"] / r1["t_rerank"] if r1["t_rerank"] > 0 else 0

        if final_same:
            n_final_same += 1
        if vector_same:
            n_vector_same += 1
        if speedup > 0:
            sum_speedup += speedup
            n_speedup += 1

        per_query_results.append({
            "query": q,
            "final_same": final_same,
            "vector_same": vector_same,
            "vector_count_1": len(r1["vector"]),
            "vector_count_2": len(r2["vector"]),
            "t_rerank_1": r1["t_rerank"],
            "t_rerank_2": r2["t_rerank"],
            "speedup": speedup,
            "final_1": r1["final"],
            "final_2": r2["final"],
        })

    print("\n" + "=" * 90)
    print(f"批量对比统计 ({n_total} 个 query)")
    print("=" * 90)
    print(f"  final top-{top_k} 完全一致: {n_final_same}/{n_total} ({100*n_final_same/n_total:.1f}%)")
    print(f"  vector 召回集合完全一致: {n_vector_same}/{n_total} ({100*n_vector_same/n_total:.1f}%)")
    if n_speedup > 0:
        avg_speedup = sum_speedup / n_speedup
        print(f"  平均 Reranker 加速比: {avg_speedup:.2f}x (×1.0 相对 ×2.0)")

    print(f"\n  {'Query':<20} {'final':<8} {'vector':<8} {'×1.0秒':<10} {'×2.0秒':<10} {'加速':<6}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    for r in per_query_results:
        final_mark = "✓" if r["final_same"] else "✗"
        vector_mark = "✓" if r["vector_same"] else "✗"
        print(f"  {r['query'][:18]:<20} {final_mark:<8} {vector_mark:<8} "
              f"{r['t_rerank_1']:<10.2f} {r['t_rerank_2']:<10.2f} {r['speedup']:<6.2f}")


if __name__ == "__main__":
    main()
