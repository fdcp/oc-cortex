"""
GraphRAG 完整 pipeline trace 调试器。
对每个 query 同时跑 use_graph_rag=False/True 两种模式, 记录每一步:
  - Stage 1: LLM 实体抽取
  - Stage 2: 实体模糊匹配
  - Stage 3: BFS 扩散
  - Stage 4: 反 IDF 累加
  - Stage 5a: Dense → tasks
  - Stage 5b: chunk_summary 检索 → 聚合到 task
  - Stage 5c: chunks_cleaned_text sparse 检索 → 聚合到 task
  - Stage 5d: 两路 chunk RRF 融合
  - Stage 5e: chunk RRF + Dense 最终 RRF
  - Stage 6: 外层 RRF (vector + graph)
  - Stage 7: Rerank 完整候选分数
  - Stage 8: 来源分布 + final Top-K
  - Stage 9: 耗时

参数:
  --rerank-multiplier 1 (跟 notebook 对齐)
  --top-k 5

输出: doc/GraphRAG_full_trace.json (12 份 trace)
       doc/GraphRAG_full_trace.md (渲染后的 markdown)

用法:
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 OPENCODE_ZEN_API_KEY=...
  python3 src/debug_full_trace.py
  python3 src/debug_full_trace.py --query "GPU" --top-k 3 --rerank-multiplier 1
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from code_p1_utils import Config  # noqa: E402

_p3_config = Config.load("config/code_p3_config.yaml")
_cache_folder = _p3_config.get("embedding.cache_folder")
if _cache_folder is None:
    _cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
_offline_mode = _p3_config.get("embedding.offline_mode", True)

from code_p3_hf_config import setup_hf_env  # noqa: E402
setup_hf_env(offline_mode=_offline_mode, cache_folder=_cache_folder)

from loguru import logger  # noqa: E402

from code_p4_searcher import SessionSearcher  # noqa: E402
from code_p5e_db import KGDatabase  # noqa: E402
from code_p5e_graph_rag import (  # noqa: E402
    GraphRAGSearcher,
    extract_query_entities,
)


# ============================================================
# 默认测试 query
# ============================================================
DEFAULT_QUERIES = [
    "GPU的对比和选型",
    "序列并行(Sequence Parallel,SP)",
    "flashattention的原理和历史",
]


# ============================================================
# helper
# ============================================================
def _safe(x, n=6):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return str(x)
        return round(x, n)
    return x


def _slugify(s: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s.strip())
    return s[:60] or "q"


# ============================================================
# 复刻 SessionSearcher.search 的 5 个子阶段
# ============================================================
def vector_5_substages(searcher: SessionSearcher, query: str, top_k: int) -> dict:
    """复刻 SessionSearcher.search 内部 5 个子阶段, 全部打点。

    输入: query, top_k (会被 searcher 内部乘 candidate_multiplier=5 得到 n_candidates=50)
    输出: dict 含 5 个子阶段的 (输入, top_k, 召回数, top-N task_id+分数)
          + 最终 candidates 列表
    """
    n_candidates = top_k * 5
    fuse_k = searcher.store.fuse_k
    retrieval_query = query  # 不拼 instruction, debug 测的是原始 query
    substages = {}

    # 1: Dense → tasks
    dense_results = searcher.store.search_dense(
        retrieval_query,
        collection=searcher.store.tasks_collection,
        top_k=n_candidates,
    )
    substages["5a_dense_tasks"] = {
        "input": {"query": retrieval_query, "collection": searcher.store.tasks_collection, "top_k": n_candidates},
        "n_results": len(dense_results),
        "top10": [
            {"task_id": r.payload.get("task_id", ""), "score": _safe(r.score, 6),
             "label": (searcher.task_map.get(r.payload.get("task_id", ""), None) and
                       searcher.task_map[r.payload.get("task_id", "")].task_label[:50] or "")}
            for r in dense_results[:10]
        ],
    }

    # 2: chunk_summary (sparse) 检索 + 聚合
    sparse_top_k = n_candidates * 3
    summary_chunk_results = searcher._search_chunks_summary(retrieval_query, sparse_top_k)
    summary_task_results = searcher._aggregate_chunks_to_tasks(summary_chunk_results)
    substages["5b_chunks_summary_aggregate"] = {
        "input": {"query": retrieval_query, "method": searcher.chunks_summary_method, "top_k_chunks": sparse_top_k},
        "n_chunk_hits": len(summary_chunk_results),
        "n_task_after_agg": len(summary_task_results),
        "top10": [
            {"task_id": r.payload.get("task_id", ""), "score": _safe(r.score, 6),
             "label": (searcher.task_map.get(r.payload.get("task_id", ""), None) and
                       searcher.task_map[r.payload.get("task_id", "")].task_label[:50] or "")}
            for r in summary_task_results[:10]
        ],
    }

    # 3: chunks_cleaned_text (sparse) 检索 + 聚合
    if searcher.store.sparse_method == "bge_m3":
        sparse_chunk_results = searcher.store.search_sparse_bge_m3(
            retrieval_query,
            collection=searcher.store.chunks_cleaned_text_collection,
            top_k=sparse_top_k,
        )
    else:
        sparse_chunk_results = searcher.store.search_sparse_bm25(
            retrieval_query,
            collection=searcher.store.chunks_cleaned_text_collection,
            top_k=sparse_top_k,
        )
    sparse_task_results = searcher._aggregate_chunks_to_tasks(sparse_chunk_results)
    substages["5c_chunks_cleaned_text_aggregate"] = {
        "input": {"query": retrieval_query, "method": searcher.store.sparse_method, "top_k_chunks": sparse_top_k},
        "n_chunk_hits": len(sparse_chunk_results),
        "n_task_after_agg": len(sparse_task_results),
        "top10": [
            {"task_id": r.payload.get("task_id", ""), "score": _safe(r.score, 6),
             "label": (searcher.task_map.get(r.payload.get("task_id", ""), None) and
                       searcher.task_map[r.payload.get("task_id", "")].task_label[:50] or "")}
            for r in sparse_task_results[:10]
        ],
    }

    # 4: 两路 chunk RRF 融合
    chunk_fused = searcher.store._rrf_fuse(
        summary_task_results,
        sparse_task_results,
        fuse_k,
    )
    substages["5d_chunk_rrf_fuse"] = {
        "input": {
            "summary_task_n": len(summary_task_results),
            "sparse_task_n": len(sparse_task_results),
            "fuse_k": fuse_k,
            "formula": f"RRF(d) = 1/({fuse_k} + rank_d),  score = sum(1/(k+rank))",
        },
        "n_fused": len(chunk_fused),
        "top10": [
            {"task_id": r.payload.get("task_id", ""), "rrf_score": _safe(r.rrf_score, 6),
             "label": (searcher.task_map.get(r.payload.get("task_id", ""), None) and
                       searcher.task_map[r.payload.get("task_id", "")].task_label[:50] or "")}
            for r in chunk_fused[:10]
        ],
    }
    # _rrf_fuse 期望 SearchResult (有 .score), 但 chunk_fused 是 HybridResult (有 .rrf_score)
    from code_p3_qdrant_store import SearchResult
    chunk_fused_as_search = [
        SearchResult(r.point_id, r.rrf_score, r.payload)
        for r in chunk_fused
    ]

    # 5: chunk RRF + Dense 最终 RRF
    fused = searcher.store._rrf_fuse(
        dense_results,
        chunk_fused_as_search,
        fuse_k,
    )
    candidates = fused[:n_candidates]
    substages["5e_final_rrf_fuse"] = {
        "input": {
            "dense_n": len(dense_results),
            "chunk_fused_n": len(chunk_fused),
            "fuse_k": fuse_k,
            "n_candidates_target": n_candidates,
        },
        "n_candidates": len(candidates),
        "top10": [
            {"task_id": c.payload.get("task_id", ""), "rrf_score": _safe(c.rrf_score, 6),
             "label": (searcher.task_map.get(c.payload.get("task_id", ""), None) and
                       searcher.task_map[c.payload.get("task_id", "")].task_label[:50] or "")}
            for c in candidates[:10]
        ],
    }

    return {
        "substages": substages,
        "candidates": [
            {"task_id": c.payload.get("task_id", ""), "rrf_score": _safe(c.rrf_score, 6),
             "label": (searcher.task_map.get(c.payload.get("task_id", ""), None) and
                       searcher.task_map[c.payload.get("task_id", "")].task_label[:50] or "")}
            for c in candidates
        ],
    }


def rerank_all_candidates(searcher: SessionSearcher, query: str, task_ids: list[str], top_k: int) -> dict:
    """对 task_ids 列表 (顺序: 5e 输出的 candidates 顺序) 全部跑 Rerank, 返回完整分数列表。

    注意: 这里直接复用 searcher.reranker, 跟 GraphRAG 内部 _rerank 一致。
    """
    if not searcher.reranker or not task_ids:
        return {"rerank_scores": [], "top_k_indices": [], "rerank_input_texts": []}

    # 准备 texts (跟 _rerank 一致: 用 task_summary)
    texts = []
    for tid in task_ids:
        t = searcher.task_map.get(tid)
        texts.append(t.task_summary if t else "")

    # 跑 rerank, top_k=全量 (返回所有分数)
    rerank_results = searcher.reranker.rank(query, texts, top_k=len(texts))

    return {
        "rerank_input_n": len(task_ids),
        "rerank_input_task_ids": task_ids,
        "rerank_input_labels": [
            (searcher.task_map.get(tid).task_label if searcher.task_map.get(tid) else "")
            for tid in task_ids
        ],
        "rerank_scores": [
            {"orig_index": rr.index, "task_id": task_ids[rr.index],
             "label": (searcher.task_map.get(task_ids[rr.index]).task_label
                       if searcher.task_map.get(task_ids[rr.index]) else ""),
             "rerank_score": _safe(rr.score, 6)}
            for rr in rerank_results
        ],
        "top5_task_ids": [task_ids[rr.index] for rr in rerank_results[:top_k]],
    }


# ============================================================
# 单次 query 单次模式 (graph_rag=True/False) 的完整 trace
# ============================================================
def trace_one(searcher: SessionSearcher, rag: GraphRAGSearcher, query: str,
              top_k: int, use_graph_rag: bool, tasks_lookup: dict) -> dict:
    """单个 run: 跑 use_graph_rag 模式, 收集所有 9 阶段数据。"""
    t0 = time.time()
    stages = {}

    # ==========================================
    # Stage 1: LLM 实体抽取 (只在 graph 模式跑)
    # ==========================================
    if use_graph_rag:
        t = time.time()
        query_entities = extract_query_entities(query)
        stages["1_llm_entity_extraction"] = {
            "input": {"query": query, "model": "hy3", "max_tokens": 2000, "temperature": 0.1},
            "output": query_entities,
            "n_extracted": len(query_entities),
            "elapsed_ms": int((time.time() - t) * 1000),
        }

        # ==========================================
        # Stage 2: 实体模糊匹配
        # ==========================================
        t = time.time()
        seed_entities = []
        match_details = []
        for e in query_entities:
            node = rag.db.get_node(e)
            if node:
                seed_entities.append(e)
                match_details.append({
                    "extracted": e, "matched": e, "match_type": "exact",
                    "task_count": node["task_count"],
                })
            else:
                matches = rag.db.search_entities(e, limit=3)
                for m in matches:
                    seed_entities.append(m["name"])
                    match_details.append({
                        "extracted": e, "matched": m["name"], "match_type": "fuzzy",
                        "task_count": m["task_count"],
                    })
        seed_entities = list(set(seed_entities))
        stages["2_seed_entity_match"] = {
            "input": {"extracted_entities": query_entities},
            "output_seed_entities": seed_entities,
            "match_details": match_details,
            "exact_count": sum(1 for d in match_details if d["match_type"] == "exact"),
            "fuzzy_count": sum(1 for d in match_details if d["match_type"] == "fuzzy"),
            "elapsed_ms": int((time.time() - t) * 1000),
        }

        # ==========================================
        # Stage 3: BFS 扩散
        # ==========================================
        t = time.time()
        if seed_entities:
            expanded = rag.db.bfs_expand(
                seed_entities,
                depth=rag.bfs_depth,
                max_nodes=rag.max_expand_nodes,
            )
        else:
            expanded = set()
        stages["3_bfs_expand"] = {
            "input": {"seed_entities": seed_entities, "depth": rag.bfs_depth,
                      "max_nodes": rag.max_expand_nodes},
            "output_expanded": sorted(expanded),
            "n_expanded": len(expanded),
            "edges_traversed_count": "见 Stage 4 详细 (从 expanded 拉所有节点的 source_tasks)",
            "elapsed_ms": int((time.time() - t) * 1000),
        }

        # ==========================================
        # Stage 4: 反 IDF 累加
        # ==========================================
        t = time.time()
        total_tasks_global = max(rag._total_tasks, 1)
        task_scores = {}
        per_entity_contrib = []
        for entity_name in expanded:
            node = rag.db.get_node(entity_name)
            if not node:
                continue
            tc = max(node["task_count"], 1)
            contrib = math.log(1 + total_tasks_global / tc)
            per_entity_contrib.append({
                "entity": entity_name,
                "task_count": tc,
                "formula": f"log(1 + {total_tasks_global}/{tc}) = {contrib:.6f}",
                "idf_contrib": _safe(contrib, 4),
                "n_source_tasks": len(node["source_tasks"]),
            })
            for tid in node["source_tasks"]:
                if tid not in task_scores:
                    task_scores[tid] = {"score": 0.0, "entities": []}
                task_scores[tid]["score"] += contrib
                task_scores[tid]["entities"].append(entity_name)

        # 按 score 降序, 截到 max_graph_tasks
        sorted_task_scores = sorted(task_scores.items(), key=lambda x: -x[1]["score"])
        graph_candidate_task_ids = [
            tid for tid, _ in sorted_task_scores[:rag.max_graph_tasks]
        ]
        stages["4_inverse_idf_accumulate"] = {
            "input": {"expanded_entities_count": len(expanded),
                      "total_tasks_global_N": total_tasks_global,
                      "max_graph_tasks": rag.max_graph_tasks,
                      "formula": "task_score = sum over its entities: log(1 + N/task_count)"},
            "output": {
                "n_unique_graph_tasks": len(task_scores),
                "n_graph_candidates_after_truncate": len(graph_candidate_task_ids),
                "per_entity_contrib_all": sorted(
                    per_entity_contrib, key=lambda x: -x["idf_contrib"]
                ),
                "per_task_score_all": [
                    {"task_id": tid, "score": _safe(info["score"], 4),
                     "n_entities": len(info["entities"]), "entities": info["entities"]}
                    for tid, info in sorted_task_scores[:rag.max_graph_tasks]
                ],
            },
            "elapsed_ms": int((time.time() - t) * 1000),
        }

    # ==========================================
    # Stage 5: 向量检索 (5 个子阶段)
    # ==========================================
    t_vec_total = time.time()
    # 跟 GraphRAG.search 一致: vector_top_k = max(top_k * rerank_multiplier, top_k)
    vector_top_k = max(int(top_k * rag.rerank_multiplier), top_k)
    vec5 = vector_5_substages(searcher, query, vector_top_k)
    # 5a-5e 写入 stages (带 input/output)
    for k, v in vec5["substages"].items():
        stages[k] = v
    stages["5_vector_search_total"] = {
        "input": {"query": query, "vector_top_k": vector_top_k,
                  "candidate_multiplier_internal": 5,
                  "fuse_k": searcher.store.fuse_k},
        "n_candidates_returned": len(vec5["candidates"]),
        "elapsed_ms": int((time.time() - t_vec_total) * 1000),
    }
    vector_candidates = vec5["candidates"]  # list of {task_id, rrf_score, label}

    # ==========================================
    # Stage 6: 外层 RRF (vector + graph), 仅 graph 模式
    # ==========================================
    if use_graph_rag:
        t = time.time()
        gw = rag.graph_channel_weight
        outer_k = rag._OUTER_RRF_K
        # 构造 rank map
        rank_v = {c["task_id"]: i + 1 for i, c in enumerate(vector_candidates)}
        rank_g = {tid: i + 1 for i, tid in enumerate(graph_candidate_task_ids)}
        INF = float("inf")
        all_tids = set(rank_v) | set(rank_g)
        fused = []
        for tid in all_tids:
            rv = rank_v.get(tid, INF)
            rg = rank_g.get(tid, INF)
            rrf = (1.0 - gw) / (outer_k + rv) + gw / (outer_k + rg)
            v_score = next((c["rrf_score"] for c in vector_candidates if c["task_id"] == tid), None)
            g_score = next((info["score"] for tk, info in sorted_task_scores if tk == tid), None)
            fused.append({
                "task_id": tid, "v_rank": rank_v.get(tid), "g_rank": rank_g.get(tid),
                "v_rrf_score": v_score, "g_raw_score": _safe(g_score, 4),
                "outer_rrf": _safe(rrf, 6),
                "label": (searcher.task_map.get(tid).task_label if searcher.task_map.get(tid) else ""),
            })
        fused.sort(key=lambda x: -x["outer_rrf"])
        stages["6_outer_rrf_merge"] = {
            "input": {
                "vector_candidates_n": len(vector_candidates),
                "graph_candidates_n": len(graph_candidate_task_ids),
                "graph_channel_weight": gw, "outer_rrf_k": outer_k,
                "formula": f"RRF(t) = (1-{gw})/({outer_k} + rank_v) + {gw}/({outer_k} + rank_g),  不在通道视为 rank=+inf",
            },
            "n_merged": len(fused),
            "merged_full": fused,
            "elapsed_ms": int((time.time() - t) * 1000),
        }
        outer_candidates_for_rerank = fused  # graph mode 下, rerank 用 outer RRF 后的全部
        final_debug = None

        # ==========================================
        # Stage 7: Rerank (完整 50+ 候选分数)
        # ==========================================
        t = time.time()
        task_ids_in_order = [c["task_id"] for c in fused]
        rerank_out = rerank_all_candidates(searcher, query, task_ids_in_order, top_k)
        stages["7_rerank_full"] = {
            "input": {
                "query": query,
                "n_candidates_in": len(task_ids_in_order),
                "top_k": top_k,
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
            },
            "output": rerank_out,
            "elapsed_ms": int((time.time() - t) * 1000),
        }
        # 来源分布
        vector_ids = set(c["task_id"] for c in vector_candidates)
        graph_ids = set(graph_candidate_task_ids)
        top5 = rerank_out["top5_task_ids"]
        source_dist = {"vector": 0, "graph": 0, "both": 0, "neither": 0}
        for tid in top5:
            in_v = tid in vector_ids
            in_g = tid in graph_ids
            if in_v and in_g:
                source_dist["both"] += 1
            elif in_g:
                source_dist["graph"] += 1
            elif in_v:
                source_dist["vector"] += 1
            else:
                source_dist["neither"] += 1
        stages["8_source_distribution"] = {
            "top5_task_ids": top5,
            "source_distribution": source_dist,
        }
    else:
        # no-graph 模式: 直接 rerank vector candidates
        t = time.time()
        task_ids_in_order = [c["task_id"] for c in vector_candidates]
        rerank_out = rerank_all_candidates(searcher, query, task_ids_in_order, top_k)
        stages["7_rerank_full"] = {
            "input": {
                "query": query,
                "n_candidates_in": len(task_ids_in_order),
                "top_k": top_k,
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                "note": "no-graph 模式, 候选直接 = vector 50",
            },
            "output": rerank_out,
            "elapsed_ms": int((time.time() - t) * 1000),
        }
        stages["8_source_distribution"] = {
            "top5_task_ids": rerank_out["top5_task_ids"],
            "source_distribution": {"vector": 5, "graph": 0, "both": 0, "neither": 0},
            "note": "no-graph 模式, 全部 vector-only",
        }
        outer_candidates_for_rerank = None
        final_debug = None

    # ==========================================
    # Stage 9: 耗时汇总 + Final Top-K 完整信息 (含 task_summary)
    # ==========================================
    final_top_k = []
    top5_ids = stages["8_source_distribution"]["top5_task_ids"]
    for rank, tid in enumerate(top5_ids, 1):
        t_info = tasks_lookup.get(tid, {})
        rerank_score = next(
            (r["rerank_score"] for r in stages["7_rerank_full"]["output"]["rerank_scores"]
             if r["task_id"] == tid),
            None,
        )
        if use_graph_rag and outer_candidates_for_rerank is not None:
            outer = next((c for c in outer_candidates_for_rerank if c["task_id"] == tid), {})
            v_rank = outer.get("v_rank")
            g_rank = outer.get("g_rank")
            outer_rrf = outer.get("outer_rrf")
            g_raw = outer.get("g_raw_score")
        else:
            v_rank = next(
                (i + 1 for i, c in enumerate(vector_candidates) if c["task_id"] == tid),
                None,
            )
            g_rank = None
            outer_rrf = next(
                (c["rrf_score"] for c in vector_candidates if c["task_id"] == tid),
                None,
            )
            g_raw = None
        final_top_k.append({
            "rank": rank, "task_id": tid,
            "label": t_info.get("task_label", ""),
            "task_summary": t_info.get("task_summary", ""),
            "rerank_score": rerank_score,
            "outer_rrf_score": outer_rrf,
            "v_rank": v_rank, "g_rank": g_rank, "g_raw_score": g_raw,
        })

    record = {
        "query": query,
        "use_graph_rag": use_graph_rag,
        "top_k": top_k,
        "rerank_multiplier": rag.rerank_multiplier,
        "graph_channel_weight": rag.graph_channel_weight,
        "outer_rrf_k": rag._OUTER_RRF_K,
        "bfs_depth": rag.bfs_depth,
        "max_expand_nodes": rag.max_expand_nodes,
        "max_graph_tasks": rag.max_graph_tasks,
        "stages": stages,
        "final_top_k": final_top_k,
        "source_distribution": stages["8_source_distribution"]["source_distribution"],
        "wall_ms": int((time.time() - t0) * 1000),
    }
    return record


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["triple", "entity", "both"], default="both")
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rerank-multiplier", type=float, default=1.0)
    ap.add_argument("--graph-channel-weight", type=float, default=0.3)
    ap.add_argument("--out-dir", default="doc/GraphRAG_full_trace_data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = args.query if args.query else DEFAULT_QUERIES
    db_choices = ["triple", "entity"] if args.db == "both" else [args.db]

    # 加载 tasks.jsonl 一次 (用于 final summary)
    tasks_lookup = {}
    with open("output/tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tasks_lookup[t["task_id"]] = t

    logger.info("初始化 SessionSearcher (共享) ...")
    searcher = SessionSearcher(config_path="config/code_p3_config.yaml")

    all_runs = []
    for db_name in db_choices:
        db_path = f"output/{db_name}/knowledge_graph.db"
        if not Path(db_path).exists():
            logger.warning(f"DB 不存在, 跳过: {db_path}")
            continue
        logger.info(f"\n{'='*60}\nDB: {db_name}\n{'='*60}")
        kg_db = KGDatabase(db_path)
        rag = GraphRAGSearcher(
            searcher=searcher,
            kg_db=kg_db,
            graph_channel_weight=args.graph_channel_weight,
            rerank_multiplier=args.rerank_multiplier,
        )
        for q in queries:
            for use_graph_rag in [False, True]:
                mode_label = "GRAPH" if use_graph_rag else "VEC-ONLY"
                logger.info(f"\n  ▶ [{db_name}/{mode_label}] {q!r}")
                rec = trace_one(
                    searcher=searcher,
                    rag=rag,
                    query=q,
                    top_k=args.top_k,
                    use_graph_rag=use_graph_rag,
                    tasks_lookup=tasks_lookup,
                )
                rec["db"] = db_name
                all_runs.append(rec)
                # 立即 dump 一份 (避免中断丢失)
                with open(out_dir / f"{db_name}_{_slugify(q)}_{'graph' if use_graph_rag else 'nograph'}.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"    wall: {rec['wall_ms']}ms, "
                    f"top5: {[r['label'][:25] for r in rec['final_top_k']]}"
                )

    # 汇总
    with open(out_dir / "_all_runs.json", "w", encoding="utf-8") as f:
        json.dump(all_runs, f, ensure_ascii=False, indent=2)
    logger.info(f"\n全部完成: {len(all_runs)} 个 run, 数据在 {out_dir}/")


if __name__ == "__main__":
    main()
