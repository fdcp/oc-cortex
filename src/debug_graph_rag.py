"""
GraphRAG 调试脚本：记录关键步骤的输出和得分。

记录范围 (与 code_p5e_graph_rag.GraphRAGSearcher.search 对齐):
  Stage 0  输入: query, 检索参数
  Stage 1  LLM 实体抽取: 抽取到的实体列表
  Stage 2  实体模糊匹配: 种子实体 (精确命中 + 模糊 top-3)
  Stage 3  BFS 扩散: 扩散节点数
  Stage 4  反 IDF 累加: 每个 task 的图谱得分及贡献实体
  Stage 5  向量检索: SessionSearcher 5 段子阶段 (Dense / chunk_summary / chunks_cleaned_text / chunk RRF / dense RRF)
  Stage 6  外层 RRF 融合: vector + graph rank-based 融合
  Stage 7  Rerank: 重排序结果
  Stage 8  来源分布: vector / graph / both 各占多少
  Stage 9  耗时: 每一阶段耗时 (ms)

输出: logs/debug_graph_rag/{db_name}_{query_slug}.json
        + logs/debug_graph_rag/_summary.jsonl (一行一个 query 摘要)

用法:
  export OPENCODE_ZEN_API_KEY=...
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
  python3 src/debug_graph_rag.py --db entity
  python3 src/debug_graph_rag.py --db triple --query "OpenCode MCP"
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path


# ============================================================
# 环境: HF 离线 + repo 根目录进 path
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# 先 set env, 再 import sentence_transformers / transformers
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
    "OpenCode",
    "OpenCode MCP",
    "FlashAttention 实现原理",
    "知识图谱 实体对齐",
    "Qwen3 Reranker 精排",
    "session 管理 数据库",
    "Markdown 文档升级",
    "深度学习 优化器",
]


# ============================================================
# 核心: 包装一层"打点" — 不改源码, 只额外记录
# ============================================================
def _slugify(s: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s.strip())
    return s[:40] or "q"


def _safe_round(x, ndigits=6):
    try:
        if isinstance(x, (int, float)):
            if math.isinf(x):
                return "inf"
            if math.isnan(x):
                return "nan"
            return round(float(x), ndigits)
    except Exception:
        return None
    return x


def _record_stage(stage_log, name, payload, t0):
    """记录一阶段: payload + 耗时 (ms)"""
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    stage_log.append({
        "stage": name,
        "elapsed_ms": int((time.time() - t0) * 1000),
        **payload,
    })


def debug_search(
    searcher: SessionSearcher,
    rag: GraphRAGSearcher,
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
    use_graph_rag: bool = True,
) -> dict:
    """复刻 GraphRAGSearcher.search 的流程, 在每个阶段额外打点。

    不改源码, 走公共 API 拿到最终结果 + debug 字典,
    再额外收集每个阶段的中间量用于分析。
    """
    t0 = time.time()
    stages: list[dict] = []

    record: dict = {
        "query": query,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "use_graph_rag": use_graph_rag,
        "graph_channel_weight": rag.graph_channel_weight,
        "bfs_depth": rag.bfs_depth,
        "max_expand_nodes": rag.max_expand_nodes,
        "max_graph_tasks": rag.max_graph_tasks,
        "rerank_multiplier": rag.rerank_multiplier,
        "outer_rrf_k": rag._OUTER_RRF_K,
        "stages": stages,
    }

    # ==========================================================
    # Stage 1: LLM 实体抽取
    # ==========================================================
    t_stage = time.time()
    query_entities = extract_query_entities(query)
    _record_stage(stages, "1_llm_entity_extraction", {
        "extracted_entities": query_entities,
        "count": len(query_entities),
    }, t_stage)
    record["query_entities"] = query_entities

    # ==========================================================
    # Stage 2: 实体模糊匹配 (种子)
    # ==========================================================
    t_stage = time.time()
    seed_entities: list[str] = []
    seed_details: list[dict] = []
    for e in query_entities:
        node = rag.db.get_node(e)
        if node:
            seed_entities.append(e)
            seed_details.append({
                "extracted": e,
                "matched": e,
                "match_type": "exact",
                "task_count": node["task_count"],
            })
        else:
            matches = rag.db.search_entities(e, limit=3)
            for m in matches:
                seed_entities.append(m["name"])
                seed_details.append({
                    "extracted": e,
                    "matched": m["name"],
                    "match_type": "fuzzy",
                    "task_count": m["task_count"],
                })
    seed_entities = list(set(seed_entities))
    _record_stage(stages, "2_seed_entity_match", {
        "seed_entities": seed_entities,
        "match_details": seed_details,
        "exact_count": sum(1 for d in seed_details if d["match_type"] == "exact"),
        "fuzzy_count": sum(1 for d in seed_details if d["match_type"] == "fuzzy"),
    }, t_stage)
    record["seed_entities"] = seed_entities

    # ==========================================================
    # Stage 3: BFS 扩散
    # ==========================================================
    t_stage = time.time()
    if seed_entities:
        expanded = rag.db.bfs_expand(
            seed_entities,
            depth=rag.bfs_depth,
            max_nodes=rag.max_expand_nodes,
        )
    else:
        expanded = set()
    _record_stage(stages, "3_bfs_expand", {
        "expanded_entities": sorted(expanded),
        "expanded_count": len(expanded),
        "seed_to_expand_ratio": (
            round(len(expanded) / max(len(seed_entities), 1), 3)
        ),
    }, t_stage)
    record["expanded_entities"] = sorted(expanded)
    record["expanded_count"] = len(expanded)

    # ==========================================================
    # Stage 4: 反 IDF 累加 (按 task 计分)
    # ==========================================================
    t_stage = time.time()
    total_tasks = max(rag._total_tasks, 1)
    task_scores: dict[str, dict] = {}
    per_entity_contrib: list[dict] = []
    for entity_name in expanded:
        node = rag.db.get_node(entity_name)
        if not node:
            continue
        tc = max(node["task_count"], 1)
        contrib = math.log(1 + total_tasks / tc)
        per_entity_contrib.append({
            "entity": entity_name,
            "task_count": tc,
            "idf_contrib": _safe_round(contrib, 4),
            "source_tasks": len(node["source_tasks"]),
        })
        for tid in node["source_tasks"]:
            if tid not in task_scores:
                task_scores[tid] = {"score": 0.0, "entities": []}
            task_scores[tid]["score"] += contrib
            task_scores[tid]["entities"].append(entity_name)

    sorted_task_scores = sorted(
        task_scores.items(), key=lambda x: -x[1]["score"]
    )[: rag.max_graph_tasks]
    graph_candidate_task_ids = [tid for tid, _ in sorted_task_scores]
    _record_stage(stages, "4_inverse_idf_accumulate", {
        "total_tasks_global": total_tasks,
        "unique_graph_tasks": len(task_scores),
        "graph_candidates_after_truncate": len(graph_candidate_task_ids),
        "top10_entity_contrib": sorted(
            per_entity_contrib, key=lambda x: -x["idf_contrib"]
        )[:10],
        "top10_task_scores": [
            {"task_id": tid, "score": _safe_round(info["score"], 4),
             "n_entities": len(info["entities"])}
            for tid, info in sorted_task_scores[:10]
        ],
        "id_contrib_distribution": _id_contrib_hist(per_entity_contrib),
    }, t_stage)

    # ==========================================================
    # Stage 5: 向量检索 (公共 API, 一次性拿到结果)
    # ==========================================================
    t_stage = time.time()
    from code_p1_utils import build_retrieval_query
    retrieval_query = build_retrieval_query(query, rag.query_instruction)
    vector_top_k = max(int(top_k * rag.rerank_multiplier), top_k)
    vector_results = searcher.search(
        query=retrieval_query,
        top_k=vector_top_k,
        skip_rerank=True,
    )
    _record_stage(stages, "5_vector_search", {
        "retrieval_query": retrieval_query,
        "vector_top_k_requested": vector_top_k,
        "vector_results_count": len(vector_results),
        "vector_score_stats": _score_stats(
            [r.hybrid_score for r in vector_results]
        ),
        "top10_vector_results": [
            {
                "rank": i + 1,
                "task_id": r.task_id,
                "hybrid_score": _safe_round(r.hybrid_score, 6),
                "label": r.task_label[:60],
            }
            for i, r in enumerate(vector_results[:10])
        ],
    }, t_stage)
    record["vector_results_count"] = len(vector_results)

    # ==========================================================
    # Stage 6 + 7: 走公共 search, 拿外层 RRF + rerank 结果 + debug
    # ==========================================================
    t_stage = time.time()
    final_results, final_debug = rag.search(
        query=query,
        top_k=top_k,
        use_reranker=use_reranker,
        use_graph_rag=use_graph_rag,
    )
    _record_stage(stages, "6_outer_rrf_merge", {
        "merged_candidates": final_debug.get("merged_candidates"),
        "outer_rrf_k": final_debug.get("outer_rrf_k"),
        "graph_channel_weight": final_debug.get("graph_channel_weight"),
    }, t_stage)
    _record_stage(stages, "7_rerank", {
        "rerank_time_ms": final_debug.get("rerank_time_ms"),
        "use_reranker": use_reranker,
        "final_count": final_debug.get("final_count"),
    }, t_stage)

    # ==========================================================
    # Stage 8: 来源分布 (自己再算一遍, 跟 final_debug 对照)
    # ==========================================================
    t_stage = time.time()
    vector_ids = {r.task_id for r in vector_results}
    graph_ids = set(graph_candidate_task_ids)
    source_dist = {"vector": 0, "graph": 0, "both": 0, "neither": 0}
    final_detail: list[dict] = []
    for r in final_results:
        in_v = r.task_id in vector_ids
        in_g = r.task_id in graph_ids
        if in_v and in_g:
            source_dist["both"] += 1
            source = "both"
        elif in_g:
            source_dist["graph"] += 1
            source = "graph"
        elif in_v:
            source_dist["vector"] += 1
            source = "vector"
        else:
            source_dist["neither"] += 1
            source = "neither"
        # 找到该 task 的图谱得分 (如果有)
        g_score = task_scores.get(r.task_id, {}).get("score")
        # 找到该 task 的向量 rank
        v_rank = next(
            (i + 1 for i, vr in enumerate(vector_results) if vr.task_id == r.task_id),
            None,
        )
        # 找到该 task 的图谱 rank
        g_rank = next(
            (i + 1 for i, (tid, _) in enumerate(sorted_task_scores) if tid == r.task_id),
            None,
        )
        final_detail.append({
            "rank": len(final_detail) + 1,
            "task_id": r.task_id,
            "source": source,
            "rerank_score": _safe_round(r.rerank_score, 6),
            "hybrid_score": _safe_round(r.hybrid_score, 6),
            "vector_rank": v_rank,
            "graph_rank": g_rank,
            "graph_raw_score": _safe_round(g_score, 6),
            "label": r.task_label[:80],
            "summary_preview": r.task_summary[:120],
        })
    _record_stage(stages, "8_source_distribution", {
        "source_dist": source_dist,
        "final_results": final_detail,
    }, t_stage)

    # ==========================================================
    # Stage 9: 耗时
    # ==========================================================
    _record_stage(stages, "9_timing", {
        "vector_time_ms": final_debug.get("vector_time_ms"),
        "graph_time_ms": final_debug.get("graph_time_ms"),
        "rerank_time_ms": final_debug.get("rerank_time_ms"),
        "total_time_ms": final_debug.get("total_time_ms"),
        "per_stage_ms": [
            {"stage": s["stage"], "elapsed_ms": s["elapsed_ms"]}
            for s in stages
        ],
    }, t_stage)

    record["source_distribution"] = source_dist
    record["final_results"] = final_detail
    record["total_time_ms"] = final_debug.get("total_time_ms")
    record["total_wall_ms"] = int((time.time() - t0) * 1000)

    return record


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "min": _safe_round(min(scores)),
        "max": _safe_round(max(scores)),
        "mean": _safe_round(sum(scores) / len(scores)),
    }


def _id_contrib_hist(per_entity_contrib: list[dict]) -> dict:
    if not per_entity_contrib:
        return {"count": 0}
    contribs = [c["idf_contrib"] for c in per_entity_contrib]
    task_counts = [c["task_count"] for c in per_entity_contrib]
    return {
        "count": len(contribs),
        "contrib_min": _safe_round(min(contribs), 4),
        "contrib_max": _safe_round(max(contribs), 4),
        "contrib_mean": _safe_round(sum(contribs) / len(contribs), 4),
        "task_count_min": min(task_counts),
        "task_count_max": max(task_counts),
        "task_count_mean": _safe_round(sum(task_counts) / len(task_counts), 2),
    }


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="GraphRAG 调试脚本: 记录每个关键步骤的输出和得分"
    )
    ap.add_argument("--db", choices=["triple", "entity", "both"], default="both",
                    help="使用哪个 KG 数据库 (默认: both)")
    ap.add_argument("--query", action="append", default=None,
                    help="单次查询 (可多次指定; 不指定则用 DEFAULT_QUERIES)")
    ap.add_argument("--queries-from-file", default=None,
                    help="从 JSONL 读 query, 每行 {query: str}")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-graph", action="store_true",
                    help="跑纯向量路径 (use_graph_rag=False)")
    ap.add_argument("--graph-channel-weight", type=float, default=None,
                    help="覆盖图谱通道权重 (默认 0.3)")
    ap.add_argument("--out-dir", default="logs/debug_graph_rag",
                    help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "_summary.jsonl"

    # 选 query
    if args.queries_from_file:
        queries = []
        with open(args.queries_from_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                queries.append(d["query"])
    elif args.query:
        queries = args.query
    else:
        queries = DEFAULT_QUERIES

    # 选 db
    db_choices = ["triple", "entity"] if args.db == "both" else [args.db]

    # 初始化 searcher (只初始化一次, 跨 db 共享, 因为 Qdrant 跟 db 无关)
    logger.info("初始化 SessionSearcher ...")
    searcher = SessionSearcher(config_path="config/code_p3_config.yaml")

    # 给每次 run 一个 summary 追加行
    summary_f = open(summary_path, "a", encoding="utf-8")

    try:
        for db_name in db_choices:
            db_path = f"output/{db_name}/knowledge_graph.db"
            if not Path(db_path).exists():
                logger.warning(f"DB 不存在, 跳过: {db_path}")
                continue
            logger.info(f"\n{'='*60}\nDB: {db_name} ({db_path})\n{'='*60}")
            kg_db = KGDatabase(db_path)
            rag_kwargs = dict(
                searcher=searcher,
                kg_db=kg_db,
                rerank_multiplier=2.0,
            )
            if args.graph_channel_weight is not None:
                rag_kwargs["graph_channel_weight"] = args.graph_channel_weight
            rag = GraphRAGSearcher(**rag_kwargs)

            stats = kg_db.get_stats()
            logger.info(
                f"  图谱: {stats['nodes']} 节点, {stats['edges']} 边, "
                f"平均 task/实体={stats['avg_task_count_per_entity']}"
            )

            for q in queries:
                logger.info(f"\n  ▶ Query: {q!r}")
                t_run = time.time()
                rec = debug_search(
                    searcher=searcher,
                    rag=rag,
                    query=q,
                    top_k=args.top_k,
                    use_reranker=not args.no_rerank,
                    use_graph_rag=not args.no_graph,
                )
                rec["db"] = db_name
                rec["db_stats"] = stats
                rec["wall_ms"] = int((time.time() - t_run) * 1000)

                slug = _slugify(q)
                out_file = out_dir / f"{db_name}_{slug}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)
                logger.info(f"    写入: {out_file}")

                # summary 一行
                try:
                    out_rel = str(out_file.resolve().relative_to(_REPO_ROOT.resolve()))
                except ValueError:
                    out_rel = str(out_file)
                summary_f.write(json.dumps({
                    "db": db_name,
                    "query": q,
                    "extracted_entities": rec["query_entities"],
                    "seed_entities": rec["seed_entities"],
                    "expanded_count": rec["expanded_count"],
                    "vector_results_count": rec["vector_results_count"],
                    "source_distribution": rec["source_distribution"],
                    "total_time_ms": rec["total_time_ms"],
                    "wall_ms": rec["wall_ms"],
                    "out_file": out_rel,
                }, ensure_ascii=False) + "\n")
                summary_f.flush()

                # 打印来源分布
                sd = rec["source_distribution"]
                logger.info(
                    f"    来源: vector={sd['vector']} graph={sd['graph']} "
                    f"both={sd['both']} neither={sd['neither']}"
                )
    finally:
        summary_f.close()

    logger.info(f"\n全部完成, summary: {summary_path}")


if __name__ == "__main__":
    main()
