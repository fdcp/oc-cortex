"""
P4 alias expansion 完整 trace 测试 (4 combo × 3 query + 3 baseline = 15 search 调用).

每条 query × 每个 combo 走完整 P4 pipeline, 全程监控 15 个状态点, 导出 JSON 供后续报告生成.

用法:
    python3 tests/test_p4/test_p4_full.py

输出:
    tests/test_p4/test_p4_full_results.json
    tests/test_p4/test_p4_full.log    (运行期详细 trace)

测试矩阵 (case_sensitive 固定 False):
    combo   match_mode       kg_auto_mode
    1       exact            entity
    2       exact            triple
    3       word_boundary    entity
    4       word_boundary    triple

每条 query: 跑 1 baseline (alias OFF) + 4 combos (alias ON) = 5 search 调用.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from code_p1_utils import Config, setup_logger
from code_p3_qdrant_store import SearchResult
from code_p4_searcher import SessionSearcher
from code_p5e_db import KGDatabase


QUERIES = [
    "我在那个opencode对话中使用github-copilot的模型了",
    "介绍一下序列并行的原理",
    "帮我找一下BF16混合精度训练中关于序列并行、上下文并行(CP)、FlashAttention相关的内容",
]

COMBOS = [
    {"match_mode": "exact", "kg_mode": "entity"},
    {"match_mode": "exact", "kg_mode": "triple"},
    {"match_mode": "word_boundary", "kg_mode": "entity"},
    {"match_mode": "word_boundary", "kg_mode": "triple"},
]

TOP_K = 5
CANDIDATE_MULTIPLIER = 5


@dataclass
class TokenTrace:
    base_tokens: list[str] = field(default_factory=list)
    pseg_tagged: list[tuple[str, str]] = field(default_factory=list)
    pos_filtered: list[tuple[str, str]] = field(default_factory=list)
    pos_dropped: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class CollectionTrace:
    idf: dict[str, float] = field(default_factory=dict)
    ranked_top_k: list[tuple[str, float]] = field(default_factory=list)
    kg_hits: list[dict] = field(default_factory=list)
    extra_terms: list[str] = field(default_factory=list)
    final_tokens: list[str] = field(default_factory=list)
    sparse_top: list[dict] = field(default_factory=list)
    aggregated_tasks: list[dict] = field(default_factory=list)


@dataclass
class PipelineTrace:
    query: str
    top_k: int
    label: str
    cfg_overrides: dict
    alias_expansion_enabled: bool
    tokenize: TokenTrace = field(default_factory=TokenTrace)
    cleaned: CollectionTrace = field(default_factory=CollectionTrace)
    summary: CollectionTrace = field(default_factory=CollectionTrace)
    dense_top: list[dict] = field(default_factory=list)
    chunk_fused_top: list[dict] = field(default_factory=list)
    candidates_top: list[dict] = field(default_factory=list)
    final_top_k: list[dict] = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    extras_summary: dict = field(default_factory=dict)
    error: str = ""


def switch_combo(searcher: SessionSearcher, combo: dict | None) -> None:
    """切换 searcher 到指定 combo; combo=None 时关闭 alias expansion。"""
    if combo is None:
        searcher.alias_expansion_enabled = False
        return

    cfg = searcher.alias_expansion_cfg
    cfg["match_mode"] = combo["match_mode"]
    cfg["kg_db_path"] = f"output/{combo['kg_mode']}/knowledge_graph.db"
    path = searcher._resolve_path(cfg["kg_db_path"])
    searcher.kg_db = KGDatabase(str(path))
    searcher.alias_expansion_enabled = True


def _search_result_to_dict(r, score_field: str = "score") -> dict:
    payload = dict(r.payload or {})
    for k in ("task_summary",):
        if k in payload and isinstance(payload[k], str) and len(payload[k]) > 120:
            payload[k] = payload[k][:120] + "..."
    return {
        "point_id": r.point_id,
        score_field: r.score,
        "task_id": payload.get("task_id", ""),
        "task_label": payload.get("task_label", "")[:80],
        "session_id": payload.get("session_id", "")[:30],
    }


def _kg_lookup(searcher: SessionSearcher, ranked: list[tuple[str, float]], match_mode: str) -> list[dict]:
    """对 ranked top-K tokens 调 KG, 返回每个 token 的 hit 详情。"""
    max_aliases = int(searcher.alias_expansion_cfg.get("max_aliases_per_match", 2))
    case_sensitive = bool(searcher.alias_expansion_cfg.get("case_sensitive", False))
    out = []
    for token, idf in ranked:
        try:
            hits = searcher.kg_db.search_entities_exact(token, limit=10, match_mode=match_mode)
        except Exception as e:
            out.append({"token": token, "idf": idf, "error": str(e), "hits": []})
            continue
        hit_records = []
        for h in hits:
            canonical = h["name"]
            aliases = list(h.get("aliases") or [])
            def _eq(a, b):
                return a == b if case_sensitive else a.lower() == b.lower()
            matched = canonical if _eq(canonical, token) else None
            if not matched:
                for a in aliases:
                    if _eq(a, token):
                        matched = a
                        break
            candidates = [canonical] + aliases
            if matched:
                candidates = [c for c in candidates if not _eq(c, matched)]
            hit_records.append({
                "name": canonical,
                "aliases": aliases,
                "task_count": h.get("task_count", 0),
                "matched_in": "name" if matched == canonical else ("alias" if matched else None),
                "extra_candidates": candidates[:max_aliases],
            })
        out.append({"token": token, "idf": idf, "hits": hit_records, "hit_count": len(hit_records)})
    return out


def trace_pipeline(searcher: SessionSearcher, query: str, top_k: int, label: str,
                   cfg_overrides: dict | None) -> PipelineTrace:
    """完整 trace 一次 search pipeline。"""
    t_total = time.time()
    switch_combo(searcher, cfg_overrides)

    trace = PipelineTrace(
        query=query,
        top_k=top_k,
        label=label,
        cfg_overrides=cfg_overrides or {"alias_expansion": "OFF"},
        alias_expansion_enabled=searcher.alias_expansion_enabled,
    )

    if cfg_overrides is None:
        trace.tokenize.base_tokens = searcher._tokenize_query(query)
        t_cleaned_tok = searcher._tokenize_query(query)
        t_summary_tok = searcher._tokenize_query(query)
    else:
        import jieba.posseg as pseg
        tagged = [(w.word, w.flag) for w in pseg.cut(query)]
        trace.tokenize.pseg_tagged = tagged
        pos_keep = set(searcher.alias_expansion_cfg.get("pos_keep", ["n", "eng", "x"]))
        min_chars = int(searcher.alias_expansion_cfg.get("min_token_chars", 2))
        trace.tokenize.pos_filtered = [(t, p) for t, p in tagged if p in pos_keep and len(t) >= min_chars]
        trace.tokenize.pos_dropped = [(t, p) for t, p in tagged if (t, p) not in trace.tokenize.pos_filtered]
        trace.tokenize.base_tokens = searcher._tokenize_query(query)
        t_cleaned_tok = searcher._expand_query_for_sparse(query, "chunks_cleaned_text")
        t_summary_tok = searcher._expand_query_for_sparse(query, "chunks_summary")

    trace.cleaned.final_tokens = t_cleaned_tok
    trace.summary.final_tokens = t_summary_tok

    n_candidates = top_k * CANDIDATE_MULTIPLIER

    try:
        t = time.time()
        dense_results = searcher.store.search_dense(
            query, collection=searcher.store.tasks_collection, top_k=n_candidates
        )
        trace.timing_ms["dense"] = int((time.time() - t) * 1000)
        trace.dense_top = [_search_result_to_dict(r, "dense_score") for r in dense_results[:n_candidates]]

        for coll_name, coll in [("cleaned", "chunks_cleaned_text"), ("summary", "chunks_summary")]:
            target = getattr(trace, coll_name)
            tokens = t_cleaned_tok if coll == "cleaned" else t_summary_tok

            t = time.time()
            sparse_chunk = searcher.store.search_sparse_bm25_tokens(
                tokens, collection=coll, top_k=n_candidates * 3
            )
            trace.timing_ms[f"sparse_bm25_{coll_name}"] = int((time.time() - t) * 1000)
            target.sparse_top = [
                {"chunk_id": r.payload.get("chunk_id", ""), "task_id": r.payload.get("task_id", ""),
                 "bm25_score": r.score, "text_preview": r.payload.get("cleaned_text", r.payload.get("summary", ""))[:80]}
                for r in sparse_chunk[:n_candidates * 3]
            ]

            t = time.time()
            task_results = searcher._aggregate_chunks_to_tasks(sparse_chunk)
            trace.timing_ms[f"aggregate_{coll_name}"] = int((time.time() - t) * 1000)
            target.aggregated_tasks = [_search_result_to_dict(r, "task_score") for r in task_results]

            if cfg_overrides is not None and tokens:
                idf_map = searcher._compute_token_idf(
                    [t for t, _ in trace.tokenize.pos_filtered], coll
                )
                target.idf = idf_map
                max_k = int(searcher.alias_expansion_cfg.get("max_idf_tokens", 3))
                idf_floor = float(searcher.alias_expansion_cfg.get("idf_floor", 0.5))
                ranked = [(t, idf_map.get(t, 0.0)) for t, _ in trace.tokenize.pos_filtered]
                ranked = [(t, s) for t, s in ranked if s >= idf_floor]
                ranked.sort(key=lambda x: -x[1])
                target.ranked_top_k = ranked[:max_k]
                target.kg_hits = _kg_lookup(searcher, target.ranked_top_k, searcher.alias_expansion_cfg["match_mode"])
                base_set = set(trace.tokenize.base_tokens)
                target.extra_terms = [t for t in tokens if t not in base_set]

        t = time.time()
        chunk_fused = searcher.store._rrf_fuse(
            trace.summary.aggregated_tasks_as_results(searcher),
            trace.cleaned.aggregated_tasks_as_results(searcher),
            searcher.store.fuse_k,
        )
        trace.timing_ms["chunk_rrf"] = int((time.time() - t) * 1000)
        chunk_fused_as_search = [
            SearchResult(r.point_id, r.rrf_score, r.payload) for r in chunk_fused
        ]
        trace.chunk_fused_top = [
            {"task_id": r.payload.get("task_id", ""),
             "task_label": r.payload.get("task_label", "")[:80],
             "rrf_score": r.rrf_score,
             "dense_rank": r.dense_rank,
             "sparse_rank": r.sparse_rank}
            for r in chunk_fused[:n_candidates]
        ]

        t = time.time()
        fused = searcher.store._rrf_fuse(
            trace.dense_top_as_results(searcher),
            chunk_fused_as_search,
            searcher.store.fuse_k,
        )
        trace.timing_ms["dense_rrf"] = int((time.time() - t) * 1000)
        candidates = fused[:n_candidates]
        trace.candidates_top = [
            {"task_id": c.payload.get("task_id", ""),
             "task_label": c.payload.get("task_label", "")[:80],
             "rrf_score": c.rrf_score,
             "dense_rank": c.dense_rank,
             "sparse_rank": c.sparse_rank,
             "dense_score": c.dense_score,
             "sparse_score": c.sparse_score}
            for c in candidates
        ]

        t = time.time()
        final = searcher.search(query, top_k=top_k)
        trace.timing_ms["search_rerank"] = int((time.time() - t) * 1000)
        trace.final_top_k = [
            {"task_id": r.task_id, "task_label": r.task_label[:80],
             "task_summary": r.task_summary[:100] + ("..." if len(r.task_summary) > 100 else ""),
             "rerank_score": r.rerank_score, "hybrid_score": r.hybrid_score,
             "chunk_count": len(r.chunks)}
            for r in final
        ]

    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
        import traceback
        trace.error += "\n" + traceback.format_exc()

    trace.timing_ms["total"] = int((time.time() - t_total) * 1000)
    return trace


def _attach_results_helpers():
    """给 CollectionTrace 加 as_results 帮助方法 (避免改 dataclass 字段)。"""
    from dataclasses import dataclass

    def aggregated_tasks_as_results(self, searcher):
        return [
            SearchResult(r["point_id"], r["task_score"], {
                "task_id": r["task_id"], "task_label": r["task_label"],
                "session_id": r["session_id"], "task_summary": "",
            })
            for r in self.aggregated_tasks
        ]

    def dense_top_as_results(self, searcher):
        from code_p3_qdrant_store import SearchResult
        return [
            SearchResult(r["point_id"], r["dense_score"], {
                "task_id": r["task_id"], "task_label": r["task_label"],
                "session_id": r["session_id"], "task_summary": "",
            })
            for r in self.dense_top
        ]

    CollectionTrace.aggregated_tasks_as_results = aggregated_tasks_as_results
    PipelineTrace.dense_top_as_results = dense_top_as_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/code_p3_config.yaml")
    parser.add_argument("--output", default="tests/test_p4/test_p4_full_results.json")
    args = parser.parse_args()

    config = Config.load(args.config)
    setup_logger(log_file=None, level="WARNING")

    print("初始化搜索引擎 ...")
    t = time.time()
    searcher = SessionSearcher(config_path=args.config)
    print(f"  耗时 {time.time()-t:.1f}s  kg_db={'None' if searcher.kg_db is None else 'OK'}\n")

    _attach_results_helpers()

    all_traces: list[dict] = []

    for qi, query in enumerate(QUERIES, 1):
        print(f"\n{'#'*70}\n# Q{qi}: {query}\n{'#'*70}")

        print(f"\n  [Baseline] alias OFF")
        baseline = trace_pipeline(searcher, query, TOP_K, "baseline", None)
        all_traces.append(asdict(baseline))
        for r in baseline.final_top_k:
            print(f"    #{r.get('rerank_score', 0):.4f}  {r.get('task_label', '')[:60]}")

        for ci, combo in enumerate(COMBOS, 1):
            label = f"combo{ci}_{combo['match_mode']}_{combo['kg_mode']}"
            print(f"\n  [{label}] match_mode={combo['match_mode']}, kg={combo['kg_mode']}")
            trace = trace_pipeline(searcher, query, TOP_K, label, combo)
            all_traces.append(asdict(trace))
            print(f"    extras(cleaned)={trace.cleaned.extra_terms}  extras(summary)={trace.summary.extra_terms}")
            for r in trace.final_top_k:
                print(f"    #{r.get('rerank_score', 0):.4f}  {r.get('task_label', '')[:60]}")
            if trace.error:
                print(f"    ERROR: {trace.error[:200]}")

    print(f"\n\n保存结果到 {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "config_used": {
                "yaml_path": args.config,
                "alias_expansion_yaml": config.get("alias_expansion", {}),
            },
            "queries": QUERIES,
            "combos": COMBOS,
            "top_k": TOP_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "traces": all_traces,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"完成。共 {len(all_traces)} 条 trace。")


if __name__ == "__main__":
    main()