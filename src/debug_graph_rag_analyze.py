"""
分析 GraphRAG 调试日志: 读取 logs/debug_graph_rag/_summary.jsonl
+ 每个 query 的详细 JSON, 输出多维度分析报告。

分析维度:
  1. 实体抽取 (Stage 1): 抽取成功率、抽取数量分布
  2. 模糊匹配 (Stage 2): 精确 vs 模糊命中率
  3. BFS 扩散 (Stage 3): seed→expanded 扩散比
  4. 反 IDF 累加 (Stage 4): 贡献分布, 热门 vs 稀有实体
  5. 向量检索 (Stage 5): 各子阶段召回数, RRF 融合收敛
  6. 外层 RRF (Stage 6): 候选数
  7. Rerank (Stage 7): rerank 时延
  8. 来源分布 (Stage 8): vector / graph / both / neither
  9. 耗时 (Stage 9): 每阶段耗时占比

输出:
  logs/debug_graph_rag/analysis.md (markdown 报告)
  logs/debug_graph_rag/analysis.json (结构化数据)
"""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _load_all(d: Path) -> list[dict]:
    records = []
    for f in sorted(d.glob("*.json")):
        if f.name == "_summary.jsonl":
            continue
        if f.name.startswith("analysis"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def _by_db(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get("db", "?"), []).append(r)
    return out


def _safe_round(x, n=4):
    try:
        if isinstance(x, (int, float)):
            if math.isinf(x) or math.isnan(x):
                return None
            return round(float(x), n)
    except Exception:
        return None
    return None


def _stat_block(values: list) -> dict:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isinf(v) and not math.isnan(v)]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": _safe_round(min(vals)),
        "max": _safe_round(max(vals)),
        "mean": _safe_round(statistics.mean(vals)),
        "median": _safe_round(statistics.median(vals)),
    }


def analyze(records: list[dict]) -> dict:
    by_db = _by_db(records)
    out: dict = {
        "totals": {
            "records": len(records),
            "dbs": list(by_db.keys()),
        },
        "per_db": {},
    }

    for db_name, db_recs in by_db.items():
        n = len(db_recs)
        d = {}

        # 1. LLM 实体抽取
        extracted_counts = [len(r.get("query_entities", [])) for r in db_recs]
        empty_extractions = sum(1 for c in extracted_counts if c == 0)
        d["1_llm_extraction"] = {
            "queries": n,
            "empty_extractions": empty_extractions,
            "empty_rate": _safe_round(empty_extractions / n, 4) if n else None,
            "extracted_count_stats": _stat_block(extracted_counts),
            "examples": [
                {"query": r["query"], "entities": r["query_entities"]}
                for r in db_recs[:3]
            ],
        }

        # 2. 模糊匹配
        seed_counts = [len(r.get("seed_entities", [])) for r in db_recs]
        no_seed = sum(1 for c in seed_counts if c == 0)
        d["2_seed_match"] = {
            "queries": n,
            "no_seed_queries": no_seed,
            "no_seed_rate": _safe_round(no_seed / n, 4) if n else None,
            "seed_count_stats": _stat_block(seed_counts),
        }

        # 3. BFS 扩散
        expanded_counts = [r.get("expanded_count", 0) for r in db_recs]
        d["3_bfs_expand"] = {
            "queries": n,
            "expanded_count_stats": _stat_block(expanded_counts),
            "max_nodes_hit": sum(1 for c in expanded_counts if c >= 30),
        }

        # 4. 反 IDF 累加 (从 stage 4 详细)
        unique_graph_tasks = []
        for r in db_recs:
            for s in r.get("stages", []):
                if s["stage"] == "4_inverse_idf_accumulate":
                    unique_graph_tasks.append(s.get("unique_graph_tasks", 0))
                    break
        d["4_idf_accumulate"] = {
            "unique_graph_tasks_stats": _stat_block(unique_graph_tasks),
        }

        # 5. 向量检索
        vec_counts = [r.get("vector_results_count", 0) for r in db_recs]
        d["5_vector_search"] = {
            "vector_results_stats": _stat_block(vec_counts),
        }

        vector_t: list[int] = []
        graph_t: list[int] = []
        rerank_t: list[int] = []
        total_t: list[int] = []
        per_stage_from_dbg: dict[str, list[int]] = {}
        for r in db_recs:
            for s in r.get("stages", []):
                if s["stage"] != "9_timing":
                    continue
                if s.get("vector_time_ms") is not None:
                    vector_t.append(s["vector_time_ms"])
                if s.get("graph_time_ms") is not None:
                    graph_t.append(s["graph_time_ms"])
                if s.get("rerank_time_ms") is not None:
                    rerank_t.append(s["rerank_time_ms"])
                if s.get("total_time_ms") is not None:
                    total_t.append(s["total_time_ms"])
                for p in s.get("per_stage_ms", []):
                    per_stage_from_dbg.setdefault(p["stage"], []).append(p["elapsed_ms"])
                break
        d["9_timing"] = {
            "vector_ms": _stat_block(vector_t),
            "graph_ms": _stat_block(graph_t),
            "rerank_ms": _stat_block(rerank_t),
            "total_ms": _stat_block(total_t),
            "per_stage_raw": {
                stage_name: _stat_block(times)
                for stage_name, times in per_stage_from_dbg.items()
            },
        }

        # 8. 来源分布 (聚合)
        source_aggr = {"vector": 0, "graph": 0, "both": 0, "neither": 0}
        for r in db_recs:
            for k, v in r.get("source_distribution", {}).items():
                source_aggr[k] = source_aggr.get(k, 0) + v
        total_final = sum(source_aggr.values())
        d["8_source_distribution"] = {
            "raw_counts": source_aggr,
            "total_final_results": total_final,
            "percentages": {
                k: _safe_round(v / total_final * 100, 2) if total_final else 0
                for k, v in source_aggr.items()
            },
            "by_query": [
                {
                    "query": r["query"],
                    "source_distribution": r["source_distribution"],
                    "expanded_count": r.get("expanded_count", 0),
                    "extracted_entities": r.get("query_entities", []),
                }
                for r in db_recs
            ],
        }

        # 整体得分分布
        rerank_scores = []
        hybrid_scores = []
        for r in db_recs:
            for fr in r.get("final_results", []):
                if fr.get("rerank_score") is not None:
                    rerank_scores.append(fr["rerank_score"])
                if fr.get("hybrid_score") is not None:
                    hybrid_scores.append(fr["hybrid_score"])
        d["score_distribution"] = {
            "rerank_score": _stat_block(rerank_scores),
            "hybrid_score": _stat_block(hybrid_scores),
        }

        out["per_db"][db_name] = d

    return out


def render_markdown(analysis: dict) -> str:
    lines = []
    lines.append("# GraphRAG 调试分析报告")
    lines.append("")
    lines.append(f"- 总查询数: **{analysis['totals']['records']}**")
    lines.append(f"- DB: {', '.join(analysis['totals']['dbs'])}")
    lines.append("")

    for db_name, d in analysis["per_db"].items():
        lines.append(f"## DB: `{db_name}`")
        lines.append("")

        # 1
        s = d["1_llm_extraction"]
        lines.append("### Stage 1: LLM 实体抽取")
        lines.append(f"- 查询数: {s['queries']}")
        lines.append(f"- 空抽取 (退化为纯向量): **{s['empty_extractions']}** ({_pct(s['empty_extractions'], s['queries'])})")
        es = s["extracted_count_stats"]
        lines.append(f"- 抽取实体数: min={es.get('min')} max={es.get('max')} mean={es.get('mean')} median={es.get('median')}")
        lines.append(f"- 示例 (前 3):")
        for ex in s["examples"]:
            lines.append(f"  - `{ex['query']}` → {ex['entities']}")
        lines.append("")

        # 2
        s = d["2_seed_match"]
        lines.append("### Stage 2: 实体模糊匹配")
        lines.append(f"- 无种子 (图谱完全不命中): **{s['no_seed_queries']}** ({_pct(s['no_seed_queries'], s['queries'])})")
        sc = s["seed_count_stats"]
        lines.append(f"- 种子实体数: min={sc.get('min')} max={sc.get('max')} mean={sc.get('mean')}")
        lines.append("")

        # 3
        s = d["3_bfs_expand"]
        lines.append("### Stage 3: BFS 扩散")
        es = s["expanded_count_stats"]
        lines.append(f"- 扩散节点数: min={es.get('min')} max={es.get('max')} mean={es.get('mean')} median={es.get('median')}")
        lines.append(f"- 命中 max_nodes (30) 上限: **{s['max_nodes_hit']}** / {d['1_llm_extraction']['queries']}")
        lines.append("")

        # 4
        s = d["4_idf_accumulate"]
        ugt = s["unique_graph_tasks_stats"]
        lines.append("### Stage 4: 反 IDF 累加")
        lines.append(f"- 唯一图谱 task 数: min={ugt.get('min')} max={ugt.get('max')} mean={ugt.get('mean')}")
        lines.append("")

        # 5
        s = d["5_vector_search"]
        lines.append("### Stage 5: 向量检索 (粗排)")
        vs = s["vector_results_stats"]
        lines.append(f"- 候选数 (top_k * 2 = 10): min={vs.get('min')} max={vs.get('max')} mean={vs.get('mean')}")
        lines.append("")

        # 8
        s = d["8_source_distribution"]
        lines.append("### Stage 8: 来源分布 (Top-K 最终结果)")
        lines.append(f"- 总最终结果数: {s['total_final_results']}")
        lines.append("| 来源 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for k in ["vector", "graph", "both", "neither"]:
            v = s["raw_counts"].get(k, 0)
            lines.append(f"| {k} | {v} | {s['percentages'].get(k, 0)}% |")
        lines.append("")
        lines.append("**按 query:**")
        lines.append("")
        lines.append("| Query | 抽取实体 | 扩散 | vector | graph | both | neither |")
        lines.append("|-------|----------|------|--------|-------|------|---------|")
        for bq in s["by_query"]:
            sd = bq["source_distribution"]
            lines.append(
                f"| `{bq['query']}` | {bq['extracted_entities']} | {bq['expanded_count']} "
                f"| {sd.get('vector', 0)} | {sd.get('graph', 0)} "
                f"| {sd.get('both', 0)} | {sd.get('neither', 0)} |"
            )
        lines.append("")

        # 9
        lines.append("### Stage 9: 阶段耗时 (ms)")
        timing = d["9_timing"]
        lines.append("**核心 4 段耗时 (来自 rag.search 内部 final_debug):**")
        lines.append("")
        lines.append("| 阶段 | count | min | max | mean | median |")
        lines.append("|------|-------|-----|-----|------|--------|")
        for stage_name, key in [
            ("vector_time (粗排)", "vector_ms"),
            ("graph_time (BFS+IDF)", "graph_ms"),
            ("rerank_time (Qwen3)", "rerank_ms"),
            ("total_time (search 整体)", "total_ms"),
        ]:
            t = timing.get(key, {})
            lines.append(
                f"| {stage_name} | {t.get('count', 0)} | {t.get('min', '-')} | {t.get('max', '-')} "
                f"| {t.get('mean', '-')} | {t.get('median', '-')} |"
            )
        lines.append("")
        lines.append("**其他细粒度阶段 (来自 debug 脚本自身计时):**")
        lines.append("")
        lines.append("| 阶段 | count | min | max | mean | median |")
        lines.append("|------|-------|-----|-----|------|--------|")
        for stage_name in [
            "1_llm_entity_extraction",
            "2_seed_entity_match",
            "3_bfs_expand",
            "4_inverse_idf_accumulate",
            "5_vector_search",
        ]:
            t = timing.get("per_stage_raw", {}).get(stage_name, {})
            if t:
                lines.append(
                    f"| {stage_name} | {t.get('count', 0)} | {t.get('min', '-')} | {t.get('max', '-')} "
                    f"| {t.get('mean', '-')} | {t.get('median', '-')} |"
                )
        lines.append("")

        # 得分
        sd = d["score_distribution"]
        lines.append("### 最终结果得分分布")
        rs = sd["rerank_score"]
        hs = sd["hybrid_score"]
        lines.append(f"- rerank_score: min={rs.get('min')} max={rs.get('max')} mean={rs.get('mean')}")
        lines.append(f"- hybrid_score: min={hs.get('min')} max={hs.get('max')} mean={hs.get('mean')}")
        lines.append("")

    lines.append("## 关键发现与建议")
    lines.append("")
    lines.append("见 analysis.md 末尾的\"洞察\"章节（自动生成）")
    return "\n".join(lines)


def _pct(num, denom):
    if not denom:
        return "0%"
    return f"{round(num / denom * 100, 1)}%"


def render_insights(analysis: dict) -> str:
    """基于分析数据自动生成洞察"""
    lines = ["## 自动洞察", ""]
    for db_name, d in analysis["per_db"].items():
        lines.append(f"### DB: `{db_name}`")
        lines.append("")

        # 空抽取
        empty_rate = d["1_llm_extraction"]["empty_rate"] or 0
        if empty_rate > 0:
            lines.append(f"- ⚠️ **{empty_rate*100:.0f}%** 的 query LLM 实体抽取返回空, "
                         f"图谱路径完全没启用 (退化为纯向量)")
        else:
            lines.append(f"- ✓ 所有 query 都成功抽取到实体")

        # 命中 BFS 上限
        max_hit = d["3_bfs_expand"]["max_nodes_hit"]
        queries_n = d["1_llm_extraction"]["queries"]
        if max_hit:
            lines.append(f"- ⚠️ **{max_hit}/{queries_n}** query 命中 `max_expand_nodes=30` 上限, "
                         f"可能图谱过密导致扩散爆炸")

        # 来源分布
        sd = d["8_source_distribution"]["raw_counts"]
        total = sum(sd.values()) or 1
        v_pct = sd.get("vector", 0) / total * 100
        g_pct = sd.get("graph", 0) / total * 100
        b_pct = sd.get("both", 0) / total * 100
        lines.append(f"- Top-K 来源: vector-only={v_pct:.1f}%, "
                     f"graph-only={g_pct:.1f}%, both={b_pct:.1f}%, neither={sd.get('neither', 0)/total*100:.1f}%")
        if b_pct > 50:
            lines.append(f"  - ✓ **图谱贡献显著** ({b_pct:.0f}% 双通道命中)")
        elif b_pct < 20 and sd.get("graph", 0) > 0:
            lines.append(f"  - ⚠️ 图谱贡献低, 大部分结果来自纯向量通道")

        # 耗时
        timing = d["9_timing"]
        if "7_rerank" in timing:
            rerank = timing["7_rerank"]
            lines.append(f"- Rerank 耗时: mean={rerank.get('mean')}ms, "
                         f"max={rerank.get('max')}ms (占总时延的瓶颈)")
        if "1_llm_entity_extraction" in timing:
            llm = timing["1_llm_entity_extraction"]
            lines.append(f"- LLM 实体抽取: mean={llm.get('mean')}ms")
        if "5_vector_search" in timing:
            v = timing["5_vector_search"]
            lines.append(f"- 向量检索: mean={v.get('mean')}ms")

        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="logs/debug_graph_rag")
    ap.add_argument("--out-md", default="logs/debug_graph_rag/analysis.md")
    ap.add_argument("--out-json", default="logs/debug_graph_rag/analysis.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"No input dir: {in_dir}")
        return

    records = _load_all(in_dir)
    if not records:
        print(f"No records found in {in_dir}")
        return

    print(f"Loaded {len(records)} records")
    analysis = analyze(records)

    # 写 json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out_json}")

    # 写 md
    md = render_markdown(analysis)
    insights = render_insights(analysis)
    full_md = md + "\n\n" + insights
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"Wrote: {args.out_md}")

    # 控制台摘要
    print("\n" + "=" * 60)
    print(insights)


if __name__ == "__main__":
    main()
