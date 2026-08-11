"""
把 doc/GraphRAG_full_trace_data/_all_runs.json 渲染成 1 个详细 markdown。
每个 run 一节, 完整展示 9 阶段 + final top-5。
"""
import json
import math
import re
from pathlib import Path


DATA_DIR = Path("doc/GraphRAG_full_trace_data")


def _fmt_score(x, n=6):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        if math.isnan(x):
            return "nan"
        return f"{x:.{n}f}"
    return str(x)


def _slug(q):
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", q).strip("_")[:60]


def render_one(rec: dict) -> str:
    lines = []
    db = rec["db"]
    mode = "GRAPH (use_graph_rag=True)" if rec["use_graph_rag"] else "VEC-ONLY (use_graph_rag=False)"
    lines.append(f"## Run: `{db}` × `{mode}` × query=`{rec['query']}`")
    lines.append("")
    lines.append(f"- top_k = {rec['top_k']}")
    lines.append(f"- rerank_multiplier = {rec['rerank_multiplier']}")
    lines.append(f"- graph_channel_weight = {rec['graph_channel_weight']}")
    lines.append(f"- outer_rrf_k = {rec['outer_rrf_k']}")
    lines.append(f"- bfs_depth = {rec['bfs_depth']}, max_expand_nodes = {rec['max_expand_nodes']}, max_graph_tasks = {rec['max_graph_tasks']}")
    lines.append(f"- 完整耗时: **{rec['wall_ms']} ms** ({rec['wall_ms']/1000:.1f}s)")
    lines.append(f"- 来源分布: `{rec['source_distribution']}`")
    lines.append("")

    stages = rec["stages"]

    if "1_llm_entity_extraction" in stages:
        s = stages["1_llm_entity_extraction"]
        lines.append("### Stage 1: LLM 实体抽取")
        lines.append(f"- **输入**: {s['input']}")
        lines.append(f"- **输出**: `{s['output']}`")
        lines.append(f"- 抽取数: {s['n_extracted']}")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    if "2_seed_entity_match" in stages:
        s = stages["2_seed_entity_match"]
        lines.append("### Stage 2: 实体模糊匹配")
        lines.append(f"- **输入**: extracted = `{s['input']['extracted_entities']}`")
        lines.append(f"- **输出 seed_entities**: `{s['output_seed_entities']}`")
        lines.append(f"- exact 命中: {s['exact_count']}, fuzzy 命中: {s['fuzzy_count']}")
        if s["match_details"]:
            lines.append("")
            lines.append("**匹配详情:**")
            lines.append("")
            lines.append("| 抽取实体 | 匹配实体 | 类型 | task_count |")
            lines.append("|----------|----------|------|------------|")
            for d in s["match_details"]:
                lines.append(f"| `{d['extracted']}` | `{d['matched']}` | {d['match_type']} | {d['task_count']} |")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    if "3_bfs_expand" in stages:
        s = stages["3_bfs_expand"]
        lines.append("### Stage 3: BFS 扩散")
        lines.append(f"- **输入**: seed = `{s['input']['seed_entities']}`, depth = {s['input']['depth']}, max_nodes = {s['input']['max_nodes']}")
        lines.append(f"- **输出 expanded ({s['n_expanded']} 节点)**: `{s['output_expanded']}`")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    if "4_inverse_idf_accumulate" in stages:
        s = stages["4_inverse_idf_accumulate"]
        out = s["output"]
        lines.append("### Stage 4: 反 IDF 累加")
        lines.append(f"- **输入**: expanded_count = {s['input']['expanded_entities_count']}, "
                     f"global N = {s['input']['total_tasks_global_N']}, "
                     f"max_graph_tasks = {s['input']['max_graph_tasks']}")
        lines.append(f"- **公式**: `{s['input']['formula']}`")
        lines.append(f"- **输出**: 唯一 graph task = {out['n_unique_graph_tasks']}, "
                     f"截断后 graph_candidates = {out['n_graph_candidates_after_truncate']}")
        lines.append("")
        lines.append("**所有实体的贡献 (按 idf 降序):**")
        lines.append("")
        lines.append("| entity | task_count | n_source_tasks | 计算 | idf_contrib |")
        lines.append("|--------|------------|----------------|------|-------------|")
        for ec in out["per_entity_contrib_all"]:
            lines.append(f"| `{ec['entity']}` | {ec['task_count']} | {ec['n_source_tasks']} | {ec['formula']} | {ec['idf_contrib']} |")
        lines.append("")
        lines.append("**所有 task 的图谱得分 (按 score 降序):**")
        lines.append("")
        lines.append("| rank | task_id | n_entities | entities | score |")
        lines.append("|------|---------|------------|----------|-------|")
        for i, t in enumerate(out["per_task_score_all"], 1):
            ents_short = ", ".join(t["entities"][:5])
            if len(t["entities"]) > 5:
                ents_short += f" ... (+{len(t['entities'])-5})"
            lines.append(f"| {i} | `{t['task_id']}` | {t['n_entities']} | {ents_short} | {t['score']} |")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    # Stage 5: 5 子阶段
    for sub_key, sub_title in [
        ("5a_dense_tasks", "Stage 5a: Dense → tasks 集合"),
        ("5b_chunks_summary_aggregate", "Stage 5b: chunks_summary 检索 + 聚合到 task"),
        ("5c_chunks_cleaned_text_aggregate", "Stage 5c: chunks_cleaned_text sparse 检索 + 聚合到 task"),
        ("5d_chunk_rrf_fuse", "Stage 5d: 两路 chunk RRF 融合"),
        ("5e_final_rrf_fuse", "Stage 5e: chunk RRF + Dense 最终 RRF"),
    ]:
        if sub_key not in stages:
            continue
        s = stages[sub_key]
        lines.append(f"### {sub_title}")
        lines.append(f"- **输入**: `{s['input']}`")
        n_field = next((v for k, v in s.items() if k.startswith("n_") and k != "n_entities"), "?")
        lines.append(f"- 召回数: {n_field}")
        if s.get("top10"):
            lines.append("")
            lines.append("**top-10:**")
            lines.append("")
            score_key = "score" if "score" in s["top10"][0] else ("rrf_score" if "rrf_score" in s["top10"][0] else None)
            if score_key:
                lines.append(f"| rank | {score_key} | task_id | label |")
                lines.append(f"|------|{'-'*10}-|---------|-------|")
                for i, r in enumerate(s["top10"], 1):
                    lines.append(f"| {i} | {_fmt_score(r[score_key])} | `{r['task_id']}` | {r['label']} |")
        lines.append("")

    if "5_vector_search_total" in stages:
        s = stages["5_vector_search_total"]
        lines.append(f"### Stage 5 总览: vector search 整体")
        lines.append(f"- 输入: `{s['input']}`")
        lines.append(f"- 候选数: {s['n_candidates_returned']}")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    if "6_outer_rrf_merge" in stages:
        s = stages["6_outer_rrf_merge"]
        lines.append("### Stage 6: 外层 RRF 融合 (vector + graph)")
        lines.append(f"- **输入**: `{s['input']}`")
        lines.append(f"- 融合候选数: {s['n_merged']}")
        lines.append("")
        lines.append("**完整融合结果 (按 outer_rrf 降序):**")
        lines.append("")
        lines.append("| rank | task_id | v_rank | g_rank | v_rrf | g_raw | outer_rrf | label |")
        lines.append("|------|---------|--------|--------|-------|-------|-----------|-------|")
        for i, c in enumerate(s["merged_full"][:30], 1):
            lines.append(
                f"| {i} | `{c['task_id']}` | {c['v_rank']} | {c['g_rank']} "
                f"| {_fmt_score(c['v_rrf_score'])} | {_fmt_score(c['g_raw_score'])} "
                f"| {_fmt_score(c['outer_rrf'])} | {c['label']} |"
            )
        if len(s["merged_full"]) > 30:
            lines.append(f"| ... | (剩余 {len(s['merged_full'])-30} 个) | | | | | | |")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    if "7_rerank_full" in stages:
        s = stages["7_rerank_full"]
        lines.append("### Stage 7: Rerank 完整候选分数")
        lines.append(f"- **输入**: `{s['input']}`")
        out = s["output"]
        lines.append(f"- **rerank_input_n**: {out['rerank_input_n']}")
        lines.append("")
        lines.append("**完整 Rerank 分数 (按分数降序):**")
        lines.append("")
        lines.append("| rr_rank | orig_index | rerank_score | task_id | label |")
        lines.append("|---------|------------|--------------|---------|-------|")
        for rr in out["rerank_scores"]:
            lines.append(
                f"| {out['rerank_scores'].index(rr)+1} "
                f"| {rr['orig_index']} | {_fmt_score(rr['rerank_score'])} "
                f"| `{rr['task_id']}` | {rr['label']} |"
            )
        lines.append("")
        lines.append(f"**Top-{len(out['top5_task_ids'])} task_ids**: `{out['top5_task_ids']}`")
        lines.append(f"- 耗时: {s['elapsed_ms']}ms")
        lines.append("")

    # Stage 8
    if "8_source_distribution" in stages:
        s = stages["8_source_distribution"]
        lines.append("### Stage 8: 来源分布")
        lines.append(f"- top5_task_ids: `{s['top5_task_ids']}`")
        lines.append(f"- source_distribution: `{s['source_distribution']}`")
        if "note" in s:
            lines.append(f"- note: {s['note']}")
        lines.append("")

    # Final Top-K with task_summary
    lines.append("### Final Top-K (含 task_summary 完整内容)")
    lines.append("")
    for r in rec["final_top_k"]:
        lines.append(f"#### #{r['rank']} `{r['task_id']}`")
        lines.append(f"- label: **{r['label']}**")
        lines.append(f"- rerank_score: `{_fmt_score(r['rerank_score'])}`")
        if r['outer_rrf_score'] is not None:
            lines.append(f"- outer_rrf_score: `{_fmt_score(r['outer_rrf_score'])}`")
        lines.append(f"- v_rank: {r['v_rank']}, g_rank: {r['g_rank']}, g_raw_score: {_fmt_score(r['g_raw_score'])}")
        lines.append("")
        lines.append("**task_summary 完整内容:**")
        lines.append("")
        lines.append("> " + r['task_summary'].replace("\n", "\n> "))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    in_path = DATA_DIR / "_all_runs.json"
    if not in_path.exists():
        print(f"Missing: {in_path}")
        return
    runs = json.load(open(in_path))
    print(f"Loaded {len(runs)} runs")

    lines = []
    lines.append("# GraphRAG 完整 Pipeline Trace (3 query × 2 db × 2 mode = 12 个 run)")
    lines.append("")
    lines.append("> 调试数据: `doc/GraphRAG_full_trace_data/*.json`")
    lines.append("> 调试脚本: `src/debug_full_trace.py`")
    lines.append("> 参数: `top_k=5, rerank_multiplier=1, graph_channel_weight=0.3, outer_rrf_k=30, bfs_depth=1, max_expand_nodes=30, max_graph_tasks=50`")
    lines.append("")
    lines.append("每节展示一个 run 的完整 9 阶段: 输入、输出、分数、计算公式。")
    lines.append("")

    # 按 db → query → mode 排序
    mode_order = {"False": 0, "True": 1}
    runs_sorted = sorted(
        runs,
        key=lambda r: (r["db"], _slug(r["query"]), mode_order[str(r["use_graph_rag"])]),
    )
    for rec in runs_sorted:
        lines.append(render_one(rec))

    out_md = Path("doc/GraphRAG_full_trace.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_md} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
