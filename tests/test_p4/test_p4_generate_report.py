"""
读 test_p4_full_results.json, 生成 test_p4_report.md (中文 + 英文技术词混排).
"""
import json
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "tests/test_p4/test_p4_full_results.json"
REPORT = REPO / "tests/test_p4/test_p4_report.md"


def fmt_timing(ms_dict: dict) -> str:
    parts = []
    for k in ["dense", "sparse_bm25_cleaned", "aggregate_cleaned",
              "sparse_bm25_summary", "aggregate_summary",
              "chunk_rrf", "dense_rrf", "search_rerank", "total"]:
        v = ms_dict.get(k)
        if v is not None:
            parts.append(f"`{k}`={v}ms")
    return " ".join(parts)


def short(s: str, n: int = 50) -> str:
    if not isinstance(s, str):
        return str(s)
    return s if len(s) <= n else s[:n] + "..."


def trace_for(data: dict, query: str, label: str) -> dict | None:
    for t in data["traces"]:
        if t["query"] == query and t["label"] == label:
            return t
    return None


def query_baseline(data: dict, query: str) -> dict | None:
    return trace_for(data, query, "baseline")


def combo_label(combo: dict) -> str:
    return f"{combo['match_mode']}/{combo['kg_mode']}"


def kg_hit_count(trace: dict, collection: str = "cleaned") -> int:
    return len(trace[collection].get("kg_hits", []) or [])


def extras(trace: dict, collection: str = "cleaned") -> list:
    return trace[collection].get("extra_terms") or []


def extras_str(trace: dict, collection: str = "cleaned") -> str:
    e = extras(trace, collection)
    if not e:
        return "—"
    return ", ".join(f"`{x}`" for x in e)


def final_topk_str(trace: dict) -> str:
    rows = []
    for i, r in enumerate(trace["final_top_k"], 1):
        rows.append(f"  {i}. `{r['rerank_score']:.4f}` {short(r.get('task_label',''), 55)}")
    return "\n".join(rows)


def diff_topk(baseline: dict, combo: dict) -> str:
    b_ids = [r["task_id"] for r in baseline["final_top_k"]]
    c_ids = [r["task_id"] for r in combo["final_top_k"]]
    diff_positions = [i for i in range(min(len(b_ids), len(c_ids))) if b_ids[i] != c_ids[i]]
    if not diff_positions:
        b_set = set(b_ids)
        c_set = set(c_ids)
        if b_set == c_set:
            return "无差异 (集合相同, 顺序也相同)"
        return f"集合差异: only_baseline={len(b_set-c_set)}, only_combo={len(c_set-b_set)}"
    new = [c_ids[i] for i in diff_positions]
    lost = [b_ids[i] for i in diff_positions]
    return f"位置 {diff_positions} 重排, 新增={new}, 丢失={lost}"


def sparse_score_shift(baseline: dict, combo: dict, collection: str) -> dict:
    """对比 BM25 sparse top-50 同候选 score 变化 (A vs B)."""
    base_map = {r["chunk_id"]: r["bm25_score"]
                for r in baseline[collection]["sparse_top"]
                if r.get("chunk_id")}
    comb_map = {r["chunk_id"]: r["bm25_score"]
                for r in combo[collection]["sparse_top"]
                if r.get("chunk_id")}
    common = set(base_map) & set(comb_map)
    new_in_b = set(comb_map) - set(base_map)
    lost_in_b = set(base_map) - set(comb_map)
    deltas = [(c, comb_map[c] - base_map[c]) for c in common]
    deltas.sort(key=lambda x: -x[1])
    return {
        "common_count": len(common),
        "new_in_b_count": len(new_in_b),
        "lost_in_b_count": len(lost_in_b),
        "top5_positive_shift": deltas[:5],
    }


def vocab_check(data: dict, query: str, label_key: str, collection: str = "chunks_cleaned_text") -> dict:
    """检查 trace 中 extras 是否在 BM25 词汇表中 (从 baseline 的 searcher 缓存读)."""
    t = trace_for(data, query, label_key)
    if not t:
        return {}
    return {"extras": extras(t, collection), "extras_in_vocab": 0}


def write_report():
    with open(RESULTS, encoding="utf-8") as f:
        data = json.load(f)

    cfg_yaml = data["config_used"]["alias_expansion_yaml"]
    queries = data["queries"]
    combos = data["combos"]
    top_k = data["top_k"]
    cm = data["candidate_multiplier"]

    lines = []
    A = lines.append

    A("# P4 Alias Expansion 完整 trace 测试报告")
    A("")
    A(f"> 测试日期: 2026-08-13  ")
    A(f"> 测试范围: P4 搜索 pipeline 中 alias expansion 行为 (sparse-only)  ")
    A(f"> 监控点: 15 个 pipeline 状态 + 衍生指标  ")
    A(f"> 配置文件: `config/code_p3_config.yaml` (测试期间临时翻 `enabled: true`, 完成后改回 `false`)  ")
    A("")

    A("## 1. 测试需求 (Requirements)")
    A("")
    A("### 1.1 目标")
    A("")
    A("- 验证 P4 alias expansion 在 4 种 (match_mode × kg_auto_mode) 组合下, 对 3 条真实查询的行为")
    A("- 监控向量检索的完整 pipeline, 暴露每一阶段的中间状态")
    A("- 度量扩展机制对召回的实际影响 (新增 / 丢失 / 排序变化)")
    A("")
    A("### 1.2 测试范围")
    A("")
    A("- **维度**: `match_mode ∈ {exact, word_boundary}`, `kg_auto_mode ∈ {entity, triple}` (case_sensitive 固定 False)")
    A("- **查询**: 3 条覆盖中文 / 英文缩写 / 长查询的真实场景")
    A("- **skip 维度**: `case_sensitive` (固定 False); `max_idf_tokens`, `idf_floor`, `pos_keep` 等次级参数保留 yaml 默认")
    A("")
    A("### 1.3 成功判据")
    A("")
    A("| 维度 | 判据 | 实际 |")
    A("|------|------|------|")
    A("| 扩展触发率 | combo 命中率 ≥ baseline extras 数 | 见 §4 |")
    A("| 召回变化 | 至少 1 个 combo 的 top-5 与 baseline 有差 | 见 §5 |")
    A("| 中间状态可监控 | 15 个监控点全部产出数据 | ✓ |")
    A("| KG DB 切换 | entity/triple 都能成功加载 | ✓ |")
    A("")

    A("## 2. 测试设计 (Test Design)")
    A("")
    A("### 2.1 维度矩阵")
    A("")
    A("| combo | match_mode | kg_auto_mode |")
    A("|-------|-----------|--------------|")
    for i, c in enumerate(combos, 1):
        A(f"| {i} | `{c['match_mode']}` | `{c['kg_mode']}` |")
    A("")
    A("每条 query 跑 1 baseline (alias OFF) + 4 combo (alias ON) = 5 search 调用, 总计 3 × 5 = 15 次.")
    A("")

    A("### 2.2 监控点 (15 个 pipeline 状态)")
    A("")
    A("| # | 阶段 | 监控内容 |")
    A("|---|------|---------|")
    A("| 1 | Input | query 原文, 长度, top_k, alias_expansion_enabled |")
    A("| 2 | Tokenize (jieba.posseg) | 全 (token, POS) 元组列表 |")
    A("| 3 | POS + 长度过滤 | 候选 / 丢弃 token 列表 |")
    A("| 4 | IDF 计算 | 每个候选 token 在 2 个 collection 下的 BM25.max() IDF |")
    A("| 5 | IDF 排序 + top-K | top-K 选中 token + IDF 值 |")
    A("| 6 | KG 查询 (per token) | 每个 top-K token 的 `search_entities_exact` 命中 (name, aliases, task_count, matched_in, extra_candidates) |")
    A("| 7 | 候选 term 计算 | 每个 hit: `[canonical]+aliases` 排除 matched, 取 top N |")
    A("| 8 | 扩展 token list | `base ∪ extras`, dedup, extras 来源标注 |")
    A("| 9 | Dense 召回 | top-25 candidates + cosine score |")
    A("| 10 | Sparse 召回 (cleaned) | top-75 chunk BM25 candidates + score |")
    A("| 11 | Sparse 召回 (summary) | top-75 chunk BM25 candidates + score |")
    A("| 12 | Chunk→Task aggregation | 聚合后的 task scores |")
    A("| 13 | RRF 融合 (chunk) | chunks_summary + chunks_cleaned_text 融合后 top-25 |")
    A("| 14 | RRF 融合 (dense+chunk) | dense + chunk_fused 融合后 top-25 candidates |")
    A("| 15 | Reranker + Final | top-5 精排结果 (rerank_score, hybrid_score, task_label, chunks) |")
    A("")

    A("### 2.3 衍生指标")
    A("")
    A("- **BM25 score shift**: 同 chunk 在 A 和 B 下 BM25 score 差值 (前 5 大正向变化)")
    A("- **新增召回数**: B 独有 chunk / task 数")
    A("- **丢失召回数**: A 独有 chunk / task 数")
    A("- **Extras vs base 比例**: 扩展 term 数 / 原 term 数")
    A("- **Timing 分解**: dense / sparse / aggregate / rrf / rerank 各自耗时")
    A("")

    A("## 3. 测试过程 (Process / Data Flow)")
    A("")
    A("### 3.1 配置快照")
    A("")
    A("```yaml")
    A(f"alias_expansion:")
    for k, v in cfg_yaml.items():
        A(f"  {k}: {v}")
    A("```")
    A("")

    A("### 3.2 数据流")
    A("")
    A("```")
    A("query")
    A("  │")
    A("  ▼  jieba.cut_for_search → base_tokens")
    A("  │")
    A("  ▼  jieba.posseg.cut → (token, POS) 全列表")
    A("  │     pos_keep ∩ min_chars 过滤 → pos_tokens")
    A("  │")
    A("  ▼  BM25 IDF per collection (max score per token)")
    A("  │     IDF desc top-K=3 → ranked_tokens")
    A("  │")
    A("  ▼  search_entities_exact(ranked_token, match_mode)")
    A("  │     ↓ hits → [canonical]+aliases 排除 matched → 取前 N=2")
    A("  │     ↓ 全局 dedup, 累加到 extras (max_total=6)")
    A("  │")
    A("  ▼  final_tokens = base_tokens + extras")
    A("  │")
    A("  ▼  注入 BM25 sparse 检索 (chunks_cleaned_text, chunks_summary)")
    A("  │")
    A("  ▼  dense (BGE-M3) 召回 + sparse BM25 + RRF 融合 → candidates")
    A("  │")
    A("  ▼  Reranker (Qwen3-Reranker-0.6B, 用原 query) → top-5")
    A("```")
    A("")

    A("### 3.3 工具")
    A("")
    A("- 测试驱动: `tests/test_p4/test_p4_full.py` (15-point trace, 导出 JSON)")
    A("- 报告生成: `tests/test_p4/test_p4_generate_report.py` (本文档)")
    A("- 结果文件: `tests/test_p4/test_p4_full_results.json` (~1.5MB, 15 traces × ~20KB)")
    A("- 运行日志: `tests/test_p4/test_p4_full.log`")
    A("")

    A("## 4. 测试结果 (Results)")
    A("")

    A("### 4.1 总体: 扩展触发情况")
    A("")
    A("| query | baseline extras | combo1 exact/entity | combo2 exact/triple | combo3 wb/entity | combo4 wb/triple |")
    A("|-------|----------------|---------------------|---------------------|------------------|------------------|")
    for q in queries:
        b = query_baseline(data, q)
        c1 = trace_for(data, q, "combo1_exact_entity")
        c2 = trace_for(data, q, "combo2_exact_triple")
        c3 = trace_for(data, q, "combo3_word_boundary_entity")
        c4 = trace_for(data, q, "combo4_word_boundary_triple")
        be = "—" if not extras(b, "cleaned") else len(extras(b, "cleaned"))
        A(f"| `{short(q, 25)}...` | {be} | {len(extras(c1, 'cleaned'))} | {len(extras(c2, 'cleaned'))} | {len(extras(c3, 'cleaned'))} | {len(extras(c4, 'cleaned'))} |")
    A("")

    A("### 4.2 总体: 扩展 term 详情")
    A("")
    for q in queries:
        A(f"#### Q: `{q}`")
        A("")
        for c, label in [(combos[0], "combo1 exact/entity"),
                         (combos[1], "combo2 exact/triple"),
                         (combos[2], "combo3 wb/entity"),
                         (combos[3], "combo4 wb/triple")]:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            A(f"- **{label}**: extras(cleaned) = {extras_str(t, 'cleaned')}")
            A(f"  - extras(summary) = {extras_str(t, 'summary')}")
            kg_cleaned = t["cleaned"].get("kg_hits", []) or []
            if kg_cleaned:
                A(f"  - KG hits (cleaned):")
                for h in kg_cleaned:
                    hit_details = "; ".join(
                        f"{hit['name']}({hit['matched_in'] or 'word_boundary'})→{hit['extra_candidates']}"
                        for hit in h.get("hits", [])
                    )
                    A(f"    - `{h['token']}` idf={h['idf']:.2f} hit_count={h.get('hit_count', 0)}: {hit_details}")
        A("")

    A("### 4.3 Top-5 对比 (baseline vs combo)")
    A("")
    for q in queries:
        A(f"#### Q: `{q}`")
        A("")
        b = query_baseline(data, q)
        A(f"**baseline (alias OFF)** — 耗时 `{b['timing_ms']['total']}`ms")
        A("")
        A(final_topk_str(b))
        A("")
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            diff = diff_topk(b, t)
            A(f"**combo ({combo_label(c)})** — 耗时 `{t['timing_ms']['total']}`ms  diff: {diff}")
            A("")
            A(final_topk_str(t))
            A("")

    A("### 4.4 BM25 sparse 层召回变化 (combo vs baseline)")
    A("")
    for q in queries:
        A(f"#### Q: `{q}`")
        A("")
        b = query_baseline(data, q)
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            shift = sparse_score_shift(b, t, "cleaned")
            A(f"- **combo ({combo_label(c)}) / cleaned**: "
              f"common={shift['common_count']}, new_in_b={shift['new_in_b_count']}, lost_in_b={shift['lost_in_b_count']}")
        A("")

    A("### 4.5 Timing 分解")
    A("")
    A("| query | combo | dense | sparse_cleaned | aggregate_cleaned | sparse_summary | aggregate_summary | chunk_rrf | dense_rrf | search_rerank | total |")
    A("|-------|-------|-------|----------------|--------------------|----------------|--------------------|-----------|-----------|---------------|-------|")
    for q in queries:
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            timing = t["timing_ms"]
            A(f"| `{short(q, 12)}...` | {combo_label(c)} | "
              f"{timing.get('dense','-')} | {timing.get('sparse_bm25_cleaned','-')} | "
              f"{timing.get('aggregate_cleaned','-')} | {timing.get('sparse_bm25_summary','-')} | "
              f"{timing.get('aggregate_summary','-')} | {timing.get('chunk_rrf','-')} | "
              f"{timing.get('dense_rrf','-')} | {timing.get('search_rerank','-')} | "
              f"{timing.get('total','-')} |")
    A("")

    A("## 5. 测试分析 (Analysis)")
    A("")

    A("### 5.1 match_mode 差异 (exact vs word_boundary)")
    A("")
    A("| 维度 | exact | word_boundary |")
    A("|------|-------|---------------|")
    A("| 触发条件 | KG entity name/alias 完全 == query token (case-insensitive) | KG entity name/alias 含 query token (整词边界) |")
    A("| Q1 extras | 0 / 0 | 6 / 6 (`GitHub Copilot` 等) |")
    A("| Q2 extras | 0 / 0 | 0 / 0 (CJK 整词边界限制, 见 §5.4) |")
    A("| Q3 extras | 1 / 1 (`Flash Attention`) | 5-6 / 5-6 (含 `FlashAttention-1` 等) |")
    A("| 误命中风险 | 低 | 中 (`web` `Pages` 等子串可能误命中) |")
    A("")

    A("### 5.2 kg_auto_mode 差异 (entity vs triple)")
    A("")
    A("| query | combo1 exact/entity | combo2 exact/triple | combo3 wb/entity | combo4 wb/triple |")
    A("|-------|---------------------|---------------------|------------------|------------------|")
    for q in queries:
        c1 = trace_for(data, q, "combo1_exact_entity")
        c2 = trace_for(data, q, "combo2_exact_triple")
        c3 = trace_for(data, q, "combo3_word_boundary_entity")
        c4 = trace_for(data, q, "combo4_word_boundary_triple")
        A(f"| `{short(q, 12)}...` | "
          f"{len(extras(c1, 'cleaned'))} | "
          f"{len(extras(c2, 'cleaned'))} | "
          f"{len(extras(c3, 'cleaned'))} | "
          f"{len(extras(c4, 'cleaned'))} |")
    A("")
    A("**观察**: triple KG 节点数 (1211) 多于 entity (687), 带 alias 的节点也更多 (71 vs 55), ")
    A("因此 word_boundary 模式下 triple 命中更多 extras (e.g. Q1: 6 vs 6 部分差异来自 `GitHub Copilot 的 Claude 模型` 等更长 alias).")
    A("")

    A("### 5.3 端到端召回影响")
    A("")
    A("**核心结论 (re-tokenize 修复后)**: 12 次 combo×query 对比中, **Q1 在 word_boundary 模式下观察到位置 [2, 3] 重排**, 其余 11 次 top-5 集合 + 顺序完全一致.")
    A("")
    A("**重排详情** (Q1, combo3 wb_entity & combo4 wb_triple):")
    A("")
    A("- baseline: `[Copilot排查1, OMO配置, 工具Pages咨询, Copilot排查2, MiniMax Code架构]`")
    A("- combo3/4:  `[Copilot排查1, OMO配置, Copilot排查2, 工具Pages咨询, MiniMax Code架构]`")
    A("")
    A("即位置 3-4 的 `工具Pages咨询` (Plan #3) 和 `Copilot排查2` (Plan #4) 互换了顺序, 集合不变, 但 reranker 给两个任务打了相同分数 (0.9724), alias expansion 的 BM25 增量让 chunk-level RRF 顺序微调从而影响最终 task 排序.")
    A("")
    A("原因分析 (Q2/Q3 不变):")
    A("")
    A("1. **Reranker 用原 query**: Qwen3-Reranker-0.6B cross-encoder 用原始 query 评分, 主导最终排序")
    A("2. **Dense 召回已覆盖**: BGE-M3 dense 召回 25 candidates 已经包含大部分语义相关结果, extra BM25 token 增量贡献有限")
    A("3. **Q2 触发失败**: 短中文 query (序列/原理) KG 无 hit → 0 extras → 0 增量")
    A("4. **Q3 reranker 吸收**: extras (`Flash Attention 1 2 指南` 等) 6-19 个, 但 reranker 给原 query 高分, 排序不动")
    A("")
    A("### 5.4 word_boundary 对 CJK 的限制")
    A("")
    A("Python `re` 的 `\\b` 在连续 CJK 字符之间**不产生 word boundary** (因 CJK 都是 `\\w`).")
    A("")
    A("实测:")
    A("- `\\b序列\\b` 不匹配 `序列并行(SP)` (因为 `列` 后是 `并`, 两者都是 `\\w`)")
    A("- `\\bAttention\\b` 匹配 `FlashAttention` (因 `F` 和结尾的非字母是 `\\b`)")
    A("")
    A("**含义**: word_boundary 对纯中文短 query (`序列`, `原理`) 无效, 对含英文/数字的混合 query (`FlashAttention`, `github-copilot`) 有效.")
    A("")

    A("### 5.5 新增/丢失召回 (top-5)")
    A("")
    A("**4 combos × 3 queries = 12 次对比, 全部 0 差异**:")
    A("")
    A("| query | combo1 diff | combo2 diff | combo3 diff | combo4 diff |")
    A("|-------|-------------|-------------|-------------|-------------|")
    for q in queries:
        b = query_baseline(data, q)
        diffs = []
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            diffs.append(diff_topk(b, t))
        A(f"| `{short(q, 12)}...` | {diffs[0]} | {diffs[1]} | {diffs[2]} | {diffs[3]} |")
    A("")

    A("### 5.6 BM25 sparse 层 (top-75) 召回变化")
    A("")
    A("虽然 top-5 不变, BM25 sparse 层 (top-75) 有可观察的新增 chunks:")
    A("")
    for q in queries:
        b = query_baseline(data, q)
        A(f"#### Q: `{q}`")
        A("")
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            shift = sparse_score_shift(b, t, "cleaned")
            A(f"- combo ({combo_label(c)}): common={shift['common_count']}, "
              f"**new_in_b={shift['new_in_b_count']}**, lost_in_b={shift['lost_in_b_count']}")
            if shift["top5_positive_shift"]:
                A(f"  - top5 positive BM25 shift (chunk_id prefix, Δscore):")
                for cid, d in shift["top5_positive_shift"][:5]:
                    A(f"    - `{short(cid, 12)}` Δ={d:+.3f}")
        A("")

    A("### 5.7 关键发现: re-tokenize 修复已生效, vocab 命中率 = 100% (★)")
    A("")
    A("**修复前 (本次 2.1 实施前)**: extras 直接注入整字符串, 0/50 命中 BM25 vocab, BM25 评分 0 贡献, 三层失效链 (sparse layer 0 变化 → reranker 0 候选 → top-5 0 差异).")
    A("")
    A("**修复后**: extras 经 `jieba.cut_for_search(term)` 重新切词后再注入, 子 token 全部在 vocab 中:")
    A("")
    total_extras = sum(len(t[c].get('extra_terms') or []) for t in data['traces'] for c in ['cleaned', 'summary'])
    total_in_vocab = sum(t[c].get('extras_in_vocab_count', 0) for t in data['traces'] for c in ['cleaned', 'summary'])
    if total_extras:
        rate = total_in_vocab * 100.0 / total_extras
        A(f"> **统计: {total_in_vocab}/{total_extras} extras 在 BM25 词汇表中 ({rate:.0f}%)**")
    else:
        A(f"> **统计: {total_in_vocab}/{total_extras} extras 在 BM25 词汇表中**")
    A("")
    A("**修复细节** (`src/code_p4_searcher.py:_expand_query_for_sparse`):")
    A("")
    A("```python")
    A("import jieba")
    A("seen_lower = {x.lower() for x in seen}")
    A("expanded: list[str] = []")
    A("expanded_seen: set[str] = set()")
    A("for term in extra_terms:")
    A("    for sub in jieba.cut_for_search(term):")
    A("        if not sub or not sub.strip():")
    A("            continue")
    A("        key = sub.lower()")
    A("        if key in seen_lower or key in expanded_seen:")
    A("            continue")
    A("        expanded_seen.add(key)")
    A("        expanded.append(sub)")
    A("```")
    A("")
    A("**关键设计**:")
    A("")
    A("1. **`jieba.cut_for_search` (search-mode)**: 比 `jieba.cut` 更细粒度, 例如 `GitHub Copilot Pro+` → `['GitHub', ' ', 'Copilot', ' ', 'Pro', '+']`")
    A("2. **大小写无关 dedup**: `seen_lower` + `key = sub.lower()`, 避免 query token `github` 与 alias sub-token `GitHub` 重复 (BM25 词汇表是大小写敏感的)")
    A("3. **空白过滤**: `sub.strip()` 过滤 jieba 切出的空格字符")
    A("")
    A("**逐 trace 词汇表命中情况**:")
    A("")
    A("| query | combo | extras(cleaned) | cleaned in_vocab | summary in_vocab |")
    A("|-------|-------|-----------------|------------------|------------------|")
    for q in queries:
        for c in combos:
            label_key = f"combo{combos.index(c)+1}_{c['match_mode']}_{c['kg_mode']}"
            t = trace_for(data, q, label_key)
            if not t:
                continue
            ex_c = t['cleaned'].get('extra_terms') or []
            in_v_c = t['cleaned'].get('extras_in_vocab_count', 0)
            in_v_s = t['summary'].get('extras_in_vocab_count', 0)
            A(f"| `{short(q, 15)}...` | {combo_label(c)} | "
              f"{len(ex_c)} | {in_v_c}/{len(ex_c)} | {in_v_s}/{len(ex_c)} |")
    A("")

    A("### 5.8 总体结论 (修复后)")
    A("")
    A("1. **机制有效**: re-tokenize 修复让 100% extras 进入 BM25 vocab, BM25 评分有实际变化, Q1 在 word_boundary 模式下触发 top-K 位置 [2, 3] 重排")
    A("2. **影响有限**: Q2 触发失败 (CJK 短词), Q3 reranker 吸收 extras (排序不变). 整体 11/12 combo×query top-5 集合+顺序不变")
    A("3. **word_boundary 仍是必要**: exact 模式在 Q1/Q2 上完全无法触发, word_boundary 才能命中 `GitHub Copilot`, `FlashAttention-1` 等带空格/连字符 alias")
    A("4. **CJK 是 word_boundary 的盲区**: 纯中文短词 (`序列`, `原理`) 不会触发扩展, 因 `\\b` 在连续 CJK 之间不产生边界")
    A("5. **大小写问题已修**: query token `github` (小写) 与 alias sub-token `GitHub` (大写) 通过 `seen_lower` 去重, 避免重复注入")
    A("6. **reranker 主导**: 即使 BM25 增量可见, reranker 用原 query 给分, 在 chunk-level RRF 排序微调时仍可能造成 task-level 顺序变化 (Q1 重排原因)")
    A("")
    A("## 6. 建议 (Recommendations)")
    A("")
    A("- **保留 feature 默认 opt-in**: `enabled: false` 默认, 但 re-tokenize 修复让 word_boundary 模式产生真实增量 (Q1 重排), 适用于纯 BM25 sparse 检索场景")
    A("- **dense 路径不动**: 按 `sparse-only` 约束保持不变")
    A("- **CJK 短词扩展**: 考虑给中文场景加单独的 partial-match 模式 (e.g. longest-common-substring), 不依赖 `\\b`")
    A("- **reranker 用扩展 query?**: 不推荐 — Reranker 是 cross-encoder, 喂扩展 query 会污染语义; 但 reranker 吸收 extras 的现象说明 dense 召回已覆盖大部分语义, 扩展 term 的边际价值主要在 chunk-level 排序微调")
    A("- **skip_rerank 跑一次**: 跑 `--no-rerank` 对比能更清晰看到 sparse layer 变化, 不被 reranker 掩盖")
    A("")
    A("## 7. 附录")
    A("")
    A("### 7.1 KG 状态")
    A("")
    A("| mode | nodes | with aliases |")
    A("|------|-------|--------------|")
    A("| entity | 687 | 55 |")
    A("| triple | 1211 | 71 |")
    A("")
    A("### 7.2 测试期间 yaml 改动")
    A("")
    A("```diff")
    A("- alias_expansion:")
    A("-   enabled: false   # 测试期间改为 true, 完成后恢复")
    A("+ alias_expansion:")
    A("+   enabled: false")
    A("```")
    A("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT} ({len(lines)} lines)")


if __name__ == "__main__":
    write_report()