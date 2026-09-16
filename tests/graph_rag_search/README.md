# Graph-RAG 检索流程 Trace 测试

本目录包含两个独立可运行的脚本，用于复现和调试 MCP `knowledge-graph` 的 Graph-RAG 检索流程。

## 背景

`code_mcp_server.py` 的 `summarize` / `graph_rag_search` 工具在生产中只返回最终结果（8 个 task + LLM 总结），不暴露中间检索阶段的细节。当出现以下问题时，需要分阶段排查：

- **召回质量下降**：是向量通道召回不足，还是图谱通道没生效？
- **耗时异常**：是 Reranker 处理量太大，还是图谱扩散太慢？
- **配置调优**：调整 `rerank_multiplier`、`graph_channel_weight` 后效果如何？

这两个脚本**手工拆分** `GraphRAGSearcher.search()` 内部流程，把 6 个阶段（向量 3 路 → chunk RRF → vector RRF → 图谱 BFS → 外层 RRF → Reranker）的中间结果全部打印出来。

## 脚本说明

### `trace_rag.py` — 单次分阶段 trace

复现 MCP `summarize(use_graph_rag=true)` 流程，逐阶段打印每路召回的 task_id、RRF 分数、来源分布。

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query` | `分布式训练` | 查询字符串 |
| `--top-k` | `8` | 最终返回结果数 |
| `--rerank-multiplier` | `1.0` | GraphRAGSearcher.rerank_multiplier |
| `--no-rerank` | False | 跳过 Reranker 精排，只打印粗排结果 |
| `--export-json PATH` | None | 导出完整结果到 JSON 文件（含每阶段 task_id 集合 + 耗时 + 来源分布）|

**典型输出**：

```
[Stage 1] GraphRAG 传 vector_top_k=8 -> SessionSearcher n_inner=40
  1a. Dense[tasks]          : limit=40, hit=40
  1b. Sparse[chunks_summary]: limit=120, chunk_hit=16, task_agg=9
  1c. Sparse[chunks_cleaned]: limit=120, chunk_hit=41, task_agg=22

[Stage 2] chunk 两路 RRF 融合: 22 task
[Stage 3] vector_results (SessionSearcher 内部 RRF): 40 task
[Stage 4] 图谱 BFS: query=['分布式训练'] seed=['分布式训练']
   扩散节点: 6 个, graph_candidates: 4
[Stage 5] 外层 RRF 融合: 40 task
   vector 独有: 36, graph 独有: 0, 共有: 4
[Stage 6] Reranker: 40 -> 8  耗时 14.7s
  [1] ses_053c0ac7..._T1  rerank=0.9897  src=both
  ...

[Export] JSON 已写入: /tmp/trace_result.json
```

**JSON 输出结构**：

```json
{
  "query": "FlashAttention",
  "top_k": 5,
  "rerank_multiplier": 1.0,
  "use_rerank": false,
  "stages": {
    "dense":          {"limit": 25, "hit": 25, "task_ids": [...]},
    "sparse_summary": {"limit": 75, "chunk_hit": 16, "task_agg": 9, "task_ids": [...]},
    "sparse_cleaned": {"limit": 75, "chunk_hit": 41, "task_agg": 22, "task_ids": [...]},
    "chunk_fused":    {"count": 22, "task_ids": [...], "summary_only": [...], "cleaned_only": [...], "shared": [...]},
    "vector":         {"count": 25, "task_ids": [...], "dense_only": [...], "chunk_only": [...], "shared": [...]},
    "graph":          {"query_entities": ["FlashAttention"], "seed_entities": ["FlashAttention"], "expanded_count": 17, "candidates": [...]},
    "merged":         {"count": 25, "task_ids": [...], "vector_only": [...], "graph_only": [...], "shared": [...]},
    "final":          {"count": 5, "rerank_time_s": 0.0, "results": [{"rank": 1, "task_id": ..., "rerank_score": ..., "src": "both", ...}, ...]}
  }
}
```

### `trace_compare_rerank_multiplier.py` — 配置对比

对比 `rerank_multiplier=1.0` vs `2.0` 两种配置下，各阶段召回集合的差异 + Reranker 耗时。

**适用场景**：小 corpus（<500 task）下，验证 `rerank_multiplier=1.0` 是否在不影响最终召回的前提下减少 Reranker 耗时。

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--query` | `分布式训练` | 查询字符串 |
| `--top-k` | `8` | 最终返回结果数 |
| `--query-list PATH` | None | 批量 query 模式：从文件读取 query 列表（一行一个），输出召回一致率统计 |

**单 query 典型输出**（末尾的对比总结）：

```
[dense]
  ×1.0: 40 个, ×2.0: 76 个, 差: 36 (×2.0 独有) / 0 (×1.0 独有)
[sparse_summary]
  ×1.0: 9 个, ×2.0: 9 个, 差: 0
...
[final]
  ×1.0: 8 -> [...]
  ×2.0: 8 -> [...]
  一致: True

[Reranker 耗时]
  ×1.0: 14.7s  (40 篇)
  ×2.0: 27.6s  (76 篇)
  加速比: ×1.0 比 ×2.0 快 1.88x
```

**批量 query 典型输出**（`--query-list` 模式）：

```
批量对比统计 (3 个 query)
  final top-8 完全一致: 1/3 (33.3%)
  vector 召回集合完全一致: 0/3 (0.0%)
  平均 Reranker 加速比: 1.66x (×1.0 相对 ×2.0)

  Query                final    vector   ×1.0秒      ×2.0秒      加速
  -------------------- -------- -------- ---------- ---------- ------
  FlashAttention       ✓        ✗        15.20      26.80      1.76
  BF16 混合精度        ✓        ✗        14.50      25.30      1.74
  序列并行             ✗        ✗        13.80      21.20      1.54
```

### `run_trace.sh` — 一键运行脚本

Shell 包装脚本，自动从 `~/.local/share/opencode/auth.json` 读取 `OPENCODE_ZEN_API_KEY` 并设置离线环境变量。

**用法**：

```bash
bash tests/graph_rag_search/run_trace.sh --help
bash tests/graph_rag_search/run_trace.sh -t "FlashAttention"      # 单次 trace
bash tests/graph_rag_search/run_trace.sh -c "序列并行"            # rerank_multiplier 对比
bash tests/graph_rag_search/run_trace.sh -b my_queries.txt       # 批量对比
bash tests/graph_rag_search/run_trace.sh -a                      # 跑全套
```

## 使用方法

### 前置条件

1. 仓库根目录已生成 Phase 1-4 产物：
   - `output/chunks.jsonl`
   - `output/tasks/*.jsonl`
   - `qdrant_data/` 或 `QDRANT_URL` 已配置的 server
   - `output/triple/knowledge_graph.db`

2. 环境变量：

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### 运行

```bash
# 仓库根目录
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph

# 单次 trace
python3 tests/graph_rag_search/trace_rag.py

# 自定义 query
python3 tests/graph_rag_search/trace_rag.py --query "FlashAttention" --top-k 5

# 跳过 Reranker 节省时间
python3 tests/graph_rag_search/trace_rag.py --no-rerank

# 导出 JSON（便于后续 diff 或画图）
python3 tests/graph_rag_search/trace_rag.py --query "FlashAttention" --export-json /tmp/trace.json

# 对比 rerank_multiplier
python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py
python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query "分布式训练"

# 批量 query 对比（需先准备 query 文件）
echo -e "分布式训练\nFlashAttention\n序列并行" > /tmp/queries.txt
python3 tests/graph_rag_search/trace_compare_rerank_multiplier.py --query-list /tmp/queries.txt

# 一键脚本（自动读 auth.json 加载 key）
bash tests/graph_rag_search/run_trace.sh -t "FlashAttention"
bash tests/graph_rag_search/run_trace.sh -c "序列并行"
bash tests/graph_rag_search/run_trace.sh -b my_queries.txt
bash tests/graph_rag_search/run_trace.sh -a
```

## 6 个阶段的含义

| Stage | 通道 | 数量参数 | 数据源 |
|-------|------|----------|--------|
| 1a | Dense[tasks] | `top_k × rerank_multiplier × 5` | Qdrant `tasks` 集合（BGE-small-zh）|
| 1b | Sparse[chunks_summary] | `top_k × rm × 5 × 3` | Qdrant `chunks_summary` + BM25 |
| 1c | Sparse[chunks_cleaned_text] | `top_k × rm × 5 × 3` | Qdrant `chunks_cleaned_text` + BM25 |
| 2 | chunk 两路 RRF 融合 | - | 1b ∪ 1c RRF 合并 |
| 3 | vector RRF 融合 | - | 1a ∪ 2 RRF 合并（SessionSearcher 内部）|
| 4 | 图谱 BFS 扩散 | `bfs_depth=2, max_nodes=30` | SQLite `output/triple/knowledge_graph.db` |
| 5 | 外层 RRF 融合 | - | 3 + 4 RRF 合并（graph_channel_weight=0.3）|
| 6 | Reranker 精排 | - | Qwen3-Reranker-0.6B |

## 关键发现（基于 76 task corpus 的实际数据）

### `rerank_multiplier=1.0` vs `2.0` 完整对比

| 指标 | ×1.0 | ×2.0 | 差异 |
|------|------|------|------|
| Dense[tasks] limit | 40 | 80 | ×2 多 40 |
| Dense[tasks] 召回 | 40 | 76 | 集合全量 |
| Sparse 两路 | 不变 | 不变 | - |
| vector RRF 后 | 40 | 76 | ×2 多 36（长尾）|
| graph_candidates | 4 | 4 | 不变 |
| merged | 40 | 76 | ×2 多 36 |
| Reranker 耗时 | **14.7s** | **27.6s** | **×1 快 47%** |
| final 8 个 | 完全相同 | 完全相同 | 一致性 True |

### 结论

- **小 corpus（<500 task）**：`rerank_multiplier=1.0` 是最优配置
  - 速度优势明显（省 13s）
  - 召回质量完全一致
  - 图谱通道相对增益更大（4/40=10% vs 4/76=5%）

- **大 corpus（>500 task）**：建议 `rerank_multiplier=2.0-3.0`
  - Dense 通道需要 buffer 召回 top_k 之后的相关任务
  - 防止 Reranker 看不到长尾但仍相关的任务

## 注意事项

- **API key 必需**：`extract_query_entities` 调 LLM，无 key 时图谱通道会失败退化为纯向量
- **离线模式**：HuggingFace 模型（embedding + Reranker）需要本地缓存，否则会卡在下载
- **不修改源码**：两个脚本都是只读调用，跑完不会修改任何 pipeline 产物
- **可重复运行**：脚本无副作用，可多次运行对比

## 相关源码

- `src/code_p5e_graph_rag.py`：GraphRAGSearcher（主流程）
- `src/code_p4_searcher.py`：SessionSearcher（3 路向量 + 内层 RRF）
- `src/code_p3_qdrant_store.py`：Phase3Store（3 个 Qdrant 集合 + BM25 索引）
- `src/code_p5e_db.py`：KGDatabase（图谱 SQLite + BFS）
- `src/code_mcp_server.py`：MCP server（生产入口，调用 GraphRAGSearcher）
- `doc/code_p4_README.md`：P4 检索流程设计文档
