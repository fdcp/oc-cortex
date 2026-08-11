# GraphRAG 调试分析报告

- 总查询数: **5**
- DB: triple

## DB: `triple`

### Stage 1: LLM 实体抽取
- 查询数: 5
- 空抽取 (退化为纯向量): **0** (0.0%)
- 抽取实体数: min=1.0 max=2.0 mean=1.6 median=2.0
- 示例 (前 3):
  - `FlashAttention 实现原理` → ['FlashAttention']
  - `OpenCode` → ['OpenCode']
  - `OpenCode MCP` → ['OpenCode', 'MCP']

### Stage 2: 实体模糊匹配
- 无种子 (图谱完全不命中): **1** (20.0%)
- 种子实体数: min=0.0 max=2.0 mean=1.2

### Stage 3: BFS 扩散
- 扩散节点数: min=0.0 max=30.0 mean=17.8 median=17.0
- 命中 max_nodes (30) 上限: **2** / 5

### Stage 4: 反 IDF 累加
- 唯一图谱 task 数: min=0.0 max=25.0 mean=11.6

### Stage 5: 向量检索 (粗排)
- 候选数 (top_k * 2 = 10): min=50.0 max=50.0 mean=50.0

### Stage 8: 来源分布 (Top-K 最终结果)
- 总最终结果数: 25
| 来源 | 数量 | 占比 |
|------|------|------|
| vector | 14 | 56.0% |
| graph | 1 | 4.0% |
| both | 10 | 40.0% |
| neither | 0 | 0.0% |

**按 query:**

| Query | 抽取实体 | 扩散 | vector | graph | both | neither |
|-------|----------|------|--------|-------|------|---------|
| `FlashAttention 实现原理` | ['FlashAttention'] | 17 | 3 | 0 | 2 | 0 |
| `OpenCode` | ['OpenCode'] | 30 | 1 | 0 | 4 | 0 |
| `OpenCode MCP` | ['OpenCode', 'MCP'] | 30 | 1 | 0 | 4 | 0 |
| `Qwen3 Reranker 精排` | ['Qwen3 Reranker', '精排'] | 0 | 5 | 0 | 0 | 0 |
| `知识图谱 实体对齐` | ['知识图谱', '实体对齐'] | 12 | 4 | 1 | 0 | 0 |

### Stage 9: 阶段耗时 (ms)
**核心 4 段耗时 (来自 rag.search 内部 final_debug):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| vector_time (粗排) | 5 | 10.0 | 13.0 | 11.2 | 11.0 |
| graph_time (BFS+IDF) | 5 | 11460.0 | 52699.0 | 23059.4 | 13742.0 |
| rerank_time (Qwen3) | 5 | 17464.0 | 19856.0 | 18656.4 | 18713.0 |
| total_time (search 整体) | 5 | 28935.0 | 72570.0 | 41728.6 | 32195.0 |

**其他细粒度阶段 (来自 debug 脚本自身计时):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| 1_llm_entity_extraction | 5 | 13387.0 | 27385.0 | 19575.0 | 18022.0 |
| 2_seed_entity_match | 5 | 1.0 | 5.0 | 2.0 | 1.0 |
| 3_bfs_expand | 5 | 0.0 | 3.0 | 1.4 | 2.0 |
| 4_inverse_idf_accumulate | 5 | 0.0 | 6.0 | 3.2 | 4.0 |
| 5_vector_search | 5 | 19.0 | 68.0 | 39.8 | 38.0 |

### 最终结果得分分布
- rerank_score: min=0.4263 max=0.9993 mean=0.7922
- hybrid_score: min=0.0086 max=0.0323 mean=0.0188

## 关键发现与建议

见 analysis.md 末尾的"洞察"章节（自动生成）

## 自动洞察

### DB: `triple`

- ✓ 所有 query 都成功抽取到实体
- ⚠️ **2/5** query 命中 `max_expand_nodes=30` 上限, 可能图谱过密导致扩散爆炸
- Top-K 来源: vector-only=56.0%, graph-only=4.0%, both=40.0%, neither=0.0%
