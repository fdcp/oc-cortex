# GraphRAG 调试分析报告

- 总查询数: **6**
- DB: entity, triple

## DB: `entity`

### Stage 1: LLM 实体抽取
- 查询数: 3
- 空抽取 (退化为纯向量): **0** (0.0%)
- 抽取实体数: min=1.0 max=3.0 mean=1.6667 median=1.0
- 示例 (前 3):
  - `GPU的对比和选型` → ['GPU']
  - `flashattention的原理和历史` → ['flashattention']
  - `序列并行(Sequence Parallel,SP)` → ['序列并行', 'Sequence Parallel', 'SP']

### Stage 2: 实体模糊匹配
- 无种子 (图谱完全不命中): **0** (0.0%)
- 种子实体数: min=1.0 max=3.0 mean=2.3333

### Stage 3: BFS 扩散
- 扩散节点数: min=27.0 max=30.0 mean=29.0 median=30.0
- 命中 max_nodes (30) 上限: **2** / 3

### Stage 4: 反 IDF 累加
- 唯一图谱 task 数: min=9.0 max=29.0 mean=16.3333

### Stage 5: 向量检索 (粗排)
- 候选数 (top_k * 2 = 10): min=50.0 max=50.0 mean=50.0

### Stage 8: 来源分布 (Top-K 最终结果)
- 总最终结果数: 15
| 来源 | 数量 | 占比 |
|------|------|------|
| vector | 10 | 66.67% |
| graph | 0 | 0.0% |
| both | 5 | 33.33% |
| neither | 0 | 0.0% |

**按 query:**

| Query | 抽取实体 | 扩散 | vector | graph | both | neither |
|-------|----------|------|--------|-------|------|---------|
| `GPU的对比和选型` | ['GPU'] | 27 | 3 | 0 | 2 | 0 |
| `flashattention的原理和历史` | ['flashattention'] | 30 | 5 | 0 | 0 | 0 |
| `序列并行(Sequence Parallel,SP)` | ['序列并行', 'Sequence Parallel', 'SP'] | 30 | 2 | 0 | 3 | 0 |

### Stage 9: 阶段耗时 (ms)
**核心 4 段耗时 (来自 rag.search 内部 final_debug):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| vector_time (粗排) | 3 | 12.0 | 17.0 | 14.3333 | 14.0 |
| graph_time (BFS+IDF) | 3 | 21480.0 | 26889.0 | 24817.3333 | 26083.0 |
| rerank_time (Qwen3) | 3 | 18881.0 | 22687.0 | 20289.3333 | 19300.0 |
| total_time (search 整体) | 3 | 44186.0 | 46203.0 | 45122.6667 | 44979.0 |

**其他细粒度阶段 (来自 debug 脚本自身计时):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| 1_llm_entity_extraction | 3 | 15559.0 | 23300.0 | 20277.3333 | 21973.0 |
| 2_seed_entity_match | 3 | 2.0 | 3.0 | 2.3333 | 2.0 |
| 3_bfs_expand | 3 | 3.0 | 5.0 | 3.6667 | 3.0 |
| 4_inverse_idf_accumulate | 3 | 3.0 | 5.0 | 4.0 | 4.0 |
| 5_vector_search | 3 | 32.0 | 86.0 | 54.3333 | 45.0 |

### 最终结果得分分布
- rerank_score: min=0.2628 max=0.9974 mean=0.7498
- hybrid_score: min=0.0088 max=0.0323 mean=0.0188

## DB: `triple`

### Stage 1: LLM 实体抽取
- 查询数: 3
- 空抽取 (退化为纯向量): **0** (0.0%)
- 抽取实体数: min=1.0 max=3.0 mean=1.6667 median=1.0
- 示例 (前 3):
  - `GPU的对比和选型` → ['GPU']
  - `flashattention的原理和历史` → ['flashattention']
  - `序列并行(Sequence Parallel,SP)` → ['序列并行', 'Sequence Parallel', 'SP']

### Stage 2: 实体模糊匹配
- 无种子 (图谱完全不命中): **0** (0.0%)
- 种子实体数: min=1.0 max=3.0 mean=2.3333

### Stage 3: BFS 扩散
- 扩散节点数: min=3.0 max=29.0 mean=17.0 median=19.0
- 命中 max_nodes (30) 上限: **0** / 3

### Stage 4: 反 IDF 累加
- 唯一图谱 task 数: min=4.0 max=6.0 mean=5.0

### Stage 5: 向量检索 (粗排)
- 候选数 (top_k * 2 = 10): min=50.0 max=50.0 mean=50.0

### Stage 8: 来源分布 (Top-K 最终结果)
- 总最终结果数: 15
| 来源 | 数量 | 占比 |
|------|------|------|
| vector | 12 | 80.0% |
| graph | 0 | 0.0% |
| both | 3 | 20.0% |
| neither | 0 | 0.0% |

**按 query:**

| Query | 抽取实体 | 扩散 | vector | graph | both | neither |
|-------|----------|------|--------|-------|------|---------|
| `GPU的对比和选型` | ['GPU'] | 3 | 5 | 0 | 0 | 0 |
| `flashattention的原理和历史` | ['flashattention'] | 29 | 5 | 0 | 0 | 0 |
| `序列并行(Sequence Parallel,SP)` | ['序列并行', 'Sequence Parallel', 'SP'] | 19 | 2 | 0 | 3 | 0 |

### Stage 9: 阶段耗时 (ms)
**核心 4 段耗时 (来自 rag.search 内部 final_debug):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| vector_time (粗排) | 3 | 11.0 | 14.0 | 13.0 | 14.0 |
| graph_time (BFS+IDF) | 3 | 17007.0 | 46145.0 | 28923.3333 | 23618.0 |
| rerank_time (Qwen3) | 3 | 18630.0 | 19374.0 | 18955.3333 | 18862.0 |
| total_time (search 整体) | 3 | 36394.0 | 64790.0 | 47892.6667 | 42494.0 |

**其他细粒度阶段 (来自 debug 脚本自身计时):**

| 阶段 | count | min | max | mean | median |
|------|-------|-----|-----|------|--------|
| 1_llm_entity_extraction | 3 | 14976.0 | 55312.0 | 31070.6667 | 22924.0 |
| 2_seed_entity_match | 3 | 2.0 | 3.0 | 2.6667 | 3.0 |
| 3_bfs_expand | 3 | 1.0 | 2.0 | 1.6667 | 2.0 |
| 4_inverse_idf_accumulate | 3 | 1.0 | 3.0 | 2.0 | 2.0 |
| 5_vector_search | 3 | 36.0 | 42.0 | 39.3333 | 40.0 |

### 最终结果得分分布
- rerank_score: min=0.2147 max=0.9974 mean=0.5807
- hybrid_score: min=0.0088 max=0.0323 mean=0.0169

## 关键发现与建议

见 analysis.md 末尾的"洞察"章节（自动生成）

## 自动洞察

### DB: `entity`

- ✓ 所有 query 都成功抽取到实体
- ⚠️ **2/3** query 命中 `max_expand_nodes=30` 上限, 可能图谱过密导致扩散爆炸
- Top-K 来源: vector-only=66.7%, graph-only=0.0%, both=33.3%, neither=0.0%

### DB: `triple`

- ✓ 所有 query 都成功抽取到实体
- Top-K 来源: vector-only=80.0%, graph-only=0.0%, both=20.0%, neither=0.0%
