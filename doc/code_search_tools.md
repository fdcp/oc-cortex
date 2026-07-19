# 检索工具对比说明

项目中有三个脚本都涉及 Qdrant 向量检索，但定位、检索深度和使用场景各不相同。本文梳理它们的异同，帮助快速选择正确的工具。

## 一览

| 维度 | `code_p3_search_demo.py` | `code_MS_inspect.py` | `code_p4_search_cli.py` |
|------|--------------------------|----------------------|--------------------------|
| 定位 | 检索实验室 | 数据质检 | 检索产品 |
| 所属阶段 | Phase 3 | 里程碑检视 | Phase 4 |
| 核心类 | `Phase3Store` | `QdrantClient` (直调) | `SessionSearcher` |
| 检索方式 | Dense / Sparse / Hybrid 三路分开展示 | 仅 Dense (top-5 命中率验证) | Hybrid + Reranker 完整流水线 |
| 结果粒度 | Chunk 级别 (chunk_id + score) | Task/Chunk 级别 (命中统计) | Task 级别 (附带 Chunk 摘要和文本预览) |
| 数据写入 | 每次运行重建索引 (upsert) | 只读 | 只读 (假设已由 `code_p3_main.py` 写入) |
| 交互模式 | 无 (批量示例查询) | 无 (全自动检视) | 有 (`--interactive`) |
| 配置文件 | `config/code_p3_config.yaml` | 硬编码 `./qdrant_data` | `config/code_p3_config.yaml` |

## 详细对比

### 1. `code_p3_search_demo.py` — 检索实验室

这是一个数据注入与检索演示一体化的脚本。每次运行时从 JSONL 文件加载 tasks、chunks、summaries，全量 upsert 到 Qdrant，然后对预设的 5 条示例查询分别执行三种检索并并排展示结果。

**检索流程：**

```
Dense 检索 → chunks_summary 集合 → SearchResult (chunk_id, score)
Sparse 检索 → chunks_cleaned_text 集合 → SearchResult (chunk_id, score)
Hybrid 跨集合检索 → RRF(dense_summary, sparse_cleaned_text) → HybridResult (RRF score + 双路排名)
```

**关键特点：**

- 三路检索独立执行、独立展示，方便对比不同策略的效果差异。
- 支持两种模式：`--mode all` 使用 bge-small-zh + BM25 全量数据；`--mode sample` 使用 Qwen3-Embedding + BGE-M3 采样 1/5 数据。
- 可通过 `--query` 自定义查询覆盖示例集。
- 运行较慢（每次重建索引），适合调参阶段验证 embedding 模型或 sparse 方法的效果。

**典型用法：**

```bash
# 全量 bge-small-zh + BM25 对比
python src/code_p3_search_demo.py --mode all --config config/code_p3_config.yaml

# 自定义查询
python src/code_p3_search_demo.py --mode all --query "分布式训练通信优化"
```

### 2. `code_MS_inspect.py` — 数据质检

这是一个里程碑检视工具，包含 6 大功能模块，检索只是其中之一。它直接操作 `QdrantClient`，不依赖 `Phase3Store`，目的是验证数据和 embedding 的质量，而非提供日常搜索能力。

**6 大功能模块：**

1. **集合统计** — 点数、维度、状态一览。
2. **样本浏览** — 随机抽取 payload 检查字段完整性。
3. **数据分析** — session/task/chunk 分布、token 压缩率、summary 覆盖率。
4. **向量近邻分析** — 检查 embedding 聚类质量，找出 cosine > 0.85 的高相似 task 对。
5. **检索质量测试** — 5 条预设查询 + 关键词匹配判定命中率（仅 Dense top-5）。
6. **Chunk Summary 质量抽样** — 检查 summary 长度分布，对比 task_summary vs chunk_summary。

**检索特点：**

检索部分（功能 5）使用最简单的方式：直接用 `QdrantClient.query_points` 对 tasks 集合做 Dense 检索，用预设关键词列表判定 top-5 是否命中。结果输出 `[OK]` 或 `[MISS]` 和命中率百分比。这是一个粗粒度的质量指标，不适用于日常搜索。

**典型用法：**

```bash
python src/code_MS_inspect.py
```

### 3. `code_p4_search_cli.py` — 检索产品

这是面向最终用户的搜索 CLI，封装了完整的 Hybrid + Reranker 精排流水线。假设 Qdrant 数据已由 Phase 3 写入，直接连接读取。

**检索流程（5 阶段）：**

```
Stage 1: Dense 检索 → tasks 集合 → task_summary 语义匹配 → n_candidates 条候选
Stage 2: Sparse 检索 → chunks_cleaned_text 集合 → BM25/BGE-M3 匹配 → 聚合到 Task 级别
Stage 3: RRF 融合 → 合并 Dense + Sparse 候选 → 截断到 n_candidates
Stage 4: Reranker 精排 → Qwen3-Reranker 对 task_summary 重新打分 → top_k
Stage 5: Chunk 展开 → 为每个结果 Task 附加关联 Chunk 的摘要和文本预览
```

**关键特点：**

- 返回 `SessionSearchResult`，包含 task_label、task_summary、rerank_score、hybrid_score，以及关联 Chunk 的 summary 和 cleaned_text_preview，信息最丰富。
- 支持 `--interactive` 交互模式，循环输入查询。
- 支持 `--no-rerank` 跳过精排，方便对比粗排和精排效果。
- Dense 检索目标是 tasks 集合（Task 级别），Sparse 检索目标是 chunks_cleaned_text 集合（Chunk 级别，聚合回 Task），实现了跨集合 Hybrid。
- 启动快（不重建索引），适合日常使用。

**典型用法：**

```bash
# 单次查询
python src/code_p4_search_cli.py --query "GPU对比分析"

# 交互模式
python src/code_p4_search_cli.py --interactive

# 跳过 Reranker 对比粗排效果
python src/code_p4_search_cli.py --query "优化器学习率" --no-rerank
```

## 检索方式深度对比

### Dense 检索

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 目标集合 | chunks_summary | tasks | tasks |
| 检索对象 | chunk summary 文本 | task_summary 文本 | task_summary 文本 |
| 模型 | 可切换 (bge-small-zh / Qwen3) | bge-small-zh (硬编码) | 由 config 决定 |
| 用途 | 对比展示 | 命中率验证 | 生产检索 |

### Sparse 检索

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 目标集合 | chunks_cleaned_text | — (无) | chunks_cleaned_text |
| 方法 | BM25 或 BGE-M3 (可切换) | — | BM25 或 BGE-M3 (config 决定) |
| Chunk→Task 聚合 | — (Chunk 级返回) | — | `_aggregate_chunks_to_tasks()` 取最高分 |

### Hybrid 融合

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 融合方式 | RRF 跨集合 (summary ↔ cleaned_text) | — | RRF 跨集合 (tasks ↔ cleaned_text) |
| Reranker | — | — | Qwen3-Reranker 精排 |
| 结果粒度 | Chunk 级 (HybridResult) | — | Task 级 (SessionSearchResult + ChunkDetail) |

## 如何选择

**日常搜索：** 用 P4 CLI。它提供最完整的信息（Task 摘要 + Chunk 预览），启动快，支持交互模式。

**调参对比：** 用 P3 demo。它能直观展示 Dense/Sparse/Hybrid 三路效果差异，方便切换 embedding 模型和 sparse 方法。

**数据验收：** 用 MS inspect。它提供全局数据质量视角，包括 embedding 聚类质量、summary 覆盖率、命中率等指标。

**典型工作流：**

```
Phase 1/2 产出数据 → code_p3_main.py 写入 Qdrant
                    → code_MS_inspect.py 验收数据质量
                    → code_p3_search_demo.py 调参对比检索策略
                    → code_p4_search_cli.py 日常使用
```
