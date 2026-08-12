# 检索工具对比说明

项目中有三个脚本都涉及 Qdrant 向量检索，但定位、检索深度和使用场景各不相同。本文梳理它们的异同，帮助快速选择正确的工具。

## 查询输入

三个检索入口都读取 `query_instruction_for_retrieval` 配置。配置非空时，实际检索查询为：

```text
query_instruction_for_retrieval + 原始 query
```

默认前缀为 `为这个句子生成表示以用于检索相关文章：`，用于给 embedding 检索模型提供查询指令。终端输出仍展示原始 query。Phase 4 的 Reranker 也接收拼接后的查询，同时继续使用独立的 `reranker.instruction` 作为 Reranker 的 `<Instruct>` 内容。

## 一览

| 维度 | `code_p3_search_demo.py` | `code_MS_inspect.py` | `code_p4_search_cli.py` |
|------|--------------------------|----------------------|--------------------------|
| 定位 | 检索实验室 | 数据质检 | 检索产品 |
| 所属阶段 | Phase 3 | 里程碑检视 | Phase 4 |
| 核心类 | `Phase3Store` | `QdrantClient` (直调) | `SessionSearcher` |
| 检索方式 | Dense / Sparse / Hybrid 三路分开展示 | 仅 Dense (top-5 命中率验证) | 两级 RRF Hybrid + Reranker |
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

这是一个里程碑检视工具，包含 6 大功能模块，检索只是其中之一。它直接操作 `QdrantClient`，不依赖 `Phase3Store`，目的是验证数据和 embedding 的质量，而非提供日常搜索能力。集合名、Qdrant 路径、embedding 模型以及输入文件路径全部从 Phase 3 配置（`config/code_p3_config.yaml`）读取，与流水线保持一致；相对路径会回退到仓库根目录解析，因此可从任意目录运行。

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
# 默认读取 config/code_p3_config.yaml
python src/code_MS_inspect.py

# 指定配置 / 调整样本数
python src/code_MS_inspect.py --config config/code_p3_config.yaml --samples 8
```

### 3. `code_p4_search_cli.py` — 检索产品

这是面向最终用户的搜索 CLI，封装了完整的 Hybrid + Reranker 精排流水线。假设 Qdrant 数据已由 Phase 3 写入，直接连接读取。

**检索流程（7 阶段）：**

```
Stage 1: Dense 检索 → tasks 集合 → task_summary 语义匹配 → n_candidates 条候选
Stage 2: chunks_summary 检索 → dense 或 sparse（sparse.chunks_summary_method）→ 聚合到 Task 级别
Stage 3: Sparse 检索 → chunks_cleaned_text 集合 → BM25/BGE-M3 匹配 → 聚合到 Task 级别
Stage 4: Chunk 路径 RRF → 合并 Stage 2 + Stage 3
Stage 5: Task 路径 RRF → 合并 Stage 1 + Stage 4 → 截断到 n_candidates
Stage 6: Reranker 精排 → Qwen3-Reranker 对 task_summary 重新打分 → top_k
Stage 7: Chunk 展开 → 为每个结果 Task 附加关联 Chunk 的摘要和文本预览
```

**关键特点：**

- 返回 `SessionSearchResult`，包含 task_label、task_summary、rerank_score、hybrid_score，以及关联 Chunk 的 summary 和 cleaned_text_preview，信息最丰富。
- 支持 `--interactive` 交互模式，循环输入查询。
- 支持 `--no-rerank` 跳过精排，默认按 `--top-k` 截断以匹配精排路径行为；配合 `--full-pool` 可返回完整候选池 (`top_k × candidate_multiplier`)，方便对比粗排召回全貌。
- Dense 检索目标是 tasks 集合（Task 级别）；两个 chunk 路径分别查询 `chunks_summary` 和 `chunks_cleaned_text`（Chunk 级别，聚合回 Task），先做 chunk 路径 RRF，再与 tasks Dense 结果做第二次 RRF。
- 启动快（不重建索引），适合日常使用。

**典型用法：**

```bash
# 单次查询
python src/code_p4_search_cli.py --query "GPU对比分析"

# 交互模式
python src/code_p4_search_cli.py --interactive

# 跳过 Reranker 对比粗排效果 (默认按 --top-k 截断)
python src/code_p4_search_cli.py --query "优化器学习率" --no-rerank

# 跳过 Reranker 并返回完整候选池
python src/code_p4_search_cli.py --query "优化器学习率" --no-rerank --full-pool
```

## 检索方式深度对比

### Dense 检索

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 目标集合 | chunks_summary | tasks | tasks |
| 检索对象 | chunk summary 文本 | task_summary 文本 | task_summary 文本 |
| 模型 | 可切换 (bge-small-zh / Qwen3) | 由 config 决定 (embedding.model) | 由 config 决定 |
| 用途 | 对比展示 | 命中率验证 | 生产检索 |

### Sparse 检索

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 目标集合 | chunks_summary / chunks_cleaned_text | — (无) | chunks_summary / chunks_cleaned_text |
| 方法 | BM25 或 BGE-M3 (可切换) | — | BM25 或 BGE-M3 (config 决定) |
| Chunk→Task 聚合 | — (Chunk 级返回) | — | `_aggregate_chunks_to_tasks()` 取最高分 |

### Hybrid 融合

| 维度 | P3 demo | MS inspect | P4 CLI |
|------|---------|------------|--------|
| 融合方式 | RRF 跨集合 (summary ↔ cleaned_text) | — | 两级 RRF：chunk 路径先融合，再与 tasks 融合 |
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
