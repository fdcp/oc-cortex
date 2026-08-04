# Phase 4: 跨 Session 搜索 + Reranker 精排

## 概述

Phase 4 在 Phase 3 的三集合向量存储基础上，构建面向最终用户的**跨 session 搜索**能力。核心流程为两级 RRF 粗排加一阶段精排（Qwen3-Reranker），返回 Task 级别的搜索结果，附带关联 Chunk 的摘要和文本预览。

**核心设计原则：Dense 搜 task_summary 做语义匹配；chunks_summary 和 chunks_cleaned_text 分别检索并映射到 task 后先做 RRF，再与 Dense task 结果做第二次 RRF；Reranker 对候选 task 做精排。**

与前序阶段的关系：

| 阶段 | 职责 | 产出 |
|------|------|------|
| Phase 1 | 原始数据清洗 | `chunks.jsonl` |
| Phase 2 | CoT 任务提取 | `tasks.jsonl` + `chunks_summary_p2.jsonl` |
| Phase 3 | 向量化 + Qdrant 存储 | Qdrant 三集合 + Sparse 索引 |
| **Phase 4** | **搜索编排 + 精排** | **`SessionSearchResult` (运行时输出)** |

## 依赖

```bash
pip install qdrant-client sentence-transformers jieba rank_bm25 transformers torch
```

Reranker 模型 (`Qwen/Qwen3-Reranker-0.6B`) 首次运行需从 HuggingFace 下载。国内环境建议设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 文件结构

```
src/
  code_p4_searcher.py     # 核心模块: SessionSearcher 类 (搜索编排)
  code_p4_reranker.py     # Reranker: Qwen3Reranker 类 (CausalLM 精排)
  code_p4_search_cli.py   # CLI 入口: 单次查询 / 交互模式

config/
  code_p3_config.yaml     # 复用 Phase 3 配置 (reranker 段为 Phase 4 新增)
```

Phase 4 没有独立的配置文件，复用 `config/code_p3_config.yaml`，其中 `reranker:` 段为 Phase 4 新增。

## Query Rewrite 实验状态

`p4-query-rewrite` 分支实现了基于 OpenCode Zen `deepseek-v4-flash-free` 的单查询改写，并将改写结果仅用于 Dense 检索。实际测试表明，Query Rewrite 的检索效果不如原有方案，因此当前不建议将该分支合并到 `main`。

当前建议使用原有检索方式：

```yaml
query_instruction_for_retrieval: "为这个句子生成表示以用于检索相关文章："

query_rewrite:
  enable: false
```

如需继续实验，可切换到 `p4-query-rewrite` 分支并设置 `query_rewrite.enable: true`。该实验分支的改动已单独提交，未合并到 `main`。

## 运行

### 单次查询

```bash
python src/code_p4_search_cli.py --query "GPU对比分析" --config config/code_p3_config.yaml
```

### 交互模式

```bash
python src/code_p4_search_cli.py --interactive --config config/code_p3_config.yaml
```

交互模式下输入查询文本回车即搜，输入 `demo` 运行 5 条预设示例查询，输入 `q` 退出。

### 跳过 Reranker (对比粗排效果)

```bash
python src/code_p4_search_cli.py --query "优化器学习率" --no-rerank
```

### 调整返回数量

```bash
python src/code_p4_search_cli.py --query "session管理" --top-k 10
```

## 配置说明

Phase 4 的配置嵌入在 `config/code_p3_config.yaml` 的 `reranker:` 段中：

### Reranker

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `reranker.model` | `Qwen/Qwen3-Reranker-0.6B` | Reranker 模型 (CausalLM 架构) |
| `reranker.device` | `cpu` | `auto` / `cpu` / `mps` / `cuda` |
| `reranker.instruction` | `Given a web search query, retrieve relevant passages.` | Reranker system instruction |
| `reranker.max_length` | `8192` | prompt 最大 token 长度 (超出截断) |
| `reranker.batch_size` | `4` | 每批 rerank 文档数 |

### 复用 Phase 3 配置

Phase 4 同时读取以下 Phase 3 配置项：

| 配置段 | 用途 |
|--------|------|
| `phase1.*` / `phase2.*` | 源数据文件路径 (chunks / tasks / summaries) |
| `embedding.*` | Dense embedding 模型和设备 (用于 Phase3Store 初始化) |
| `sparse.*` | Sparse 方法选择 (BM25/BGE-M3)、参数和 RRF 融合参数 |
| `qdrant.*` | Qdrant 数据路径和集合名 |

## 核心设计

### 七阶段搜索流水线

`SessionSearcher.search()` 封装了完整的搜索流程：

```
用户查询
  │
  ├─ Stage 1: Dense → tasks 集合
  │   语义匹配 task_summary，取 top_k × candidate_multiplier 条候选
  │
  ├─ Stage 2: chunks_summary 检索
  │   由 sparse.chunks_summary_method 配置为 dense 或 sparse
  │   → _aggregate_chunks_to_tasks() 聚合为 task 级
  │
  ├─ Stage 3: chunks_cleaned_text 检索
  │   BM25 关键词匹配 或 BGE-M3 sparse 向量匹配 cleaned_text
  │   → _aggregate_chunks_to_tasks() 聚合为 task 级
  │
  ├─ Stage 4: Chunk 路径 RRF
  │   chunks_summary→task + chunks_cleaned_text→task
  │
  ├─ Stage 5: Task 路径 RRF
  │   Dense(tasks.task_summary) + Stage 4 结果 → 截断到 n_candidates
  │
  ├─ Stage 6: Reranker 精排 (可选, --no-rerank 跳过)
  │   Qwen3-Reranker 对 task_summary 重新打分 → top_k
  │
  └─ Stage 7: Chunk 展开
      为每个结果 task 查找关联 chunk，附加 summary 和 cleaned_text 预览
```

### 候选倍数 (candidate_multiplier)

粗排候选池大小为 `top_k × candidate_multiplier`（默认 5）。例如 `top_k=5` 时，最终 RRF 会产出最多 25 个候选 task，随后 Reranker 从中精排出 5 个。倍数越大，Reranker 看到的候选越多，精度可能更高但耗时更长。

### Sparse Chunk→Task 聚合

Sparse 检索（BM25 或 BGE-M3）在 chunk 级别执行（因为 cleaned_text 存储在 chunk 集合中），需要映射回 Task 级别才能与 Dense 的 task 级结果做 RRF 融合：

```
Sparse chunk hits:
  chunk_A (task_1, score=8.2)
  chunk_B (task_1, score=6.1)    →  task_1: score=8.2 (取最高)
  chunk_C (task_2, score=7.5)    →  task_2: score=7.5
```

`_aggregate_chunks_to_tasks()` 遍历 chunk 级结果，通过 `chunk_to_task` 映射表找到所属 task，同一 task 保留最高分数。映射后的 SearchResult 使用 task 级的 stable UUID 作为 point_id，确保 RRF 融合时正确去重。

### Qwen3-Reranker 精排

Qwen3-Reranker 是基于 CausalLM 架构的重排序模型，打分机制如下：

```
输入 prompt:
  <|im_start|>system
  Judge whether the Document matches the Intent. ...<|im_end|>
  <|im_start|>user
  <Instruct>: Given a web search query, retrieve relevant passages.
  <Query>: {query}
  <Document>: {task_summary}<|im_end|>
  <|im_start|>assistant
  <think>
  </think>

输出:
  取最后一个有效 token 位置对 "yes" / "no" 两个 token 的 logit
  score = softmax(yes_logit, no_logit)[yes]
```

**与 Bi-Encoder (embedding) 的区别：** Bi-Encoder 预计算向量，查询时只做 cosine 距离；Reranker 是 Cross-Encoder，将 query 和 document 拼接后过完整 transformer，推理更慢但精度更高。因此 Reranker 只作用于粗排后的少量候选（通常 20-30 个），而非全量数据。

### 搜索结果结构

`SessionSearchResult` 包含完整的检索信息：

```python
@dataclass
class SessionSearchResult:
    task_id: str                    # task 唯一标识
    session_id: str                 # 所属 session
    task_label: str                 # task 简短标签 (Phase 2 LLM 生成)
    task_summary: str               # task 详细摘要 (Phase 2 LLM 生成)
    rerank_score: float             # Reranker 精排分数 (0~1)
    hybrid_score: float             # RRF 粗排分数
    chunks: list[ChunkDetail]       # 关联 chunk 详情列表
```

每个 `ChunkDetail` 包含：

| 字段 | 说明 |
|------|------|
| `chunk_id` | chunk 唯一标识 |
| `session_id` | 所属 session |
| `turn_index` | 对话轮次序号 |
| `summary` | chunk 摘要 (Phase 2 生成) |
| `cleaned_text_preview` | cleaned_text 前 200 字预览 |
| `raw_size_tokens` | 原始文本 token 数 |
| `cleaned_size_tokens` | 清洗后 token 数 |

### 内存数据结构

`SessionSearcher.__init__()` 在初始化时加载所有源数据到内存 map，用于运行时的快速查找：

| 数据结构 | 类型 | 用途 |
|----------|------|------|
| `task_map` | `dict[task_id → Task]` | Reranker 后取 task 详情 |
| `chunk_map` | `dict[chunk_id → Chunk]` | Chunk 展开时取 chunk 数据 |
| `summaries` | `dict[chunk_id → str]` | Chunk 展开时取 summary |
| `chunk_to_task` | `dict[chunk_id → task_id]` | Sparse chunk→task 聚合 |

BM25 索引在初始化时按需重建（`_rebuild_bm25_index()`），仅当 `sparse.method` 为 `bm25` 时执行。BGE-M3 模式下 sparse vectors 已存储在 Qdrant 中，无需内存重建。

## 性能参考 (Apple M2 CPU)

| 操作 | 数量 | 耗时 |
|------|------|------|
| SessionSearcher 初始化 | 31 tasks / 83 chunks | ~5s (含模型加载) |
| Sparse 索引重建 (BM25) | 83 cleaned_text | ~0.5s |
| Dense 检索 (bge-small-zh) | 单次 | <0.02s |
| Sparse 检索 (BM25) | 单次 | <0.01s |
| Sparse 检索 (BGE-M3) | 单次 | ~0.1-0.5s (含 sparse encode) |
| RRF 融合 | 25 候选 | <0.001s |
| Reranker 精排 (Qwen3-0.6B) | 25 候选, batch=4 | ~8-15s (CPU) |
| 单次搜索总耗时 (含 Reranker) | top_k=5 | ~10-20s |
| 单次搜索总耗时 (跳过 Reranker) | top_k=5 | <0.1s |

Reranker 在 CPU 上耗时显著（每条候选 ~0.3-0.6s）。生产环境建议使用 GPU，或设置 `--no-rerank` 仅用粗排。

## 检索结果示例

### 精排模式 (Hybrid + Reranker)

查询: "GPU对比分析 A800 H100 H800"

```
#1  rerank=0.9987  hybrid=0.0323
    Task:   数据中心GPU性能对比与选型分析
    摘要:   详细对比了V100/A800/H100/H200/H800等数据中心GPU的性能参数...
    关联 Chunks (5 个):
      [1] ses_..._c1 (turn 3, raw=2450tok, clean=1820tok)
          摘要: 对比了A800与H100的算力、带宽、功耗指标...

#2  rerank=0.8543  hybrid=0.0309
    Task:   A800形态识别方法
    摘要:   讨论了如何通过PCIe拓扑识别A800的不同形态...
```

### 粗排模式 (--no-rerank)

查询: "GPU对比分析 A800 H100 H800"

```
#1  rerank=0.0000  hybrid=0.0323
    Task:   数据中心GPU性能对比与选型分析

#2  rerank=0.0000  hybrid=0.0315
    Task:   生成优化器v2指南
```

精排模式下 Reranker 能将"GPU对比"直接排到第一（rerank=0.9987），而粗排模式下 RRF 融合分数差异较小（0.0323 vs 0.0315），排序稳定性较低。

## 与 Phase 3 检索的差异

| 维度 | Phase 3 (search_demo) | Phase 4 (searcher) |
|------|----------------------|-------------------|
| Dense 目标 | chunks_summary | tasks |
| Sparse 聚合 | Chunk 级直接返回 | Chunk→Task 聚合 |
| Reranker | 无 | Qwen3-Reranker 精排 |
| 结果粒度 | Chunk (chunk_id + score) | Task (label + summary + chunks) |
| 数据写入 | 每次运行 upsert | 只读 (假设已写入) |
| 定位 | 检索策略对比 | 生产搜索 |

详细的三脚本对比参见 `doc/code_search_tools.md`。
