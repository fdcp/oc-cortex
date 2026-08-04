# 检索原理说明

本文档解释项目中使用的五种检索概念的原理，以及它们在本项目中的具体实现。

## 1. Dense 检索 (稠密检索)

### 原理

Dense 检索基于 Bi-Encoder 架构。核心思想是将文本编码为一个固定维度的稠密向量（embedding），通过向量间的余弦相似度衡量语义相关性。

```
Bi-Encoder 架构:

  Query 文本 ──→ [Transformer Encoder] ──→ query_vec (512/1024 维)
                                                    ↕ cosine 相似度
  Doc 文本   ──→ [Transformer Encoder] ──→ doc_vec  (512/1024 维)
```

**编码阶段（离线）：** 将所有文档通过 embedding 模型编码为向量，存入向量数据库（本项目使用 Qdrant）。

**检索阶段（在线）：** 将用户 query 编码为向量，在向量数据库中做近似最近邻搜索（ANN），返回 cosine 相似度最高的 top-K 文档。

**Cosine 相似度公式：**

```
sim(q, d) = (q · d) / (||q|| × ||d||)
```

值域 [-1, 1]，越接近 1 表示语义越相似。实际使用时通常对向量做 L2 归一化，此时 cosine 等价于内积。

**特点：**
- 语义理解能力强，能匹配同义词和近义表达（"GPU对比" 能匹配 "数据中心算力选型"）
- 文档向量可预计算、可缓存，在线只需编码 query + ANN 检索，速度快
- 但对精确关键词不敏感（"A800" 这样的专有名词可能被泛化）

### 项目实现

本项目使用 `sentence-transformers` 库加载 embedding 模型，默认 `BAAI/bge-small-zh-v1.5`（512 维，中文优化），可选 `Qwen/Qwen3-Embedding-0.6B`（1024 维，精度更高）。

三个 Qdrant 集合分别存储不同文本的 dense 向量：

| 集合 | 嵌入文本 | 语义侧重 |
|------|---------|---------|
| tasks | `task_summary` (Phase 2 LLM 生成) | 高层语义：这个 session 做了什么 |
| chunks_summary | `summary` (Phase 2 LLM 生成) | 中层语义：这一轮对话的摘要 |
| chunks_cleaned_text | `cleaned_text()` (格式化对话原文) | 底层语义：完整对话内容 |

> 三个集合在 BM25 / BGE-M3 两种模式下的向量存储方式（unnamed dense vs named {dense, sparse}）详见：[Phase 3 向量存储架构图](diagram_phase3_vector_store.html)

Dense 检索调用 `Phase3Store.search_dense()`，内部使用 Qdrant 的 `query_points()` 做 cosine ANN 搜索。

## 2. Sparse 检索 (稀疏检索)

Sparse 检索基于关键词精确匹配，不依赖语义理解，擅长捕捉专有名词、代码标识符等精确术语。

### 2.1 BM25

BM25 (Best Matching 25) 是经典的关键词打分算法，由 TF-IDF 演化而来。

**打分公式：**

```
BM25(q, d) = Σ IDF(qi) × tf(qi,d) × (k1+1) / (tf(qi,d) + k1×(1 - b + b×|d|/avgdl))
```

其中：
- `tf(qi,d)` — 查询词 qi 在文档 d 中的词频
- `IDF(qi)` — 逆文档频率，衡量词的稀有程度：`log((N - n(qi) + 0.5) / (n(qi) + 0.5))`
- `k1` — 词频饱和参数（默认 1.5），词频越高打分越高但趋于饱和
- `b` — 文档长度归一化参数（默认 0.75），长文档中命中权重更低
- `|d|/avgdl` — 文档长度与平均文档长度的比值

**直觉理解：** BM25 的核心逻辑是——稀有的词比常见的词重要，在短文档中命中比在长文档中命中更有价值。

**中文处理：** BM25 需要分词。本项目使用 jieba 分词（`search` 模式），将中文文本切分为词序列后再建索引。

### 2.2 BGE-M3 Sparse

BGE-M3 是 BAAI 的多语言多粒度检索模型，同时产出 dense embedding 和 learned sparse vectors。

**与 BM25 的区别：** BM25 是基于统计的无监督方法（词频 + IDF），BGE-M3 sparse 是通过 transformer 学到的稀疏表示——每个 token 有一个权重，权重由模型根据上下文决定。

**本项目的近似实现：** 由于 `sentence-transformers` 版本的 BGE-M3 不支持原生 `sparse_vec` 输出，本项目采用 token embedding 近似：

```
1. BGE-M3 tokenizer 分词 → token IDs
2. Transformer forward → last_hidden_state (token embeddings)
3. 每个 token embedding 取 L2 norm 作为权重
4. 按 token ID max-pool（同一 token 取最大权重）
5. 过滤 special tokens
→ SparseVector(indices=token_ids, values=weights)
```

这些 sparse vectors 存入 Qdrant 的 named sparse vectors，检索时通过 Qdrant 原生 sparse vector search 完成。

### 项目实现

| 模式 | 方法 | 索引位置 | 适用场景 |
|------|------|---------|---------|
| BM25 | `search_sparse_bm25()` | in-memory (rank_bm25 库) | 开发环境，数据量 < 10K |
| BGE-M3 | `search_sparse_bge_m3()` | Qdrant sparse vectors | 生产环境，大规模数据 |

BM25 索引在 `chunks_summary` 和 `chunks_cleaned_text` 两个集合上构建。BGE-M3 sparse vectors 在三个集合上都存储；P4 可按 `sparse.chunks_summary_method` 将 `chunks_summary` 配置为 dense 或 sparse，并始终检索 `chunks_cleaned_text` 的 sparse 路径。

## 3. Hybrid 检索 (混合检索)

### 原理

Hybrid 检索将 Dense 和 Sparse 的结果合并，取各自优势：Dense 捕捉语义相似，Sparse 捕捉关键词精确匹配。融合算法决定了如何合并两路排序。

**RRF (Reciprocal Rank Fusion)：**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `rank_i(d)` — 文档 d 在第 i 路检索结果中的排名（从 1 开始）
- `k` — 平滑参数（默认 60），防止排名靠前的文档得分过于主导

**直觉理解：** RRF 不看原始分数（因为 Dense cosine 和 BM25 score 量纲不同、不可比），只看排名。一个文档在两路检索中都排名靠前，它的 RRF 分数就会显著高于只在一路中出现的文档。

**示例：** 假设 k=60

```
文档 A:  Dense 排名第 1, BM25 排名第 3
         RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

文档 B:  Dense 排名第 2, BM25 未命中
         RRF = 1/(60+2) = 0.01613

文档 C:  Dense 排名第 10, BM25 排名第 1
         RRF = 1/(60+10) + 1/(60+1) = 0.01429 + 0.01639 = 0.03068
```

文档 A 因两路都靠前而获得最高分。文档 C 虽然 BM25 排名第一，但 Dense 排名太低，综合分不如 A。

### 项目实现

本项目的 Hybrid 检索是**跨集合、分层融合**的（cross-collection hierarchical fusion）：

```
chunks_summary → task 映射 (dense 或 sparse) ─┐
                                                ├── 第一层 RRF → chunk task 结果
chunks_cleaned_text → task 映射 (sparse)     ─┘
                                                        │
Dense → tasks 集合 (task_summary 语义) ─────────────────┴── 第二层 RRF
```

三路搜的是不同集合或不同文本表示，因此能覆盖不同类型的查询意图。两个 chunk 路径先分别聚合到 task 级（同一 task 取最高分），再互相做第一层 RRF；第一层结果随后与 Dense 的 task 级结果做第二层 RRF。

跨集合 Hybrid 相比同集合 Hybrid 的优势：`chunks_summary` 提供可配置的语义或关键词路径，`chunks_cleaned_text` 保留完整原始对话中的具体术语和代码，`tasks.task_summary` 提供高层语义，三者互补性更强。

## 4. 粗排 (Coarse Ranking)

### 原理

粗排是搜索流水线的第一阶段，目标是从全量文档中快速筛选出少量候选，为后续精排提供输入。核心诉求是**召回率**（尽量不漏）和**速度**（尽量快），对排序精度要求不高。

**典型的粗排策略：**
- Dense ANN 检索：毫秒级，从百万文档中召回 top-100
- Sparse BM25 检索：毫秒级，关键词命中召回
- Hybrid RRF：合并两路，互补召回

粗排后的候选数量通常是最终结果的 3-10 倍（比如要返回 5 条结果，粗排召回 25 条候选）。

### 项目实现

本项目的粗排由 Stage 1-5 组成：

```
Stage 1: Dense → tasks 集合
         编码 query → Qdrant cosine ANN → top (top_k × 5) 个 task
         耗时: < 0.02s

Stage 2: chunks_summary → task 集合
          dense 或 sparse，由 sparse.chunks_summary_method 配置
          top (top_k × 15) 个 chunk → 聚合为 task 级

Stage 3: Sparse → chunks_cleaned_text 集合
         BM25: jieba 分词 → BM25Okapi 打分 → top (top_k × 15) 个 chunk
         BGE-M3: sparse encode → Qdrant sparse search → top (top_k × 15) 个 chunk
         → 聚合为 task 级 (同一 task 取最高分)
         耗时: < 0.01s (BM25) / ~0.1-0.5s (BGE-M3)

Stage 4: Chunk 路径 RRF
          合并 Stage 2 + Stage 3 的 task 级结果

Stage 5: Task 路径 RRF
          合并 Stage 1 + Stage 4 的 task 级结果 → top (top_k × 5) 个候选
         耗时: < 0.001s

粗排总耗时: < 0.5s
```

粗排产出的是 `HybridResult` 列表，每条包含 `task_id`、`rrf_score`、以及两路的排名信息。这些候选被传给精排阶段做二次打分。

**候选倍数 (`candidate_multiplier`)** 控制粗排的召回宽度。默认值 5 意味着 `top_k=5` 时粗排召回 25 个候选。倍数越大，精排看到的候选越多（可能提高最终精度），但精排耗时也线性增长。

## 5. 精排 (Fine Ranking / Reranking)

### 原理

精排是搜索流水线的第二阶段，目标是对粗排产出的少量候选做精确排序。核心诉求是**精度**，对速度要求相对宽松。

**Cross-Encoder 架构：**

```
Cross-Encoder:

  [Query + Document] ──→ [Transformer Encoder] ──→ relevance score
```

与 Bi-Encoder 的区别：

| 维度 | Bi-Encoder (Dense 检索) | Cross-Encoder (Reranker) |
|------|----------------------|-------------------------|
| 编码方式 | query 和 document 各自独立编码 | query + document 拼接后联合编码 |
| 交互深度 | 无交互，只在最后算 cosine | 每一层 attention 都有交互 |
| 预计算 | document 向量可离线预计算 | 不可预计算，每次必须联合 forward |
| 推理速度 | 快（query 编码 + ANN） | 慢（每对 query-doc 都要过完整模型） |
| 精度 | 中 | 高 |
| 适用阶段 | 粗排（全量文档） | 精排（少量候选） |

### 项目实现

本项目使用 `Qwen/Qwen3-Reranker-0.6B` 做精排，这是一个基于 CausalLM 架构的 reranker。

**打分机制：**

Qwen3-Reranker 不是传统的回归打分模型（直接输出一个 float），而是通过分类方式计算相关性——让模型判断 query-document 对是否相关（yes/no），用两个 token 的 logit 差异作为分数：

```
Prompt 构造:
  system: "Judge whether the Document matches the Intent.
           Your answer must be either yes or no."
  user:   <Instruct>: Given a web search query, retrieve relevant passages.
          <Query>: {用户查询}
          <Document>: {task_summary}
  assistant: <think> </think>

模型推理:
  logits = model.forward(prompt)[last_token_position]
  yes_logit = logits[tokenizer.encode("yes")]
  no_logit  = logits[tokenizer.encode("no")]

打分:
  score = softmax([yes_logit, no_logit])[0]
        = exp(yes_logit) / (exp(yes_logit) + exp(no_logit))
```

score 接近 1 表示模型认为 document 与 query 高度相关，接近 0 表示不相关。

**为什么用 CausalLM 做 Reranker：** 传统 Cross-Encoder 用 `[CLS]` token 的表示做回归打分。Qwen3-Reranker 利用 CausalLM 的 next-token prediction 能力，通过 yes/no 二分类的方式将生成任务转化为判别任务。这种设计复用了大规模预训练的 CausalLM 能力，在训练数据较少的情况下也能获得较好的泛化。

**精排流程：**

```
输入: query + 25 个候选 task 的 task_summary
分批: batch_size=4, 共 7 batch
每批:
  1. 构造 4 条 prompt (query + 各自的 task_summary)
  2. tokenizer batch encode (padding, truncation, max_length=8192)
  3. model forward → logits
  4. 提取每条的 yes/no logit → score
汇总: 25 个 score → 排序 → 取 top_k
```

精排耗时（CPU）：每条候选 ~0.3-0.6s，25 条候选约 8-15s。GPU 上可降到 1-2s。

**`--no-rerank` 模式：** 跳过精排，直接返回粗排的 RRF 排序结果。适用于对延迟敏感或不需要最高精度的场景。

## 完整流水线总览

```
用户 Query: "GPU对比分析 A800 H100 H800"
  │
  │ ┌─────────────── 粗排 (< 0.5s) ───────────────┐
  │ │                                                │
  ├─┤ Stage 1: Dense (bge-small-zh)                  │
  │ │   query embedding → Qdrant cosine search       │
  │ │   → 25 个 task 候选                            │
  │ │                                                │
  ├─┤ Stage 2: Sparse (BM25 / BGE-M3)               │
  │ │   query 分词/编码 → 匹配 cleaned_text           │
  │ │   → 75 个 chunk 命中 → 聚合为 ~15 个 task       │
  │ │                                                │
  ├─┤ Stage 3: RRF 融合                              │
  │ │   合并两路 task 排名 → 25 个候选 task            │
  │ └────────────────────────────────────────────────┘
  │
  │ ┌─────────────── 精排 (8-15s CPU) ──────────────┐
  │ │                                                │
  ├─┤ Stage 4: Reranker (Qwen3-Reranker-0.6B)       │
  │ │   25 条 prompt → CausalLM → yes/no score       │
  │ │   → top 5 精排结果                              │
  │ │                                                │
  │ └────────────────────────────────────────────────┘
  │
  │ ┌─────────────── 展开 ──────────────────────────┐
  │ │                                                │
  └─┤ Stage 5: Chunk 展开                            │
    │   为每个 task 查找关联 chunk                     │
    │   → 附加 summary + cleaned_text_preview         │
    └────────────────────────────────────────────────┘
  │
  ▼
最终结果: 5 条 SessionSearchResult
  #1 rerank=0.9987  数据中心GPU性能对比与选型分析  (5 chunks)
  #2 rerank=0.8543  A800形态识别方法               (2 chunks)
  #3 rerank=0.7210  GPU集群拓扑设计                (3 chunks)
  ...
```
