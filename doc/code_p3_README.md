# Phase 3: Qdrant 三集合向量存储 + 混合检索

## 概述

Phase 3 将 Phase 1 (chunks) 和 Phase 2 (tasks + chunk summaries) 的输出向量化，写入本地 Qdrant 三集合，并提供**稠密检索 + 稀疏检索**的混合检索能力。

三个集合各有侧重：`tasks` 集合存储 task_summary 的向量，适合按"做了什么"语义检索；`chunks_summary` 集合存储 chunk summary 的向量，适合按单轮摘要检索；`chunks_cleaned_text` 集合存储 cleaned_text 的向量和 BM25 索引，适合按关键词精确匹配。

**核心设计原则：Dense 用 summary（语义聚合），Sparse 用 cleaned_text（原始对话关键词）。**

混合检索架构按环境可配置：

| 环境 | Dense 模型 | Sparse 方法 | 融合方式 |
|------|-----------|-------------|---------|
| 开发 (Apple M2 CPU) | `BAAI/bge-small-zh-v1.5` (512 维) | BM25 + jieba 分词 | Python 级 RRF |
| 上线 (NVIDIA GPU) | `Qwen/Qwen3-Embedding-0.6B` (1024 维) | BGE-M3 sparse vectors | Qdrant 原生 RRF |

## 依赖

```bash
pip install qdrant-client sentence-transformers jieba rank_bm25
```

首次运行会自动从 HuggingFace 下载 embedding 模型。国内环境建议设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 文件结构

```
config/code_p3_config.yaml   # 配置 (dense/sparse 模型、Qdrant 路径、离线/缓存)
code_p3_hf_config.py         # HuggingFace 环境配置 (离线模式 + 缓存目录)
code_p3_qdrant_store.py      # 核心模块: Phase3Store 类 (存储 + 检索)
code_p3_main.py              # 写入入口: 加载数据 -> embedding -> upsert
code_p3_search_demo.py       # 检索演示: dense / sparse / hybrid 对比
```

## 运行

### 写入数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
python code_p3_main.py
python code_p3_main.py --config config/code_p3_config.yaml
```

### 检索演示

```bash
# 全量数据 bge-small-zh + BM25
python code_p3_search_demo.py --mode all

# 1/5 数据 Qwen3 + BGE-M3 (示例)
python code_p3_search_demo.py --mode sample

# 自定义查询
python code_p3_search_demo.py --mode all --query "GPU对比分析"
```

## 配置说明 (code_p3_config.yaml)

### 稠密检索 (embedding)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding.model` | `BAAI/bge-small-zh-v1.5` | Dense embedding 模型 |
| `embedding.dim` | `512` | 向量维度 (与模型匹配) |
| `embedding.batch_size` | `32` | 每批 embed 条数; CPU 建议 4-32, GPU 可调到 64+ |
| `embedding.device` | `cpu` | `auto` / `cpu` / `mps` / `cuda` |

### 稀疏检索 (sparse)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `sparse.method` | `bm25` | `bm25` (开发) 或 `bge_m3` (上线) |
| `sparse.bge_m3_model` | `BAAI/bge-m3` | BGE-M3 模型路径 (bge_m3 模式使用) |
| `sparse.jieba_mode` | `search` | jieba 分词模式 (bm25 使用) |
| `sparse.bm25_params.k1` | `1.5` | BM25 词频饱和参数 |
| `sparse.bm25_params.b` | `0.75` | BM25 文档长度归一化参数 |
| `sparse.fuse_k` | `60` | RRF 融合参数 k |

### Qdrant

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `qdrant.path` | `./qdrant_data` | 文件持久化路径 |
| `qdrant.collections.tasks` | `tasks` | task 集合名 |
| `qdrant.collections.chunks_summary` | `chunks_summary` | chunk summary 集合名 |
| `qdrant.collections.chunks_cleaned_text` | `chunks_cleaned_text` | chunk cleaned_text 集合名 |

## 核心设计

### 三集合架构

**tasks 集合** -- 每个 point 代表一个 session 级任务：
- Dense 向量来源: `task_summary` (50-200 字, Phase 2 LLM 生成)
- 无 BM25 索引 (Phase 4 通过两阶段查询处理)
- Payload: `task_id`, `session_id`, `task_label`, `task_summary`, `chunk_ids`, `created_at`

**chunks_summary 集合** -- 每个 point 代表一个对话轮次的摘要：
- Dense 向量来源: `summary` (Phase 2 生成的 chunk 摘要)
- 无 BM25 索引
- Payload: `chunk_id`, `session_id`, `turn_index`, `task_id`, `summary`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`

**chunks_cleaned_text 集合** -- 每个 point 代表一个对话轮次的原始对话：
- Dense 向量来源: `cleaned_text` (格式化后的用户/助手对话)
- BM25 索引来源: `cleaned_text` (同一文本)
- Payload: `chunk_id`, `session_id`, `turn_index`, `task_id`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`

### Cross-Collection 混合检索

Dense 和 Sparse 搜索不同集合，利用各自的文本优势：

```
用户查询
  ├─ Dense:  query embedding → Qdrant cosine search
  │          Phase 3: 搜 chunks_summary (语义匹配摘要)
  │          Phase 4: 搜 tasks (语义匹配 task_summary)
  ├─ Sparse: jieba 分词 → BM25 打分
  │          始终搜 chunks_cleaned_text (关键词匹配原始对话)
  │          Phase 4: chunk 级结果 → chunk_to_task 映射 → 聚合为 task 级
  └─ RRF:    Reciprocal Rank Fusion 融合两路结果 → 最终 top-K
```

Phase 3 (search_demo) 使用 `search_hybrid_cross_collection()` 直接做跨集合 RRF 融合。
Phase 4 (searcher) 实现两阶段查询：Dense→tasks + BM25→chunks_cleaned_text→task 映射→RRF。

**RRF 融合公式:**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中 k 默认 60，rank 从 1 开始。两路都有排名的文档会获得更高融合分数。

### BM25 模式 (开发)

- 使用 jieba 中文分词构建 BM25 索引 (in-memory, rank_bm25 库)
- BM25 索引仅在 `chunks_cleaned_text` 集合的 upsert 时构建
- `tasks` 和 `chunks_summary` 集合只做 dense 向量存储
- 检索时 Dense 从 Qdrant 获取，BM25 从内存获取，Python 层面做 RRF 融合
- 适合快速迭代，数据量 < 10K 时性能很好

### BGE-M3 模式 (上线)

- 使用 BGE-M3 模型生成 sparse vectors，存入 Qdrant 的 named sparse vectors
- Qdrant 集合配置为 named vectors: `dense` + `sparse`
- 检索时可利用 Qdrant 原生的 Prefetch + FusionQuery 做 RRF 融合
- 适合大规模数据，sparse vectors 索引由 Qdrant 管理

**Sparse vector 计算方式:**

sentence-transformers 版本的 BGE-M3 不支持 `sparse_vec` 输出（原版 FlagEmbedding 才有），因此采用 token embedding 近似：

1. 用 BGE-M3 tokenizer 分词
2. 通过 transformer forward pass 获取 token embeddings (last_hidden_state)
3. 计算每个 token embedding 的 L2 norm 作为权重
4. 按 token ID max-pool（同一 token 取最大权重）
5. 过滤 special tokens (`<s>`, `<pad>`, `</s>`, `<unk>`)

这是一种近似方案，在生产环境建议使用 FlagEmbedding 原版的 `BGEM3FlagModel` 获取真正的 learned sparse vectors。

### 确定性 UUID

使用 `_stable_uuid()` 从业务 ID 的 MD5 生成 Qdrant point id，保证幂等 upsert。

### Chunk-Task 关联

`upsert_chunks_summary()` 和 `upsert_chunks_cleaned_text()` 都接受 `tasks` 参数，构建 `chunk_id → task_id` 反向映射，写入 chunk payload。Phase 4 的两阶段查询还利用 `chunk_to_task` 映射将 BM25 的 chunk 级结果聚合回 task 级。

## 性能参考 (Apple M2 CPU)

### bge-small-zh-v1.5 + BM25 (全量数据)

| 操作 | 数量 | 耗时 |
|------|------|------|
| Dense embedding (tasks) | 29 | ~0.8s |
| Dense embedding (chunks_summary) | 83 | ~0.6s |
| Dense embedding (chunks_cleaned_text) | 83 | ~2.7s |
| BM25 索引 (chunks_cleaned_text) | 83 | ~0.5s |
| 单次混合检索 | - | <0.02s |

### Qwen3-Embedding-0.6B + BGE-M3 (1/5 数据示例)

| 操作 | 数量 | 耗时 |
|------|------|------|
| Dense embedding (tasks) | 5 | ~11s |
| Dense embedding (chunks_summary) | 17 | ~预估 30s |
| Dense embedding (chunks_cleaned_text) | 17 | ~预估 15min (CPU 长文本慢) |
| 单次混合检索 | - | ~0.2s |

CPU 模式下 Qwen3 对长文本 chunk embedding 很慢（每条约 30-120s）。生产环境建议使用 GPU。BGE-M3 sparse 计算（token embedding L2 norm）相对快得多。

## 检索结果示例

### Phase 3 Cross-Collection 检索 (全量数据, bge-small-zh + BM25)

查询: "GPU对比分析"

```
Dense [chunks_summary] #1:  [ses_...c7]  详细解析了TP=2与SP=4的2D mesh配置...     (score=0.592)
Sparse [chunks_cleaned_text] #1: [ses_...c3]  用户要求生成v2版本...                  (score=5.487)
Hybrid #1: [ses_...c13]  FA-2的并行策略类比为warp粒度的序列并行...  D#2 S#8  RRF=0.0308
```

Dense 和 Sparse 搜索不同集合：Dense 在 summary 中找语义相关，Sparse 在原始对话中找关键词命中。

### Phase 4 两阶段查询 (全量数据, bge-small-zh + BM25)

查询: "GPU对比分析"

```
Dense[tasks]:        25 个候选 task
BM25[cleaned_text]:  38 chunk 命中 → 聚合为 15 个 task
RRF 融合:            25 个候选 task

#1  数据中心GPU性能对比与选型  hybrid=0.0323  (5 chunks: V100/A800/H100/H200/H800 完整对比)
#2  生成优化器v2指南          hybrid=0.0315
#3  SDPA与FlashAttention原理  hybrid=0.0310
#4  A800形态识别方法          hybrid=0.0309
#5  NCCL通信性能诊断与优化    hybrid=0.0303
```

BM25 搜 cleaned_text 后通过 chunk_to_task 映射聚合，与 dense 搜 task_summary 做 RRF 融合。"数据中心GPU性能对比与选型" 因 5 个关联 chunk 在 BM25 中多次命中，获得最高的 RRF 分数。
