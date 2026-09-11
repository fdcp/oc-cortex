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
- BM25 索引来源: `summary` (同上文本, `upsert_chunks_summary` 同样调用 `build_bm25_index`,见 `code_p3_qdrant_store.py:633`)
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
- BM25 索引在 `chunks_summary` (LLM 摘要) 和 `chunks_cleaned_text` (cleaned_text) 两个集合的 upsert 时都构建
- `tasks` 集合只做 dense 向量存储 (无 BM25)
- 检索时 Dense 从 Qdrant 获取，BM25 从内存获取，Python 层面做 RRF 融合
- 适合快速迭代，数据量 < 10K 时性能很好

### 存储分离: Dense vs Sparse

**核心事实** (用户最容易踩坑的认知误区):

| 数据类型 | 存哪里 | 持久化 |
|---|---|---|
| Dense 向量 (BGE-small-zh 512d) | Qdrant 服务端 | ✅ 跟着 Qdrant 走 (snapshot/-v) |
| Payload (chunk_id, task_id, summary 等) | Qdrant 服务端 | ✅ 同上 |
| BM25 倒排索引 | **Python 进程内存** (`Phase3Store._bm25_index[collection]`) | ❌ **不落盘,进程死就没了** |

```
           ┌───────────────────────────────────────────────┐
           │  Qdrant Server (持久化)                       │
           │   tasks                  ──→  dense vectors   │
           │   chunks_summary         ──→  dense vectors   │
           │   chunks_cleaned_text    ──→  dense vectors   │
           └───────────────────────────────────────────────┘
                              ↕  HTTP (qdrant-client)
           ┌───────────────────────────────────────────────┐
           │  Phase3Store 进程内存 (不持久化)              │
           │   _bm25_index[chunks_summary]       (LLM 摘要) │
           │   _bm25_index[chunks_cleaned_text]  (cleaned)  │
           └───────────────────────────────────────────────┘
                              ↕  rank_bm25 BM25Okapi
                          Sparse 检索
```

**为什么 BM25 不进 Qdrant?** Qdrant 原生支持的是 BGE-M3 那样的 named sparse vector (dense + sparse 一起存),不直接支持 BM25 这类外部倒排索引算法;BM25 用 `rank_bm25` 库在 Python 进程内现算。

**当前默认 config 下,3 个 collection 的实际用途**:

| Collection | Dense 用? | BM25 用? | 实际检索路径 |
|---|---|---|---|
| `tasks` | ✅ P4 dense search 主路径 | ❌ (无 BM25) | `search_dense(query)` |
| `chunks_summary` | ❌ (config `sparse.chunks_summary_method: sparse`) | ✅ 内存 BM25 (LLM 摘要) | `search_sparse_bm25_tokens` |
| `chunks_cleaned_text` | ❌ (P4 代码无 dense 路径) | ✅ 内存 BM25 (cleaned_text) | `search_sparse_bm25_tokens` |

**配置开关 `sparse.chunks_summary_method`**:

- `sparse` (默认): chunks_summary 走 BM25/BGE-M3 内存检索
- `dense`: chunks_summary 改走 dense 检索,直接用 Qdrant 已存的 BGE(summary) 向量 (绕过 BM25,失去 alias expansion 能力)

`chunks_cleaned_text` 没有这个开关,P4 代码里**没有 dense 路径**,那批 dense 向量**始终是死数据** (~960 KB in 234 points)。

**BM25 持久化边界**:

| 场景 | Dense 还在? | BM25 还在? |
|---|---|---|
| Qdrant 容器销毁/重启 | ✅ (snapshot / -v 恢复) | ✅ 不受影响 (本来就不在 Qdrant) |
| P3 进程退出 | ✅ | ❌ 进程内存丢了 |
| P4 新进程启动 | ✅ | ⚠️ `code_p4_searcher.py:_rebuild_bm25_index()` 自动从 Qdrant 拉文本重算 |
| Qdrant 容器重建 + P4 重启 | ✅ (从 snapshot 恢复) | ❌ 需先跑 P3 或等 P4 自动 rebuild |

**Q: 同一 chunk 为什么存 2 个 collection?**

| Collection | Dense 向量化的文本 | BM25 索引化的文本 | 召回特点 |
|---|---|---|---|
| `chunks_summary` | LLM 摘要 (精炼) | LLM 摘要 | 同义词友好,代码符号 / 路径 / 报错串可能丢 |
| `chunks_cleaned_text` | cleaned_text (原始) | cleaned_text | 字面值命中强,调试 / 错误消息 / 类名 / 库名必备 |

两条 sparse 路径在 P4 RRF 融合时**互补**,任意删一条都会掉召回 (尤其是查询包含报错字符串、库名、文件名时)。

### BM25 索引的不可变性 + 持久化优化方向

**观察**: 对一个固定的 corpus (chunks_summary / chunks_cleaned_text), 4 个 BM25 值在 build 之后就不再变:

| 值 | 来源 | 变不变 |
|---|---|---|
| TF (`doc_freqs`) | 单 doc 内的 token 计数 | doc 文本 + jieba 分词不变 → 不变 |
| DF | 跨 doc 数 token | corpus 集合不变 → 不变 |
| IDF | DF + N → 公式 | DF + N 不变 → 不变 |
| `avgdl` | `sum(doc_len) / N` | corpus 集合不变 → 不变 |

`rank_bm25.BM25Okapi` 在 build 时一次性算好塞进实例属性,运行时只读不重算 — 这本身就是一种"内存 cache"。

**问题**: 内存 cache 进程死就没了。P4 新进程必须 `_rebuild_bm25_index()` 走一遍 jieba 切词 + 重建倒排表,在 234 doc / ~50 token/doc 量级实测 ~1-2s。

**优化方向** (按实现代价递增):

| 级别 | 做法 | 落盘内容 | 体积 | load 时间 |
|---|---|---|---|---|
| 轻量 | pickle 整个 `BM25Okapi` 实例 | `output/bm25_cache/{c}.pkl` | ~30 KB | ~10 ms |
| 中等 | pickle `tokenized_corpus` + 元数据 | `output/bm25_cache/{c}_tokens.pkl` | ~12 KB | ~50 ms (现算 BM25) |
| 完整 | 上面 + corpus hash 校验,corpus 变了自动 invalidate | + `output/bm25_cache/{c}.sha256` | + 32 B | 多 ~50-200 ms hash |

**关键约束: corpus hash 校验**

严格说语料不是 immutable 的:

- 重跑 P2 → `chunks_summary` collection 内容变 (LLM 摘要可能改写),`doc_freqs` 全错位 → IDF 全错
- 重跑 P1 → `chunks.jsonl` 变 → `chunks_cleaned_text` 变 → 同上
- 手动 `delete + recreate` collection → 同上

**所以持久化的 cache 必须带 corpus fingerprint** (e.g.):

- 简单版: `sha256(concat(point_id + payload.summary))` (chunks_summary)
- 严格版: Qdrant collection 的 `points_count` + 排序后 point_id 列表的 sha256
- 校验流程: load cache 时重算 hash → 与缓存的 hash 比对 → 不一致就 rebuild

**当前代码现状**: `Phase3Store` **不落盘 BM25 cache**,每次 P4 cold start 都重算。如果要加,改动点:

- `code_p3_qdrant_store.py:_save_bm25_cache()` (build 末尾调)
- `code_p3_qdrant_store.py:_load_bm25_cache(corpus_hash)` (init 时调)
- `code_p4_searcher.py:_rebuild_bm25_index()` (fallback, corpus hash 不匹配时调)

**性能 vs 正确性 trade-off**: 不做 hash 校验 → corpus 变了 cache 还在 → 检索结果悄悄错位;做了 hash 校验 → 多 ~50-200 ms hash 计算,换来正确性。

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
