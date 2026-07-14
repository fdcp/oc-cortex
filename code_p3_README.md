# Phase 3: Qdrant 双集合向量存储 + 混合检索

## 概述

Phase 3 将 Phase 1 (chunks) 和 Phase 2 (tasks + chunk summaries) 的输出向量化，写入本地 Qdrant 双集合，并提供**稠密检索 + 稀疏检索**的混合检索能力。

两个集合各有侧重：`tasks` 集合存储 task_summary 的向量，适合按"做了什么"检索；`chunks` 集合存储 summary + cleaned_text 的向量，适合按"具体细节"检索。

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
code_p3_config.yaml        # 配置 (dense/sparse 模型、Qdrant 路径)
code_p3_qdrant_store.py    # 核心模块: Phase3Store 类 (存储 + 检索)
code_p3_main.py            # 写入入口: 加载数据 -> embedding -> upsert
code_p3_search_demo.py     # 检索演示: dense / sparse / hybrid 对比
```

## 运行

### 写入数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
python code_p3_main.py
python code_p3_main.py --config code_p3_config.yaml
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
| `qdrant.collections.chunks` | `chunks` | chunk 集合名 |

## 核心设计

### 双集合架构

**tasks 集合** -- 每个 point 代表一个 session 级任务：
- 向量来源: `task_summary` (50-200 字, Phase 2 LLM 生成)
- Payload: `task_id`, `session_id`, `task_label`, `task_summary`, `chunk_ids`, `created_at`

**chunks 集合** -- 每个 point 代表一个对话轮次：
- 向量来源: `summary + "\n\n" + cleaned_text`
- Payload: `chunk_id`, `session_id`, `turn_index`, `task_id`, `summary`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`

### 混合检索流程

```
用户查询
  ├─ Dense:  embedding → Qdrant cosine search → top-K dense 结果
  ├─ Sparse: jieba 分词 → BM25 打分 → top-K sparse 结果
  │          (或 BGE-M3 sparse vector → Qdrant sparse search)
  └─ RRF:    Reciprocal Rank Fusion 融合两路结果 → 最终 top-K
```

**RRF 融合公式:**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中 k 默认 60，rank 从 1 开始。两路都有排名的文档会获得更高融合分数。

### BM25 模式 (开发)

- 使用 jieba 中文分词构建 BM25 索引 (in-memory, rank_bm25 库)
- upsert 时同步构建 BM25 索引，Qdrant 只存 dense vectors
- 检索时分别从 Qdrant (dense) 和内存 (BM25) 获取结果，Python 层面做 RRF 融合
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

`upsert_chunks()` 接受 `tasks` 参数，构建 `chunk_id → task_id` 反向映射，写入 chunk payload。

## 性能参考 (Apple M2 CPU)

### bge-small-zh-v1.5 + BM25 (全量数据)

| 操作 | 数量 | 耗时 |
|------|------|------|
| Dense embedding (tasks) | 27 | ~1.2s |
| Dense embedding (chunks) | 86 | ~2.6s |
| BM25 索引 (tasks) | 27 | ~0.3s |
| BM25 索引 (chunks) | 86 | ~0.2s |
| 单次混合检索 | - | <0.02s |

### Qwen3-Embedding-0.6B + BGE-M3 (1/5 数据示例)

| 操作 | 数量 | 耗时 |
|------|------|------|
| Dense embedding (tasks) | 5 | ~11s |
| Sparse embedding (tasks, BGE-M3) | 5 | ~4s |
| Dense embedding (chunks) | 17 | ~1415s (~23.5min) |
| Sparse embedding (chunks, BGE-M3) | 17 | ~49s |
| 单次混合检索 | - | ~0.2s |

CPU 模式下 Qwen3 对长文本 chunk embedding 很慢（每条约 30-120s）。生产环境建议使用 GPU。BGE-M3 sparse 计算（token embedding L2 norm）相对快得多。

## 检索结果示例

### BM25 模式 (全量数据, bge-small-zh)

查询: "GPU对比分析 A800 H100 H800"

```
Dense #1:  [ses_...c2]  系统对比 A800/A100/H100/H200/H800 ...  (score=0.762)
Sparse #1: [ses_...c2]  系统对比 A800/A100/H100/H200/H800 ...  (score=26.678)
Hybrid #1: [ses_...c2]  D#1 S#1  RRF=0.0328  ← 两路都是 #1, 融合后仍然是 #1
```

### BGE-M3 模式 (1/5 数据, Qwen3 + BGE-M3 sparse)

查询: "GPU对比"

```
Dense #1:  [ses_...c10] FA-2 相对 FA-1 的关键改进 ...                  (score=0.587)
Sparse #1: [ses_...c1]  V100 算力参数明确: FP32 14 TFLOPS ...          (score=1561.3)
Hybrid #1: [ses_...c2]  系统对比 A800/A100/H100/H200/H800 ...  D#2 S#2 RRF=0.0323
```

Dense 和 Sparse 给出了不同的 top-1（语义相似度 vs token 匹配），RRF 融合后系统对比 GPU 的 chunk 升到 #1（两路都排名靠前）。
