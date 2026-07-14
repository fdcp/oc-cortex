## Phase 3 运行指南

Phase 3 将 Phase 1 (chunks) 和 Phase 2 (tasks + chunk summaries) 的输出向量化，写入 Qdrant 双集合，并提供稠密 + 稀疏混合检索。

### 前置条件

需要先完成 Phase 1 和 Phase 2，确保以下文件存在：

```
output/tasks.jsonl              # Phase 2 输出: task 列表
output/chunks.jsonl             # Phase 1 输出: chunk 列表
output/chunks_summary_p2.jsonl  # Phase 2 输出: chunk summary
```

### 安装依赖

```bash
pip install qdrant-client sentence-transformers jieba rank_bm25 loguru pyyaml
```

BGE-M3 模式额外需要 transformers（通常随 sentence-transformers 自动安装）。

### 模型加载 (离线模式)

默认使用**离线模式**：仅从本地缓存加载模型，不联网检查更新。模型缓存在 `~/.cache/huggingface/hub/`。

首次运行需要下载模型，国内环境设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果本地没有模型，代码会自动切换到在线模式下载。也可以通过配置文件指定自定义缓存路径（见下方配置）。

---

### 模式 A: 开发环境 (Apple M2 CPU)

模型组合：`bge-small-zh-v1.5` (dense, 512 维) + `BM25 + jieba` (sparse)

#### 1. 配置文件 code_p3_config.yaml

```yaml
embedding:
  model: "BAAI/bge-small-zh-v1.5"
  dim: 512
  batch_size: 32
  device: "cpu"
  offline_mode: true      # 离线模式，仅使用本地缓存
  cache_folder: null      # null 使用默认值 ~/.cache/huggingface/hub/

sparse:
  method: "bm25"
  jieba_mode: "search"
  bm25_params:
    k1: 1.5
    b: 0.75
  fuse_k: 60

qdrant:
  path: "./qdrant_data"
  collections:
    tasks: "tasks"
    chunks: "chunks"
```

#### 2. 写入数据

```bash
cd /path/to/oc_sess_graph
python3 code_p3_main.py --config code_p3_config.yaml
```

全量数据 (27 tasks + 86 chunks) 预计耗时 ~5 秒。

#### 3. 检索演示

```bash
# 使用内置示例查询
python3 code_p3_search_demo.py --mode all

# 自定义查询
python3 code_p3_search_demo.py --mode all --query "GPU对比分析"

# 调整返回数量
python3 code_p3_search_demo.py --mode all --query "优化器学习率" --top-k 10
```

输出会依次展示 Dense 检索、Sparse 检索 (BM25)、Hybrid 检索 (RRF 融合) 三组结果。

---

### 模式 B: 生产环境 (NVIDIA GPU)

模型组合：`Qwen3-Embedding-0.6B` (dense, 1024 维) + `BGE-M3 sparse vectors` (sparse)

#### 1. 配置文件 code_p3_config.yaml

```yaml
embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"
  dim: 1024
  batch_size: 16
  device: "cuda"
  offline_mode: true      # 离线模式，仅使用本地缓存
  cache_folder: null      # null 使用默认值 ~/.cache/huggingface/hub/

sparse:
  method: "bge_m3"
  bge_m3_model: "BAAI/bge-m3"
  fuse_k: 60

qdrant:
  path: "./qdrant_data"
  collections:
    tasks: "tasks"
    chunks: "chunks"
```

> **device 选项说明：** `auto` 自动检测, `cpu` 强制 CPU, `mps` Apple Silicon GPU (有挂起风险，暂不推荐), `cuda` NVIDIA GPU。

#### 2. 写入数据

```bash
cd /path/to/oc_sess_graph
python3 code_p3_main.py --config code_p3_config.yaml
```

GPU 环境 (如 A100) 全量数据预计 10-30 秒。CPU 环境需要数分钟到数十分钟。

#### 3. 检索演示

```bash
# 1/5 数据采样演示 (自动使用 Qwen3 + BGE-M3, 写入 ./qdrant_data_sample)
python3 code_p3_search_demo.py --mode sample

# 自定义查询
python3 code_p3_search_demo.py --mode sample --query "GPU对比"
```

> **注意：** `--mode sample` 会自动切换到 Qwen3 + BGE-M3 组合，采样 1/5 数据，使用独立的 `./qdrant_data_sample` 目录，不会影响全量数据。

CPU 模式下 Qwen3 对长文本 embedding 很慢（每条约 30-120 秒），17 个 chunk 约需 23 分钟。生产环境强烈建议 GPU。

---

### 两种模式配置对比

```yaml
# ==================== 模式 A: 开发环境 ====================
embedding:
  model: "BAAI/bge-small-zh-v1.5"    # 33M 参数, ~130MB
  dim: 512
  batch_size: 32
  device: "cpu"
  offline_mode: true
  cache_folder: null                  # 默认 ~/.cache/huggingface/hub/

sparse:
  method: "bm25"                      # 内存索引, 无需额外模型
  jieba_mode: "search"
  bm25_params:
    k1: 1.5
    b: 0.75
  fuse_k: 60

# ==================== 模式 B: 生产环境 ====================
embedding:
  model: "Qwen/Qwen3-Embedding-0.6B" # 600M 参数, ~1.2GB
  dim: 1024
  batch_size: 16
  device: "cuda"
  offline_mode: true
  cache_folder: null                  # 默认 ~/.cache/huggingface/hub/

sparse:
  method: "bge_m3"                    # BGE-M3 sparse vectors, ~1.3GB
  bge_m3_model: "BAAI/bge-m3"
  fuse_k: 60
```

### 两种模式性能对比

| | 模式 A (开发) | 模式 B (生产) |
|---|---|---|
| Dense 模型 | bge-small-zh-v1.5 (33M) | Qwen3-Embedding-0.6B (600M) |
| Dense 维度 | 512 | 1024 |
| Sparse 方法 | BM25 + jieba (内存索引) | BGE-M3 sparse vectors (Qdrant) |
| 模型下载量 | ~130MB | ~2.5GB (Qwen3 + BGE-M3) |
| 全量数据写入 | ~5s (CPU) | ~10-30s (GPU) |
| 检索延迟 | <20ms | <200ms |
| 适用数据量 | <10K | 不限 (Qdrant 索引) |

### 配置项说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding.offline_mode` | `true` | 离线模式，仅使用本地缓存，不联网检查更新 |
| `embedding.cache_folder` | `null` | 模型缓存目录，`null` 使用 `~/.cache/huggingface/hub/` |
| `embedding.device` | `cpu` | `auto` 自动检测, `cpu` / `mps` / `cuda` |

### 已知问题

- **jieba + setuptools 83**：setuptools 80+ 移除了 `pkg_resources.resource_stream`，jieba 加载字典会报错。代码已内置 patch，无需手动处理。
- **BGE-M3 sparse 近似**：sentence-transformers 版本的 BGE-M3 不支持 `sparse_vec` 输出，代码使用 token embedding L2 norm 作为近似。如需精确的 learned sparse vectors，可使用 FlagEmbedding 原版的 `BGEM3FlagModel`。
- **MPS 设备**：Apple Silicon 的 MPS 后端有 embedding 挂起风险，建议暂用 `cpu`。
