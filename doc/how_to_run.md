## Phase 3 运行指南

Phase 3 将 Phase 1 (chunks) 和 Phase 2 (tasks + chunk summaries) 的输出向量化，写入 Qdrant 三集合 (tasks / chunks_summary / chunks_cleaned_text)，Dense 用 summary 语义匹配，BM25 用 cleaned_text 关键词匹配。

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

---

## Phase 4 运行指南: 跨 Session 搜索

Phase 4 实现完整的跨 session 搜索流程：自然语言查询 → 混合检索粗排 → Qwen3-Reranker 精排 → 取回 chunk 原文。

### 前置条件

1. Phase 1-3 已完成，Qdrant 数据已写入：

```
qdrant_data/                       # Phase 3 输出: Qdrant 持久化数据
output/tasks.jsonl                  # Phase 2 输出
output/chunks.jsonl                 # Phase 1 输出
output/chunks_summary_p2.jsonl      # Phase 2 输出
```

2. Reranker 模型已缓存（首次需下载，约 1.2GB）：

```bash
# 国内环境设置镜像
export HF_ENDPOINT=https://hf-mirror.com
python3 -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-Reranker-0.6B --repo-type model
```

下载完成后模型缓存在 `~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/`，之后无需联网。

### 安装依赖

```bash
pip install qdrant-client sentence-transformers transformers jieba rank_bm25 loguru pyyaml
```

### Reranker 配置 (code_p3_config.yaml)

```yaml
reranker:
  model: "Qwen/Qwen3-Reranker-0.6B"
  device: "cpu"            # auto / cpu / mps / cuda
  instruction: "Given a web search query, retrieve relevant passages."
  max_length: 8192
  batch_size: 4
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `reranker.model` | `Qwen/Qwen3-Reranker-0.6B` | Reranker 模型 (CausalLM 架构, ~1.2GB) |
| `reranker.device` | `cpu` | 推理设备，GPU 可大幅加速 |
| `reranker.instruction` | 见上 | 自定义任务指令 (可改为中文) |
| `reranker.max_length` | `8192` | 最大输入长度 (token) |
| `reranker.batch_size` | `4` | 推理 batch size |

### 运行命令

```bash
cd /path/to/oc_sess_graph

# 单次查询 (带 Reranker 精排)
python3 code_p4_search_cli.py --query "GPU对比分析"

# 跳过 Reranker (仅粗排, 毫秒级)
python3 code_p4_search_cli.py --query "GPU对比分析" --no-rerank

# 调整返回数量
python3 code_p4_search_cli.py --query "优化器学习率" --top-k 10

# 交互模式 (循环查询)
python3 code_p4_search_cli.py --interactive

# 指定配置文件
python3 code_p4_search_cli.py --config code_p3_config.yaml --query "session管理"
```

### 搜索流程

```
Query "GPU对比分析"
  │
  ├─ 1. Hybrid 粗排 (bge-small-zh dense + BM25 sparse + RRF 融合)
  │     → 27 tasks 中取 Top-15 候选 (top_k * 5)
  │
  ├─ 2. Reranker 精排 (Qwen3-Reranker-0.6B, CausalLM yes/no logits)
  │     → 15 候选 → Top-3 精排结果
  │
  └─ 3. Chunk 展开
        → 每个 task 关联的 chunk 原文 (summary + cleaned_text 预览)
```

### 输出格式

每条搜索结果包含：

- `rerank score` — Reranker 精排分数 (0~1, softmax)
- `hybrid score` — RRF 混合检索粗排分数
- `task_label` + `task_summary` — 任务标签和摘要
- `chunks` — 关联的对话轮次详情 (summary + cleaned_text 预览)

### 性能参考 (CPU: Apple M2)

| 阶段 | 耗时 |
|------|------|
| 初始化 (加载模型 + BM25 索引) | ~1s |
| 粗排 (Hybrid: dense + BM25 + RRF) | ~0.2s |
| Reranker (15 篇文档, batch=4) | ~135s |
| **总计 (带 Reranker)** | **~136s** |
| **总计 (skip Reranker)** | **~0.2s** |

> CPU 模式下 Reranker 较慢 (~9s/篇)，生产环境建议使用 GPU。Reranker 是可选的，`--no-rerank` 可跳过精排直接用粗排结果。

### 技术说明

- **Reranker 架构**: Qwen3-Reranker-0.6B 是 CausalLM (非 CrossEncoder)，通过计算 "yes"/"no" token 的 softmax 概率得到相关性分数
- **Prompt 格式**: 使用 chat template `<Instruct> + <Query> + <Document>` 格式，系统 prompt 要求模型回答 "yes" 或 "no"
- **离线模式**: 复用 Phase 3 的 `offline_mode` 配置，模型加载后不需要网络
- **数据预加载**: 启动时一次性将 tasks/chunks/summaries 读入内存，查询时直接查 map

---

## 里程碑检视工具

`code_MS_inspect.py` 是第一个里程碑的数据检视工具，替代 Qdrant Dashboard（项目用嵌入式 Qdrant，无 Web UI）。

### 前置条件

Phase 1-3 已完成，Qdrant 数据已写入 `./qdrant_data/`。

### 运行命令

```bash
cd /path/to/oc_sess_graph
# 默认读取 config/code_p3_config.yaml (集合名/路径/模型均从配置读取)
python3 src/code_MS_inspect.py

# 可指定配置或调整样本数, 相对路径会回退到仓库根目录, 可从任意 CWD 运行
python3 src/code_MS_inspect.py --config config/code_p3_config.yaml --samples 8
```

### 检视内容

脚本依次执行 6 项检查：

1. **集合统计** — 三个集合 (tasks / chunks_summary / chunks_cleaned_text) 的点数、维度、状态
2. **样本浏览** — 随机抽取 5 个 task 和 5 个 chunk 的 payload
3. **数据分析** — session/task/chunk 分布、token 压缩率、summary 覆盖率
4. **向量近邻分析** — 抽样 10 个 task，检查 cosine > 0.85 的近重复对
5. **检索质量测试** — 5 组预设查询的 Dense 检索命中率
6. **Chunk Summary 质量** — 长度分布、过短 summary 列表、task vs chunk summary 对比

### 调优笔记

`code_MS_tuning_notes.md` 记录了基于检视数据的 prompt 调优分析和改进建议，包括发现的问题、优先级和是否需要重跑的判断。
