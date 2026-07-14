# Phase 3: Qdrant 双集合向量存储

## 概述

Phase 3 将 Phase 1 (chunks) 和 Phase 2 (tasks + chunk summaries) 的输出向量化，写入本地 Qdrant 双集合，为后续的语义检索和图谱构建提供基础。

两个集合各有侧重：`tasks` 集合存储 task_summary 的向量，适合按"做了什么"检索；`chunks` 集合存储 summary + cleaned_text 的向量，适合按"具体细节"检索。

## 依赖

```bash
pip install qdrant-client sentence-transformers
```

首次运行会自动从 HuggingFace 下载 embedding 模型。国内环境建议设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 文件结构

```
code_p3_config.yaml      # 配置 (模型/维度/设备/Qdrant路径)
code_p3_qdrant_store.py  # 核心模块: Phase3Store 类
code_p3_main.py          # 主入口: 加载数据 -> embedding -> upsert
```

## 运行

```bash
# 默认配置
export HF_ENDPOINT=https://hf-mirror.com
python code_p3_main.py

# 自定义输入路径
python code_p3_main.py --tasks ./output/tasks.jsonl --chunks ./output/chunks.jsonl --summaries ./output/chunks_summary_p2.jsonl

# 自定义配置
python code_p3_main.py --config code_p3_config.yaml
```

## 配置说明 (code_p3_config.yaml)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding.model` | `Qwen/Qwen3-Embedding-0.6B` | 本地 embedding 模型 (sentence-transformers 加载) |
| `embedding.dim` | `1024` | 向量维度 (与模型匹配) |
| `embedding.batch_size` | `4` | 每批 embed 条数; CPU 模式建议 4, GPU 可调到 16-32 |
| `embedding.device` | `cpu` | `auto` / `cpu` / `mps` / `cuda`; MPS 有挂起风险暂用 cpu |
| `qdrant.path` | `./qdrant_data` | Qdrant 文件持久化路径 (无需 Docker) |
| `qdrant.collections.tasks` | `tasks` | task 集合名 |
| `qdrant.collections.chunks` | `chunks` | chunk 集合名 |

## 核心设计

### 双集合架构

**tasks 集合** -- 每个 point 代表一个 session 级任务：
- 向量来源: `task_summary` (50-200 字, Phase 2 LLM 生成)
- Payload: `task_id`, `session_id`, `task_label`, `task_summary`, `chunk_ids`, `created_at`

**chunks 集合** -- 每个 point 代表一个对话轮次：
- 向量来源: `summary + "\n\n" + cleaned_text` (summary 来自 Phase 2 CoT, cleaned_text 来自 Phase 1 清洗)
- Payload: `chunk_id`, `session_id`, `turn_index`, `task_id`, `summary`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`

### 确定性 UUID

使用 `_stable_uuid()` 从业务 ID (task_id / chunk_id) 的 MD5 生成 Qdrant point id。这保证了：
- 相同内容 upsert 到同一个 point (幂等)
- 重跑不会产生重复数据
- 无需先删后写

### 设备自动检测

`_detect_device("auto")` 按以下优先级选择设备：
1. macOS + Apple Silicon → MPS (但有挂起风险, 当前默认 cpu)
2. Linux/Windows + NVIDIA → CUDA
3. 其他 → CPU

### Chunk 文本构建

每个 chunk 的 embed 文本由两部分拼接：
1. Phase 2 生成的 `summary` (一句话总结该轮次做了什么)
2. Phase 1 生成的 `cleaned_text()` (结构化的 user/assistant/tool/mcp 内容)

summary 在前提供语义锚点，cleaned_text 在后补充细节。如果两者都为空则跳过该 chunk。

### Chunk-Task 关联

`upsert_chunks()` 接受可选的 `tasks` 参数，从中构建 `chunk_id → task_id` 的反向映射，写入 chunk payload 的 `task_id` 字段。这样从 chunk 可以追溯到它属于哪个 task。

## 性能参考

在 Apple Silicon (M 系列) CPU 模式下，Qwen3-Embedding-0.6B：

| 数据 | 数量 | batch_size | 耗时 |
|------|------|-----------|------|
| tasks (短文本 ~50 字) | 27 | 4 | ~91s |
| chunks (长文本 avg ~1900 字) | 86 | 4 | ~60-70min |

CPU 模式对长文本较慢。如果有 NVIDIA GPU，将 device 设为 `cuda` 可大幅提速。

## 输出验证

运行完成后，日志会打印两个集合的统计：

```
Phase 3 完成
  总耗时: XXXs
  输入 task: 27, upsert: 27
  输入 chunk: 86, upsert: 86
  chunk summary 匹配: 86
  集合 'tasks': vectors=27, points=27, status=green
  集合 'chunks': vectors=86, points=86, status=green
```

Qdrant 数据持久化在 `./qdrant_data/` 目录，后续 Phase 4 (语义检索) 直接读取。
