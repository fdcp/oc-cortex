# Phase 5: 知识图谱构建

## 概述

Phase 5 从 Phase 2 产出的 task summaries 中构建知识图谱，包含三个核心步骤：

1. **三元组抽取** — LLM 从每个 task summary 中抽取 `(head, relation, tail)` 三元组
2. **实体对齐** — 将实体 embed 后写入 Qdrant `entities` 集合，检索相似度 > 0.92 的候选对，LLM 二次确认是否合并
3. **图谱构建** — 用 NetworkX MultiDiGraph 组装节点和边，持久化为 gpickle + JSON

最终可生成 pyvis 交互式 HTML 可视化，在浏览器中浏览知识网络。

**核心设计原则：实体必须是跨 session 可复用的概念（项目/模块/文件/工具/技术），排除属性值和一次性术语。关系用精确动词，禁止"是"、"包含"等笼统关系。**

## 依赖

```bash
pip install networkx pyvis numpy
```

已有依赖（Phase 2/3 共用）：`openai`、`sentence-transformers`、`qdrant-client`、`loguru`

## 文件结构

```
code_p5_config.yaml        # 配置 (LLM/embedding/Qdrant/KG/可视化)
code_p5_models.py          # 数据模型: Triple, Entity, KGStats
code_p5_kg_builder.py      # 核心模块: KGBuilder 类 (抽取 + 对齐 + 建图)
code_p5_main.py            # 主入口: 串联全流程
code_p5_visualize.py       # pyvis 可视化: 生成 HTML
code_p5_benchmark.py       # 模型对比测试: 多模型成功率/质量/耗时对比
```

### 输出文件

```
output/triples_p5.jsonl             # 三元组 (增量写入, 支持断点续传)
output/entities_p5.jsonl            # 对齐后的实体列表
output/knowledge_graph.gpickle      # NetworkX 图 (pickle)
output/knowledge_graph.json         # 图数据 (JSON, 供可视化)
output/knowledge_graph.html         # pyvis 交互式可视化
```

## 运行

### 前置条件

需要 Phase 2 的输出文件：`output/tasks.jsonl`

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
```

### 基本用法

```bash
# 完整流程 (三元组抽取 + 实体对齐 + 建图 + 可视化)
python code_p5_main.py --config code_p5_config.yaml --visualize

# 快速验证 (3 个 task, 跳过对齐)
python code_p5_main.py --config code_p5_config.yaml --limit 3 --skip-alignment

# 跳过抽取 (加载已有三元组, 只重建图)
python code_p5_main.py --config code_p5_config.yaml --skip-extraction --visualize

# 调整并发数
python code_p5_main.py --config code_p5_config.yaml --concurrency 8 --visualize
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `code_p5_config.yaml` | 配置文件路径 |
| `--concurrency` | 配置文件中指定 | 并发线程数 |
| `--limit N` | 全部 | 只处理前 N 个 task |
| `--skip-alignment` | false | 跳过实体对齐 |
| `--skip-extraction` | false | 跳过三元组抽取, 加载已有文件 |
| `--visualize` | false | 构建完成后自动生成 HTML |

### 单独运行可视化

```bash
# 从 gpickle 加载
python code_p5_visualize.py

# 从 JSON 加载
python code_p5_visualize.py --input ./output/knowledge_graph.json --format json

# 自定义参数
python code_p5_visualize.py --max-nodes 200 --drop-isolated --physics barnesHut
```

## 架构说明

### 三元组抽取流程

```
tasks.jsonl → 逐 task 调用 LLM → 解析 JSON → 增量写入 triples_p5.jsonl
                                          ↑
                                  content_retries 重试
                              (应对 JSON 解析失败的情况)
```

- 每个 task 一次 LLM 调用，prompt 引导抽取 3-8 个三元组
- 支持并发 (ThreadPoolExecutor)，默认 4 线程
- 增量写入 JSONL：每个 task 完成后立即追加，中断后可断点续传
- Prompt 严格要求：实体必须是跨 session 可复用概念，关系用精确动词

### 实体对齐流程

```
收集所有实体 → embed → 写入 Qdrant entities 集合
                              ↓
                    逐实体检索 Top-5 相似实体 (threshold=0.92)
                              ↓
                    LLM 二次确认 (MERGE / KEEP)
                              ↓
                    Union-Find 传递归一化 → canonical name map
```

LLM 确认支持两种模式，通过 `llm.merge_batch_enable` 控制：

- **单条模式**（默认，`merge_batch_enable: false`）— 每对实体独立 LLM 调用，ThreadPoolExecutor 并发，结果稳定
- **批量模式**（`merge_batch_enable: true`）— 每批发送多对，减少 API 调用次数，但依赖模型稳定输出 JSON 数组

> **实测结论**：nemotron-3-ultra-free 批量模式 JSON 数组输出不稳定（16 对仅匹配 1/16），降级补充后反而更慢。单条并发模式质量一致且更快，推荐作为默认。

- Qdrant `entities` 集合复用 Phase 3 存储路径
- Union-Find 保证传递性：A↔B 合并 + B↔C 合并 → A/B/C 全部归一
- canonical name 策略：取最短实体名

### 图谱构建

- NetworkX `MultiDiGraph`：允许多条边 (同一对节点间可有不同关系)
- 节点属性：`entity_type`、`aliases`、`source_tasks`
- 边属性：`relation`、`weight`(置信度)、`source_task`

### 可视化

- pyvis (基于 vis.js) 生成交互式 HTML
- 节点颜色按 entity_type 映射 (concept=绿, module=青, tool=桃, bug=黄, file=紫)
- 节点大小按 degree 缩放
- forceAtlas2Based 物理引擎自动布局

## 配置说明

`code_p5_config.yaml` 中的关键配置项：

| 配置项 | 说明 |
|--------|------|
| `llm.model` | LLM 模型名 (默认 nemotron-3-ultra-free) |
| `llm.concurrency` | 并发线程数 (默认 4) |
| `llm.max_tokens` | 三元组抽取 max_tokens (默认 4000) |
| `llm.merge_batch_enable` | 是否启用批量实体合并确认 (默认 false) |
| `llm.merge_batch_size` | 批量模式每批发送的实体对数 (默认 20) |
| `embedding.model` | embedding 模型 (复用 Phase 3 的 bge-small-zh-v1.5) |
| `qdrant.entities_collection` | 实体集合名 (默认 entities) |
| `knowledge_graph.entity_alignment_threshold` | 对齐相似度阈值 (默认 0.92) |
| `visualization.max_nodes` | 可视化最大节点数 (默认 500) |

## 模型对比与优化

### 问题背景

初版使用 `deepseek-v4-flash-free` 跑全量 29 个 task，成功率仅 24%（7/29）。主要失败原因：模型将 JSON 输出放入 `reasoning_content` 而非 `content`，导致 `content` 为空、解析失败。

### 三项优化

1. **Prompt 强化** — 在 `TRIPLE_EXTRACTION_PROMPT` 中明确要求"必须直接在 content 中输出 JSON，不要把 JSON 放在 reasoning/思考过程中"，引导模型正确分离推理和输出。

2. **reasoning_content 回退解析** — `_call_llm` 中当 `content` 为空时，尝试从 `reasoning_content` 中用 `_extract_json_from_response` 提取 JSON，而不是直接丢弃。

3. **模型选型** — 对比 `deepseek-v4-flash-free`、`nemotron-3-ultra-free`、`mimo-v2.5-free` 三个模型。

### 10 task 采样对比结果

| 模型 | 成功率 | 三元组数 | content 为空次数 | reasoning 次数 | 总耗时 |
|------|--------|----------|-----------------|---------------|--------|
| deepseek-v4-flash-free | 90% (9/10) | 50 | 3 | 10 | 159.5s |
| **nemotron-3-ultra-free** | **100% (10/10)** | **84** | **0** | **0** | **47.6s** |
| mimo-v2.5-free | 100% (10/10) | 54 | 0 | 0 | 143.0s |

`nemotron-3-ultra-free` 全面最优：100% 成功率、三元组数量最多、零 content 空、速度最快（比 deepseek 快 3.3 倍）。

> **注**：Prompt 和解析优化后，deepseek 的成功率也从 24% 提升到 90%（10 task 采样），其中 3 次 content 为空均通过 reasoning_content JSON 提取成功恢复。

### 全量运行结果 (nemotron-3-ultra-free, 29 tasks)

切换模型后全量运行，结果大幅改善：

| 指标 | 优化前 (deepseek) | 优化后 (nemotron) |
|------|-------------------|-------------------|
| 成功率 | 24% (7/29) | **100% (29/29)** |
| 三元组总数 | 26 | **241** |
| 原始实体数 | — | 303 |
| 对齐后实体数 | — | 296 (合并 10 对) |
| 图谱节点数 | — | 296 |
| 图谱边数 | — | 241 |
| 总耗时 | — | 166.3s |

**实体对齐合并示例**：OpenCode/opencode、Git 操作/Git操作、PCIE形态/PCIe形态、BF16 训练指南/BF16 训练指南 v3/v4 等。

### 实体对齐模式对比 (nemotron-3-ultra-free, 303 实体, 16 对候选)

| 方案 | 对齐耗时 | 合并对 | 唯一实体 | 说明 |
|------|---------|--------|---------|------|
| 原始单条确认 | 92.3s | 10 | 296 | 无并发优化 |
| 批量三级策略 | 183.0s | 10 | 296 | batch 仅 1/16 成功, 降级补充 |
| **单条+并发 (当前默认)** | **54.2s** | **10** | **296** | concurrency=4, 快 1.7x / 3.4x |

三种方案质量完全一致（296 唯一实体 / 10 对合并），单条并发模式在对齐阶段分别比原始单条和批量三级策略快 1.7 倍和 3.4 倍。

**Top-10 高度节点**（跨 session 连通性体现）：

| 实体 | 类型 | 度 |
|------|------|----|
| 优化器综合指南 | concept | 14 |
| OpenCode | concept | 13 |
| BF16 训练指南 | concept | 10 |
| oh-my-openagent | concept | 8 |
| DataCollatorForSeq2Seq | concept | 8 |
| Ulysses | concept | 7 |
| OpenCode Zen平台 | concept | 6 |
| SP_mm.md | file | 6 |
| 大模型分布式训练并行 | concept | 6 |
| NVIDIA | concept | 6 |

### 模型对比测试脚本

如需自行对比其他模型，使用 `code_p5_benchmark.py`：

```bash
# 默认对比三个模型, 10 个 task
python code_p5_benchmark.py

# 自定义 task 数量
python code_p5_benchmark.py --limit 5

# 指定模型
python code_p5_benchmark.py --models deepseek-v4-flash-free,nemotron-3-ultra-free

# 调整并发数
python code_p5_benchmark.py --concurrency 8
```

结果输出到 `output/benchmark_results.json`，包含每个 task 的成功率、三元组数、content 是否为空、是否检测到 reasoning 等详细诊断信息。
