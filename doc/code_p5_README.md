# Phase 5: 知识图谱构建

## 概述

Phase 5 从 Phase 2 产出的 task summaries 中构建知识图谱，支持两种抽取模式：

### triple 模式（三元组抽取）

1. **三元组抽取** — LLM 从每个 task summary 中抽取 `(head, relation, tail)` 三元组
2. **实体对齐** — 将实体 embed 后写入 Qdrant `entities` 集合，检索相似度 > 0.92 的候选对，LLM 二次确认是否合并
3. **图谱构建** — 用 NetworkX MultiDiGraph 组装节点和边，持久化为 gpickle + JSON

### entity 模式（直接实体提取 + 倒排索引）

1. **实体提取** — LLM 从每个 task summary 中直接提取 5-10 个关键实体（prompt 更简单，成功率更高）
2. **实体对齐** — 同上（Qdrant embedding + LLM 确认 + UnionFind）
3. **共现图谱** — 同一 task 内的实体两两连边，构建 co-occurrence 图谱
4. **倒排索引** — 输出 entity → [task_ids] 映射，支持跨 session 关联查询

**核心设计原则：实体必须是跨 session 可复用的概念（项目/模块/文件/工具/技术），排除属性值和一次性术语。**

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

### 输出文件（按模式隔离）

```
output/
├── triple/                        # triple 模式输出
│   ├── triples.jsonl              #   三元组 (增量写入, 支持断点续传)
│   ├── entities.jsonl             #   对齐后的实体列表
│   ├── knowledge_graph.gpickle    #   NetworkX 图 (pickle)
│   ├── knowledge_graph.json       #   图数据 (JSON)
│   └── knowledge_graph.html       #   pyvis 交互式可视化
└── entity/                        # entity 模式输出
    ├── entity_extract.jsonl       #   每 task 实体提取结果 (增量写入)
    ├── entities.jsonl             #   对齐后的实体列表
    ├── inverted_index.json        #   倒排索引 (canonical → [task_ids])
    ├── knowledge_graph.gpickle    #   共现图谱 (pickle)
    ├── knowledge_graph.json       #   共现图谱 (JSON)
    └── knowledge_graph.html       #   pyvis 交互式可视化
```

两种模式的输出完全隔离，互不干扰，可分别用于后续搜索或总结。

## 运行

### 前置条件

需要 Phase 2 的输出文件：`output/tasks.jsonl`

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
```

### 基本用法

```bash
# 完整流程 (根据配置文件的 extraction_mode 自动选择模式)
python code_p5_main.py --config code_p5_config.yaml --visualize

# 快速验证 (3 个 task, 跳过对齐)
python code_p5_main.py --config code_p5_config.yaml --limit 3 --skip-alignment

# 跳过抽取 (加载已有结果, 只重建图)
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
| `--skip-extraction` | false | 跳过抽取, 加载已有文件 |
| `--visualize` | false | 构建完成后自动生成 HTML |

### 切换模式

修改 `code_p5_config.yaml` 中的 `extraction_mode` 即可：

```yaml
knowledge_graph:
  extraction_mode: "entity"   # "triple" | "entity"
```

- `triple` — 抽取三元组，构建关系图谱（边 = head→relation→tail）
- `entity` — 直接抽取实体，构建共现图谱（边 = 同 task 内两两共现）+ 倒排索引

### 单独运行可视化

```bash
# 从 gpickle 加载 (默认)
python code_p5_visualize.py

# 从 JSON 加载
python code_p5_visualize.py --input ./output/entity/knowledge_graph.json --format json

# 自定义参数
python code_p5_visualize.py --max-nodes 200 --drop-isolated --physics barnesHut
```

## 架构说明

### triple 模式流程

```
tasks.jsonl → 逐 task 调用 LLM (TRIPLE_EXTRACTION_PROMPT)
                              ↓
                     解析 JSON → 增量写入 triples.jsonl
                              ↓
                     collect_entities() → entity_map
                              ↓
                     align_entities() → canonical_map
                              ↓
                     build_graph() → MultiDiGraph
```

- 每个 task 一次 LLM 调用，prompt 引导抽取 3-8 个三元组
- 支持并发 (ThreadPoolExecutor)，默认 4 线程
- 增量写入 JSONL：每个 task 完成后立即追加，中断后可断点续传
- Prompt 严格要求：实体必须是跨 session 可复用概念，关系用精确动词

### entity 模式流程

```
tasks.jsonl → 逐 task 调用 LLM (ENTITY_EXTRACTION_PROMPT)
                              ↓
                     解析 JSON {"entities": [...]} → 增量写入 entity_extract.jsonl
                              ↓
                     构建倒排索引 {entity: [task_ids]}
                              ↓
                     collect_entities_from_index() → entity_map
                              ↓
                     align_entities() → canonical_map
                              ↓
                     build_cooccurrence_graph() → MultiDiGraph (共现边)
                              ↓
                     save_inverted_index() → inverted_index.json
```

- Prompt 更简单：只要求输出 5-10 个实体名，无需关系
- LLM 成功率更高（输出格式简单，解析稳定）
- 共现边：同一 task 内的实体两两配对（限 n≤15 避免 C(n,2) 爆炸）
- 倒排索引：按 canonical name 聚合，包含 aliases 和 task_ids

### 实体对齐流程（两种模式共用）

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

> **实测结论**：nemotron-3-ultra-free 批量模式 JSON 数组输出不稳定，降级补充后反而更慢。单条并发模式质量一致且更快，推荐作为默认。

- Qdrant `entities` 集合复用 Phase 3 存储路径
- Union-Find 保证传递性：A↔B 合并 + B↔C 合并 → A/B/C 全部归一
- canonical name 策略：取最短实体名

### 图谱对比

| 特征 | triple 模式 | entity 模式 |
|------|------------|------------|
| 边含义 | 语义关系 (使用/依赖/修复) | 共现关系 (同 task 出现) |
| 边数量 | ≈ 三元组数 | ≈ C(n,2) × task 数 (较多) |
| 额外产出 | — | inverted_index.json |
| 适用场景 | 关系推理、因果链分析 | 跨 session 关联查询、实体聚类 |

### 可视化

- pyvis (基于 vis.js) 生成交互式 HTML
- 节点颜色按 entity_type 映射 (concept=绿, module=青, tool=桃, bug=黄, file=紫, project=蓝)
- 节点大小按 degree 缩放
- forceAtlas2Based 物理引擎自动布局
- 自定义 tooltip：HTML 渲染（粗体标题 + 换行），跟随鼠标移动
- 满屏显示 (100vh × 100%)

## 配置说明

`code_p5_config.yaml` 中的关键配置项：

| 配置项 | 说明 |
|--------|------|
| `knowledge_graph.extraction_mode` | 抽取模式: `triple` \| `entity` |
| `knowledge_graph.triple_output_dir` | triple 模式输出目录 (默认 `./output/triple`) |
| `knowledge_graph.entity_output_dir` | entity 模式输出目录 (默认 `./output/entity`) |
| `knowledge_graph.entity_alignment_threshold` | 对齐相似度阈值 (默认 0.92) |
| `llm.model` | LLM 模型名 (默认 nemotron-3-ultra-free) |
| `llm.concurrency` | 并发线程数 (默认 4) |
| `llm.max_tokens` | 三元组抽取 max_tokens (默认 4000) |
| `llm.merge_batch_enable` | 是否启用批量实体合并确认 (默认 false) |
| `llm.merge_batch_size` | 批量模式每批发送的实体对数 (默认 20) |
| `embedding.model` | embedding 模型 (复用 Phase 3 的 bge-small-zh-v1.5) |
| `qdrant.entities_collection` | 实体集合名 (默认 entities) |
| `visualization.max_nodes` | 可视化最大节点数 (默认 500) |

## 模型对比与优化

### 问题背景

初版使用 `deepseek-v4-flash-free` 跑全量 29 个 task，成功率仅 24%（7/29）。主要失败原因：模型将 JSON 输出放入 `reasoning_content` 而非 `content`，导致 `content` 为空、解析失败。

### 三项优化

1. **Prompt 强化** — 明确要求"必须直接在 content 中输出 JSON，不要把 JSON 放在 reasoning/思考过程中"。

2. **reasoning_content 回退解析** — 当 `content` 为空时，从 `reasoning_content` 中用 `_extract_json_from_response` 提取 JSON。

3. **模型选型** — 对比三个模型后选择 `nemotron-3-ultra-free`。

### 10 task 采样对比结果

| 模型 | 成功率 | 三元组数 | content 为空次数 | reasoning 次数 | 总耗时 |
|------|--------|----------|-----------------|---------------|--------|
| deepseek-v4-flash-free | 90% (9/10) | 50 | 3 | 10 | 159.5s |
| **nemotron-3-ultra-free** | **100% (10/10)** | **84** | **0** | **0** | **47.6s** |
| mimo-v2.5-free | 100% (10/10) | 54 | 0 | 0 | 143.0s |

### 两种模式 10 task 对比 (nemotron-3-ultra-free)

| 指标 | entity 模式 | triple 模式 |
|------|------------|------------|
| 抽取耗时 | 17.8s | 48.9s |
| 对齐耗时 | 57.3s | 102.6s |
| 总耗时 | **75.2s** | 151.5s |
| 实体数 | 88 → 85 | 108 → 105 |
| 合并对 | 7 | 14 |
| 图谱节点 | 85 | 105 |
| 图谱边 | 378 (共现) | 86 (关系) |

entity 模式总耗时快 2 倍，prompt 更简单成功率更高，适合快速构建跨 session 关联索引。triple 模式提供语义关系，适合关系推理和因果链分析。

### 实体对齐模式对比 (nemotron-3-ultra-free, 303 实体, 16 对候选)

| 方案 | 对齐耗时 | 合并对 | 唯一实体 | 说明 |
|------|---------|--------|---------|------|
| 原始单条确认 | 92.3s | 10 | 296 | 无并发优化 |
| 批量三级策略 | 183.0s | 10 | 296 | batch 仅 1/16 成功, 降级补充 |
| **单条+并发 (当前默认)** | **54.2s** | **10** | **296** | concurrency=4, 快 1.7x / 3.4x |

### 模型对比测试脚本

```bash
python code_p5_benchmark.py                   # 默认对比三个模型, 10 个 task
python code_p5_benchmark.py --limit 5         # 自定义 task 数量
python code_p5_benchmark.py --concurrency 8   # 调整并发数
```

结果输出到 `output/benchmark_results.json`。
