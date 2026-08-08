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
code_p5_config.yaml        # 配置 (LLM/embedding/Qdrant/KG/可视化/P5e)
code_p5_models.py          # 数据模型: Triple, Entity, KGStats
code_p5_kg_builder.py      # 核心模块: KGBuilder 类 (抽取 + 对齐 + 建图, 含 post_init 模型路由)
code_p5_main.py            # 主入口: 串联全流程 (末尾自动导出 SQLite, P5e 集成)
code_p5_visualize.py       # pyvis 可视化: 生成 HTML
code_p5_benchmark.py       # 模型对比测试: 多模型成功率/质量/耗时对比

# Phase 5e 扩展 (SQLite 持久化 + Graph-RAG 搜索, 详见 doc/code_p5e.md)
code_p5e_db.py             # SQLite 持久化: KGDatabase (BFS / 模糊搜索 / 统计)
code_p5e_graph_rag.py      # Graph-RAG 搜索: 查询实体抽取 + 向量+图谱融合检索
```

### 输出文件（按模式隔离）

```
output/
├── triple/                        # triple 模式输出
│   ├── triples.jsonl              #   三元组 (增量写入, 支持断点续传)
│   ├── entities.jsonl             #   对齐后的实体列表
│   ├── knowledge_graph.gpickle    #   NetworkX 图 (pickle)
│   ├── knowledge_graph.json       #   图数据 (JSON)
│   ├── knowledge_graph.db         #   SQLite 持久化 (P5e 产物, sqlite.enabled=true 时生成)
│   └── knowledge_graph.html       #   pyvis 交互式可视化
└── entity/                        # entity 模式输出
    ├── entity_extract.jsonl       #   每 task 实体提取结果 (增量写入)
    ├── entities.jsonl             #   对齐后的实体列表
    ├── inverted_index.json        #   倒排索引 (canonical → [task_ids])
    ├── knowledge_graph.gpickle    #   共现图谱 (pickle)
    ├── knowledge_graph.json       #   共现图谱 (JSON)
    ├── knowledge_graph.db         #   SQLite 持久化 (P5e 产物, sqlite.enabled=true 时生成)
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

> 从仓库根目录运行(`code_p5_main.py` 在 `src/` 下,配置在 `config/` 下)。

```bash
# 完整流程 (根据配置文件的 extraction_mode 自动选择模式)
python3 src/code_p5_main.py --config config/code_p5_config.yaml --visualize

# CLI 临时覆盖抽取模式 (不改配置文件, 同时切换 llm.triple_model / llm.entity_model)
python3 src/code_p5_main.py --config config/code_p5_config.yaml --extraction_mode triple

# 快速验证 (3 个 task, 跳过对齐)
python3 src/code_p5_main.py --config config/code_p5_config.yaml --limit 3 --skip-alignment

# 跳过抽取 (加载已有结果, 只重建图)
python3 src/code_p5_main.py --config config/code_p5_config.yaml --skip-extraction --visualize

# 强制重抽 (删除已有 triples/entities 后从零开始, 与 --skip-extraction 互斥)
python3 src/code_p5_main.py --config config/code_p5_config.yaml --force

# 调整并发数
python3 src/code_p5_main.py --config config/code_p5_config.yaml --concurrency 8 --visualize
```

### 一键脚本 (run_phase5.sh)

仓库根目录的 `run_phase5.sh` 会**顺序跑 `triple` + `entity` 两种模式**（抽取 ×2 + 可视化 ×2 + SQLite 导出 ×2），避免手动来回切 `--extraction_mode`。每次执行都 `tee` 完整日志到 `logs/phase5_triple.log` / `logs/phase5_entity.log`。

```bash
# 断点续传 (默认; 已有 triples / entity_extract 文件会被复用)
bash run_phase5.sh

# 强制重抽 (透传 --force 给两次 python, 删除已有抽取结果)
bash run_phase5.sh --force

# 自定义配置 (通过环境变量)
P5_CONFIG=config/my_p5.yaml bash run_phase5.sh
```

| 名称 | 默认值 | 说明 |
|------|--------|------|
| `P5_CONFIG` | `config/code_p5_config.yaml` | 配置文件路径 |
| `--force` (位置参数) | — | 透传给两次 `python3 src/code_p5_main.py --force`，删除已有抽取结果 |

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/code_p5_config.yaml` | 配置文件路径 |
| `--concurrency` | 配置文件中指定 | 并发线程数 |
| `--limit N` | 全部 | 只处理前 N 个 task |
| `--skip-alignment` | false | 跳过实体对齐 (每个实体保持原名) |
| `--skip-extraction` | false | 跳过抽取, 加载已有 triples / entity_extract 文件 |
| `--force` | false | 强制重新抽取, 删除已有 triples / entity_extract 后从零开始, 与 `--skip-extraction` 互斥 |
| `--visualize` | false | 构建完成后自动运行 pyvis 可视化 |
| `--extraction_mode` | 配置文件中指定 | 覆盖 `knowledge_graph.extraction_mode` (`triple` \| `entity`), 同时决定使用 `llm.triple_model` 还是 `llm.entity_model` |

### 切换模式

修改 `config/code_p5_config.yaml` 中的 `extraction_mode` 即可（默认 `entity`，启动更快、跨 session 关联更密）：

```yaml
knowledge_graph:
  extraction_mode: "entity"   # "triple" | "entity"  (默认 entity)
```

- `triple` — 抽取三元组，构建关系图谱（边 = head→relation→tail）
- `entity` — 直接抽取实体，构建共现图谱（边 = 同 task 内两两共现）+ 倒排索引

如果只想临时切换、不改配置文件，可以用 CLI 覆盖：

```bash
python3 src/code_p5_main.py --config config/code_p5_config.yaml --extraction_mode triple
```

`--extraction_mode` 覆盖配置文件的同时，`KGBuilder.post_init()` 会按模式选择 `llm.triple_model` 或 `llm.entity_model`。

### 单独运行可视化

图谱已经构建完成（`knowledge_graph.gpickle` 或 `.json` 存在）后，可以**只重跑 HTML 这一步**，不再走抽取 / 对齐 / 建图全流程。常用于调整可视化参数、换物理引擎布局、HTML 丢失后重画。

> `--input` / `--output` 都不传时,会自动按 P5 产物目录约定探测(`output/triple/` → `output/entity/`, `.gpickle` 优先 / `.json` 兜底),HTML 写到与 `--input` 同目录,**不会污染 `output/` 根目录**。需要覆盖时再显式传。

```bash
# 不传参: 自动探测 (推荐; 跑完 P5 后直接用)
python3 src/code_p5_visualize.py

# 显式指定图谱 (HTML 默认会写到同目录)
python3 src/code_p5_visualize.py --input output/triple/knowledge_graph.gpickle
python3 src/code_p5_visualize.py --input output/entity/knowledge_graph.json --format json

# 调小节点数、丢孤立节点、换物理引擎
python3 src/code_p5_visualize.py --max-nodes 200 --drop-isolated --physics barnesHut
```

支持的物理引擎：`forceAtlas2Based`（默认）/ `barnesHut` / `repulsion`。

## Phase 5e 集成

P5 主流程已经默认接入 P5e：每次跑完 P5，只要 `sqlite.enabled: true`（默认），就会把图谱导出到 `{output_dir}/knowledge_graph.db`，并在日志末尾打印节点/边数与平均每实体关联 task 数。

P5e 提供两类能力，完整文档见 `doc/code_p5e.md`：

- **SQLite 持久化 (`src/code_p5e_db.py`)** — `KGDatabase` 类，支持 `import_graph` / `get_node` / `get_edges` / `bfs_expand` / `search_entities` / `get_top_entities` / `get_stats`，毫秒级查询
- **Graph-RAG 搜索 (`src/code_p5e_graph_rag.py`)** — 在 P4 向量检索之上叠加图谱扩散：查询 → LLM 实体抽取 → 模糊匹配图谱节点 → BFS 扩散 → 合并向量结果 → Reranker 排序

如果暂时不想写 SQLite，可以在 `config/code_p5_config.yaml` 里关掉：

```yaml
sqlite:
  enabled: false
```

> `graph_rag.*` 段是 P5e 搜索侧的配置，P5 主流程不直接消费，只会被 `GraphRAGSearcher` 读取。

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

`config/code_p5_config.yaml` 中的关键配置项（按段分组）：

### LLM

> P5 实际上跑 LLM 时有 **3 个独立模型槽**：`triple_model`（三元组抽取）/ `entity_model`（实体抽取）/ `merge_model`（实体合并确认）。`KGBuilder.post_init()` 会按 `extraction_mode` 自动选 `triple_model` 或 `entity_model`。`merge_model` 始终用于合并阶段，因此拆出来单独配。
>
> 旧字段 `llm.model` 仍保留为兜底：`triple_model` / `entity_model` / `merge_model` 任一未配时回退到 `model`；如果 `model` 也未配，则默认 `deepseek-v4-flash-free`。
>
> `api_key_env` / `base_url` 字段已从本配置移除，统一由 `opencode_models.yaml` 提供（见下）。

| 配置项 | 说明 |
|--------|------|
| `llm.triple_model` | 三元组抽取模型 (默认 `nemotron-3-ultra-free`) |
| `llm.entity_model` | 实体抽取模型 (默认 `hy3`) |
| `llm.merge_model` | 实体合并确认模型 (默认 `deepseek-v4-flash-free`，轻量模型；reasoning 模型在此场景下收益小、还可能因 max_tokens 不够返回空) |
| `llm.model` | 兜底模型，`triple_model` / `entity_model` / `merge_model` 任一未配时使用 |
| `llm.concurrency` | 并发线程数 (默认 4) |
| `llm.max_retries` | API 调用最大重试次数 (默认 3) |
| `llm.content_retries` | `content` 为空时的额外重试次数 (默认 2) |
| `llm.timeout` | API 超时秒数 (默认 120) |
| `llm.max_tokens` | 三元组抽取 max_tokens (默认 **6000**) |
| `llm.merge_max_tokens` | 实体合并确认 max_tokens (默认 **420**)；reasoning 模型（如 `hy3`）建议 ≥ 2000，否则 thinking 预算都不够 |
| `llm.merge_batch_enable` | 是否启用批量合并确认 (默认 `false`，推荐单条并发) |
| `llm.merge_batch_size` | 批量模式每批发送的实体对数 (默认 20) |

### OpenCode 模型池

> `api_key_env` / `base_url` 不再写在 P5 配置里。`KGBuilder.post_init()` 在 LLM 调用前会从 `opencode_models.config_path` 读 base_url，从 `~/.local/share/opencode/auth.json` 读 api_key。模型名相同的请求走同一条 base_url。

| 配置项 | 说明 |
|--------|------|
| `opencode_models.config_path` | 模型池配置路径 (默认 `config/opencode_models.yaml`，相对路径优先 CWD、再退到项目根) |

### Embedding

| 配置项 | 说明 |
|--------|------|
| `embedding.model` | embedding 模型 (默认 `BAAI/bge-small-zh-v1.5`，复用 Phase 3) |
| `embedding.dim` | 向量维度 (默认 512) |
| `embedding.batch_size` | 批量大小 (默认 32) |
| `embedding.device` | 计算设备 (默认 `cpu`) |
| `embedding.offline_mode` | HF 离线模式 (默认 `true`)，启动时由 `code_p3_hf_config.setup_hf_env()` 强制设置 |
| `embedding.cache_folder` | HF 缓存目录 (默认 `null`，走 `~/.cache/huggingface`) |

### Qdrant

| 配置项 | 说明 |
|--------|------|
| `qdrant.path` | 嵌入式 Qdrant 存储路径 (默认 `./qdrant_data`) |
| `qdrant.entities_collection` | 实体集合名 (默认 `entities`) |

### 知识图谱

| 配置项 | 说明 |
|--------|------|
| `knowledge_graph.extraction_mode` | 抽取模式: `triple` \| `entity` (默认 `entity`) |
| `knowledge_graph.triple_output_dir` | triple 模式输出目录 (默认 `./output/triple`) |
| `knowledge_graph.entity_output_dir` | entity 模式输出目录 (默认 `./output/entity`) |
| `knowledge_graph.entity_alignment_threshold` | 实体对齐相似度阈值 (默认 0.92)，超过即交给 LLM 二次确认 |

### 可视化

| 配置项 | 说明 |
|--------|------|
| `visualization.backend` | 可视化后端 (默认 `pyvis`) |
| `visualization.height` | HTML 画布高度 (默认 800) |
| `visualization.width` | HTML 画布宽度 (默认 1200) |
| `visualization.physics_solver` | pyvis 物理引擎 (默认 `forceAtlas2Based`) |
| `visualization.max_nodes` | 可视化最大节点数 (默认 500) |
| `visualization.drop_isolated_nodes` | 是否剔除孤立节点 (默认 `false`) |

### SQLite 持久化 (P5e)

| 配置项 | 说明 |
|--------|------|
| `sqlite.enabled` | P5 主流程末尾是否自动导出 SQLite (默认 `true`) |
| `sqlite.db_path` | 自定义 SQLite 路径 (默认 `{output_dir}/knowledge_graph.db`) |

### Graph-RAG 搜索 (P5e)

> P5 主流程不直接消费此段，只会被 `GraphRAGSearcher` 读取，详细用法见 `doc/code_p5e.md`。

| 配置项 | 说明 |
|--------|------|
| `graph_rag.enabled` | 是否启用 Graph-RAG 增强 (默认 `true`) |
| `graph_rag.bfs_depth` | BFS 扩散深度 (默认 1) |
| `graph_rag.max_expand_nodes` | BFS 最大扩散节点数 (默认 30) |
| `graph_rag.max_graph_tasks` | 图谱扩散最多引入的 task 数 (默认 50) |
| `graph_rag.graph_weight` | 图谱扩散结果的权重加成 (默认 0.3) |

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
