# OpenCode Session Knowledge Graph

从 [OpenCode](https://github.com/sst/opencode) 的会话历史中提取结构化知识，构建跨 session 知识图谱，支持语义搜索、图谱查询、主题总结和决策溯源。

## 快速开始

```bash
# 1. 安装
bash install.sh
source .venv/bin/activate

# 2. 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 3. 离线模式（避免 HuggingFace 超时）
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 4. 端到端运行（Phase 1→5e）
python3 src/code_p1_main.py && \
python3 src/code_p2_main.py && \
python3 src/code_p3_main.py && \
python3 src/code_p5_main.py

# 5. 搜索
python3 src/code_p4_search_cli.py --query "FlashAttention 实现原理"

# 6. 主题总结 + 决策溯源
python3 src/code_p6b_cli.py summary "FlashAttention"
python3 src/code_p6b_cli.py trace "FlashAttention"
```

## 目录结构

```
oc_sess_graph/
├── src/                     # Python 源码（扁平结构，code_p{N}_ 前缀区分 phase）
│   ├── code_p1_*.py         # Phase 1: 数据预处理
│   ├── code_p2_*.py         # Phase 2: 任务提取
│   ├── code_p3_*.py         # Phase 3: 向量存储
│   ├── code_p4_*.py         # Phase 4: 跨 Session 搜索
│   ├── code_p5_*.py         # Phase 5: 知识图谱
│   ├── code_p5e_*.py        # Phase 5e: SQLite + Graph-RAG
│   ├── code_p6_*.py         # Phase 6: 跨 Session 总结
│   ├── code_p6b_*.py        # Phase 6b: 决策溯源 + 骨架总结
│   ├── code_mcp_*.py        # MCP Server (标准协议)
│   └── code_MS_inspect.py   # Milestone 检查工具
├── config/                  # YAML 配置文件
├── prompts/                 # LLM Prompt 模板（从代码中拆出）
├── doc/                     # 各 Phase 详细文档
├── output/                  # 流水线输出数据
│   ├── chunks.jsonl         # P1 产物
│   ├── tasks.jsonl          # P2 产物
│   ├── chunks_summary_p2.jsonl  # P2 产物
│   ├── entity/              # P5 entity 模式产物
│   └── triple/              # P5 triple 模式产物
├── tests/                   # 测试 / 演示文件
├── logs/                    # 执行日志
├── data/                    # 输入数据
├── lib/                     # pyvis 前端资源
├── qdrant_data/             # Qdrant 嵌入式数据库
├── 图谱方案.md               # 系统总体设计文档
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
└── install.sh               # 安装脚本
```

---

## Phase 详解

每个 Phase 的说明包含：前置条件、执行命令、预期输出、配置说明。

**所有命令均假设在项目根目录执行，且已激活虚拟环境。**

### 环境变量（每个 Phase 都需要）

```bash
# 激活虚拟环境
source .venv/bin/activate

# API Key（Phase 2 及以后需要）
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# HuggingFace 离线模式（避免模型下载超时）
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
```

---

### Phase 1: 数据预处理

**做什么**: 从 OpenCode 会话数据（JSONL 或 SQLite）加载对话，按 user message 边界切分为 chunk，清洗 tool call 输出降噪。

**前置条件**:
- OpenCode 会话数据（JSONL 文件 或 `~/.local/share/opencode/opencode.db`）
- 配置文件 `config/code_p1_config.yaml`（已配置数据路径）

**执行**:
```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 默认配置（不带任何数据源参数时，按 --sqlite > --mock > --source >
# config.project.mock_data > config.opencode.session_source 的优先级选择数据源）
python3 src/code_p1_main.py 2>&1 | tee logs/phase1.log

# 自定义配置文件
python3 src/code_p1_main.py --config config/code_p1_config.yaml 2>&1 | tee logs/phase1.log

# 从 OpenCode SQLite 数据库加载（推荐，直接读 opencode.db）
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db 2>&1 | tee logs/phase1.log

# 只跑前 N 个 session（快速验证）
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3

# 用 mock 数据（无需真实 session）
python3 src/code_p1_main.py --mock
```

**输出**:
- `output/chunks.jsonl` — 清洗后的 chunk 数据（每行一个 JSON，字段见 `Chunk` 模型：`chunk_id`, `session_id`, `turn_index`, `user_message`, `assistant_messages`, `tool_calls`, `mcp_calls`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`, `task_summary`；`task_summary` 在 P1 阶段恒为 `null`，由 Phase 2 填充）

**配置** (`config/code_p1_config.yaml`，仅列代码实际读取的项；其余 see `doc/code_p1_README.md` 中的 wired/advisory 对照):
- `opencode.session_filter.type` — session 类型过滤（默认 `root`，对应 SQLite `parent_id IS NULL`）
- `opencode.session_filter.min_user_messages` — 最少 user 消息数（默认 2）
- `opencode.session_source` — 无 `--sqlite`/`--source`/`--mock` 时的默认 JSONL/JSON 路径
- `project.mock_data` — 无命令行数据源时是否回退 mock 数据
- `logging.level` / `logging.file` — 日志配置

> 注: YAML 中的 `chunking.*` 与 `content_cleaning.*` 段当前为 advisory,chunker 与 cleaner 已在源码中 hardcode 行为,改 YAML 不会生效。详见 `doc/code_p1_README.md`。

---

### Phase 2: 任务提取

**做什么**: 用 LLM（CoT 三步提示）从每个 session 的 chunk 中提取结构化任务（task_id, task_label, task_summary, chunk_ids），同时生成 chunk 摘要。

**前置条件**:
- Phase 1 产物 `output/chunks.jsonl`
- `OPENCODE_ZEN_API_KEY` 环境变量

**执行**:
```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

python3 src/code_p2_main.py 2>&1 | tee logs/phase2.log

# 自定义并发数
python3 src/code_p2_main.py --config config/code_p2_config.yaml --concurrency 8 2>&1 | tee logs/phase2.log
```

**输出**:
- `output/tasks.jsonl` — 提取的任务列表（每行一个 JSON）
- `output/chunks_summary_p2.jsonl` — 每个 chunk 的 LLM 摘要

**配置** (`config/code_p2_config.yaml`):
- `llm.model` — LLM 模型（默认 `nemotron-3-ultra-free`）
- `llm.base_url` — API 地址（默认 `https://opencode.ai/zen/v1`）
- `llm.concurrency` — 并发数（默认 4）

---

### Phase 3: 向量存储

**做什么**: 将 Phase 1/2 产物写入 Qdrant 三个集合（tasks, chunks_summary, chunks_cleaned_text），构建 Dense + BM25 混合检索索引。

**前置条件**:
- Phase 1: `output/chunks.jsonl`
- Phase 2: `output/tasks.jsonl`, `output/chunks_summary_p2.jsonl`

**执行**:
```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 写入 Qdrant（首次运行会下载 embedding 模型 ~100MB）
python3 src/code_p3_main.py 2>&1 | tee logs/phase3.log

# 搜索演示（交互式）
python3 src/code_p3_search_demo.py --query "序列并行" --top-k 5
```

**输出**:
- `qdrant_data/` — Qdrant 嵌入式数据库（3 个集合）
  - `tasks` — 任务向量
  - `chunks_summary` — chunk 摘要向量
  - `chunks_cleaned_text` — 清洗后文本向量

**配置** (`config/code_p3_config.yaml`):
- `embedding.dense_model` — Dense 模型（默认 `BAAI/bge-small-zh-v1.5`）
- `sparse.bm25_tokenizer` — BM25 分词（默认 `jieba`）
- `qdrant.path` — Qdrant 数据路径（默认 `./qdrant_data`）

---

### Phase 4: 跨 Session 搜索

**做什么**: 两级 RRF 查询：Dense 搜 `tasks` 集合（task_summary 语义匹配）；`chunks_summary` 按配置使用 Dense 或 Sparse，`chunks_cleaned_text` 使用 Sparse，两路分别映射回 task 后先做 RRF，再与 Dense task 结果做 RRF → Qwen3-Reranker 精排 → Chunk 详情展开。

**前置条件**:
- Phase 3 产物（Qdrant 三个集合已填充）

**执行**:
```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 单次查询
python3 src/code_p4_search_cli.py --query "FlashAttention 实现原理" --top-k 5

# 跳过 Reranker（仅看粗排 RRF 效果）
python3 src/code_p4_search_cli.py --query "FlashAttention 实现原理" --no-rerank

# 交互式模式
python3 src/code_p4_search_cli.py --interactive
```

**输出**: 终端打印搜索结果（task_label、rerank_score、hybrid_score、task_summary 及关联 chunk 摘要/预览）

**配置** (`config/code_p3_config.yaml` 的 `reranker` 段，复用 `sparse.*`/`embedding.*`/`qdrant.*`):
- `reranker.model` — Reranker 模型（默认 `Qwen/Qwen3-Reranker-0.6B`）
- `reranker.device` — 推理设备（默认 `cpu`，可改 `mps`/`cuda`）
- `sparse.method` — 稀疏检索方法（`bm25` 或 `bge_m3`）
- `sparse.chunks_summary_method` — `chunks_summary` 的检索方式（`sparse` 或 `dense`）
- `sparse.fuse_k` — RRF 融合常数（默认 60）

详见 `doc/code_p4_README.md`。

---

### Phase 5: 知识图谱

**做什么**: 用 LLM 从 task summary 中提取三元组 (head, relation, tail)，实体对齐（Qdrant 相似度 + LLM 判断），构建 NetworkX MultiDiGraph。

**前置条件**:
- Phase 2: `output/tasks.jsonl`
- Phase 3: Qdrant 已初始化
- `OPENCODE_ZEN_API_KEY`

**执行**:
```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# Triple 模式（默认，推荐；由 config/code_p5_config.yaml 的 knowledge_graph.extraction_mode 决定）
python3 src/code_p5_main.py 2>&1 | tee logs/phase5.log

# Entity 模式（共现图谱；无 --mode 参数，需将配置文件 knowledge_graph.extraction_mode 改为 "entity"，
# 或另存一份 config/code_p5_config_entity.yaml 并通过 --config 指定）
python3 src/code_p5_main.py --config config/code_p5_config_entity.yaml 2>&1 | tee logs/phase5_entity.log

# 跳过对齐 + 可视化
python3 src/code_p5_main.py --skip-alignment --visualize

# 限制处理数（调试）
python3 src/code_p5_main.py --limit 5
```

**输出**:
- `output/triple/` — Triple 模式产物
  - `triples.jsonl` — 提取的三元组
  - `entities.jsonl` — 对齐后的实体
  - `knowledge_graph.gpickle` — NetworkX 图
  - `knowledge_graph.html` — pyvis 交互式可视化
  - `knowledge_graph.db` — SQLite 持久化（Phase 5e）
- `output/entity/` — Entity 模式产物（同上结构）

**配置** (`config/code_p5_config.yaml`):
- `extraction_mode` — `triple` 或 `entity`
- `alignment.threshold` — 实体对齐相似度阈值（默认 0.92）
- `sqlite.enabled` — 是否导出 SQLite（默认 true）

---

### Phase 5e: SQLite + Graph-RAG

**做什么**: Phase 5 的扩展。SQLite 持久化（KGDatabase）+ Graph-RAG 搜索（向量检索 + 图谱 BFS 扩散 + Reranker 合并排序）。

**前置条件**:
- Phase 5: `output/triple/knowledge_graph.db` 已生成（Phase 5 自动导出）

**此 Phase 无独立 CLI**，通过以下方式使用：

**Python API**:
```python
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher
from code_p4_searcher import SessionSearcher

db = KGDatabase("output/triple/knowledge_graph.db")
searcher = SessionSearcher("config/code_p3_config.yaml")
rag = GraphRAGSearcher(searcher, db)

results, debug = rag.search("FlashAttention", top_k=5)
for r in results:
    print(f"[{r.task_label}] score={r.rerank_score:.3f}")
```

**MCP 集成**: 见下方 MCP Server 章节。

---

### Phase 6: 跨 Session 总结

**做什么**: 检索与查询相关的 Top-K Task → 收集 chunk 内容 → Token 预算裁剪 → LLM 生成主题性总结。

**前置条件**:
- Phase 2-4 产物
- `OPENCODE_ZEN_API_KEY`

**执行**:
```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 基本总结
python3 src/code_p6_cli.py "我最近做过的 FlashAttention 相关工作"

# 详细模式 + 指定 top-k
python3 src/code_p6_cli.py "序列并行和 FlashAttention 的关系" --top-k 5 --detail preview

# 时间范围过滤
python3 src/code_p6_cli.py "性能优化" --time-from 2026-07-01 --time-to 2026-07-15

# Graph-RAG 增强
python3 src/code_p6_cli.py "opencode 功能" --graph-rag

# JSON 输出
python3 src/code_p6_cli.py "RoPE 位置编码" --json
```

**输出**: 终端打印 LLM 生成的主题总结（300-800 字，含 task_id 引用）

**Chunk 详情级别** (`--detail`):
- `summary` — 仅 chunk 摘要（轻量，默认）
- `preview` — 摘要 + 前 500 字原文
- `full` — 完整原文（token 消耗大）

---

### Phase 6b: 决策溯源 + 骨架总结

**做什么**: 
1. **决策溯源** — 从知识图谱中多跳追踪技术决策链（选用/选型/排除/替换/依赖关系）
2. **骨架总结** — 在 Phase 6 基础上注入图谱结构和决策链上下文，生成带因果关系的技术总结

**前置条件**:
- Phase 5e: `output/triple/knowledge_graph.db`
- Phase 4: Qdrant + Reranker 就绪
- `OPENCODE_ZEN_API_KEY`

**执行**:
```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 决策溯源
python3 src/code_p6b_cli.py trace "FlashAttention"
python3 src/code_p6b_cli.py trace "OpenCode" --hops 2 --json

# 搜索图谱实体
python3 src/code_p6b_cli.py search "Flash"

# 骨架总结（含决策链注入）
python3 src/code_p6b_cli.py summary "FlashAttention"

# 骨架总结（不注入决策链）
python3 src/code_p6b_cli.py summary "FlashAttention" --no-trace

# 完整参数
python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 5 --hops 2 --json
```

**输出**:
- `trace` — 决策链（按类别分组：选用/选型/排除/替换/依赖）
- `summary` — 结构化总结（400-1000 字，含决策主线 + 选型对比 + 技术细节 + 建议）

**决策关系分类** (`config/code_p6b_config.yaml`):
- 选用类: 使用, 集成了, 实现, 支持, 通过兼容层加载
- 选型类: 对比, 对比了, 区别于, 适用于, 适合
- 排除类: 禁用, 缺少, 阻塞于, 阉割了
- 替换类: 替代了, 新增了, 补充了
- 依赖类: 属于, 包含, 对应

---

### MCP Server: 知识图谱工具服务

**做什么**: 将知识图谱封装为标准 MCP Server（JSON-RPC 2.0 over stdio），提供 5 个工具供 Claude Code / Codex / OpenCode / QoderWork 直接加载。

**前置条件**:
- Phase 5: `output/entity/knowledge_graph.db`（默认配置，见 `config/code_mcp_config.yaml` 的 `server.db_path`）
- `pip install mcp` 已安装

**工具列表**:
| 工具 | 说明 |
|------|------|
| `query_kg` | BFS 扩散查询，获取实体关联 task |
| `search_entities` | 模糊搜索实体 |
| `get_entity_info` | 实体详情（节点 + 边） |
| `graph_rag_search` | Graph-RAG 增强搜索 |
| `get_kg_stats` | 图谱统计信息 |

**启动**:
```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# stdio 模式（供客户端配置使用）
python3 src/code_mcp_server.py

# HTTP 调试模式
python3 src/code_mcp_server.py --http
```

**客户端配置**:

**QoderWork** — 在 Connectors 设置中粘贴:
```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/absolute/path/to/src/code_mcp_server.py"],
      "env": {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1"
      },
      "cwd": "/absolute/path/to/project/root"
    }
  }
}
```

**Claude Code** — `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/absolute/path/to/src/code_mcp_server.py"]
    }
  }
}
```

**OpenCode** — `opencode.json`:
```json
{
  "mcp": {
    "knowledge-graph": {
      "type": "stdio",
      "command": "python3",
      "args": ["/absolute/path/to/src/code_mcp_server.py"]
    }
  }
}
```

---

### MS: Milestone 检查工具

**做什么**: 检查 Qdrant 数据质量、检索效果评估、collection 统计。

**前置条件**:
- Phase 3: Qdrant 已填充

**执行**:
```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

python3 src/code_MS_inspect.py
```

---

## 完整工作流

从零开始的端到端流水线，每一步都可直接复制执行：

### 全流程一键脚本

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 环境准备
source .venv/bin/activate
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

echo "=== Phase 1: 数据预处理 ==="
python3 src/code_p1_main.py 2>&1 | tee logs/phase1.log

echo "=== Phase 2: 任务提取 ==="
python3 src/code_p2_main.py 2>&1 | tee logs/phase2.log

echo "=== Phase 3: 向量存储 ==="
python3 src/code_p3_main.py 2>&1 | tee logs/phase3.log

echo "=== Phase 5: 知识图谱 (triple) ==="
python3 src/code_p5_main.py 2>&1 | tee logs/phase5.log

# Entity 模式无独立 --mode 参数，需要单独的配置文件（knowledge_graph.extraction_mode: entity）
# echo "=== Phase 5: 知识图谱 (entity) ==="
# python3 src/code_p5_main.py --config config/code_p5_config_entity.yaml 2>&1 | tee logs/phase5_entity.log

echo ""
echo "=== 完成! ==="
echo "搜索: python3 src/code_p4_search_cli.py --query '你的问题'"
echo "总结: python3 src/code_p6b_cli.py summary '你的主题'"
```

### 分步执行（带详细参数）

```bash
# ============================================================
# Step 0: 环境准备（每次新开终端都需要执行）
# ============================================================
cd ~/Desktop/oc_sess_graph
source .venv/bin/activate

export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# ============================================================
# Step 1: Phase 1 — 数据预处理
# ============================================================
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --config config/code_p1_config.yaml 2>&1 | tee logs/phase1.log

# 验证
wc -l output/chunks.jsonl  # 应输出 chunk 数量

# ============================================================
# Step 2: Phase 2 — 任务提取
# ============================================================
python3 src/code_p2_main.py --config config/code_p2_config.yaml 2>&1 | tee logs/phase2.log

# 验证
wc -l output/tasks.jsonl output/chunks_summary_p2.jsonl

# ============================================================
# Step 3: Phase 3 — 向量存储
# ============================================================
python3 src/code_p3_main.py --config config/code_p3_config.yaml 2>&1 | tee logs/phase3.log

# 验证搜索
python3 src/code_p3_search_demo.py --query "测试查询" --top-k 3

# ============================================================
# Step 4: Phase 4 — 跨 Session 搜索（可选验证）
# ============================================================
python3 src/code_p4_search_cli.py --query "FlashAttention 实现" --top-k 5

# ============================================================
# Step 5: Phase 5 — 知识图谱 (Triple 模式)
# ============================================================
python3 src/code_p5_main.py --config config/code_p5_config.yaml 2>&1 | tee logs/phase5.log

# 验证
ls -la output/triple/knowledge_graph.db

# ============================================================
# Step 6: Phase 5 — 知识图谱 (Entity 模式, 可选)
# 无独立 --mode 参数，需将 config/code_p5_config.yaml 的
# knowledge_graph.extraction_mode 改为 "entity"（或另存一份指定 --config）
# ============================================================
python3 src/code_p5_main.py --config config/code_p5_config_entity.yaml 2>&1 | tee logs/phase5_entity.log

# ============================================================
# Step 7: Phase 6 — 跨 Session 总结
# ============================================================
python3 src/code_p6_cli.py "我最近做的技术工作" --top-k 5 --detail summary

# ============================================================
# Step 8: Phase 6b — 决策溯源
# ============================================================
python3 src/code_p6b_cli.py trace "FlashAttention"
python3 src/code_p6b_cli.py search "序列"

# ============================================================
# Step 9: Phase 6b — 骨架总结
# ============================================================
python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 5

# ============================================================
# Step 10: MCP Server（供客户端集成，保持运行）
# ============================================================
python3 src/code_mcp_server.py  # stdio 模式，由客户端启动
```

---

### 输出清理（code_cleanup）

按 phase 选择性清理 `output/`、`qdrant_data/`、`logs/` 下的 pipeline 产物。默认 dry-run，不加 `--yes` 不删任何文件。

```bash
# 1. 列出所有 phase 产物 (path/exists/size)
python3 src/code_cleanup.py list

# 2. 预览删除计划 (dry-run, 只打印不删)
python3 src/code_cleanup.py clean --p2

# 3. 真删 Phase 2 产物 (tasks.jsonl + chunks_summary_p2.jsonl + .p2_checkpoint.json)
python3 src/code_cleanup.py clean --p2 --yes

# 4. 级联删除: 清 Phase 1 并连带删除依赖它的 P2/P3/P5 产物
python3 src/code_cleanup.py clean --p1 --cascade --yes

# 5. 清空所有 phase 产物 (含 checkpoint 与 Qdrant 向量库)
python3 src/code_cleanup.py clean --all --yes
```

说明：
- checkpoint 与 phase 强绑：清 P1/P2 必带对应 checkpoint，`--all` 一并清。
- `--cascade` 默认关闭；关闭时真删后会 WARNING 列出下游孤儿产物。
- 不触碰 `tests/` 下的 benchmark 副产物，不提供 backup/undo（删除不可逆，先 dry-run 确认）。
- 详见 `doc/code_cleanup_README.md`。

---

## 配置参考

| 配置文件 | Phase | 主要配置项 |
|---------|-------|-----------|
| `config/code_p1_config.yaml` | P1 | 数据路径, session 过滤, chunk 切分, 内容清洗 |
| `config/code_p2_config.yaml` | P2 | LLM 模型, API 地址, 并发数, 重试策略 |
| `config/code_p3_config.yaml` | P3, P4, P6, P6b | Embedding 模型, BM25 参数, Qdrant 路径, Reranker 模型, RRF 融合 |
| `config/code_p5_config.yaml` | P5, P5e | LLM 模型, 实体对齐阈值, 提取模式, SQLite 导出, Graph-RAG 参数 |
| `config/code_p6_config.yaml` | P6 | 总结 LLM, token 预算, 搜索参数, 时间过滤 |
| `config/code_p6b_config.yaml` | P6b | 决策关系分类, BFS 参数, 数据库路径 |
| `config/code_mcp_config.yaml` | MCP | 服务 DB 路径, 上下文注入参数 |

注: Phase 3 的配置文件 (`code_p3_config.yaml`) 被 P3/P4/P6/P6b 共用。

## 更多文档

- `doc/` — 各 Phase 详细设计文档
- `prompts/` — 所有 LLM Prompt 模板
- `图谱方案.md` — 系统总体设计（7 阶段规划）
- `tests/` — 测试和演示 notebook
