<div align="center">

# OpenCode Session Knowledge Graph

从 [OpenCode](https://github.com/sst/opencode) 会话历史中提取结构化知识，构建跨 session 知识图谱，支持语义搜索、Graph-RAG、主题总结与决策溯源。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qdrant](https://img.shields.io/badge/Qdrant-embedded-red.svg)](https://qdrant.tech/)
[![MCP](https://img.shields.io/badge/MCP-stdio-orange.svg)](https://modelcontextprotocol.io/)
[![Status](https://img.shields.io/badge/status-active-green.svg)]()

</div>

## 知识图谱效果

P5 支持两种提取模式（`run_phase5.sh` 一次跑完两种 + 可视化）：

**Triple 模式** — 三元组提取，hub-and-spoke 拓扑，稀疏分支，便于查看关系链路

![Triple 模式知识图谱](doc/triple-knowledge_graph.png)

**Entity 模式** — 实体共现，稠密 clique 聚簇，聚簇间桥接，平均度数更高

![Entity 模式知识图谱](doc/entity-knowledge_graph.png)

打开 `output/{triple,entity}/knowledge_graph.html` 即可在浏览器中交互（缩放、拖拽、悬停查看实体元数据与来源 task）。

## 功能

- **跨 Session 搜索** — Dense + BM25 + RRF 融合 + Qwen3-Reranker 精排
- **知识图谱构建** — LLM 提取三元组 / 实体共现，实体对齐（向量相似度 + LLM + UnionFind 合并），NetworkX 图
- **Graph-RAG** — 向量检索 + 图谱 BFS 扩散 + Reranker 融合
- **跨 Session 总结** — 检索 Top-K Task → Token 预算裁剪 → LLM 生成主题总结
- **决策溯源** — 多 hop 追踪技术决策链（选用 / 选型 / 排除 / 替换 / 依赖）
- **MCP Server** — 标准协议，支持 Claude Code / Codex / OpenCode / QoderWork 开箱接入

## 架构

```mermaid
graph LR
    A[OpenCode Sessions] -->|P1 切分/清洗| B[chunks.jsonl]
    B -->|P2 LLM 提取| C[tasks.jsonl + chunk summary]
    C -->|P3 向量化| D[(Qdrant<br/>3 collections)]
    C -->|P5 LLM 抽取| E[(Knowledge Graph<br/>SQLite)]
    D --> F[P4 混合搜索]
    D --> G[P6 总结]
    E --> G
    E --> H[P6b 决策溯源]
    D & E --> I[Graph-RAG]
    E --> J[MCP Server]
```

| Phase | 说明 | 详细文档 |
|-------|------|----------|
| **P1** 数据预处理 | Session → chunk，按 user message 边界切分，清洗 tool 输出 | [`doc/code_p1_README.md`](doc/code_p1_README.md) |
| **P2** | LLM CoT 提取结构化任务 + chunk 摘要 | [`doc/code_p2_README.md`](doc/code_p2_README.md) |
| **P3** 向量存储 | Qdrant 三个集合，Dense 支持 BM25 索引 | [`doc/code_p3_README.md`](doc/code_p3_README.md) |
| **P4** 跨 Session 搜索 | 两级 RRF → Reranker 精排 → chunk 展开 | [`doc/code_p4_README.md`](doc/code_p4_README.md) ✨核心 |
| **P5** 知识图谱 | `run_phase5.sh` 串行跑 triple + entity 两种模式，实体对齐，NetworkX 图 + pyvis 可视化 | [`doc/code_p5_README.md`](doc/code_p5_README.md) ✨核心 |
| **P5e** SQLite + Graph-RAG | 图谱持久化 + 向量检索 + 图 BFS 扩散 + Reranker 合并 | [`doc/code_p5e.md`](doc/code_p5e.md) |
| **P6** 跨 Session 总结 | Top-K Task → Token 裁剪 → LLM 主题总结 | [`doc/code_p6.md`](doc/code_p6.md) |
| **P6b** 决策溯源 + 骨架总结 | 多 hop 决策链追踪 + 图结构注入的因果总结 | [`doc/code_p6b.md`](doc/code_p6b.md) |
| **MCP** MCP Server | 5 个工具，Claude Code / Codex / OpenCode / QoderWork 即插即用 | [`doc/code_mcp.md`](doc/code_mcp.md) |

## 快速开始

```bash
# 1. 安装依赖
bash install.sh
source .venv/bin/activate

# 2. 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 3. HuggingFace 离线模式（避免模型下载超时）
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1

# 4. 端到端运行（P1 → P5）
python3 src/code_p1_main.py && \
python3 src/code_p2_main.py && \
python3 src/code_p3_main.py && \
bash run_phase5.sh

# 5. 搜索
python3 src/code_p4_search_cli.py --query "FlashAttention 实现原理"

# 6. 主题总结 + 决策溯源
python3 src/code_p6b_cli.py summary "FlashAttention"
python3 src/code_p6b_cli.py trace "FlashAttention"
```

> 数据源默认读取 `~/.local/share/opencode/opencode.db`，也可用 `--sqlite <path>` 或 `--mock` 指定。

## 目录结构

```
oc_sess_graph/
├── src/                  # Python 源码（扁平结构，code_p{N}_ 前缀区分 phase）
├── config/               # YAML 配置文件
├── prompts/             # LLM Prompt 模板
├── doc/                  # 各 Phase 详细文档 + 系统架构图（SVG）
├── output/               # 流水线输出（gitignore）
├── qdrant_data/          # 嵌入式 Qdrant（gitignore）
├── tests/                # P2/P5 benchmark 脚本
├── lib/                  # pyvis 前端资源
├── requirements.txt
└── install.sh            # 安装脚本（集安装 + 模型下载）
```

## 使用示例

```bash
# 跨 Session 搜索
python3 src/code_p4_search_cli.py --query "序列并行和 FlashAttention 的关系" --top-k 5

# 决策溯源（图谱多 hop 追踪）
python3 src/code_p6b_cli.py trace "OpenCode" --hops 2 --json

# 骨架总结（注入决策链 + 图结构）
python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 5

# Graph-RAG 增强总结
python3 src/code_p6_cli.py "opencode 功能" --graph-rag

# 时间范围过滤
python3 src/code_p6_cli.py "性能优化" --time-from 2026-07-01 --time-to 2026-07-15
```

## MCP 集成

将知识图谱作为 MCP Server 加载到你的 AI 编码工具，开箱即用 5 个工具：

| 工具 | 说明 |
|------|------|
| `query_kg` | BFS 扩散查询，获取实体关联 task |
| `search_entities` | 模糊搜索实体 |
| `get_entity_info` | 实体详情（节点 + 边） |
| `graph_rag_search` | Graph-RAG 增强搜索 |
| `get_kg_stats` | 图谱统计信息 |

<details>
<summary><b>客户端配置</b></summary>

**OpenCode** — `opencode.json`:
```json
{
  "mcp": {
    "knowledge-graph": {
      "type": "stdio",
      "command": "python3",
      "args": ["/absolute/path/to/src/code_mcp_server.py"],
      "env": { "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1" }
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
      "args": ["/absolute/path/to/src/code_mcp_server.py"],
      "env": { "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1" }
    }
  }
}
```

**QoderWork** — 在 Connectors 设置中粘贴同样的 `mcpServers` 块即可。

</details>

## 配置参考

| 配置文件 | Phase | 主要配置项 |
|---------|-------|-----------|
| `config/code_p1_config.yaml` | P1 | 数据路径, session 过滤 |
| `config/code_p2_config.yaml` | P2 | LLM 模型, API 地址, 并发数, 重试策略 |
| `config/code_p3_config.yaml` | P3/P4/P6/P6b | Embedding 模型, BM25 参数, Qdrant 路径, Reranker 模型, RRF 融合 |
| `config/code_p5_config.yaml` | P5/P5e | LLM 模型, 实体对齐阈值, 提取模式, SQLite 导出, Graph-RAG 参数 |
| `config/code_p6_config.yaml` | P6 | 总结 LLM, token 预算, 搜索参数, 时间过滤 |
| `config/code_p6b_config.yaml` | P6b | 决策关系分类, BFS 参数, 数据库路径 |
| `config/code_mcp_config.yaml` | MCP | 服务 DB 路径, 上下文注入参数 |

> 注：`code_p3_config.yaml` 被 P3/P4/P6/P6b 共用；P6/P6b 的 YAML 为 advisory，实际参数硬编码在 `code_p6_summarizer.py` / `code_p6b_skeleton.py`。

## 输出清理（code_cleanup）

按 phase 选择性清理流水线产物，默认 dry-run，需显式 `--yes` 才真删。详见 `doc/code_cleanup_README.md`。

```bash
# 列出所有 phase 产物
python3 src/code_cleanup.py list

# 预览删除计划
python3 src/code_cleanup.py clean --p2

# 真删 + 级联删除下游孤儿
python3 src/code_cleanup.py clean --p1 --cascade --yes
```

## 更多文档

- `doc/` — 各 Phase 详细设计文档 + 系统架构 SVG 图（`fig1`~`fig6`） + 各 phase README
- `doc/code_p4_README.md` / `doc/code_p5_README.md` — 搜索与图谱核心模块设计
- `prompts/` — 所有 LLM Prompt 模板
- `tests/` — P2 / P5 benchmark 脚本（独立 CLI 程序，非 pytest）