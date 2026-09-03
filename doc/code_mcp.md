## MCP Server 封装：知识图谱工具服务（标准 MCP 协议）

### 概述

将 Phase 5 知识图谱（SQLite）+ Phase 3 向量库 + Phase 6/6b 总结能力，封装为标准 MCP Server，使用 `mcp` Python SDK + stdio 传输。
**一套实现，多处可用**：Claude Code、Codex、OpenCode、QoderWork 均可直接接入。

当前版本暴露 **10 个工具**：

| 类别 | 工具 | 用途 |
|------|------|------|
| 检索（KG / 向量） | `query_kg` `search_entities` `get_entity_info` `graph_rag_search` | 知识图谱 + 向量混合检索 |
| 统计 | `get_kg_stats` | 图谱健康检查 |
| 浏览 | `list_sessions` `get_session_tasks` | 跨 session 历史浏览 |
| 溯源 | `trace_decision` | 决策链追踪（不调 LLM） |
| 总结（LLM） | `summarize` `skeleton_summarize` | 自由叙事总结 / 骨架+决策链总结 |

### 文件列表

| 文件 | 说明 |
|------|------|
| `src/code_mcp_server.py` | 标准 MCP Server（stdio 传输，FastMCP） |
| `src/code_mcp_server_http.py` | HTTP REST 版本（调试/备用，非标准 MCP） |
| `src/code_mcp_client.py` | Python 客户端 + `ContextInjector` 上下文注入器 |
| `config/code_mcp_config.yaml` | 服务配置（DB 路径、tasks 文件）+ 各客户端 MCP 配置示例 |
| `scripts/run_kg_mcp.sh` | opencode 启动 wrapper：从 auth.json 抽 key + 设离线环境（不入库） |
| `.opencode/opencode.jsonc` | opencode 项目级 MCP 注册（timeout 300000ms） |

---

### MCP 工具详解

#### 1. `query_kg(entity, depth=2, max_nodes=30)` — BFS 扩散检索

**用途**：从种子实体出发，沿知识图谱 BFS 扩散找到关联实体及其 task 列表。
**典型场景**：用户说"继续搞 XX" / "上次做 XX 怎么样了" 时，自动注入相关历史 task 到上下文。
**何时用** vs `graph_rag_search`：
- `query_kg` 是**结构化扩散**（沿图谱边走），确定性、可解释，适合"已知实体 → 关联任务"
- `graph_rag_search` 是**语义检索**（向量相似度），模糊查询时召回更全

**参数**：
- `entity`：种子实体名（支持模糊匹配，找不到时自动建议最近似的 1 个）
- `depth`：BFS 跳数（1=直接邻居，2=二跳，默认 2；3+ 易爆量）
- `max_nodes`：扩散节点上限，默认 30

**返回**：
```json
{
  "seed_entity": "FlashAttention",
  "depth": 2,
  "expanded_entities": ["Compute-Bound", "Online Softmax", ...],  // 15 个
  "related_tasks": [
    {
      "task_id": "ses_xxx_T2",
      "task_label": "Roofline模型算术强度推导",
      "task_summary": "背景：基于 Roofline 模型...",
      "created_at": "2026-08-04T01:49:08"
    }
  ],
  "elapsed_ms": 2
}
```

---

#### 2. `search_entities(query, limit=10)` — 实体模糊搜索

**用途**：列出与关键词匹配的实体名（部分匹配 + 别名命中）。
**典型场景**：用户说的实体名跟图谱里不完全一致时（如"flash attention" → "FlashAttention"），先 `search_entities` 找到准确名，再 `query_kg`。

**返回**：
```json
{
  "query": "FlashAttention",
  "results": [
    {"name": "FlashAttention", "entity_type": "concept", "aliases": ["Flash Attention"], "source_tasks": [...], "task_count": 3},
    {"name": "FlashAttention-2", "entity_type": "concept", "aliases": [], "source_tasks": [...], "task_count": 1}
  ],
  "count": 2
}
```

---

#### 3. `get_entity_info(entity)` — 实体详情

**用途**：查看某个实体的所有元数据 + 入边 + 出边。
**典型场景**：了解实体"连接到什么"、"被谁引用"，配合 `query_kg` 补充细节。

**返回**：
```json
{
  "name": "FlashAttention",
  "entity_type": "concept",
  "aliases": ["Flash Attention"],
  "source_tasks": ["ses_xxx_T2", "ses_xxx_T3", "ses_yyy_T1"],
  "task_count": 3,
  "edges_out": [{"head": "FlashAttention", "tail": "Online Softmax", "relation": "使用", "weight": 0.95, "source_task": "...", "extraction_mode": "triple"}],
  "edges_in": [...]
}
```

---

#### 4. `graph_rag_search(query, top_k=5, use_graph=true)` — Graph-RAG 增强搜索

**用途**：结合向量检索（Dense + BM25 + RRF + Reranker）和知识图谱扩散，返回跨 session 相关 task。
**典型场景**：用户问"我最近做过的性能优化" / "跟 FA 相关的所有工作" 等开放性查询。
**何时开图谱** vs 关图谱：
- 开（默认）：先向量召回，再 BFS 扩散补充关联 task → 召回更全
- 关：纯向量检索 → 速度快但漏掉图谱关联

**参数**：
- `query`：自然语言查询
- `top_k`：返回 task 数，默认 5
- `use_graph`：是否启用图谱扩散，默认 true

**性能**：
- 首次调用：~15-20s（含 Qwen3-Reranker-0.6B 模型加载）
- 后续调用：~10-15s（reranker warm）

---

#### 5. `get_kg_stats()` — 图谱健康检查

**用途**：返回 db 路径、节点数、边数、平均每实体关联 task 数。
**典型场景**：排查"为什么搜不到" / 验证 P5 是否跑完。

**返回**：
```json
{"db_path": "output/triple/knowledge_graph.db", "nodes": 1211, "edges": 3353, "avg_task_count_per_entity": 1.09}
```

---

#### 6. `list_sessions(limit=10)` — 浏览最近 session

**用途**：列出最近活跃的 session（按 `last_task_at` 倒序），包含 task 数、首末任务时间、task 标题列表。
**典型场景**：用户问"我最近做过哪些项目" / 在多个 session 之间导航。
**下一步**：拿到 `session_id` 后调用 `get_session_tasks` 下钻。

**返回**：
```json
{
  "session_count": 26,
  "sessions": [
    {
      "session_id": "ses_0870d73a7ffeh...",
      "task_count": 21,
      "first_task_at": "2026-08-04T10:32:28",
      "last_task_at": "2026-08-04T11:27:05",
      "task_labels": ["澄清图谱构建流程...", "评估 Dense Embedding 替换方案", ...]
    }
  ]
}
```

---

#### 7. `get_session_tasks(session_id)` — 单 session 完整 task 列表

**用途**：拿某个 session 的完整 task 摘要（按 `created_at` 升序）。
**典型场景**：`list_sessions` 找到感兴趣的 session 后，下钻看完整对话轨迹。
**注意**：`session_id` 不是 `task_id`（后者带 `_TN` 后缀）。

**返回**：
```json
{
  "session_id": "ses_0870d73a7ffeh...",
  "task_count": 21,
  "tasks": [{"task_id": "..._T1", "task_label": "...", "task_summary": "...", "created_at": "..."}]
}
```

如果传入无效 ID（不是已注册 session），返回：
```json
{"error": "未找到 session: ses_xxx", "session_id": "...", "task_count": 0, "tasks": []}
```

---

#### 8. `trace_decision(entity, max_hops=2)` — 决策链追踪（不调 LLM）

**用途**：从实体出发，沿 5 类决策关系（**选用** / **选型** / **排除** / **替换** / **依赖**）做 BFS，返回完整决策步骤及来源 task。
**典型场景**：
- "为什么选 FlashAttention 而不是标准 Attention？"
- "FA-2 跟 FA-1 的差异原因"
- "BF16 指南 v4 改了哪些内容"

**何时用** vs `skeleton_summarize`：
- `trace_decision`：**纯 KG 查询，毫秒级**，返回结构化步骤（不要 LLM）
- `skeleton_summarize`：在 `trace_decision` 基础上 + LLM 生成自然语言总结（秒级），但需要等 LLM

**5 类决策关系**：
| category | 含义 | 例子 |
|----------|------|------|
| 选用 | "使用了" | FlashAttention → 使用 → Online Softmax |
| 选型 | "对比了" | FlashAttention → 对比了 → 标准Attention |
| 排除 | "排除了" | (稀疏) |
| 替换 | "替代了" / "新增了" | FA-2 → 替代了 → FA-1 |
| 依赖 | "属于" / "包含" | FlashAttention → 属于 → Compute-Bound |

**返回**：
```json
{
  "root_entity": "FlashAttention",
  "hop_count": 3,
  "visited_entity_count": 43,
  "visited_entities": [...],
  "steps": [
    {
      "entity": "FlashAttention", "related_entity": "Compute-Bound",
      "relation": "属于", "category": "依赖", "hop": 1, "direction": "out",
      "source_task": "ses_xxx_T2", "weight": 0.9
    }
  ],
  "elapsed_ms": 19
}
```

---

#### 9. `summarize(query, top_k=8, time_from, time_to, use_graph_rag=true)` — 自由叙事总结（P6）

**用途**：检索 Top-K 相关 task → 收集 chunk 内容 → LLM 生成自然语言主题总结。
**典型场景**：
- "总结 XX 的相关工作" 等开放性查询
- "我最近做过的性能优化工作"

**参数**：
- `query`：自然语言查询
- `top_k`：检索 task 数，默认 8
- `time_from` / `time_to`：可选，日期范围（YYYY-MM-DD）
- `use_graph_rag`：是否启用图谱增强检索（默认 true）

**何时用** vs `skeleton_summarize`：
- `summarize`：**自由叙事**，没有骨架结构 / 决策链注入，更像一篇综述
- `skeleton_summarize`：**结构化骨架**，输出"决策逻辑 → 选型对比 → 技术细节 → 评价建议"

**性能**：
- 冷启动：~55s（首次需加载 SessionSummarizer + embedding + Reranker + LLM client）
- warm 调用：~55-70s（瓶颈是 LLM 推理）
- **MCP timeout 必须 ≥ 300000ms**（5min），否则冷启动会超时

**返回**：
```json
{
  "query": "FlashAttention",
  "summary": "# FlashAttention 技术深度解：...\n## 1. 性能的理论源...",
  "sources": [
    {"task_id": "...", "task_label": "Roofline...", "rerank_score": 0.998, "chunk_count": 3, "created_at": "..."}
  ],
  "total_tokens": 1074,
  "elapsed_ms": 55023,
  "debug": {"retrieved_tasks": 2, "graph": {"vector_results": 20, "rerank_time_ms": 7979, ...}}
}
```

---

#### 10. `skeleton_summarize(topic, max_tasks=5, use_decision_trace=true)` — 骨架总结（P6b）

**用途**：定位主题实体 → BFS 扩散 → 决策链追踪 → 骨架构建 → LLM 生成结构化总结。
**典型场景**：
- "总结 XX 的决策" / "总结 XX 的演进"
- 需要"为什么这么做"的因果链解释

**参数**：
- `topic`：主题关键词（如 'FlashAttention'）
- `max_tasks`：最多纳入 task 数，默认 5
- `use_decision_trace`：是否注入决策链上下文，默认 true

**何时用** vs `summarize`：
- 想看"决策依据 / 选型过程" → 用本工具
- 想看"完整综述 / 全貌介绍" → 用 `summarize`

**返回**：相比 `summarize` 多 3 个字段：
```json
{
  "topic": "FlashAttention",
  "summary": "# FlashAttention 技术决策源总结\n## 核心定位：...",
  "skeleton": "### FlashAttention (concept)\n  决策关系: → Compute-Bound (属于); ← 训练加速三件套 (包含); ...\n  - [Roofline...] (2026-08-04): 背景：...",
  "decision_chain": {
    "root_entity": "FlashAttention",
    "hop_count": 3,
    "step_count": 49,
    "steps": [...]  // 同 trace_decision 结构
  },
  "task_summaries": [...],
  "elapsed_ms": 35525,
  "debug": {...}
}
```

`skeleton` 字段是 Markdown 文本，可直接喂给后续 Agent 作为上下文。

---

### 工具搭配模式（实战推荐）

| 用户意图 | 推荐工具链 |
|----------|-----------|
| "继续搞 XX" | `search_entities` → `query_kg`（注入关联 task 摘要） |
| "上次为什么选 XX" | `trace_decision`（毫秒级，不调 LLM） |
| "总结 XX 的演进" | `skeleton_summarize`（带决策骨架） |
| "总结 XX 的相关工作" | `summarize`（自由叙事） |
| "我最近做了什么" | `list_sessions` → `get_session_tasks` |
| "跟 XX 相关的所有 task" | `graph_rag_search`（语义检索 + 图谱扩散） |
| "XX 跟 YY 的对比" | `trace_decision('XX')` + `trace_decision('YY')` → 自己 diff |
| "图谱健康检查" | `get_kg_stats` |

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 安装依赖
pip3 install mcp[cli] pyyaml --quiet

# 进入项目目录
cd ~/Desktop/oc_sess_graph

# 确保 Phase 5 数据已生成（MCP 默认使用 triple 模式，见 config/code_mcp_config.yaml）
ls output/triple/knowledge_graph.db
# 如果不存在: python3 src/code_p5_main.py --mode triple --limit 10
```

#### 1. Claude Code 接入

```bash
# 方法 1: 命令行添加（推荐）
claude mcp add knowledge-graph \
  python3 /Users/zhaoxiuwei/Desktop/oc_sess_graph/src/code_mcp_server.py \
  -e OPENCODE_ZEN_API_KEY "$(python3 -c \
    "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")"

# 验证
claude mcp list
# 预期: knowledge-graph (healthy)

# 在 Claude Code 中使用（agent 会自动发现工具）:
# 直接对话: "帮我查一下 OpenCode 相关的历史任务"
# Claude Code 会自动调用 query_kg(entity="OpenCode")
```

```json
// 方法 2: 手动编辑 ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/src/code_mcp_server.py"],
      "env": {
        "OPENCODE_ZEN_API_KEY": "<your-key>"
      }
    }
  }
}
```

#### 2. Codex 接入

```json
// 编辑 ~/.codex/config.json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/src/code_mcp_server.py"],
      "env": {
        "OPENCODE_ZEN_API_KEY": "<your-key>"
      }
    }
  }
}
```

#### 3. OpenCode 接入（推荐 wrapper 方式）

项目级 `.opencode/opencode.jsonc`：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "knowledge-graph": {
      "type": "local",
      "command": ["bash", "scripts/run_kg_mcp.sh"],
      "enabled": true,
      "timeout": 300000
    }
  }
}
```

配套 `scripts/run_kg_mcp.sh`（gitignored，25 行 wrapper）：
```bash
#!/usr/bin/env bash
# 从 ~/.local/share/opencode/auth.json 抽 key 并设离线环境
AUTH_FILE="${OPENCODE_AUTH_FILE:-$HOME/.local/share/opencode/auth.json}"
export OPENCODE_ZEN_API_KEY="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["opencode-go"]["key"])' "$AUTH_FILE")"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
exec python3 "$(cd "$(dirname "$0")/.." && pwd)/src/code_mcp_server.py" "$@"
```

> **timeout 必须 ≥ 300000ms**（5 分钟），否则 `summarize` / `skeleton_summarize` 冷启动会超时。

#### 4. QoderWork / MiniMax Code 接入

在连接器 / 自定义 MCP 中手动添加：

| 字段 | 值 |
|------|---|
| Server 名称 | `knowledge-graph` |
| 传输方式 | stdio |
| 命令 | `python3` |
| 参数 | `/Users/zhaoxiuwei/Desktop/oc_sess_graph/src/code_mcp_server.py` |
| 环境变量 | `OPENCODE_ZEN_API_KEY` = `<your-key>` |
| 超时 | `300000`（默认 30000 太短） |

或使用 wrapper：
| 命令 | `bash` |
| 参数 | `/Users/zhaoxiuwei/Desktop/oc_sess_graph/scripts/run_kg_mcp.sh` |
| 环境变量 | 全空 |

#### 5. MCP Inspector 调试

```bash
# 使用 MCP Inspector 交互式测试工具
npx @modelcontextprotocol/inspector python3 src/code_mcp_server.py

# 浏览器打开后：
# 1. 点击 "Tools" 标签查看 10 个工具定义
# 2. 选择 query_kg，填入 entity="OpenCode"，depth=2
# 3. 点击 "Call Tool" 查看返回结果
```

#### 6. HTTP 模式调试（备选）

```bash
# 以 streamable-http 模式启动（适合 curl 调试）
python3 src/code_mcp_server.py --http &
# 服务运行在 http://127.0.0.1:8000/mcp

# 或以 SSE 模式启动
python3 src/code_mcp_server.py --sse &
# SSE 端点: http://127.0.0.1:8000/sse
```

#### 7. Client CLI 验证

```bash
# 前置：先启动 HTTP 版本 Server（code_mcp_client.py 通过 HTTP 调用）
python3 src/code_mcp_server_http.py &

# 设置 API Key（context 命令需要 LLM 抽取实体）
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# BFS 查询（通过 src/code_mcp_client.py，调用 HTTP 版本）
python3 src/code_mcp_client.py query OpenCode 2

# 上下文注入演示
python3 src/code_mcp_client.py context "继续搞 RoPE 位置编码的优化"
# 预期:
#   查询: 继续搞 RoPE 位置编码的优化
#   实体: ['RoPE', '位置编码'] → 命中: ['RoPE位置编码']
#   关联 task: N
#   === 注入上下文 ===
#   ## 跨会话记忆（来自知识图谱）
#   ...
```

---

### 架构

```
Claude Code / Codex / OpenCode / QoderWork / MiniMax Code
  │
  ├─ MCP 协议 (stdio)
  │   ├─ initialize → server capabilities
  │   ├─ tools/list → 10 个工具定义 (JSON Schema)
  │   └─ tools/call → query_kg / search_entities / ... / summarize / skeleton_summarize
  │                      ├─ KGDatabase (SQLite)        ← P5e
  │                      ├─ SessionSearcher (Qdrant)   ← P4
  │                      ├─ DecisionTracer             ← P6b
  │                      └─ SessionSummarizer          ← P6 / P6b
  │
  └─ 自动工具发现
      └─ Agent 根据用户意图自动选择工具调用
          "继续搞 XX" → query_kg(entity="XX")
          "为什么选 XX" → trace_decision(entity="XX")
          "总结 XX 的演进" → skeleton_summarize(topic="XX")
          "总结 XX 的相关工作" → summarize(query="XX")
```

### 性能基线（实测）

| 工具 | 冷启动 | warm | 备注 |
|------|--------|------|------|
| `query_kg` | ~5ms | ~5ms | 纯 SQLite BFS |
| `search_entities` | ~1ms | ~1ms | SQLite LIKE |
| `get_entity_info` | ~1ms | ~1ms | SQLite 单节点 + 边 |
| `get_kg_stats` | ~1ms | ~1ms | SQLite 聚合 |
| `trace_decision` | ~20ms | ~20ms | BFS 沿 5 类决策关系 |
| `list_sessions` | ~5ms | ~5ms | 内存索引 |
| `get_session_tasks` | ~5ms | ~5ms | 内存过滤 |
| `graph_rag_search` | ~17s | ~10-15s | 含 Qwen3-Reranker-0.6B 加载 |
| `summarize` | ~55-70s | ~55s | 含 SessionSummarizer + LLM |
| `skeleton_summarize` | ~160s (cold) / ~35s (warm) | ~50s | warm 后复用 SessionSummarizer |

> **MCP server timeout 必须 ≥ 300000ms**。`summarize` 和 `skeleton_summarize` 冷启动 > 180s，需要 5 分钟预算。

### 上下文压缩效果

| 指标 | 无记忆 | MCP 注入 |
|------|--------|----------|
| 上下文大小 | 0 tokens | ~200-500 tokens (3-5 task 摘要) |
| 跨 session 连续性 | 无 | Agent 自动调用 query_kg 关联历史 task |
| 工具调用延迟 | — | ~5-10ms (BFS 查询) |

当 task 数量增长到 100+，MCP 注入的上下文压缩比会更显著：从需要加载全部历史 session（~50K tokens）压缩到仅注入相关 task 摘要（~500 tokens），压缩率约 99%。

### 已知约束

1. **Qdrant portalocker**: 同 `qdrant_data/` 路径只允许一个进程访问。`graph_rag_search` 和 `skeleton_summarize` / `summarize` 共享 `SessionSearcher` 实例，避免多实例互斥。
2. **HF 离线**: `src/code_mcp_server.py` 模块顶层强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，避免 huggingface_hub 内部 httpx.Client GC 关闭导致 "Cannot send a request" 错误。
3. **冷启动超长**: `summarize` 首次调用需加载 embedding + Reranker + LLM client，~55s。`skeleton_summarize` 首次 ~160s（cold SessionSummarizer）。建议在 prompt 引导先调便宜的 `summarize` 预热。
4. **stdio pipe 卡死风险**: opencode client 在某次请求超时会断开 stdio pipe，导致 MCP server 进程残留。修复方法：杀掉残留 PID (`lsof qdrant_data/.lock` 找持有者) + 重启 opencode。

### 依赖

```
mcp[cli]    — MCP Python SDK (FastMCP + 传输层)
pyyaml      — YAML 配置解析
```

已有的项目依赖（无需额外安装）：`code_p5e_db.KGDatabase`、`code_p5e_graph_rag`、`code_p4_searcher.SessionSearcher`、`code_p6_summarizer.SessionSummarizer`、`code_p6b_skeleton.SummarySkeleton`。