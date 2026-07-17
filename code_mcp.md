## MCP Server 封装：知识图谱工具服务（标准 MCP 协议）

### 概述

将 Phase 5 知识图谱（SQLite）封装为标准 MCP Server，使用 `mcp` Python SDK + stdio 传输。
**一套实现，四处可用**：Claude Code、Codex、OpenCode、QoderWork 均可直接加载。

### 文件列表

| 文件 | 说明 |
|------|------|
| `code_mcp_server.py` | 标准 MCP Server（stdio/SSE/streamable-http 三种传输可选） |
| `code_mcp_server_http.py` | HTTP REST 版本（调试/备用，非标准 MCP） |
| `code_mcp_client.py` | Python 客户端：实体抽取 + `ContextInjector` 上下文注入器 |
| `code_mcp_config.yaml` | 服务配置 + 各客户端 MCP 配置示例 |

### MCP 工具

| 工具 | 说明 |
|------|------|
| `query_kg(entity, depth, max_nodes)` | BFS 扩散查询：从种子实体出发获取关联 task（跨会话记忆核心） |
| `search_entities(query, limit)` | 模糊搜索实体名称 |
| `get_entity_info(entity)` | 实体详情：类型、别名、关联 task、出入边 |
| `graph_rag_search(query, top_k, use_graph)` | Graph-RAG 增强搜索（向量 + 图谱 + Reranker） |
| `get_kg_stats()` | 图谱统计：节点数、边数、平均 task 数 |

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 安装依赖
pip3 install mcp[cli] pyyaml --quiet

# 进入项目目录
cd ~/Desktop/oc_sess_graph

# 确保 Phase 5 数据已生成
ls output/triple/knowledge_graph.db
# 如果不存在: python3 code_p5_main.py --limit 10
```

#### 1. Claude Code 接入

```bash
# 方法 1: 命令行添加（推荐）
claude mcp add knowledge-graph \
  python3 /Users/zhaoxiuwei/Desktop/oc_sess_graph/code_mcp_server.py \
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
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/code_mcp_server.py"],
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
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/code_mcp_server.py"],
      "env": {
        "OPENCODE_ZEN_API_KEY": "<your-key>"
      }
    }
  }
}
```

#### 3. OpenCode 接入

```json
// 编辑项目根目录 .opencode/mcp.json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/code_mcp_server.py"],
      "env": {
        "OPENCODE_ZEN_API_KEY": "<your-key>"
      }
    }
  }
}
```

#### 4. QoderWork 接入

在 QoderWork 设置 → 连接器 → 自定义 MCP 中手动添加，粘贴配置：

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/code_mcp_server.py"],
      "cwd": "/Users/zhaoxiuwei/Desktop/oc_sess_graph",
      "env": {
        "OPENCODE_ZEN_API_KEY": "<your-key>"
      }
    }
  }
}
```

> Server 会自动以脚本所在目录为基准解析相对路径，`cwd` 可省略。

#### 5. MCP Inspector 调试

```bash
# 使用 MCP Inspector 交互式测试工具
npx @modelcontextprotocol/inspector python3 code_mcp_server.py

# 浏览器打开后：
# 1. 点击 "Tools" 标签查看 5 个工具定义
# 2. 选择 query_kg，填入 entity="OpenCode"，depth=2
# 3. 点击 "Call Tool" 查看返回结果
```

#### 6. HTTP 模式调试（备选）

```bash
# 以 streamable-http 模式启动（适合 curl 调试）
python3 code_mcp_server.py --http &
# 服务运行在 http://127.0.0.1:8000/mcp

# 或以 SSE 模式启动
python3 code_mcp_server.py --sse &
# SSE 端点: http://127.0.0.1:8000/sse
```

#### 7. Client CLI 验证

```bash
# 设置 API Key（context 命令需要 LLM 抽取实体）
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# BFS 查询（通过 code_mcp_client.py，调用 HTTP 版本）
python3 code_mcp_client.py query OpenCode 2

# 上下文注入演示
python3 code_mcp_client.py context "继续搞 RoPE 位置编码的优化"
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
Claude Code / Codex / OpenCode / QoderWork
  │
  ├─ MCP 协议 (stdio)
  │   ├─ initialize → server capabilities
  │   ├─ tools/list → 5 个工具定义 (JSON Schema)
  │   └─ tools/call → query_kg / search_entities / ...
  │                      └─ KGDatabase (SQLite)
  │
  └─ 自动工具发现
      └─ Agent 根据用户意图自动选择工具调用
          "继续搞 XX" → query_kg(entity="XX")
          "搜索 YY"  → search_entities(query="YY")
```

### 上下文压缩效果

| 指标 | 无记忆 | MCP 注入 |
|------|--------|----------|
| 上下文大小 | 0 tokens | ~200-500 tokens (3-5 task 摘要) |
| 跨 session 连续性 | 无 | Agent 自动调用 query_kg 关联历史 task |
| 工具调用延迟 | — | ~5-10ms (BFS 查询) |

当 task 数量增长到 100+，MCP 注入的上下文压缩比会更显著：从需要加载全部历史 session（~50K tokens）压缩到仅注入相关 task 摘要（~500 tokens），压缩率约 99%。

### 依赖

```
mcp[cli]    — MCP Python SDK (FastMCP + 传输层)
pyyaml      — YAML 配置解析
```

已有的项目依赖（无需额外安装）：`code_p5e_db.KGDatabase`、`code_p5e_graph_rag`（可选，graph_rag_search 需要）。
