## MCP Server 封装：知识图谱工具服务

### 概述

将 Phase 5 知识图谱（SQLite）封装为 HTTP API 服务，提供 MCP 兼容的工具发现和调用接口，供 OpenCode 或其他 AI Agent 直接集成，实现跨会话记忆。

### 新增文件

| 文件 | 说明 |
|------|------|
| `code_mcp_server.py` | FastAPI MCP Server：5 个工具端点 + MCP 协议兼容层 + 健康检查 |
| `code_mcp_client.py` | MCP Client：HTTP 客户端 + 实体抽取 + `ContextInjector` 上下文注入器 |
| `code_mcp_config.yaml` | 服务配置：端口、数据库路径、上下文注入参数 |

### MCP 工具列表

| 工具 | 方法 | 说明 |
|------|------|------|
| `query_kg` | GET `/query_kg` | BFS 扩散查询：从种子实体出发，获取关联 task（核心工具） |
| `search_entities` | GET `/search_entities` | 模糊搜索实体名称 |
| `get_entity_info` | GET `/get_entity_info` | 实体详情：类型、别名、关联 task、出入边 |
| `graph_rag_search` | GET `/graph_rag_search` | Graph-RAG 增强搜索（向量 + 图谱 + Reranker） |
| `get_stats` | GET `/get_stats` | 图谱统计信息 |

MCP 协议端点：

| 端点 | 说明 |
|------|------|
| GET `/mcp/tools/list` | 工具发现：返回所有工具定义（JSON Schema） |
| POST `/mcp/tools/call` | 工具调用：统一入口 `{"tool": "query_kg", "arguments": {...}}` |

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 安装依赖
pip3 install fastapi uvicorn requests pyyaml --quiet

# 设置 API Key（如需 graph_rag_search）
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 进入项目目录
cd ~/Desktop/oc_sess_graph

# 确保 Phase 5 数据已生成
ls output/triple/knowledge_graph.db
# 如果不存在，先运行: python3 code_p5_main.py --limit 10
```

#### 1. 启动 MCP Server

```bash
# 后台启动
python3 code_mcp_server.py &
# 或: uvicorn code_mcp_server:app --host 0.0.0.0 --port 8000 &

# 等待启动完成（约 2 秒）
sleep 2

# 健康检查
curl -s http://localhost:8000/health | python3 -m json.tool
# 预期: {"status": "ok", "db_exists": true, "db_path": "output/triple/knowledge_graph.db", "tools_count": 5}
```

#### 2. 工具发现

```bash
# 列出所有 MCP 工具
curl -s http://localhost:8000/mcp/tools/list | python3 -m json.tool
# 预期: 返回 5 个工具定义，每个包含 name, description, inputSchema
```

#### 3. 核心工具：query_kg（BFS 扩散查询）

```bash
# 从 "OpenCode" 实体出发，BFS 扩散 2 跳
curl -s "http://localhost:8000/query_kg?entity=OpenCode&depth=2" | python3 -m json.tool

# 预期输出（关键字段）:
# {
#   "seed_entity": "OpenCode",
#   "depth": 2,
#   "expanded_entities": ["OpenCode", "opencode.db", "session管理", ...],
#   "related_tasks": [
#     {"task_id": "...", "task_label": "opencode_auth", "task_summary": "..."},
#     ...
#   ],
#   "elapsed_ms": 5
# }
```

#### 4. 模糊搜索实体

```bash
curl -s "http://localhost:8000/search_entities?query=RoPE" | python3 -m json.tool
# 预期: 匹配到 "RoPE位置编码" 等实体
```

#### 5. MCP 协议调用

```bash
# 通过 MCP 统一入口调用
curl -s -X POST http://localhost:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "query_kg", "arguments": {"entity": "OpenCode", "depth": 1}}' \
  | python3 -m json.tool

# 预期: {"tool": "query_kg", "result": {...}, "elapsed_ms": ...}
```

#### 6. Client CLI 验证

```bash
# 健康检查
python3 code_mcp_client.py health

# 列出工具
python3 code_mcp_client.py tools

# BFS 查询
python3 code_mcp_client.py query OpenCode 2

# 模糊搜索
python3 code_mcp_client.py search RoPE

# 上下文注入（核心功能）
python3 code_mcp_client.py context "继续搞 RoPE 位置编码的优化"
# 预期输出:
#   查询: 继续搞 RoPE 位置编码的优化
#   实体: ['RoPE', '位置编码', '优化'] → 命中: ['RoPE位置编码']
#   关联 task: N
#   === 注入上下文 ===
#   ## 跨会话记忆（来自知识图谱）
#   基于查询中识别的实体 [RoPE位置编码]，以下是相关的历史任务上下文：
#   1. **rope_impl** (session: ...)
#      RoPE 旋转位置编码的实现细节...
```

#### 7. OpenCode 集成验证

```bash
python3 -c "
from code_mcp_client import init_opencode, on_new_session

# Step 1: 初始化（注册 MCP 工具）
tools = init_opencode()
print(f'注册工具: {[t[\"name\"] for t in tools]}')

# Step 2: 新 session 启动 → 自动注入上下文
context = on_new_session('继续搞 RoPE 位置编码的优化')
if context:
    print('注入上下文:')
    print(context[:300])
else:
    print('(无相关上下文)')
"
```

#### 8. 停止服务

```bash
# 查找并终止
kill $(lsof -ti:8000)
# 或: pkill -f "code_mcp_server"
```

---

### 架构

```
OpenCode / AI Agent
  │
  ├─ 新 session 启动
  │   └─ on_new_session(query)
  │       ├─ extract_session_entities(query) → ["RoPE", "位置编码"]
  │       ├─ MCPClient.query_kg("RoPE位置编码", depth=1)
  │       │   └─ HTTP GET /query_kg
  │       └─ ContextInjector._format_context() → 注入 system prompt
  │
  └─ MCP 工具调用
      └─ POST /mcp/tools/call
          └─ 分发到 query_kg / search_entities / graph_rag_search / ...
                └─ KGDatabase (SQLite)
```

### 上下文压缩效果

在 10 task 测试集上的效果估算：

| 指标 | 无记忆 | MCP 注入 |
|------|--------|----------|
| 上下文大小 | 0 tokens | ~200-500 tokens (3-5 task 摘要) |
| 跨 session 连续性 | 无 | 自动关联历史 task |
| 首次响应延迟 | — | +50-100ms (BFS 查询) |

当 task 数量增长到 100+，MCP 注入的上下文压缩比会更显著：从需要加载全部历史 session（~50K tokens）压缩到仅注入相关 task 摘要（~500 tokens），压缩率约 99%。

### 依赖

```
fastapi     — HTTP 框架
uvicorn     — ASGI 服务器
requests    — HTTP 客户端（code_mcp_client.py）
pyyaml      — YAML 配置解析
```

已有的项目依赖（无需额外安装）：`code_p5e_db.KGDatabase`、`code_p5e_graph_rag`（可选，graph_rag_search 需要）。
