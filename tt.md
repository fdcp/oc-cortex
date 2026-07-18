# 端到端测试记录

日期: 2026-07-18
环境: macOS (darwin arm64), Python 3.10, 无 venv (系统 Python)

## 测试前状态

```bash
cd ~/Desktop/oc_sess_graph
# 仓库刚完成目录重组: src/, config/, doc/, prompts/, tests/
# 所有 Python 文件已从根目录移到 src/
# 所有 YAML 配置已移到 config/
# 源码中的默认配置路径已从 code_pX_config.yaml 改为 config/code_pX_config.yaml
# MCP Server 的 _project_root 已修正为 Path(__file__).parent.parent
```

---

## Phase 1: 数据预处理

### 命令

```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p1_main.py --config config/code_p1_config.yaml --sqlite ~/.local/share/opencode/opencode.db 2>&1
```

### 说明

config/code_p1_config.yaml 中 `mock_data: true`，直接运行 `python3 src/code_p1_main.py` 会生成 mock 数据（仅 4 chunks）。
需要加 `--sqlite` 参数从真实 OpenCode 数据库加载。

### 结果: 成功

```
Phase 1 完成
  输入 session: 16
  过滤后 session: 16
  生成 chunk: 83
  输出文件: ./output/chunks.jsonl
```

### 关键日志

- 从 SQLite 查询到 21 个 session，加载 16 个（跳过 5 个）
- 各 session chunk 数: 2~13 不等
- 后处理: 合并空 chunk + 去重 user_message (3 个 session 触发)
- 示例 chunk: `ses_12c6bd8dfffevT51PypMW2v5Mx` 的 token 占比 raw=245 → cleaned=173

---

## Phase 2: 任务提取

### 命令

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
python3 src/code_p2_main.py --config config/code_p2_config.yaml 2>&1
```

### 结果: 成功

```
Phase 2 完成
  输入 session: 16
  生成 task: 29
  chunk 含总结: 83/83
  输出文件: ./output/tasks.jsonl
  chunk 总结: ./output/chunks_summary_p2.jsonl
  平均每 session task 数: 1.8
```

### 关键日志

- 并发 4 个 LLM 调用 (nemotron-3-ultra-free via OpenCode Zen API)
- 29 个 task 分布: 大多数 session 1-2 个 task，最多 3 个
- 示例 task: "RoPE预计算函数解析", "本地opencode会话清查", "mavis-branch溯源及MiniMax Code关系确认"

---

## Phase 3: 向量存储

### 命令

```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p3_main.py --config config/code_p3_config.yaml 2>&1
```

### 结果: 成功

```
Phase 3 完成 (总耗时 5.1s)
  输入 task: 29, upsert: 29
  输入 chunk: 83
  集合 'tasks': vectors=30, points=30, status=green
  集合 'chunks_summary': vectors=83, points=83, status=green
  集合 'chunks_cleaned_text': vectors=83, points=83, status=green
```

### 关键日志

- Dense embedding: BAAI/bge-small-zh-v1.5 (CPU, 从缓存加载)
- Embedding 分 3 批: 32+32+19 = 83 向量, 耗时 2.6s
- BM25 索引: jieba 分词, 83 条文档, 耗时 0.50s
- 全部 upsert 完成

---

## Phase 4: 跨 Session 搜索

### 命令

```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p4_search_cli.py --config config/code_p3_config.yaml --query "FlashAttention 实现原理" --top-k 3 2>&1
```

### 结果: 成功

输出 3 个匹配 task，每个包含关联 chunk 的摘要和预览:
1. **SDPA/FlashAttention 理论与实现深度分析** — score 最高，包含 13 个 chunk
2. **序列并行(SP)完整流程图补全与v3渲染优化** — 相关 chunk 涉及 TP/SP 对比
3. 其他相关结果

### 关键日志

- Dense + BM25 → RRF 融合 → Qwen3-Reranker-0.6B 重排序
- 搜索结果包含 chunk 摘要 + 前 200 字预览

---

## Phase 5: 知识图谱 (Triple 模式)

### 命令

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p5_main.py --config config/code_p5_config.yaml --skip-alignment 2>&1
```

### 说明

使用 `--skip-alignment` 跳过实体对齐（加速测试，之前已对齐过）。
完整运行应不加此参数。

### 结果: 成功

```
Phase 5 总耗时: 14.3s

SQLite 数据库: output/triple/knowledge_graph.db
  节点数: 337, 边数: 765
  平均每个实体关联 task 数: 1.06

Top-10 高度节点:
  优化器学习指南 (degree=14)
  opencode (degree=11)
  DataCollatorForSeq2Seq (degree=8)
  OpenCode (degree=7)
  Bridge (degree=7)
```

### 关键日志

- 从 29 个 task 中提取三元组 (LLM 并发调用)
- 示例三元组: `(BF16训练指南v3) --[缺少]--> (梯度累积)`, `(BF16训练指南v4) --[补充了]--> (训练加速三件套)`
- 自动导出 SQLite: 337 nodes, 765 edges

---

## Phase 5e: SQLite + Graph-RAG (内置在 Phase 5 中)

### 验证

Phase 5 自动导出 SQLite 数据库，通过 Phase 6b 的 trace 命令验证:

```bash
python3 src/code_p6b_cli.py trace "FlashAttention" 2>&1
```

### 结果: 成功

```
决策链 [FlashAttention] (4 步, 2 跳)

### 依赖类
  - FlashAttention → Compute-Bound (属于, 1跳)
  - Compute-Bound ← FlashAttention (属于, 2跳)

### 选型类
  - FlashAttention ← PyTorch SDPA (对比了, 1跳)
  - PyTorch SDPA → FlashAttention (对比了, 2跳)
```

---

## Phase 6: 跨 Session 总结

### 命令

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p6_cli.py "FlashAttention 相关的工作" --top-k 3 --detail summary 2>&1
```

### 结果: 成功

输出一份完整的主题总结（约 1000 字），包含:
- 5 个章节: 分析背景 → Roofline 模型 → 在线 Softmax → FA-2 并行策略 → 技术决策点回顾
- task_id 引用: `ses_0c5b6171bfferME7He2F8G6UuJ_T2`
- 后续建议: H100 适配, Ring Attention 混合, 稀疏 Attention, 反向传播权衡

### 关键日志

- 检索 3 个 task (Dense+BM25→RRF→Reranker)
- 仅 1 个 task 与 FlashAttention 真正相关，LLM 正确识别并聚焦
- chunk_detail_level=summary: 使用 chunk 摘要作为 LLM 输入

---

## Phase 6b: 决策溯源 + 骨架总结

### 6b-1: 决策溯源

```bash
python3 src/code_p6b_cli.py trace "FlashAttention" 2>&1
```

**结果: 成功** (已在 Phase 5e 验证中展示)

### 6b-2: 实体搜索

```bash
python3 src/code_p6b_cli.py search "序列" --limit 5 2>&1
```

**结果: 成功**

```
  序列并行 [concept] task_count=2
  warp粒度的序列并行 [concept] task_count=1
  跨设备长序列训练提供了直观理解 [concept] task_count=1
  序列长度 [concept] task_count=1
  长序列训练性能 10-30% [concept] task_count=1
```

### 6b-3: 骨架总结

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 3 2>&1
```

**结果: 成功**

输出结构化总结，包含:
- 图谱结构: FlashAttention 节点 + 决策关系 (→ Compute-Bound, ← PyTorch SDPA)
- 5 个章节:
  1. 核心决策背景: Memory-Bound → Compute-Bound 范式转移
  2. 选型对比决策: FlashAttention vs PyTorch SDPA (含后端分发逻辑)
  3. 代际演进决策: FA-1 → FA-2 架构重构 (含对比表格)
  4. 关键技术细节: 在线 Softmax 数值稳定性
  5. 整体评价与后续建议 (4 条具体建议)
- 参与 task: 1 个 (SDPA/FlashAttention 理论与实现深度分析)

**与 Phase 6 对比**: Phase 6b 骨架总结多了决策因果链和选型对比维度，输出更结构化。

---

## MCP Server

### 命令

```bash
export TRANSFORMERS_OFFLINE=1 && export HF_HUB_OFFLINE=1
python3 -c "
import sys; sys.path.insert(0, 'src')
from code_mcp_server import get_db
db = get_db()
stats = db.get_stats()
print(f'MCP Server DB: {stats[\"nodes\"]} 节点, {stats[\"edges\"]} 边')
print('MCP Server 模块加载验证通过')
" 2>&1
```

### 结果: 成功

```
MCP Server DB: 337 节点, 765 边
MCP Server 模块加载验证通过
```

### 说明

MCP Server 使用 stdio transport，由客户端 (Claude Code / QoderWork) 启动。
此处仅验证模块加载和 DB 连接，完整测试需通过 MCP 客户端发起 JSON-RPC 调用。

---

## 测试汇总

| Phase | 命令 | 结果 | 耗时 | 产物 |
|-------|------|------|------|------|
| P1 | `python3 src/code_p1_main.py --sqlite ...` | 成功 | <1s | 83 chunks |
| P2 | `python3 src/code_p2_main.py` | 成功 | ~4min | 29 tasks + 83 summaries |
| P3 | `python3 src/code_p3_main.py` | 成功 | 5.1s | Qdrant 3 集合 (30+83+83) |
| P4 | `python3 src/code_p4_search_cli.py --query ...` | 成功 | ~3s | 搜索结果 |
| P5 | `python3 src/code_p5_main.py --skip-alignment` | 成功 | 14.3s | 337 nodes, 765 edges, SQLite DB |
| P5e | (通过 P6b trace 验证) | 成功 | <1ms | SQLite 查询正常 |
| P6 | `python3 src/code_p6_cli.py "..." --top-k 3` | 成功 | ~30s | 主题总结 |
| P6b-trace | `python3 src/code_p6b_cli.py trace "FlashAttention"` | 成功 | <1ms | 4 步决策链 |
| P6b-search | `python3 src/code_p6b_cli.py search "序列"` | 成功 | <1ms | 5 个实体 |
| P6b-summary | `python3 src/code_p6b_cli.py summary "FlashAttention"` | 成功 | ~20s | 骨架总结 |
| MCP | Python import + DB 加载 | 成功 | <1s | 337 nodes, 765 edges |

## 遇到的问题

### 问题 1: Phase 1 mock 数据覆盖

**现象**: 直接运行 `python3 src/code_p1_main.py` 生成 mock 数据（4 chunks），覆盖了真实的 83 chunks。
**原因**: `config/code_p1_config.yaml` 中 `mock_data: true`。
**修复**: 使用 `--sqlite ~/.local/share/opencode/opencode.db` 参数从真实数据加载。
**建议**: 默认配置应设置 `mock_data: false`，或增加覆盖确认提示。

### 无其他问题

目录重组后所有 import 路径正常，配置文件路径正确更新为 `config/` 前缀，
MCP Server 的 `__file__` 路径解析正确（`.parent.parent` 回到项目根）。
