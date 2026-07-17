## Phase 6: 跨 Session 总结

### 概述

基于 Phase 4 搜索 + Phase 5 知识图谱，对多个相关 session 的内容进行主题性总结。

流程：Query → SessionSearcher 检索 Top-K Task → (可选) Graph-RAG 图谱增强 → 收集 Task 关联 Chunk 内容 → 时间范围过滤 → Token 预算裁剪 → LLM 生成主题总结 → 返回结构化结果。

### 新增文件

| 文件 | 说明 |
|------|------|
| `code_p6_summarizer.py` | 核心模块：`SessionSummarizer` 类，含 `summarize`、`_search_with_graph`、`_filter_by_time`、`_gather_chunk_content`、`_call_llm` |
| `code_p6_cli.py` | CLI 入口：argparse 封装，支持 `--graph-rag`、`--time-from/to`、`--detail`、`--json` 等参数 |
| `code_p6_config.yaml` | Phase 6 配置：LLM 模型、token 预算、搜索默认参数、时间过滤 |

### 依赖

| 模块 | 用途 |
|------|------|
| `code_p1_utils` | `count_tokens`、`safe_truncate` |
| `code_p4_searcher` | `SessionSearcher`、`SessionSearchResult` |
| `code_p5e_db` | `KGDatabase`（可选，Graph-RAG 模式） |
| `code_p5e_graph_rag` | `GraphRAGSearcher`（可选，Graph-RAG 模式） |

### 配置

`code_p6_config.yaml`：

```yaml
summarizer:
  model: "nemotron-3-ultra-free"
  max_context_tokens: 12000
  max_tasks: 10
  max_output_tokens: 2000
  temperature: 0.3

search:
  top_k: 8
  use_graph_rag: false
  chunk_detail_level: "summary"   # summary | preview | full
```

### Chunk 详情级别

| 级别 | 说明 | Token 消耗 |
|------|------|-----------|
| `summary` | 仅 chunk 摘要（来自 `chunks_summary`） | 低 |
| `preview` | chunk 摘要 + 前 500 字原文 | 中 |
| `full` | chunk 完整 `cleaned_text()` | 高 |

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 进入项目目录
cd ~/Desktop/oc_sess_graph

# 确认前置数据就绪
ls output/tasks.jsonl           # Phase 2 产物
ls output/chunks_summary_p2.jsonl  # Phase 2 产物
# Qdrant 集合 tasks, chunks_summary, chunks_cleaned_text 已存在（Phase 3/4）
# (可选) output/triple/knowledge_graph.db 已存在（Phase 5e，Graph-RAG 需要）
```

#### 1. CLI 基本用法

```bash
# 基本查询
python3 code_p6_cli.py "我最近做过的 FlashAttention 相关工作"

# 指定 top-k 和 detail 级别
python3 code_p6_cli.py "序列并行和 FlashAttention 的关系" --top-k 5 --detail preview

# JSON 输出（方便程序消费）
python3 code_p6_cli.py "RoPE 位置编码" --json

# 不包含 chunk 原文（仅用 task 摘要，速度快）
python3 code_p6_cli.py "性能优化" --no-chunks
```

**预期输出**：

```
初始化 SessionSummarizer (config=code_p3_config.yaml)...
初始化完成 (2.3s)

查询: "序列并行和 FlashAttention 的关系"
参数: top_k=5, graph_rag=False, detail=summary
正在检索和生成总结...

======================================================================
  跨 Session 总结
  查询: "序列并行和 FlashAttention 的关系"
  来源: 5 个任务 | 2450 tokens | 118320ms
======================================================================

[LLM 生成的主题总结，300-800 字，包含 task_id 引用...]

----------------------------------------------------------------------
参与总结的任务:
  1. [FlashAttention 实现] score=0.892 (ses_1022f74...)
     实现了 FlashAttention 的 CUDA kernel...
  2. [序列并行优化] score=0.856 (ses_1022f74...)
     ...
```

#### 2. 时间范围过滤

```bash
# 指定时间范围
python3 code_p6_cli.py "性能优化工作" \
  --time-from 2026-07-01 --time-to 2026-07-15
```

#### 3. Graph-RAG 增强

```bash
# 启用图谱增强检索（需要 Phase 5e SQLite 数据库）
python3 code_p6_cli.py "opencode 有哪些功能" --graph-rag
```

Graph-RAG 会在向量检索基础上叠加图谱扩散：从查询中抽取实体 → 模糊匹配图谱 → BFS 扩散 → 收集关联 task → 与向量结果合并去重 → Reranker 排序。

#### 4. Python API 调用

```python
from code_p6_summarizer import SessionSummarizer

summarizer = SessionSummarizer(
    config_path="code_p3_config.yaml",
    max_context_tokens=12000,
)

# 基本总结
result = summarizer.summarize("我最近做过的 FlashAttention 相关工作")
print(result.summary)

# 带时间范围 + Graph-RAG
result = summarizer.summarize(
    query="性能优化",
    top_k=10,
    time_range=("2026-07-01", "2026-07-15"),
    use_graph_rag=True,
    chunk_detail_level="preview",
)

# 结构化结果
print(f"总结: {result.summary}")
print(f"来源: {len(result.sources)} 个任务")
print(f"Token: {result.total_tokens}, 耗时: {result.elapsed_ms}ms")
for s in result.sources:
    print(f"  [{s.task_label}] score={s.rerank_score:.3f} ({s.task_id})")
```

---

### 端到端测试结果

查询："序列并行和 FlashAttention 的关系"

| 指标 | 值 |
|------|------|
| 检索到的 Task 数 | 5 |
| 输入 Token 数 | 2450 |
| 总耗时 | 118s（含 Reranker 加载） |
| Chunk 详情级别 | summary |
| LLM 模型 | nemotron-3-ultra-free |

输出质量：LLM 生成了结构化的中文主题总结，按"序列并行原理 → FlashAttention 实现 → 两者协同关系"组织，包含具体 task_id 引用，覆盖了关键技术决策和实现细节。

### 架构

```
SessionSummarizer.summarize(query)
  │
  ├─ Stage 1: 检索
  │    ├─ use_graph_rag=False → SessionSearcher.search()
  │    └─ use_graph_rag=True  → GraphRAGSearcher.search()
  │
  ├─ Stage 2: 时间过滤
  │    └─ time_range 参数 → _filter_by_time()
  │
  ├─ Stage 3: 收集内容
  │    ├─ chunk_detail_level="summary" → chunks_summary 摘要
  │    ├─ chunk_detail_level="preview" → 摘要 + 前 500 字
  │    └─ chunk_detail_level="full"    → cleaned_text() 全文
  │    └─ Token 预算裁剪 (max_context_tokens)
  │
  └─ Stage 4: LLM 生成
       └─ nemotron-3-ultra-free → 300-800 字主题总结
```
