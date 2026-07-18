## Phase 5 Extension: SQLite 迁移 + Graph-RAG 集成

### 概述

本次扩展为 Phase 5 知识图谱新增两项能力：

1. **SQLite 持久化** — 将 NetworkX 图谱导出到 SQLite 数据库（nodes/edges 两张表），支持毫秒级查询、BFS 扩散、模糊搜索
2. **Graph-RAG 搜索** — 在 Phase 4 向量检索基础上，叠加图谱扩散逻辑：从查询中抽取实体 → BFS 扩散 → 收集关联 task → 与向量结果合并 → Reranker 排序

### 新增文件

| 文件 | 说明 |
|------|------|
| `code_p5e_db.py` | SQLite 持久化模块：`KGDatabase` 类，含 `import_graph`、`get_node`、`get_edges`、`bfs_expand`、`search_entities`、`get_top_entities`、`get_stats` |
| `code_p5e_graph_rag.py` | Graph-RAG 搜索模块：`extract_query_entities` 查询实体抽取、`GraphRAGSearcher` 增强搜索器 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `code_p5_kg_builder.py` | `save_graph()` 新增 `db_path` 和 `extraction_mode` 参数，自动导出到 SQLite |
| `code_p5_main.py` | 两个分支（entity/triple）都传入 `db_path`，末尾增加 SQLite 摘要输出 |
| `code_p5_config.yaml` | 新增 `sqlite` 和 `graph_rag` 配置段 |

### 配置

`code_p5_config.yaml` 新增配置段：

```yaml
# SQLite 持久化
sqlite:
  enabled: true
  # db_path 默认为 {output_dir}/knowledge_graph.db

# Graph-RAG 搜索
graph_rag:
  enabled: true
  bfs_depth: 1              # BFS 扩散深度
  max_expand_nodes: 30      # BFS 最大扩散节点数
  max_graph_tasks: 50       # 图谱扩散最多引入的 task 数
  graph_weight: 0.3         # 图谱扩散结果的权重加成
```

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 进入项目目录
cd ~/Desktop/oc_sess_graph
```

#### 1. 运行 triple 模式（生成 SQLite）

```bash
# 确认配置为 triple 模式
grep extraction_mode code_p5_config.yaml
# 预期: extraction_mode: "triple"

# 清理旧输出并运行 (10 个 task)
rm -rf output/triple
python3 code_p5_main.py --limit 10
```

**预期输出**（关键行）：

```
SQLite 导入完成: 97 节点, 81 边 → output/triple/knowledge_graph.db
SQLite 数据库: output/triple/knowledge_graph.db
  节点数: 97, 边数: 81
  平均每个实体关联 task 数: 1.05
```

**验证产物**：

```bash
ls -la output/triple/
# 预期文件:
#   knowledge_graph.db       (~53KB, SQLite 数据库)
#   knowledge_graph.gpickle  (~10KB, NetworkX 序列化)
#   knowledge_graph.json     (~34KB, JSON 格式)
#   entities.jsonl           (~15KB, 实体列表)
#   triples.jsonl            (~12KB, 三元组列表)
```

#### 2. 运行 entity 模式（生成 SQLite）

```bash
# 切换到 entity 模式
sed -i '' 's/extraction_mode: "triple"/extraction_mode: "entity"/' code_p5_config.yaml

# 清理旧输出并运行
rm -rf output/entity
python3 code_p5_main.py --limit 10
```

**预期输出**（关键行）：

```
SQLite 导入完成: 84 节点, 372 边 → output/entity/knowledge_graph.db
SQLite 数据库: output/entity/knowledge_graph.db
  节点数: 84, 边数: 372
  平均每个实体关联 task 数: 1.08
```

**验证产物**：

```bash
ls -la output/entity/
# 预期文件:
#   knowledge_graph.db       (~106KB, SQLite 数据库，共现边更多所以更大)
#   knowledge_graph.gpickle
#   knowledge_graph.json
#   entities.jsonl
#   entity_extract.jsonl
#   inverted_index.json
```

#### 3. SQLite 查询验证

```bash
python3 -c "
from code_p5e_db import KGDatabase

db = KGDatabase('output/triple/knowledge_graph.db')

# 统计
print(db.get_stats())
# {'db_path': 'output/triple/knowledge_graph.db', 'nodes': 97, 'edges': 81, ...}

# 查询节点
print(db.get_node('OpenCode'))
# {'name': 'OpenCode', 'entity_type': 'concept', 'source_tasks': [...], 'task_count': 6}

# 查询边
for e in db.get_edges('OpenCode', direction='out')[:5]:
    print(f'  {e[\"head\"]} --[{e[\"relation\"]}]--> {e[\"tail\"]}')

# BFS 扩散
expanded = db.bfs_expand(['OpenCode'], depth=1)
print(f'BFS(OpenCode): {len(expanded)} 节点 → {sorted(expanded)[:10]}')

# 模糊搜索
print(db.search_entities('RoPE', limit=3))
# [{'name': 'RoPE位置编码', ...}, ...]

# Top 实体
for e in db.get_top_entities(5):
    print(f'  {e[\"name\"]} ({e[\"entity_type\"]}) — {e[\"task_count\"]} tasks')
"
```

#### 4. Graph-RAG 端到端验证

```bash
python3 -c "
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import extract_query_entities

db = KGDatabase('output/triple/knowledge_graph.db')

# 测试查询实体抽取
queries = [
    'token过期bug怎么排查',
    'opencode有哪些功能和命令',
    'RoPE位置编码实现细节',
    'BF16训练精度问题',
]
for q in queries:
    entities = extract_query_entities(q)
    # 模糊匹配到图谱节点
    seed = set()
    for e in entities:
        if db.get_node(e):
            seed.add(e)
        else:
            for m in db.search_entities(e, limit=3):
                seed.add(m['name'])
    # BFS 扩散
    expanded = db.bfs_expand(list(seed), depth=1, max_nodes=20)
    # 收集关联 task
    task_ids = set()
    for ent in expanded:
        task_ids.update(db.get_tasks_by_entity(ent))
    print(f'  \"{q}\"')
    print(f'    实体: {entities} → 命中图谱: {len(seed)} → 扩散: {len(expanded)} → 关联 task: {len(task_ids)}')
"
```

**预期输出**：

```
  "token过期bug怎么排查"
    实体: [...] → 命中图谱: N → 扩散: M → 关联 task: K
  "opencode有哪些功能和命令"
    实体: ['opencode'] → 命中图谱: 3 → 扩散: 19 → 关联 task: 4
  "RoPE位置编码实现细节"
    实体: ['RoPE', '位置编码', '旋转位置编码'] → 命中图谱: 1 → 扩散: N → 关联 task: K
  "BF16训练精度问题"
    实体: ['BF16', '训练精度'] → 命中图谱: 1 → 扩散: N → 关联 task: K
```

#### 5. Graph-RAG 完整搜索（需 Phase 4 数据就绪）

```bash
python3 -c "
from code_p4_searcher import SessionSearcher
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher

# 初始化
searcher = SessionSearcher('code_p3_config.yaml')
kg_db = KGDatabase('output/triple/knowledge_graph.db')
rag = GraphRAGSearcher(searcher, kg_db, bfs_depth=1, graph_weight=0.3)

# 搜索
results, debug = rag.search('opencode有哪些命令和功能', top_k=5)
for r in results:
    print(f'  [{r.rerank_score:.3f}] {r.task_label}')
print(f'调试信息: {debug}')
"
```

**预期输出**：搜索结果中包含通过图谱扩散发现的关联 task（纯向量检索可能遗漏的），`debug["source_distribution"]` 显示来源分布。

---

### 两种模式对比（10 tasks 基准）

| 指标 | triple 模式 | entity 模式 |
|------|------------|------------|
| 总耗时 | ~167s | ~97s |
| 实体数（对齐后） | 97 | 84 |
| 边数 | 81（关系边） | 372（共现边） |
| SQLite 大小 | ~53KB | ~106KB |
| BFS(OpenCode) 扩散 | 14 节点 | 20 节点 |
| 适用场景 | 精确关系推理 | 快速关联发现 |

### SQLite 表结构

```sql
CREATE TABLE nodes (
    name TEXT PRIMARY KEY,
    entity_type TEXT DEFAULT '',
    aliases TEXT DEFAULT '[]',        -- JSON array
    source_tasks TEXT DEFAULT '[]',   -- JSON array of task_ids
    task_count INTEGER DEFAULT 0
);

CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    head TEXT NOT NULL,
    tail TEXT NOT NULL,
    relation TEXT DEFAULT '',
    weight REAL DEFAULT 1.0,
    source_task TEXT DEFAULT '',
    extraction_mode TEXT DEFAULT 'triple',  -- 'triple' | 'entity'
    FOREIGN KEY (head) REFERENCES nodes(name),
    FOREIGN KEY (tail) REFERENCES nodes(name)
);

-- 索引
CREATE INDEX idx_edges_head ON edges(head);
CREATE INDEX idx_edges_tail ON edges(tail);
CREATE INDEX idx_edges_relation ON edges(relation);
CREATE INDEX idx_nodes_task_count ON nodes(task_count DESC);
```

### Graph-RAG 搜索流程

```
Query ("opencode有哪些命令")
  │
  ├─ Stage 1: 向量检索 (Phase 4 SessionSearcher)
  │    Dense + BM25 → RRF → 候选池 (top_k * 2)
  │
  ├─ Stage 2: 图谱扩散
  │    ├─ LLM 实体抽取: ["opencode"]
  │    ├─ 模糊匹配图谱: ["OpenCode", "opencode.db", "opencode session"]
  │    ├─ BFS 扩散 (depth=1): 19 个关联实体
  │    └─ 收集 task_ids: 4 个关联 task
  │
  ├─ Stage 3: 合并候选池
  │    向量结果 ∪ 图谱结果 (去重, 图谱命中加分)
  │
  └─ Stage 4: Reranker 排序 → Top-K
```

### VEC-RAG vs Graph-RAG 实测对比（10 tasks, triple 模式）

使用 5 条查询对比纯向量检索 (VEC-RAG) 与图谱增强检索 (Graph-RAG) 的效果：

| 查询 | VEC-RAG Top-3 | Graph-RAG Top-3 | 差异 |
|------|--------------|----------------|------|
| opencode命令 | session_info, session_list, opencode_auth | opencode_auth, session_list, session_info | Graph-RAG 额外发现 2 个向量遗漏的 task（免费模型清单、session列表） |
| RoPE位置编码 | rope_impl, model_debug, bf16_precision | 同 VEC-RAG | 无差异（向量已经很强） |
| BF16训练精度 | bf16_precision, model_debug, rope_impl | 同 VEC-RAG | 无差异 |
| session管理 | session_list, session_info, opencode_auth | 同 VEC-RAG | 无差异 |
| opencode配置 | opencode_auth, session_info, session_list | 同 VEC-RAG | 无差异 |

**结论**：在 10 task 规模下，Graph-RAG 仅对 1/4 条查询有增量贡献（发现向量检索遗漏的关联 task）。当图谱规模增长到 100+ task、实体关系更密集时，BFS 扩散的跨跳关联优势预计会更显著。

#### 对比测试代码

```bash
python3 -c "
from code_p4_searcher import SessionSearcher
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher

searcher = SessionSearcher('code_p3_config.yaml')
kg_db = KGDatabase('output/triple/knowledge_graph.db')
rag = GraphRAGSearcher(searcher, kg_db, bfs_depth=1, graph_weight=0.3)

queries = [
    'opencode命令',
    'RoPE位置编码',
    'BF16训练精度',
    'session管理',
    'opencode配置',
]
for q in queries:
    # VEC-RAG: 纯向量 (skip graph)
    vec_results, _ = rag.search(q, top_k=5, use_graph_rag=False)
    # Graph-RAG: 向量 + 图谱扩散
    graph_results, debug = rag.search(q, top_k=5, use_graph_rag=True)
    print(f'查询: {q}')
    print(f'  VEC-RAG:   {[r.task_label for r in vec_results[:3]]}')
    print(f'  Graph-RAG: {[r.task_label for r in graph_results[:3]]}')
    print(f'  来源: {debug.get(\"source_distribution\", {})}')
    print()
"
```

### Bug 修复记录

开发过程中遇到并修复的问题：

| 问题 | 原因 | 修复 |
|------|------|------|
| `ImportError: cannot import name 'call_llm'` | `code_p1_utils` 中不存在 `call_llm` 函数 | 改用 `openai.OpenAI` 直接创建客户端，通过 `_get_client()` 辅助函数 |
| `AttributeError: 'KGBuilder' has no attribute 'stats'` | `save_graph` 引用了不存在的 `self.stats.extraction_mode` | 新增 `extraction_mode` 参数，由调用方显式传入 |
| `ModelError: Model qwen3-235b-a22b is not supported` | OpenCode Zen API 不支持 qwen3 系列模型 | 改用 `nemotron-3-ultra-free`，新增 `DEFAULT_MODEL` 常量 |
| `ValueError: empty separator` | `content.split("")` 使用空字符串作分隔符 | 改用 `re.sub()` 去除 think 标签 |
| `TypeError: search() got unexpected keyword argument 'chunk_top_k'` | `SessionSearcher.search()` 签名为 `(query, top_k, candidate_multiplier, skip_rerank)` | 简化为 `search(query, top_k, skip_rerank=True)` |
| `AttributeError: 'Qwen3Reranker' has no attribute 'rerank'` | Reranker 方法名为 `rank()` 而非 `rerank()` | 改为 `self.searcher.reranker.rank(query, texts, top_k=...)` |
| `AttributeError: 'NoneType' has no attribute 'strip'` | LLM 偶尔返回 `message.content = None` | 在 `.strip()` 前加 None 检查，为空时跳过当前重试 |
