# GraphRAG 调试分析报告

> 分支: `debug_graphRAG`
> 调试数据: `logs/debug_graph_rag/*.json` (5 个 query, triple 模式)
> 脚本: `src/debug_graph_rag.py` (记录) + `src/debug_graph_rag_analyze.py` (聚合分析)

## 1. 调试方法

为复现 `code_p5e_graph_rag.GraphRAGSearcher.search` 流程的中间量, 不修改源码,
新增 `src/debug_graph_rag.py` 通过公共 API + 中间量采样, 在 9 个关键阶段打点:

| Stage | 内容 | 数据来源 |
|-------|------|----------|
| 1 | LLM 实体抽取 | `extract_query_entities()` 返回值 + 耗时 |
| 2 | 实体模糊匹配 | `KGDatabase.get_node()` + `search_entities()` |
| 3 | BFS 扩散 | `KGDatabase.bfs_expand()` |
| 4 | 反 IDF 累加 | 复刻 `math.log(1 + N / tc)` 累加, 记录每个实体的贡献 |
| 5 | 向量检索 | `SessionSearcher.search(skip_rerank=True)` |
| 6+7 | 外层 RRF + Rerank | `GraphRAGSearcher.search()` + `final_debug` |
| 8 | 来源分布 | 自行对照 `vector_results`/`graph_candidates` 集合 |
| 9 | 耗时 | `final_debug["vector_time_ms"/"graph_time_ms"/"rerank_time_ms"/"total_time_ms"]` |

每次 query 写到 `logs/debug_graph_rag/{db}_{query_slug}.json`, 同时追加一行到 `_summary.jsonl`。

## 2. 测试集 (5 个 query)

| Query | 设计意图 |
|-------|----------|
| `OpenCode` | 命中 graph 的 hub 实体 (task_count=17) |
| `OpenCode MCP` | 双实体, 验证多实体融合 |
| `FlashAttention 实现原理` | 技术长尾, 验证稀疏召回 |
| `知识图谱 实体对齐` | 中文 + 复合概念, 验证中文抽取 |
| `Qwen3 Reranker 精排` | 不在 graph 的新概念, 验证退化路径 |

## 3. 关键阶段结果 (triple DB, 1211 节点 / 3353 边)

### 3.1 Stage 1 — LLM 实体抽取

| 指标 | 值 |
|------|-----|
| 抽取成功率 | **5/5 (100%)** |
| 抽取实体数 (min/max/mean) | 1 / 2 / **1.6** |
| 平均耗时 | **19,575 ms** (占总时延 ~47%) |
| 模型 | `hy3` (OpenCode Zen) |

**观察**: 抽取稳健, 全部 query 都拿到 1-2 个实体。耗时是 GraphRAG 的最大瓶颈之一。

### 3.2 Stage 2 — 实体模糊匹配

| Query | 抽取 | 精确命中 | 模糊匹配 | 种子实体数 |
|-------|------|----------|----------|------------|
| `OpenCode` | OpenCode | ✅ | 0 | 1 |
| `OpenCode MCP` | OpenCode, MCP | 2 ✅ | 0 | 2 |
| `FlashAttention 实现原理` | FlashAttention | 0 | **1 (FlashAttention)** | 1 |
| `知识图谱 实体对齐` | 知识图谱, 实体对齐 | 0 | **2 (两个都模糊命中)** | 2 |
| `Qwen3 Reranker 精排` | Qwen3 Reranker, 精排 | 0 | 0 ❌ | **0 (退化)** |

**观察**:
- 精确命中需要大小写、符号完全一致; 中英混排、含空格的复合概念几乎都走模糊
- `Qwen3 Reranker 精排` 完全没命中 → 退化为纯向量路径, Top-K 全部 vector-only
- 建议: 抽取后大小写归一化、移除空格后再做精确匹配 (减少对 LIKE 的依赖)

### 3.3 Stage 3 — BFS 扩散

| 指标 | 值 |
|------|-----|
| 扩散节点数 (min/max/mean) | 0 / 30 / 17.8 |
| 命中 `max_expand_nodes=30` | **2/5 (40%)** |

**问题**: `OpenCode` 类 hub 节点 1-hop 出度大, 频繁触发 `max_nodes` 截断。
**建议**:
- 扩散时按 `task_count` 升序 (稀有优先) 可保留更长的尾部
- 或对 hub 节点 (`task_count > threshold`) 在 BFS 内部降权
- 或增加 `bfs_topological` 选项: 优先扩散叶子

### 3.4 Stage 4 — 反 IDF 累加

| Query | global N | unique graph tasks | 截断后 |
|-------|----------|-------------------|--------|
| `OpenCode` | 76 | 23 | 23 |
| `OpenCode MCP` | 76 | 24 | 24 |
| `FlashAttention` | 76 | 15 | 15 |
| `知识图谱` | 76 | 8 | 8 |
| `Qwen3 Reranker` | 76 | 0 | 0 (退化) |

**OpenCode 的实体贡献 (task_count→idf):**

| 实体 | task_count | idf = log(1 + 76/tc) |
|------|------------|---------------------|
| OpenCode | **17** | 1.70 (hub, 被压低 ✓) |
| opencode.db | 4 | 3.04 |
| /compact | 1 | 4.34 |
| ...其余 27 个都是 task_count=1 → 4.34 |

**观察**: 反 IDF 公式工作正确, hub (OpenCode) 贡献 1.70, 长尾 (task_count=1) 贡献 4.34。
**潜在问题**: 长尾太"平"了 — 90% 实体都是 1 task, 4.34 一致。
**建议**: 可以叠加 `log(1 + depth)` 或扩展到 2-hop 但更严格的"尾部优先"截断。

### 3.5 Stage 5 — 向量检索

| 指标 | 值 |
|------|-----|
| 候选数 (top_k × rerank_multiplier) | 50 (top_k=5, multiplier=2.0, 但 `max()` 保证 ≥ top_k) |

**注**: `rerank_multiplier=2.0` 实际产出 50 候选, 远超 top_k=10。这是因为 `max(int(top_k*2), top_k)` 是 10, 但 `searcher.search` 内部有自己的 `candidate_multiplier=5`, 所以 10×5=50。这是预期行为, 但命名"rerank_multiplier"会误导, 实际是 searcher 内部 N 倍。

### 3.6 Stage 6 — 外层 RRF 融合

| 参数 | 值 |
|------|-----|
| `outer_rrf_k` | 30 |
| 融合后候选 | 30-51 (vector 50 + graph 0-25) |
| `graph_channel_weight` | 0.3 (vector 拿 0.7) |

**观察**: graph-only=1/25 极低, 但 both=10/25 (40%) 高, 说明 graph 的主要价值是**给 vector 已召回的 task 加权**, 而不是引入新 task。

### 3.7 Stage 7 — Rerank

| 指标 | 值 |
|------|-----|
| 输入候选 | 30-51 (外层 RRF 后) |
| 输出 | top_k (5) |
| 平均耗时 | **18,656 ms** (Qwen3-Reranker-0.6B on MPS) |

**瓶颈**: 17-19s/query, 占总时延 60%+。Qwen3-Reranker-0.6B 是 CausalLM 一次 forward, 文档越长越慢。
**建议**:
- 限制 rerank 输入: 最多 N (而不是全 50 候选)
- 改用 cross-encoder (如 BGE-reranker) 速度会快 5-10x
- 缓存 (相同 query + topK 候选 hash) — 同一会话内可复用

### 3.8 Stage 8 — 来源分布 (Top-K)

| 来源 | 数量 | 占比 |
|------|------|------|
| vector-only | 14 | 56% |
| graph-only | 1 | 4% |
| **both (双通道)** | **10** | **40%** |
| neither | 0 | 0% |

**关键结论**: 图谱通道**有效** (40% 双通道), 但**几乎不独立贡献** (graph-only 4%)。
图谱的价值 ≈ "给 vector 已召回的 task 加权, 提升它们进入 Top-K 的概率"。

### 3.9 Stage 9 — 耗时分布

| 阶段 | mean (ms) | mean (%) | 备注 |
|------|----------|----------|------|
| LLM 实体抽取 | 19,575 | 47% | hy3 reasoning 模型 |
| Rerank | 18,656 | 45% | Qwen3-Reranker-0.6B on MPS |
| 向量检索 | 11 | 0.03% | BGE-small-zh CPU, 50 候选 |
| BFS + IDF 累加 | < 10 | < 0.05% | SQLite 极快 |
| **合计 (典型 query)** | ~38,000 | 100% | |

**注**: 上表的"graph_time"在 `final_debug` 里实际是 `LLM 抽取 + BFS + IDF` 总和, 真实 BFS+IDF 部分 < 10ms。

## 4. 一个代表性 query 的完整链路

`triple_OpenCode.json` 详细复盘:

```
Query: "OpenCode"
  ↓ [Stage 1] LLM 18s
  实体: ['OpenCode']
  ↓ [Stage 2] 1ms
  精确命中: OpenCode (task_count=17, hub)
  ↓ [Stage 3] 1ms
  BFS 1-hop: 30 节点 (命中 max_nodes 上限)
  ↓ [Stage 4] 1ms
  反 IDF 累加: 23 unique tasks
  ↓ [Stage 5] 向量检索 10ms
  50 候选 (vector 通道), top5 都是 OpenCode 相关
  ↓ [Stage 6] 外层 RRF
  50 vector + 23 graph → 51 unique, 排序
  ↓ [Stage 7] Rerank 17.5s
  Qwen3-Reranker 精排 → top5
  ↓ [Stage 8] 来源分布
  vector=1, graph=0, both=4, neither=0
  ↓ [Stage 9] 总耗时 28.9s
```

Top-5 最终结果:

| # | source | rerank | hybrid | v_rank | g_rank | task |
|---|--------|--------|--------|--------|--------|------|
| 1 | **B** | 0.983 | 0.032 | 1 | 1 | 澄清 OpenCode /fork 命令 |
| 2 | V | 0.979 | 0.010 | 39 | — | 排查 Git push 网络故障 |
| 3 | **B** | 0.958 | 0.024 | 10 | 17 | Repo架构定位确认 |
| 4 | **B** | 0.924 | 0.030 | 2 | 7 | 排查 codegraph MCP 注册机制 |
| 5 | **B** | 0.905 | 0.023 | 12 | 15 | 澄清图谱构建流程及模型职责 |

**观察**:
- graph 通道把"Repo架构定位确认" (vector 第 10) 提升到 #3 (因为图谱 g_rank=17 + rerank 同意)
- "排查 Git push 网络故障" 纯 vector (vector_rank=39, 图谱无 → vector 单独拉到)
- rerank 同意 graph 的 4 个 B 推荐 — 没有"rerank 拉走 graph 推荐"的反例

## 5. 关键问题与改进建议

### P0: 性能瓶颈

1. **Rerank 占 45% 时延** (17-19s/查询)
   - 方案 A: 换 cross-encoder (BGE-reranker-base) 速度提升 5-10x
   - 方案 B: 输入截断到 top-20 而非 top-51
   - 方案 C: 引入 cache (query+topK hash)

2. **LLM 实体抽取占 47%** (18-20s/查询)
   - 方案 A: 允许 `extract_query_entities` 失败时 fallback 到 query 词直接 LIKE 搜索
   - 方案 B: 缓存 (query → entities), 命中率应该很高
   - 方案 C: 用更小的模型 (`gpt-4o-mini` 而非 reasoning 模型)

### P1: 图谱召回质量

3. **graph-only = 4%** (图谱几乎不独立贡献)
   - 接受: 这是 RRF 加性融合的特性, graph 用作"加权"而非"独立通道"
   - 改进: 可以把 `graph_channel_weight` 默认从 0.3 提到 0.4-0.5, 让 graph-only 更显眼

4. **`max_expand_nodes=30` 频繁被命中** (40% query)
   - 在 dense hub (OpenCode) 上尤其明显
   - 改进: BFS 内部按 `1/task_count` 升序扩散 (稀有优先), 而不是按遍历顺序

5. **反 IDF 公式对长尾不区分** (task_count=1 全都贡献 4.34)
   - 改进: 引入 entity 类型 (Tool/Concept/Code) 差异化贡献

### P2: 健壮性

6. **`SessionSearcher()` 默认 config_path 解析错误**
   - 源码: `Path(config_path).resolve().parent.parent`
   - 默认 `config_path="code_p3_config.yaml"` (无 `config/` 前缀) → `_repo_root` 错位到上层目录
   - 修复建议: 默认改为 `"config/code_p3_config.yaml"`, 或在搜索前显式校验

7. **Qdrant 单例锁** (`.lock` 文件残留)
   - 调试时遇到: 上次进程崩溃后锁文件残留
   - 现状: 手动 `rm qdrant_data/.lock` 解决
   - 改进: 启动时检查并清理 stale lock

## 6. 复现命令

```bash
# 1. 切到调试分支
git checkout debug_graphRAG

# 2. 准备环境
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export OPENCODE_ZEN_API_KEY=...
# (如果 qdrant_data/.lock 还在, 删除: rm qdrant_data/.lock)

# 3. 跑单条调试
python3 src/debug_graph_rag.py --db entity --query "OpenCode" --top-k 5
# → logs/debug_graph_rag/entity_OpenCode.json

# 4. 跑全量 (8 query × 2 db ≈ 10 分钟)
python3 src/debug_graph_rag.py --db both

# 5. 聚合分析
python3 src/debug_graph_rag_analyze.py
# → logs/debug_graph_rag/analysis.{md,json}
```

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `src/debug_graph_rag.py` | 单次 query 调试器, 记录 9 阶段 |
| `src/debug_graph_rag_analyze.py` | 聚合多个 JSON, 生成 analysis |
| `logs/debug_graph_rag/*.json` | 每个 query 一份详细记录 |
| `logs/debug_graph_rag/_summary.jsonl` | 一行一 query 的摘要 |
| `logs/debug_graph_rag/analysis.md` | 自动生成的多维度报告 |
| `logs/debug_graph_rag/analysis.json` | 结构化分析数据 |
| `doc/debug_graphRAG_analysis.md` | 本文档 (人工分析 + 改进建议) |
