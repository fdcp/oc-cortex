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

---

# 附录: 第二批 3 个 query 的针对性 case 分析

> 查询: `GPU的对比和选型`, `序列并行(Sequence Parallel,SP)`, `flashattention的原理和历史`
> 运行时间: 2026-08-12, dual DB (triple + entity)
> 数据: `doc/debug_graph_rag_data/{triple,entity}_{query}.json`

## 1. 三 query 总览 (Top-5 最终结果)

### 1.1 triple DB

| Query | 抽取实体 | 扩散 | 来源分布 | top-1 命中 | top-1 rerank |
|-------|----------|------|----------|------------|--------------|
| `GPU的对比和选型` | `['GPU']` | 3 | V=5, B=0 | NeMo Megatron Bridge 定位 | 0.89 |
| `序列并行(Sequence Parallel,SP)` | `[序列并行, Sequence Parallel, SP]` | 19 | V=2, **B=3** | **SP原理剖析与Mermaid图解迭代** | **0.997** |
| `flashattention的原理和历史` | `['flashattention']` | 29 | V=5, B=0 | NeMo Megatron Bridge 定位 | 0.82 |

### 1.2 entity DB

| Query | 抽取实体 | 扩散 | 来源分布 | top-1 命中 | top-1 rerank |
|-------|----------|------|----------|------------|--------------|
| `GPU的对比和选型` | `['GPU']` | 27 | V=3, B=2 | **NVIDIA SP与Ulysses并行策略对比** | 0.94 |
| `序列并行(Sequence Parallel,SP)` | `[序列并行, Sequence Parallel, SP]` | 30 | V=2, **B=3** | **SP原理剖析与Mermaid图解迭代** | **0.997** |
| `flashattention的原理和历史` | `['flashattention']` | 30 | V=5, B=0 | NeMo Megatron Bridge 定位 | 0.82 |

## 2. 关键发现

### 2.1 ✅ 序列并行 query — GraphRAG 最佳案例

抽取到 3 个实体变体 (`序列并行`, `Sequence Parallel`, `SP`), fuzzy 匹配还额外找到 `Sequence Parallelism`。
**图谱贡献显著**:
- triple 模式: 3/5 Top-K = B (双通道命中)
- entity 模式: 3/5 Top-K = B
- 两次都把 **"SP原理剖析与Mermaid图解迭代"** 推到 #1, rerank 0.997 (几乎满分)
- graph 通道拉上来的高价值 task: `并行训练尾部填充代码解析`, `NVIDIA SP与Ulysses并行策略对比`

**结论**: 当用户查询含**同义词/缩写/全称**时, LLM 抽取做对了, fuzzy 匹配也对了, graph 通道就能稳定贡献 Top-K。

### 2.2 ⚠️ GPU query — LLM 抽取粒度太粗

LLM 抽到 `['GPU']` (仅 1 个宽泛词), 实际查询"GPU 对比和选型"应该导向 **A800/H100/H800** 等具体型号。
- triple 模式: BFS 只扩散 **3 节点** → 4 个 graph_candidate 全是围绕 `GPU` 主题的 task, 跟"对比/选型"语义不匹配
- vector 通道: top-5 命中 `AllReduce算法原理`, `SP原理剖析`, `FA状态变量Shape` — **没有任何一个**真在做 GPU 选型
- entity 模式稍好 (BFS 27 节点, 2 个 both 命中), 但 top-1 仍是 `NVIDIA SP与Ulysses并行策略对比` (跟 GPU 选型半相关)

**结论**: 抽取阶段是**单点失败导致整体失败**。如果 LLM 抽到 `A800, H100, H800` 三个具体型号, 整个 pipeline 会完全不一样。

**建议**:
- 优化 `QUERY_ENTITY_PROMPT`: 明确要求"抽取查询中**具体实例、型号、专有名词**; 跳过通用主题词"
- 或者在图谱里手动 seed 一个"GPU 选型 → A800/H100/H800" 的强关联

### 2.3 ❌ flashattention query — 双重失效 (图谱 & rerank)

这是最让人警醒的 case:
- LLM 抽到 `['flashattention']` (小写), fuzzy 匹配到 `FlashAttention-1, FlashAttention-2, FlashAttention` (3 个变体)
- BFS 扩散 **29 节点**, unique_graph_tasks=5, top-1 task 得分 51.45 (n_entities=12)
- 但 **Top-5 全部是 V (vector-only)**, graph 信号 0 贡献
- 真正的问题: vector top-1 已经是 `FA状态变量Shape澄清` (跟"原理和历史"匹配度低), rerank 之后**更糟**:
  - rerank 把 `NeMo Megatron Bridge定位与价值辨析` 推到 #1 (rerank=0.82)
  - 4 个跟 FA 相关的候选 (FA状态变量Shape澄清, SP原理剖析, 并行训练尾部填充, FA1与FA2底层实现) **全部掉出 Top-5**

**根因分析**:
- `task_summary` 的向量特征不区分"原理/历史/细节/状态变量", BGE-small-zh 把所有 FlashAttention 相关 task 召回, 但区分度低 (hybrid_score 都在 0.028-0.031)
- Qwen3-Reranker 看到 "flashattention 的原理和历史" → 偏好"广义价值/定位"类回答 (跟训练/系统话题相关) → 把 NeMo Megatron Bridge 推到第一
- 这是 **reranker 偏置问题**: 在弱相关候选中, 它更倾向"看起来高级"的内容

**结论**: GraphRAG 的有效边界是**当图谱有强相关节点**时。如果查询对象在图谱中只有弱关联 (FA 节点的 1-hop 都是"训练系统" 而不是"原理/论文"), 整条 pipeline 都会偏向错误的语义。

**建议**:
- chunks_summary 集合的 embedding 用更大模型 (BGE-large) 区分"原理/历史/实现细节"
- reranker 输入截断 (top-20 而非 top-50) — 去掉长尾噪声
- graph_channel_weight 提到 0.5: 让图谱信号更"敢"覆盖 rerank

## 3. triple vs entity 模式对比

| 维度 | triple | entity |
|------|--------|--------|
| 节点数 | 1211 | 687 (更少) |
| 边数 | 3353 (稀疏) | **10757 (稠密)** |
| BFS 扩散 | 3-29 节点 (灵活) | 27-30 节点 (易撞顶) |
| Top-K 来源 (mean) | V=80%, B=20% | V=67%, B=33% |
| 图谱贡献度 | 较低 | **较高** (因为边更密, 1-hop 命中更多) |
| 典型场景 | 关系稀疏, 关注 hub 链路 | 共现稠密, 关注"同时提到 X 和 Y"的 session |

**结论**:
- **entity 模式**对"短查询 + 共现召回"更友好 (BFS 扩散 27-30 节点, 召回更广)
- **triple 模式**对"长查询 + 关系推理"更友好 (节点少但关系明确)
- 3 个 query 中, entity 模式在 2/3 上比 triple 给出更对题的 top-1

## 4. 时延 (mean, ms)

| 阶段 | triple (3 query) | entity (3 query) |
|------|------------------|------------------|
| LLM 实体抽取 | 31,070 | 20,277 |
| vector 检索 | 13 | 14 |
| graph (BFS+IDF) | 28,923 (含 LLM) | 24,817 (含 LLM) |
| rerank | 18,955 | 20,289 |
| **total (search)** | 47,893 | 45,123 |

**观察**: triple 模式的 LLM 抽取时延波动大 (15-55s, max=55s), 跟 query 长度/复杂度相关;
entity 模式稳定在 15-23s 区间。

## 5. 总体结论

| | 序列并行 | GPU | flashattention |
|---|---------|-----|----------------|
| 抽取质量 | ✅ 多变体 | ❌ 主题词太粗 | ⚠️ 大小写不一致 |
| 图谱召回 | ✅ 命中 SP 家族 | ❌ 4 节点太少 | ❌ 5 节点不在 top-5 |
| 融合效果 | ✅ 60% B | ❌ 0% B (triple) | ❌ 0% B (两边) |
| rerank 有效 | ✅ #1 完美 | ⚠️ #1 跑偏 | ❌ 错把 Megatron 推上 |
| 总体可用 | **⭐⭐⭐⭐⭐** | ⭐⭐ | ⭐ |

**最有价值的 query**: `序列并行` — 抽取 → 模糊 → BFS → RRF → Rerank 全链路都对
**最值得修的环节**:
1. 抽取 prompt 引导"具体实体优先于主题词"
2. rerank 输入截断 (top-20) 减少噪声影响
3. graph_channel_weight 默认从 0.3 提到 0.4-0.5, 让图谱信号更敢表达


## DB: `triple`

### Query: `GPU的对比和选型`  *(file: triple_GPU的对比和选型.json)*
- 抽取: `['GPU']` → 种子 `['GPU']` → BFS 3 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0323 | AllReduce算法原理与历史溯源 |
| 2 | 0.0310 | GPU算力对比与选型分析 |
| 3 | 0.0307 | NVIDIA SP与Ulysses并行策略对比 |
| 4 | 0.0303 | Roofline模型算术强度推导 |
| 5 | 0.0302 | 开发浮点数格式转换与教学网页 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 7.61 | 2 | `ses_0c5b6171bfferME7He2F8G6UuJ_T2` |
| 2 | 4.34 | 1 | `ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1` |
| 3 | 3.27 | 1 | `ses_0c5b6171bfferME7He2F8G6UuJ_T3` |
| 4 | 3.27 | 1 | `ses_0c9e4a359ffeGDSrVSG2jNVB6n_T1` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | vector | 0.8932 | 0.0167 | 12 | None | NeMo Megatron Bridge定位与价值辨析 |
| 2 | vector | 0.8366 | 0.0090 | 48 | None | FSDP与SDPA概念澄清 |
| 3 | vector | 0.7098 | 0.0149 | 17 | None | FA1与FA2底层实现及并行类比 |
| 4 | vector | 0.6002 | 0.0088 | 50 | None | 模型可用性确认 |
| 5 | vector | 0.5244 | 0.0101 | 39 | None | 梳理 opencode 会话分布及属性 |

---

### Query: `序列并行(Sequence_Parallel_SP)`  *(file: triple_序列并行_Sequence_Parallel_SP_.json)*
- 抽取: `['序列并行', 'Sequence Parallel', 'SP']` → 种子 `['序列并行', 'SP', 'Sequence Parallelism']` → BFS 19 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0325 | SP原理剖析与Mermaid图解迭代 |
| 2 | 0.0310 | RoPE预计算函数实现解析 |
| 3 | 0.0306 | NVIDIA SP与Ulysses并行策略对比 |
| 4 | 0.0302 | 并行训练尾部填充代码解析 |
| 5 | 0.0286 | FA状态变量Shape澄清 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 34.07 | 8 | `ses_12c6bd8dfffevT51PypMW2v5Mx_T1` |
| 2 | 27.29 | 7 | `ses_0a16b26dcffewGBp0akSLB5IZG_T1` |
| 3 | 14.94 | 4 | `ses_12c6bd8dfffevT51PypMW2v5Mx_T2` |
| 4 | 11.28 | 3 | `ses_0e8898453ffeS7ByDo9zjP8Z8h_T2` |
| 5 | 8.69 | 2 | `ses_0c5b6171bfferME7He2F8G6UuJ_T4` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | both | 0.9974 | 0.0323 | 1 | 1 | SP原理剖析与Mermaid图解迭代 |
| 2 | both | 0.8205 | 0.0303 | 3 | 3 | NVIDIA SP与Ulysses并行策略对比 |
| 3 | both | 0.5583 | 0.0294 | 4 | 4 | 并行训练尾部填充代码解析 |
| 4 | vector | 0.3558 | 0.0130 | 24 | None | 技术笔记静态站点整合与本地实施 |
| 5 | vector | 0.2147 | 0.0101 | 39 | None | Repo改动分析与Commit规划 |

---

### Query: `flashattention的原理和历史`  *(file: triple_flashattention的原理和历史.json)*
- 抽取: `['flashattention']` → 种子 `['FlashAttention-2 工作划分', 'FlashAttention-2', 'FlashAttention']` → BFS 29 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0313 | FA状态变量Shape澄清 |
| 2 | 0.0306 | SP原理剖析与Mermaid图解迭代 |
| 3 | 0.0296 | 并行训练尾部填充代码解析 |
| 4 | 0.0296 | FA1与FA2底层实现及并行类比 |
| 5 | 0.0289 | Roofline模型算术强度推导 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 51.45 | 12 | `ses_0c5b6171bfferME7He2F8G6UuJ_T4` |
| 2 | 46.03 | 11 | `ses_0c5b6171bfferME7He2F8G6UuJ_T2` |
| 3 | 19.29 | 5 | `ses_0c5b6171bfferME7He2F8G6UuJ_T3` |
| 4 | 16.30 | 4 | `ses_0c9e4a359ffeGDSrVSG2jNVB6n_T1` |
| 5 | 4.34 | 1 | `ses_12c6bd8dfffevT51PypMW2v5Mx_T1` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | vector | 0.8245 | 0.0179 | 9 | None | NeMo Megatron Bridge定位与价值辨析 |
| 2 | vector | 0.5588 | 0.0184 | 8 | None | ZeRO显存优化原理与通信推导 |
| 3 | vector | 0.2926 | 0.0095 | 44 | None | 整合优化专利交底书生成Ver2 |
| 4 | vector | 0.2628 | 0.0194 | 6 | None | 技术笔记静态站点整合与本地实施 |
| 5 | vector | 0.2613 | 0.0143 | 19 | None | 重构消息编号系统增加轮次与局部编号 |

---


## DB: `entity`

### Query: `GPU的对比和选型`  *(file: entity_GPU的对比和选型.json)*
- 抽取: `['GPU']` → 种子 `['GPU']` → BFS 27 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0323 | AllReduce算法原理与历史溯源 |
| 2 | 0.0310 | GPU算力对比与选型分析 |
| 3 | 0.0307 | NVIDIA SP与Ulysses并行策略对比 |
| 4 | 0.0303 | Roofline模型算术强度推导 |
| 5 | 0.0302 | 开发浮点数格式转换与教学网页 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 39.93 | 10 | `ses_053c0ac72ffe883RpIvqU1LUEw_T3` |
| 2 | 34.07 | 8 | `ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1` |
| 3 | 33.00 | 8 | `ses_0c5b6171bfferME7He2F8G6UuJ_T2` |
| 4 | 10.21 | 3 | `ses_053c0ac72ffe883RpIvqU1LUEw_T2` |
| 5 | 6.93 | 2 | `ses_0c5b6171bfferME7He2F8G6UuJ_T3` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | both | 0.9433 | 0.0295 | 3 | 6 | NVIDIA SP与Ulysses并行策略对比 |
| 2 | both | 0.8783 | 0.0238 | 16 | 5 | FA状态变量Shape澄清 |
| 3 | vector | 0.8366 | 0.0090 | 48 | None | FSDP与SDPA概念澄清 |
| 4 | vector | 0.8032 | 0.0175 | 10 | None | P5测试脚本对比与能力确认 |
| 5 | vector | 0.6002 | 0.0088 | 50 | None | 模型可用性确认 |

---

### Query: `序列并行(Sequence_Parallel_SP)`  *(file: entity_序列并行_Sequence_Parallel_SP_.json)*
- 抽取: `['序列并行', 'Sequence Parallel', 'SP']` → 种子 `['序列并行', 'SP', 'Sequence Parallelism']` → BFS 30 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0325 | SP原理剖析与Mermaid图解迭代 |
| 2 | 0.0310 | RoPE预计算函数实现解析 |
| 3 | 0.0306 | NVIDIA SP与Ulysses并行策略对比 |
| 4 | 0.0302 | 并行训练尾部填充代码解析 |
| 5 | 0.0286 | FA状态变量Shape澄清 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 49.69 | 12 | `ses_12c6bd8dfffevT51PypMW2v5Mx_T1` |
| 2 | 41.00 | 10 | `ses_0c5b6171bfferME7He2F8G6UuJ_T4` |
| 3 | 24.03 | 6 | `ses_0a16b26dcffewGBp0akSLB5IZG_T1` |
| 4 | 14.94 | 4 | `ses_12c6bd8dfffevT51PypMW2v5Mx_T2` |
| 5 | 4.56 | 2 | `ses_0a53b6807ffex4xr4a0wNdi5EQ_T1` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | both | 0.9974 | 0.0323 | 1 | 1 | SP原理剖析与Mermaid图解迭代 |
| 2 | vector | 0.9797 | 0.0189 | 7 | None | Roofline模型算术强度推导 |
| 3 | both | 0.9442 | 0.0206 | 26 | 7 | Session 历史追溯与总结 |
| 4 | vector | 0.8804 | 0.0219 | 2 | None | RoPE预计算函数实现解析 |
| 5 | both | 0.8205 | 0.0300 | 3 | 4 | NVIDIA SP与Ulysses并行策略对比 |

---

### Query: `flashattention的原理和历史`  *(file: entity_flashattention的原理和历史.json)*
- 抽取: `['flashattention']` → 种子 `['FlashAttention-2', 'FlashAttention-1', 'FlashAttention']` → BFS 30 节点

**Stage 5: 向量检索 top-5 (粗排)**

| # | hybrid_score | label |
|---|--------------|-------|
| 1 | 0.0313 | FA状态变量Shape澄清 |
| 2 | 0.0306 | SP原理剖析与Mermaid图解迭代 |
| 3 | 0.0296 | 并行训练尾部填充代码解析 |
| 4 | 0.0296 | FA1与FA2底层实现及并行类比 |
| 5 | 0.0289 | Roofline模型算术强度推导 |

**Stage 4: 图谱扩散 top-5 (反 IDF 累加)**

| # | score | n_ent | task_id |
|---|-------|-------|---------|
| 1 | 41.00 | 10 | `ses_0c5b6171bfferME7He2F8G6UuJ_T4` |
| 2 | 33.00 | 8 | `ses_0c5b6171bfferME7He2F8G6UuJ_T3` |
| 3 | 33.00 | 8 | `ses_0c5b6171bfferME7He2F8G6UuJ_T2` |
| 4 | 20.65 | 5 | `ses_0c9e4a359ffeGDSrVSG2jNVB6n_T1` |
| 5 | 7.61 | 2 | `ses_053c0ac72ffe883RpIvqU1LUEw_T3` |

**Stage 8: 最终 Top-5 (RRF + Rerank)**

| # | src | rerank | hybrid | v_rank | g_rank | label |
|---|-----|--------|--------|--------|--------|-------|
| 1 | vector | 0.8245 | 0.0179 | 9 | None | NeMo Megatron Bridge定位与价值辨析 |
| 2 | vector | 0.6960 | 0.0119 | 29 | None | Session 历史追溯与总结 |
| 3 | vector | 0.4610 | 0.0115 | 31 | None | 编写优化器基础指南v1 |
| 4 | vector | 0.3193 | 0.0091 | 47 | None | 分析 omo 模型配置及影响 |
| 5 | vector | 0.2628 | 0.0194 | 6 | None | 技术笔记静态站点整合与本地实施 |

---


---

# 附录: GPU query 融合失效的根因分析 (用户追问)

> 用户疑问: "图谱占比 0.3, 不应该这个结果"
> 实际上, **外层 RRF 阶段表现正确**, 真正的失效发生在 **Rerank 阶段**。

## 1. 重新跑一遍 RRF (验证)

外层 RRF 公式 (gw=0.3, k=30):
- vector top-5 (只 vector 通道) RRF: `0.7/(30+v_rank)` → 0.0226, 0.0219, 0.0212, 0.0206, 0.0200
- graph 命中 task 加上 graph 通道: `0.7/(30+v_rank) + 0.3/(30+g_rank)` → 0.0303, 0.0294

**外层 RRF 后排序**:

| RRF 排名 | task | v_rank | g_rank | RRF | 来源 |
|---------|------|--------|--------|-----|------|
| 1 | Roofline模型算术强度推导 | 4 | 1 | 0.0303 | graph+vector |
| 2 | 开发浮点数格式转换 | 5 | 2 | 0.0294 | graph+vector |
| 3 | AllReduce算法原理 | 1 | - | 0.0226 | vector |
| 4 | GPU算力对比与选型分析 | 2 | - | 0.0219 | vector |
| 5 | NVIDIA SP与Ulysses | 3 | - | 0.0212 | vector |

✅ **RRF 阶段行为正确** — vector top-5 中前 3 个 (AllReduce, GPU算力, NVIDIA SP) 都进 RRF 前 5, 图谱通道靠双通道命中 (Roofline, 浮点转换) 拉了 2 个进来。

## 2. Rerank 阶段: 顺序被完全推翻

**Rerank 之后 (50 候选 → 5)**:

| Final # | task | v_rank | rerank_score | RRF (RRF 排名) |
|---------|------|--------|--------------|----------------|
| 1 | NeMo Megatron Bridge | 12 | 0.893 | 0.017 (#13) |
| 2 | FSDP与SDPA | 48 | 0.837 | 0.009 (#42) |
| 3 | FA1与FA2 | 17 | 0.710 | 0.015 (#21) |
| 4 | 模型可用性确认 | 50 | 0.600 | 0.009 (#50) |
| 5 | 梳理 opencode | 39 | 0.524 | 0.010 (#33) |

❌ **Rerank 完全无视 RRF 排序** — Final Top-5 的 v_rank 是 [12, 48, 17, 50, 39], 几乎都是向量排名靠后的 task, RRF 分数 0.009-0.017 远低于 vector top-5 的 0.022+。

## 3. 为什么 Rerank 把"GPU算力"踢出 Top-5

直接调 `Qwen3Reranker` 重新打分, 单独对比这 10 个候选:

| 排名 | label | rerank_score | 类型 |
|------|-------|--------------|------|
| 1 | FSDP与SDPA概念澄清 | **0.9797** | final #2 |
| 2 | NVIDIA SP与Ulysses | 0.9433 | vector #3 |
| 3 | 模型可用性确认 | 0.0981 | final #4 |
| 4 | 开发浮点数转换 | 0.0401 | vector #5 |
| 5 | **GPU算力对比与选型分析** | **0.0278** | vector #2 |
| 6 | FA1与FA2 | 0.0223 | final #3 |
| 7 | Roofline | 0.0062 | vector #4 |
| 8 | AllReduce | 0.0044 | vector #1 |
| 9 | **NeMo Megatron Bridge** | **0.0001** | final #1 |
| 10 | 梳理 opencode | 0.0000 | final #5 |

**惊人发现**: 我重新跑 10 候选 Rerank, **"GPU算力对比" 排第 5 (0.028), "NeMo Megatron Bridge" 排倒数第 2 (0.0001)**。这跟原始 final 完全相反!

## 4. 根因: Rerank 候选截断 + 顺序偏置

当我用 vector top-10 跑 Rerank, 选出来的 top-5 是: NVIDIA SP / P5测试 / 浮点转换 / ZeRO / GPU算力。**没有 NeMo Megatron Bridge**。

但 GraphRAG 真实跑的是 50 候选 (vector 全 50), 选出的 top-5 全是 v_rank=12-50, **包含 NeMo Megatron Bridge**。

这就说明:
1. **Rerank 在 10 候选 vs 50 候选下行为不同** — 大候选池里 Qwen3-Reranker 倾向于"长文本+复杂概念", 把 NeMo Megatron Bridge 这类"概念辨析" 推上去
2. **Rerank 完全无视 RRF 排序** — 它独立打分, 不考虑 RRF 已经做了"channel balance"
3. **GW=0.3 在 Rerank 面前无效** — 既然 Rerank 选 5 个全是 vector-only, graph_channel_weight 根本不影响最终 Top-5

## 5. 对"图谱占比 0.3" 的回答

**用户疑问**: "图谱占比 0.3, 不应该这个结果"

**真实情况**:
- graph_channel_weight=0.3 (vector 拿 0.7) 在 **RRF 阶段**工作正确: vector top-5 排在 RRF 前 5
- 但 **Rerank 阶段** 完全覆盖了 RRF 排序, 把 vector 排名 12-50 的 task 推上 final
- 也就是说, **rerank 才是 GraphRAG 质量的决定因素, RRF 排序基本被 rerank 推翻了**

**调 GW=0.3 → 0.5 不会有改善**, 因为 rerank 不看 RRF 分数。

**真正需要修的**:
1. **Rerank 输入截断**: 只 rerank vector+graph 共同候选 或 vector top-20, 防止长尾噪声
2. **RRF+Rerank 分数融合**: `final = α·rerank + (1-α)·RRF`, α < 1 让 RRF 仍有发言权
3. **Query 改写**: "GPU 的对比" 改写为 "GPU 硬件 选型 V100 H100", 避免 Qwen3-Reranker 把它误判为"GPU 训练系统"
4. **Rerank 候选打分检查**: 在 GraphRAG 加 `_rerank_pre_filter` 阶段, 去掉 RRF 排名最差的 N 个再 rerank

## 6. 总结表

| 阶段 | 行为 | vector top-5 命运 |
|------|------|------------------|
| Vector 检索 | 召回 50 候选 | AllReduce, **GPU算力**, NVIDIA SP, Roofline, 浮点转换 |
| RRF 融合 | 排序 | ✅ Top-5 中有 3-4 个 vector top-5 |
| **Rerank** | **50 → 5** | **❌ 全部踢出, 换成 v_rank 12-50** |
| Final | Top-5 | NeMo Megatron Bridge, FSDP, FA1, 模型可用性, opencode |

**核心结论**: GraphRAG 的瓶颈不在 RRF 权重调参, 而在 Rerank 行为 — 它独立判断、独立覆盖, 跟 RRF 排序完全脱钩。

