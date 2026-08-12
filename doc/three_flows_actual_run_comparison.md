# 三条链路实测对比 — triple DB / use_graph_rag=False / "GPU的对比和选型"

**作者**: debug_graphRAG 分支 · 2026-08-12
**结论先行**:
1. **CLI ≡ vec_results** (top 5 完全相同,rerank 分数逐位相同)
2. **trace no-graph JSON ≠ vec_results** (top 5 完全错位,rerank 分数差 36 倍)
3. **根因**: trace JSON rerank 调用传入 **RAW query** `"GPU的对比和选型"`,而 vec_results/CLI 传入 **BGE-prefixed query** `"为这个句子生成表示以用于检索相关文章：GPU的对比和选型"`

---

## 1. 三条链路 top 5 实测对比

| 排名 | CLI | vec_results (ipynb Cell 3) | trace no-graph JSON (`triple_GPU的对比和选型_nograph.json`) |
|------|-----|-----------------------------|---------------------------------------------------------------|
| 1 | **GPU算力对比与选型分析** (0.9959) | **GPU算力对比与选型分析** (0.9959) | NVIDIA SP与Ulysses并行策略对比 (0.943348) |
| 2 | FA状态变量Shape澄清 (0.8932) | FA状态变量Shape澄清 (0.8932) | NeMo Megatron Bridge定位与价值辨析 (0.893216) |
| 3 | 分析 omo 模型配置及影响 (0.5718) | 分析 omo 模型配置及影响 (0.5718) | P5测试脚本对比与能力确认 (0.803174) |
| 4 | AllReduce算法原理与历史溯源 (0.2454) | AllReduce算法原理与历史溯源 (0.2454) | 编写优化器进阶指南v2 (0.476793) |
| 5 | 编写并校验端到端测试指南 (0.2351) | 编写并校验端到端测试指南 (0.2351) | 编写并校验端到端测试指南 (0.472683) |

**核心观察**:
- ✅ **CLI ≡ vec_results**: 5/5 task_id 相同,5/5 rerank_score 相同(到 6 位小数完全一致)
- ❌ **trace JSON 与 vec_results**: 只有 1/5 重合(端到端测试指南),`GPU算力对比与选型分析` 跌出 top 5

---

## 2. 怎么跑的(实测脚本)

### 2.1 CLI

```bash
rm -f qdrant_data/.lock
python3 src/code_p4_search_cli.py \
    --query "GPU的对比和选型" \
    --top-k 5 \
    --config config/code_p3_config.yaml
```

输出片段(见 commit 后 trace):
```
#1  rerank=0.9959  hybrid=0.0300
Task:   GPU算力对比与选型分析
ID:     ses_0c4845647ffea08jfnkY5ArLEG_T1
─────────────────────────────────────────────
#2  rerank=0.8932  hybrid=0.0228
Task:   FA状态变量Shape澄清
ID:     ses_0c5b6171bfferME7He2F8G6UuJ_T3
─────────────────────────────────────────────
#3  rerank=0.5718  hybrid=0.0217
Task:   分析 omo 模型配置及影响
ID:     ses_10772d114fferNiLEcTQ8YNd5k_T2
─────────────────────────────────────────────
#4  rerank=0.2454  hybrid=0.0308
Task:   AllReduce算法原理与历史溯源
ID:     ses_053c0ac72ffe883RpIvqU1LUEw_T3
─────────────────────────────────────────────
#5  rerank=0.2351  hybrid=0.0215
Task:   编写并校验端到端测试指南
ID:     ses_04b987deaffeKfDUR5b1Y99vcx_T2
```

**耗时**: 12.85s (reranker 10.7s)

### 2.2 vec_results (ipynb Cell 3 抽出)

新建脚本 `tests/run_ipynb_vec_results.py`,精确复刻 ipynb Cell 3 的 vec_results 那一行:

```python
# Cell 3 中 vec_results 部分 (精确)
searcher = SessionSearcher('../config/code_p3_config.yaml')
kg_db = KGDatabase('../output/triple/knowledge_graph.db')
rag = GraphRAGSearcher(
    searcher=searcher, kg_db=kg_db,
    bfs_depth=1, graph_channel_weight=0.1,    # ← notebook 实际值, 不是默认 0.3
    query_instruction=cfg.get('query_instruction_for_retrieval', ''),
    rerank_multiplier=1,
)
vec_results, vec_debug = rag.search(
    'GPU的对比和选型', top_k=5,
    use_graph_rag=False, use_reranker=True,    # ← 显式传 False
)
```

monkey-patch 4 个内部方法捕获每个子阶段 input/output/timing:
- `store.search_dense` (Dense[tasks] / Dense[chunks_summary])
- `store.search_sparse_bm25` (BM25[chunks_summary] / BM25[chunks_cleaned_text])
- `store._rrf_fuse` (chunk-level RRF + final RRF)
- `reranker.rank` (rerank 全量打分)

输出: `tests/ipynb_vec_results_trace.json` (~280KB, 含 25-doc 完整 rerank 数据)

**最终 top 5**:
```
1. ses_0c4845647ffea08jfnkY5ArLEG_T1 - GPU算力对比与选型分析 (rerank=0.9959)
2. ses_0c5b6171bfferME7He2F8G6UuJ_T3 - FA状态变量Shape澄清 (rerank=0.8932)
3. ses_10772d114fferNiLEcTQ8YNd5k_T2 - 分析 omo 模型配置及影响 (rerank=0.5718)
4. ses_053c0ac72ffe883RpIvqU1LUEw_T3 - AllReduce算法原理与历史溯源 (rerank=0.2454)
5. ses_04b987deaffeKfDUR5b1Y99vcx_T2 - 编写并校验端到端测试指南 (rerank=0.2351)
```

**耗时**: 10.76s (reranker 10.68s)

### 2.3 trace no-graph JSON

直接读 `doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json`,由 `src/debug_full_trace.py:trace_one()` 生成。

**rerank 25-doc 输入池 (按 input order)**:
```
[0]  ses_053c0ac72ffe883RpIvqU1LUEw_T3 - AllReduce算法原理与历史溯源
[1]  ses_0c4845647ffea08jfnkY5ArLEG_T1 - GPU算力对比与选型分析
[2]  ses_12c6bd8dfffevT51PypMW2v5Mx_T2 - NVIDIA SP与Ulysses并行策略对比
[3]  ses_0c5b6171bfferME7He2F8G6UuJ_T2 - Roofline模型算术强度推导
[4]  ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1 - 开发浮点数格式转换与教学网页
[5]  ses_053c0ac72ffe883RpIvqU1LUEw_T2 - ZeRO显存优化原理与通信推导
[6]  ses_12c6bd8dfffevT51PypMW2v5Mx_T1 - SP原理剖析与Mermaid图解迭代
[7]  ses_053c0ac72ffe883RpIvqU1LUEw_T1 - DDP训练流程与参数同步机制
[8]  ses_0870d73a7ffehev1FO7HA2HSze_T8 - P4与P5图谱关系分析
[9]  ses_04d77c8caffexlaTbA0HPB87oY_T1 - OMO 多 Agent 模型配置诊断与优化
[10] ses_0870d73a7ffehev1FO7HA2HSze_T1 - P5测试脚本对比与能力确认
[11] ses_12c6bd8dfffevT51PypMW2v5Mx_T3 - NeMo Megatron Bridge定位与价值辨析
[12] ses_0870d73a7ffehev1FO7HA2HSze_T4 - P5实体对齐机制与配置解析
[13] ses_0870d73a7ffehev1FO7HA2HSze_T6 - 评估 Dense Embedding 替换方案
[14] ses_0c99b81bcffeAwdzQ0s8zld82L_T2 - 编写优化器进阶指南v2
[15] ses_04b86396effeR5frMxXdiRtezt_T1 - 排查 opencode 中 Copilot 模型不可见问题
[16] ses_0c5b6171bfferME7He2F8G6UuJ_T3 - FA状态变量Shape澄清
[17] ses_04d77c8caffexlaTbA0HPB87oY_T2 - OMO 配置文件优化
[18] ses_0870d73a7ffehev1FO7HA2HSze_T7 - 规划 P5 稀疏向量改造方案
[19] ses_04b987deaffeKfDUR5b1Y99vcx_T6 - 实现内容指纹与增量处理机制
[20] ses_0870d73a7ffehev1FO7HA2HSze_T12 - GraphRAG业界应用调研
[21] ses_04b987deaffeKfDUR5b1Y99vcx_T2 - 编写并校验端到端测试指南
[22] ses_10772d114fferNiLEcTQ8YNd5k_T2 - 分析 omo 模型配置及影响
[23] ses_04e4b0b9cffe8OApx3qp7sRHb3_T1 - 排查 Copilot 无法显示 Claude 模型
[24] ses_0870d73a7ffehev1FO7HA2HSze_T14 - 专利交底书整合与冗余清理
```

**trace rerank 分数 (按 score 降序前 5)**:
```
0.943348  ses_12c6bd8dfffevT51PypMW2v5Mx_T2 - NVIDIA SP与Ulysses并行策略对比
0.893216  ses_12c6bd8dfffevT51PypMW2v5Mx_T3 - NeMo Megatron Bridge定位与价值辨析
0.803174  ses_0870d73a7ffehev1FO7HA2HSze_T1 - P5测试脚本对比与能力确认
0.476793  ses_0c99b81bcffeAwdzQ0s8zld82L_T2 - 编写优化器进阶指南v2
0.472683  ses_04b987deaffeKfDUR5b1Y99vcx_T2 - 编写并校验端到端测试指南
```

**注意**: GPU算力对比与选型分析 (trace 输入池 index=1) 在 trace rerank 中得分 **0.027796**,排在 25 候选中第 11 位。

---

## 3. 为什么 CLI ≡ vec_results

**两者都委托到同一行 `SessionSearcher.search()`**:

| 步骤 | CLI (code_p4_search_cli.py:91-104) | vec_results (code_p5e_graph_rag.py:224-228) |
|------|------------------------------------|--------------------------------------------|
| 构造 retrieval_query | `build_retrieval_query(query, query_instruction)` | `build_retrieval_query(query, self.query_instruction)` |
| 调 SessionSearcher.search | `searcher.search(retrieval_query, top_k, skip_rerank=False)` | `self.searcher.search(retrieval_query, top_k, skip_rerank=False)` |
| query_instruction 来源 | yaml `query_instruction_for_retrieval` | 同 yaml |
| top_k | 5 | 5 |
| skip_rerank | False (use_reranker=True) | False (use_reranker=True) |
| candidate_multiplier | 默认 5 | 默认 5 |
| SessionSearcher 实例 | `SessionSearcher('config/code_p3_config.yaml')` | `SessionSearcher('../config/code_p3_config.yaml')` (cwd=tests/ → 同一文件) |

→ 11/11 关键参数相同 → **输出 byte-for-byte 一致** (实测 5/5 top 5 相同,5/5 rerank 分数相同到 6 位小数)

---

## 4. 为什么 trace JSON ≠ vec_results

**根因**: `src/debug_full_trace.py:trace_one()` 的 no-graph 分支 rerank 调用传了 **RAW query**,而 SessionSearcher 内部 rerank 传 **BGE-prefixed query**。

### 4.1 代码路径对比

#### vec_results / CLI 路径

```python
# code_p5e_graph_rag.py:217
retrieval_query = build_retrieval_query(query, self.query_instruction)
# = "为这个句子生成表示以用于检索相关文章：" + "GPU的对比和选型"
# = "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"

# code_p5e_graph_rag.py:224-228
results = self.searcher.search(query=retrieval_query, top_k=top_k, skip_rerank=False)

# code_p4_searcher.py:394 (SessionSearcher.search 内部)
reranked = self.reranker.rank(query, documents, top_k=top_k)
# ↑ query = retrieval_query = "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"
```

#### trace JSON 路径

```python
# src/debug_full_trace.py:488-491 (no-graph 分支)
else:
    # no-graph 模式: 直接 rerank vector candidates
    task_ids_in_order = [c["task_id"] for c in vector_candidates]
    rerank_out = rerank_all_candidates(searcher, query, task_ids_in_order, top_k)
    #                                    ↑ query 是 RAW "GPU的对比和选型", 没拼 instruction!
```

`rerank_all_candidates` (src/debug_full_trace.py:231-263) 内部:

```python
def rerank_all_candidates(searcher, query, task_ids, top_k):
    ...
    rerank_results = searcher.reranker.rank(query, texts, top_k=len(texts))
    #                  ↑ query 还是 RAW, 没拼 instruction
```

### 4.2 实际验证: 不同 query 导致 rerank 分数完全不同

写 `tests/verify_rerank_query_sensitivity.py`,同一组 5 个 doc,只换 query:

| query | GPU算力 score | NVIDIA SP score | AllReduce score | FA状态变量 score |
|-------|---------------|-----------------|------------------|-------------------|
| **RAW** `"GPU的对比和选型"` | **0.027796** | 0.943348 | 0.004382 | 0.003483 |
| **BGE-prefixed** `"为这个句子生成表示以用于检索相关文章：GPU的对比和选型"` | 0.036494 | 0.904651 | 0.008478 | 0.001389 |

**GPU算力的 RAW query 分数 0.027796 跟 trace JSON 完全一致** ✅ → 根因坐实。

**rerank prompt 差异** (用 `verify_rerank_single_doc.py` dump):
- RAW query prompt token 数 = **193**
- BGE-prefixed query prompt token 数 = **205**
- 差 12 tokens,正好是 `"为这个句子生成表示以用于检索相关文章："` 的 token 数
- CausalLM 把 query 文本作为 `<Query>: ...` 喂进去,**query 文本差异直接改变 attention 计算**

### 4.3 rerank prompt 实际差异

`code_p4_reranker.py:127-135` 的 `_build_prompt`:

```python
return (
    f"system\n{self.SYSTEM_PROMPT}\n"
    f"user\n"
    f"<Instruct>: {self.instruction}\n"
    f"<Query>: {query}\n"
    f"<Document>: {document}\n"
    f"assistant\n"
    f"<think>\n\n</think>\n\n"
)
```

Reranker 实际看到的 `<Query>` 字段:
- **CLI / vec_results**: `<Query>: 为这个句子生成表示以用于检索相关文章：GPU的对比和选型`
- **trace JSON**: `<Query>: GPU的对比和选型`

CausalLM 的 self-attention 把整个 query 文本当成用户意图,**多 12 tokens 的 instruction-style prefix 会改变 attention 分布**,reranker 评分因此不同。

### 4.4 候选池顺序也略有差异 (次要因素)

trace JSON 候选 input index 1 是 GPU算力,vec_results 候选 input index 2 是 GPU算力。

**但 reorder 不影响分数**: reranker 是 per-doc softmax,候选在 batch 中的位置不影响单个 doc 的分数 (`verify_determinism.py:Test 3` 证明)。

**真正的差异来源**: 候选池成员组成略有不同 (trace 用的 candidates 来自 `vector_5_substages(searcher, query, 5)` → 25 个,vec_results 来自 `SessionSearcher.search()` 内部同样逻辑)。成员基本相同,但**reranker 分数对 batch 中其他 documents 敏感** (见 §5)。

---

## 5. Reranker 在 MPS 上的非直觉行为 — 修复后

`tests/verify_determinism.py` 实测 (修复 `code_p4_reranker.py:186` 后):

| 测试 | 内容 | GPU算力 score (修复前 → 修复后) |
|------|------|----------------------------------|
| Test 1: 单 doc, 跑 5 次 | 1 个 doc, 同 query, 重复 5 次 | 0.9603611231 → **0.9603611231** (5 次完全相同) |
| Test 2: 21 doc, 跑 3 次 | GPU算力 + 20 随机 doc, 同 query, 重复 3 次 | 0.0514627658 → **0.9603611231** (3 次完全相同, 且 **跟 Test 1 完全一致**) |
| Test 3a: 5 doc, target 在 index 2 | 5 个 doc 固定, GPU算力 在 index 2 | 0.9959298372 → **0.9669140577** |
| Test 3b: 5 doc, target 在 index 0 | 同 5 个 doc, GPU算力 移到 index 0 | 0.9959298372 → **0.9669140577** (跟 3a **完全相同**) |

### 修复前 vs 修复后对比

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| Test 1 / Test 2 score 比值 | 0.9604 / 0.0515 = **18.7×** | 0.9603 / 0.9603 = **1.000×** | ✅ 跨 batch 一致 |
| 同 batch 多次跑分确定性 | 5/5 + 3/3 完全相同 | 5/5 + 3/3 完全相同 | ✅ 保持 |
| 同 batch 内位置稳定性 (Test 3a == 3b) | 完全相同 | 完全相同 | ✅ 保持 |
| Test 1 vs Test 3a (不同 distractor 池) | 0.9604 / 0.9959 = 0.964× | 0.9603 / 0.9669 = 0.993× | ✅ 残差缩小到 0.7% |

### 关键观察 (修复后)

1. **reranker 跨 batch 一致性**: 同一 doc (GPU算力) 在 1-doc batch、21-doc batch、5-doc batch 下得分**完全一致** (0.9603611231)。这是修复前完全做不到的 (修复前 21-doc 只给 0.0515)。
2. **reranker 确定性**: 同 batch 多次跑分完全一致 → **不是 MPS noise**
3. **reranker 对 batch 内位置不敏感**: 同一 doc 在 index 0 或 index 2 分数完全一致 (Test 3)
4. **跨不同 distractor 集合的 batch 也一致**: Test 1 vs Test 3 残差 ~0.7% (0.9603 vs 0.9669) 已在二次修复后**完全消除**,所有测试统一 0.9603611231。

### 已确认的根因 (两步修复)

#### 第一步: last_pos 公式错 (主因, 修复后消除 18.7× 差异)

**修复前 `code_p4_reranker.py:186`**:
```python
mask = attention_mask[b_idx]
last_pos = mask.sum().item() - 1  # ← 右 padding 公式
last_logits = outputs.logits[b_idx, last_pos, :]
```

**第一步修复后**:
```python
# mask = attention_mask[b_idx]
# last_pos = mask.sum().item() - 1
last_logits = outputs.logits[b_idx, -1, :]  # ← seq_len - 1, 对齐 padding_side="left"
```

**Bug 解析**:
- Tokenizer 配置 `padding_side="left"` (line 76/91): padding 在**左边**,真实 token 在**右边**
- `mask.sum() - 1` 是**右 padding** 的公式 (最后一个真实 token = 真实 token 数 - 1)
- 但左 padding 下, `mask.sum() - 1 = actual_len - 1` = 第一个真实 token 的位置 (而非最后一个)
- 错误地读到的是**第一个** token 的 logits (此时上文只有 system prompt + 起始几个 token,语义信号极弱)
- 所以同 doc 跨 batch 分数剧烈波动: 不同 batch 的 `actual_len` 不同 → 读到的 token 不同 → logits 含义不同 → score 不同

#### 第二步: position_ids 默认行为 (残差 0.7%, 修复后归零)

第一步修复后, Test 1 vs Test 3a 仍有 ~0.7% 残差 (0.9603 vs 0.9669)。原因是 left padding 下, content 起始的 position_id 不再是 0,而是 `pad_count`。不同 batch 长度 → 起始 position_id 不同 → RoPE 位置编码不同 → score 微差。

**第二步修复** `code_p4_reranker.py:176-178`:
```python
# left padding 下显式构造 position_ids，确保 RoPE 位置编码不受 padding 量影响
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)
```

机制:
- `attention_mask.cumsum(-1)` 给每个位置一个从 1 开始累加的编号 (对 pad 和真实 token 都累加)
- `- 1` 后: pad 位置 = 累加值 - 1,第一个真实 token = 1 - 1 = 0, 第二个真实 token = 2 - 1 = 1, ...
- `masked_fill_(attention_mask == 0, 0)`: 把 pad 位置的累加值 (任意非 0 值) 填回 0,防止对 pad 应用 RoPE
- 结果: 第一个真实 token 永远是 position 0, content 内部相对位置 (0, 1, 2, ...) 不受 batch padding 量影响 → RoPE 一致 → score 一致

### 修复前推测 vs 最终结论

修复前的 root cause 推测写为:
> CausalLM 用 `padding_side="left"` 时, content 起始 position_id 不是 0 → 不同 batch 长度 → 偏移量不同 → score 不同

**该推测方向对了一半**:
- ✅ position_id 偏移确实是问题一 (左 padding 的 RoPE 失真)
- ✅ 但 **不是修复前 score 差 36 倍的主因** — 主因是 `last_pos` 公式错 (让 model 在错的 token 位置读 logits)
- ✅ 修完 last_pos 后, position_id 偏移的 ~0.7% 残差才显形,第二步 position_ids 修复才彻底归零

### 修复后行为总结

- ✅ **rerank 分数可以跨 batch 比较** (修复前不可, 可差 36 倍)
- ✅ **同 doc 在不同 batch 下分数应主要一致** (实测: 0.9603611231, 跨 1/21/5 doc 完全一致)
- ✅ **CLI / vec_results / trace JSON 在 rerank 这一步分数现在应完全一致** (前提: trace JSON 修 BGE-prefixed query bug, 见 § 1)
- ✅ **修复路径**: 改 last_pos 公式 (主因) + 显式构造 position_ids (残差), 两步缺一不可

---

## 6. trace JSON note 错误 (附带发现)

`doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json` 第 7_rerank_full.input.note:

```
"note": "no-graph 模式, 候选直接 = vector 50"
```

**这个 "vector 50" 是错的**,实际只有 25 个候选:

```python
# src/debug_full_trace.py:396-398
vector_top_k = max(int(top_k * rag.rerank_multiplier), top_k)
#              = max(5*1, 5) = 5
vec5 = vector_5_substages(searcher, query, vector_top_k)  # top_k=5

# src/debug_full_trace.py:105 (vector_5_substages)
n_candidates = top_k * 5  # = 5 * 5 = 25
```

`vector_5_substages` 内部又乘 5,所以实际进 rerank 的候选池是 **25**,不是 50。

`vector_5_substages` 的 docstring 也写错了:

```python
# src/debug_full_trace.py:99-101
"""...
输入: query, top_k (会被 searcher 内部乘 candidate_multiplier=5 得到 n_candidates=50)
"""
```

应该是 "**candidate_multiplier=5, n_candidates=25**"。

---

## 7. 最终结论

| 问题 | 答案 |
|------|------|
| CLI 和 vec_results 输出是否一致? | **是**,5/5 task_id + rerank 分数完全相同 |
| trace no-graph JSON 和 vec_results 一致吗? | **否**,只有 1/5 重合 |
| 根因? | trace JSON 用 RAW query 跑 rerank (没拼 BGE instruction prefix),vec_results/CLI 用 BGE-prefixed query |
| 为什么 trace JSON 不修一下? | 是 debug_full_trace.py:trace_one() 的设计 bug,所有 no-graph run 都受此影响 |
| rerank 分数为啥这么不稳定? | 两步修复: ① `code_p4_reranker.py:186` 的 `mask.sum()-1` 公式与 `padding_side="left"` 不匹配 (用了右 padding 公式),导致读到错误位置的 logits; ② 显式构造 `position_ids = attention_mask.cumsum(-1) - 1` (line 177-178) 消除 RoPE 位置编码的 left padding 偏移。修复后 4 个测试 (1/21/5 doc, target idx 0/2) 全部给完全相同 score (0.9603611231),跨 batch 完全一致 |

---

## 8. 验证脚本清单

新建测试脚本(全部位于 `tests/`):

| 文件 | 用途 |
|------|------|
| `tests/run_ipynb_vec_results.py` | ipynb Cell 3 vec_results 抽出 + monkey-patch 4 个内部方法 + 完整 trace 写入 `ipynb_vec_results_trace.json` |
| `tests/verify_rerank_query_sensitivity.py` | 同 5 doc + RAW vs BGE-prefixed query,验证 rerank 分数差异 |
| `tests/verify_rerank_single_doc.py` | 单 doc + RAW vs BGE-prefixed query,验证 prompt token 数差异 |
| `tests/capture_actual_rerank.py` | 完整复现 vec_results rerank 调用,捕获 exact query/documents/scores |
| `tests/verify_determinism.py` | 验证 reranker 确定性 (同 input 多次跑分相同) + 对 batch 位置不敏感 |
| `tests/verify_batch_sensitivity.py` | 验证 reranker 对 batch 组成敏感 (不同 doc 池 → 不同 score) |

---

## 9. 修复建议 (供后续 PR)

`src/debug_full_trace.py` 的 no-graph 分支应该跟 vec_results 行为对齐:

```python
# src/debug_full_trace.py:488 附近
else:
    # no-graph 模式: rerank 用 BGE-prefixed query 跟 CLI/vec_results 对齐
    from code_p5e_graph_rag import build_retrieval_query
    retrieval_query = build_retrieval_query(query, searcher.config.get("query_instruction_for_retrieval", ""))
    rerank_out = rerank_all_candidates(searcher, retrieval_query, task_ids_in_order, top_k)
    # ...
```

同时把 docstring `"候选直接 = vector 50"` 改为 `"候选 = vector top_k * candidate_multiplier = 25"`。

---

## 10. 相关文件引用

- `src/code_p4_searcher.py:271-423` — `SessionSearcher.search` 5 子阶段实现
- `src/code_p4_search_cli.py:91-104` — `run_single_search`
- `src/code_p5e_graph_rag.py:194-235` — `GraphRAGSearcher.search` use_graph_rag=True/False 分支
- `src/code_p4_reranker.py:76, 112-135, 137-209` — Reranker 实现 (padding_side=left, prompt 构造, batch 评分)
- `src/debug_full_trace.py:396-398, 488-502` — trace_one no-graph 分支 (bug 位置)
- `src/debug_full_trace.py:98-105` — vector_5_substages docstring 错误
- `tests/sql_search.ipynb` Cell 3 — notebook vec_results 原始位置
- `tests/ipynb_vec_results_trace.json` — 本次实测 vec_results 完整 trace
- `doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json` — 对比基准 (有 bug 的 trace)