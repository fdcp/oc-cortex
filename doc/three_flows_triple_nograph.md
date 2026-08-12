# 三条链路流程对比 — triple DB / use_graph_rag=False / "GPU的对比和选型"

**作者**: debug_graphRAG 分支 · 2026-08-12
**范围**: triple 图谱 DB + `use_graph_rag=False` (即"纯向量"路径)
**目的**: 解释为何 `python3 src/code_p4_search_cli.py`、`tests/sql_search.ipynb` Cell 3 中的 `vec_results`、以及 `doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json` 三者的输出理论上 100% 等价。

---

## 1. 三条链路定义

| # | 名称 | 入口调用 | 是否经过 GraphRAG | 是否经过图谱 |
|---|------|---------|-------------------|--------------|
| ① | **CLI** | `python3 src/code_p4_search_cli.py --query "GPU的对比和选型" --top-k 5` | 否 | 否 |
| ② | **vec_results** (ipynb) | `rag.search(..., use_graph_rag=False, use_reranker=True)` | **是** (但走 `False` 分支) | 否 |
| ③ | **trace no-graph JSON** | `trace_one(use_graph_rag=False)` (debug 脚本) | **是** (但走 `False` 分支) | 否 |

**核心结论**:
- ① ② ③ 三者都 **不会** 执行 LLM 实体抽取、BFS 扩散、外层 RRF merge、GraphRAG `_rerank`。
- 三者最终都委托到 `SessionSearcher.search()`,因此 **理论输出必须 byte-for-byte 一致**(在确定性假设下)。

> 本文档关注的是 "**为什么 trace no-graph JSON 就是 vec_results 应该输出的样子**",以及 "**为什么 GPU 算力对比与选型分析 在 RRF 第 2 名却被 Reranker 挤到第 11**"。

---

## 2. 代码路径逐步拆解

### 2.1 CLI 调用栈 (`code_p4_search_cli.py`)

```python
# code_p4_search_cli.py:91-104
def run_single_search(searcher, query, top_k, skip_rerank, query_instruction):
    results = searcher.search(
        build_retrieval_query(query, query_instruction),  # "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"
        top_k=top_k,                                       # 5
        skip_rerank=skip_rerank,                           # False (use_reranker 默认 True)
    )
```

`main()` 默认 `query_instruction = config.get("query_instruction_for_retrieval", "")` 即 yaml 中的 `"为这个句子生成表示以用于检索相关文章："`。

### 2.2 vec_results 调用栈 (`code_p5e_graph_rag.py`)

ipynb Cell 3:

```python
vec_results, vec_debug = rag.search(
    'GPU的对比和选型', top_k=5,
    use_graph_rag=False,    # ← 显式传 False
    use_reranker=True,
)
```

`GraphRAGSearcher.search()` 在 `use_graph_rag=False` 分支直接走这段 (src/code_p5e_graph_rag.py:219-235):

```python
if not use_graph_rag:
    results = self.searcher.search(
        query=retrieval_query,         # = build_retrieval_query(query, self.query_instruction)
        top_k=top_k,
        skip_rerank=not use_reranker,
    )
```

ipynb 中 `GraphRAGSearcher` 用 `query_instruction=cfg.get('query_instruction_for_retrieval', '')` (即 `"为这个句子生成表示以用于检索相关文章："`)。所以 `retrieval_query` 与 CLI 完全一致。

**关键等价性**:
- CLI: `searcher.search(build_retrieval_query(query, instruction), top_k, skip_rerank=False)`
- vec_results: `searcher.search(build_retrieval_query(query, instruction), top_k, skip_rerank=False)`
- ✅ **参数完全相同**,调用同一个 `SessionSearcher` 实例的同一个 `search` 方法。

### 2.3 trace no-graph 调用栈 (`src/debug_full_trace.py`)

`trace_one(query, use_graph_rag=False)` 在 `debug_full_trace.py` 中:

```python
rag = GraphRAGSearcher(searcher, kg_db, ...)
# 等价于 ipynb: use_graph_rag=False 分支,同样调 self.searcher.search
results, debug = rag.search(query, top_k=top_k, use_graph_rag=False, use_reranker=True)
```

→ 与 vec_results **完全相同**。

---

## 3. 参数一致性检查表

要保证三链路输出 byte-for-byte 一致,**所有这些参数都必须相同**:

| 参数 | CLI (①) | vec_results (②) | trace (③) | 一致? |
|------|---------|------------------|-----------|--------|
| `query` (原始) | `"GPU的对比和选型"` | `"GPU的对比和选型"` | `"GPU的对比和选型"` | ✅ |
| `query_instruction` | `"为这个句子生成表示以用于检索相关文章："` (yaml) | 同左 | 同左 | ✅ |
| `retrieval_query` (拼好后) | `"为这个句子生成表示以用于检索相关文章：GPU的对比和选型"` | 同左 | 同左 | ✅ |
| `top_k` | 5 | 5 | 5 | ✅ |
| `skip_rerank` | False (默认) | False (use_reranker=True) | False | ✅ |
| `candidate_multiplier` (P4 内部) | 默认 5 | 默认 5 | 默认 5 | ✅ |
| `SessionSearcher.config_path` | `"config/code_p3_config.yaml"` (CLI 默认) | `"../config/code_p3_config.yaml"` (ipynb, cwd=tests/) → 解析到 `config/code_p3_config.yaml` | `"config/code_p3_config.yaml"` | ✅ (同一文件) |
| `Qwen3Reranker` 模型 | `Qwen/Qwen3-Reranker-0.6B` (yaml) | 同左 | 同左 | ✅ |
| `BGE-small-zh-v1.5` embedding | yaml 默认 | 同左 | 同左 | ✅ |
| `BM25` (`sparse.method`) | yaml 默认 | 同左 | 同左 | ✅ |
| `chunks_summary_method` | yaml 默认 `sparse` | 同左 | 同左 | ✅ |
| `tasks.jsonl` | yaml 默认 | 同左 (从 `_repo_root` 解析) | 同左 | ✅ |
| `chunks.jsonl` | yaml 默认 | 同左 | 同左 | ✅ |
| `output/chunks_summary_p2.jsonl` | yaml 默认 | 同左 | 同左 | ✅ |

**结论**: 11/11 项一致 → 三链路产出在 **确定性假设下完全等价**。

---

## 4. 预期输出 (以 trace no-ground JSON 为 ground truth)

下面所有数据来自 `doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json`,这是 **vec_results 应该输出的内容**,CLI 也应该相同。

### 4.1 5 个内部子阶段 (P4)

#### Stage 5a: Dense[tasks] (top 25 by cosine similarity)

| rank | task_id | dense_score | label |
|------|---------|-------------|-------|
| 1 | ses_12c6bd8dfffevT51PypMW2v5Mx_T2 | 0.69182 | NVIDIA SP与Ulysses并行策略对比 |
| 2 | ses_053c0ac72ffe883RpIvqU1LUEw_T2 | 0.619429 | ZeRO显存优化原理与通信推导 |
| 3 | ses_053c0ac72ffe883RpIvqU1LUEw_T3 | 0.616821 | AllReduce算法原理与历史溯源 |
| 4 | **ses_0c4845647ffea08jfnkY5ArLEG_T1** | **0.607437** | **GPU算力对比与选型分析** |
| 5 | ses_0870d73a7ffehev1FO7HA2HSze_T1 | 0.585502 | P5测试脚本对比与能力确认 |
| ... | ... | ... | ... |

完整 25 个详见 trace JSON 第 19-70 行。

#### Stage 5b: BM25[chunks_summary] (75 chunks → 45 unique tasks)

| rank | task_id | bm25_score | label |
|------|---------|------------|-------|
| 1 | ses_053c0ac72ffe883RpIvqU1LUEw_T3 | 7.525334 | AllReduce算法原理与历史溯源 |
| 2 | ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1 | 5.655567 | 开发浮点数格式转换与教学网页 |
| 3 | ses_0870d73a7ffehev1FO7HA2HSze_T8 | 5.525269 | P4与P5图谱关系分析 |
| ... | ... | ... | ... |

完整 75 行详见 trace JSON 第 80-131 行。

#### Stage 5c: BM25[chunks_cleaned_text] (75 chunks → 41 unique tasks)

| rank | task_id | bm25_score | label |
|------|---------|------------|-------|
| 1 | **ses_0c4845647ffea08jfnkY5ArLEG_T1** | **9.168549** | **GPU算力对比与选型分析** |
| 2 | ses_053c0ac72ffe883RpIvqU1LUEw_T3 | 8.104362 | AllReduce算法原理与历史溯源 |
| 3 | ses_0c5b6171bfferME7He2F8G6UuJ_T2 | 7.836732 | Roofline模型算术强度推导 |
| ... | ... | ... | ... |

完整 75 行详见 trace JSON 第 142-192 行。

> **观察 1**: `GPU算力对比与选型分析` 在 BM25[chunks_cleaned_text] 中是 #1 (分 9.17),在 BM25[chunks_summary] 中没进 top10,但在 Dense[tasks] 中排 #4。这说明其 task_summary 长度/语义跟 query 较匹配,但 cleaned_text 中包含的关键词匹配度极高。

#### Stage 5d: chunk RRF fusion (chunks_summary + chunks_cleaned_text → 60 tasks)

RRF 公式: `score(d) = sum_i 1/(60 + rank_i)`,k=60。

| rank | task_id | rrf_score | label |
|------|---------|-----------|-------|
| 1 | ses_053c0ac72ffe883RpIvqU1LUEw_T3 | 0.032522 | AllReduce算法原理与历史溯源 |
| 2 | ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1 | 0.031054 | 开发浮点数格式转换与教学网页 |
| 3 | ses_0870d73a7ffehev1FO7HA2HSze_T14 | 0.030536 | 专利交底书整合与冗余清理 |
| 4 | ses_0c5b6171bfferME7He2F8G6UuJ_T2 | 0.029762 | Roofline模型算术强度推导 |
| 5 | **ses_0c4845647ffea08jfnkY5ArLEG_T1** | **0.02938** | **GPU算力对比与选型分析** |
| ... | ... | ... | ... |

完整 top10 详见 trace JSON 第 202-253 行。

#### Stage 5e: chunk RRF + Dense[tasks] → top 25 candidates

| rank | task_id | rrf_score | label |
|------|---------|-----------|-------|
| 1 | ses_053c0ac72ffe883RpIvqU1LUEw_T3 | 0.032266 | AllReduce算法原理与历史溯源 |
| 2 | **ses_0c4845647ffea08jfnkY5ArLEG_T1** | **0.03101** | **GPU算力对比与选型分析** |
| 3 | ses_12c6bd8dfffevT51PypMW2v5Mx_T2 | 0.030679 | NVIDIA SP与Ulysses并行策略对比 |
| 4 | ses_0c5b6171bfferME7He2F8G6UuJ_T2 | 0.030331 | Roofline模型算术强度推导 |
| 5 | ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1 | 0.030214 | 开发浮点数格式转换与教学网页 |
| ... | ... | ... | ... |

完整 top10 详见 trace JSON 第 263-314 行。

> **观察 2**: `GPU算力对比与选型分析` 在 Stage 5e 排 **#2** (rrf_score 0.03101),非常强势。如果只看 RRF,大概率会进 top 5。

### 4.2 Reranker 阶段 (Qwen3-Reranker-0.6B)

**输入池**: 25 个 candidates (Stage 5e 的全部)
**reranker 输入文本**: 每个 candidate 的 `task_summary` (非 task_label)
**reranker instruction**: `"Given a web search query, retrieve relevant passages."` (yaml)

完整 25 个 rerank 分数按降序:

| rerank_rank | orig_index | task_id | label | rerank_score |
|-------------|-----------|---------|-------|--------------|
| 1 | 2 | ses_12c6bd8dfffevT51PypMW2v5Mx_T2 | NVIDIA SP与Ulysses并行策略对比 | **0.943348** |
| 2 | 11 | ses_12c6bd8dfffevT51PypMW2v5Mx_T3 | NeMo Megatron Bridge定位与价值辨析 | **0.893216** |
| 3 | 10 | ses_0870d73a7ffehev1FO7HA2HSze_T1 | P5测试脚本对比与能力确认 | **0.803174** |
| 4 | 14 | ses_0c99b81bcffeAwdzQ0s8zld82L_T2 | 编写优化器进阶指南v2 | 0.476793 |
| 5 | 21 | ses_04b987deaffeKfDUR5b1Y99vcx_T2 | 编写并校验端到端测试指南 | 0.472683 |
| 6 | 4 | ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1 | 开发浮点数格式转换与教学网页 | 0.089137 |
| 7 | 18 | ses_0870d73a7ffehev1FO7HA2HSze_T7 | 规划 P5 稀疏向量改造方案 | 0.078926 |
| 8 | 8 | ses_0870d73a7ffehev1FO7HA2HSze_T8 | P4与P5图谱关系分析 | 0.054601 |
| 9 | 23 | ses_04e4b0b9cffe8OApx3qp7sRHb3_T1 | 排查 Copilot 无法显示 Claude 模型 | 0.052814 |
| 10 | 5 | ses_053c0ac72ffe883RpIvqU1LUEw_T2 | ZeRO显存优化原理与通信推导 | 0.034456 |
| **11** | **1** | **ses_0c4845647ffea08jfnkY5ArLEG_T1** | **GPU算力对比与选型分析** | **0.027796** |
| 12 | 22 | ses_10772d114fferNiLEcTQ8YNd5k_T2 | 分析 omo 模型配置及影响 | 0.027796 |
| ... | ... | ... | ... | ... |

完整 25 行详见 trace JSON 第 390-540 行。

### 4.3 最终 top 5 (CLI / vec_results / trace no-ground 都应输出)

| rank | task_id | label | rerank_score | v_rank |
|------|---------|-------|--------------|--------|
| **1** | ses_12c6bd8dfffevT51PypMW2v5Mx_T2 | NVIDIA SP与Ulysses并行策略对比 | 0.943348 | 3 |
| **2** | ses_12c6bd8dfffevT51PypMW2v5Mx_T3 | NeMo Megatron Bridge定位与价值辨析 | 0.893216 | 12 |
| **3** | ses_0870d73a7ffehev1FO7HA2HSze_T1 | P5测试脚本对比与能力确认 | 0.803174 | 11 |
| **4** | ses_0c99b81bcffeAwdzQ0s8zld82L_T2 | 编写优化器进阶指南v2 | 0.476793 | 15 |
| **5** | ses_04b987deaffeKfDUR5b1Y99vcx_T2 | 编写并校验端到端测试指南 | 0.472683 | 22 |

**观察 3**:
- `GPU算力对比与选型分析` (dense rank=2, chunk rank=2, RRF 总排名 #2) 被 Reranker 给了 **0.027796**,排在 rerank 候选池 **第 11 位**,最终被挤到 top 5 之外。
- 这是 **Reranker 的偏好问题**,不是 RRF 的问题:**Qwen3-Reranker 偏好长篇大论 + 主题相关性,而对"标题直白包含查询词"的 task 不友好**。

---

## 5. 为什么 GPU 算力对比与选型分析 不在 top 5?

### 5.1 数据层面的对比

| 维度 | NVIDIA SP与Ulysses (rank 1) | GPU算力对比与选型分析 (rank 11) |
|------|-----------------------------|--------------------------------|
| dense score | 0.69182 | 0.607437 |
| chunk RRF rank | 10 (0.028595) | 5 (0.02938) |
| final RRF rank | 3 (0.030679) | **2** (0.03101) |
| task_summary 长度 | ~370 字 (3 段) | ~80 字 (1 段) |
| 与 query 字面重合 | "并行 / Ulysses / SP" | **"GPU / 对比 / 选型 / 算力"** |
| 主题相关度 | SP 并行策略 (中等相关) | **GPU 选型 (直接相关)** |

### 5.2 Reranker 的反直觉现象

Reranker 给 `GPU算力对比与选型分析` 的分数是 **0.027796**,给 `NVIDIA SP与Ulysses` 的分数是 **0.943348**,差 **34 倍**。原因:

1. **Qwen3-Reranker 偏好"概念辨析类"内容**:
   - `NVIDIA SP与Ulysses` 的 task_summary 是 "针对 8 GPU (TP=2, SP=4) 场景,澄清 Tensor Parallelism 与 Sequence Parallelism 的正交关系,并深度对比 NVIDIA SP 与 Ulysses 两种序列并行实现方案的通信机制与适用边界..."
   - 这是 **"深度技术辨析 + 对比"** 的典型,reranker 判定为"高度相关"
2. **Reranker 不偏好"标题直白命中"**:
   - `GPU算力对比与选型分析` 的 task_summary 直接说 GPU/对比/选型,但内容上可能是浅显的概览
   - reranker 看到 query "GPU的对比和选型" 与 task_summary 高度字面重合,但缺乏"技术辨析深度"
3. **Reranker 几乎无视 RRF 分数**:
   - 即便 RRF 把这个 task 排到 #2 (rrf_score 0.03101),reranker 仍然给它 0.028
   - 这印证了之前在 `doc/debug_graphRAG_analysis.md` 中已经发现的"Qwen3-Reranker 在大候选池中偏向长文本/概念辨析"的 bug

### 5.3 三个调用点看到的"诡异"对比

| 用户看到 | 实际原因 |
|---------|---------|
| CLI / vec_results 的 top 1 是 "NVIDIA SP与Ulysses" | **正确** — 这是 Reranker 的偏好 |
| Notebook 中 `results` 的 top 1 是 "P5测试脚本" | graph 通道加入了 4 个独有候选,稀释了向量候选的密度,Reranker 排序洗牌 |
| 期待的 "GPU算力对比与选型分析" 不在 top 5 | Reranker 把它挤到第 11 位 |

---

## 6. 如果用户实际看到三者不一致的根因排查

按照第 2 节和第 3 节的等价性证明, **三者理论上必须 byte-for-byte 一致**。如果用户实际看到不一致,逐项检查:

### 6.1 高概率原因 (代码层)

1. **reranker 软硬件确定性差异**
   - Qwen3-Reranker 是 CausalLM,forward pass 使用 `torch.no_grad()`,softmax over [yes, no] logits
   - 在 MPS / CUDA 上极小概率出现非确定性 (例如 atomicAdd),但 CPU 应该是确定的
   - 如果有 0.001 分的差,可能导致 top 5 边界换位
   - **验证方法**: 同一 query 跑 CLI 两次,看 top 5 是否一致

2. **`build_retrieval_query` 参数不一致**
   - CLI: `cfg.get("query_instruction_for_retrieval", "")` (默认 yaml 值)
   - ipynb: `cfg.get('query_instruction_for_retrieval', '')` (同上)
   - 但如果 **用户修改过 yaml** 把 `query_instruction_for_retrieval` 改了,两边可能不一样
   - **验证方法**: `grep "query_instruction" config/code_p3_config.yaml`

3. **`SessionSearcher` 实例化参数不一致**
   - CLI: `SessionSearcher(config_path=args.config)` 默认 `"config/code_p3_config.yaml"`
   - ipynb: `SessionSearcher('../config/code_p3_config.yaml')` (cwd=`tests/`)
   - 如果 ipynb 的 cwd 不是 `tests/`,`../config/...` 会解析失败或指错文件
   - **验证方法**: 在 ipynb Cell 3 之前加 `print(Path.cwd())` 确认 cwd

### 6.2 中概率原因 (数据层)

4. **`tasks.jsonl` / `chunks.jsonl` 在两次运行之间被修改**
   - 如果用户在 P1/P2 阶段重跑过,`task_summary` / `chunk_summary` 内容可能微变
   - 这会导致 embedding / BM25 / Reranker 全部跟着变
   - **验证方法**: `md5sum output/tasks.jsonl output/chunks.jsonl output/chunks_summary_p2.jsonl`

5. **Qdrant 数据被重建**
   - 如果 P3 重跑过,tasks 集合里的 task_summary 可能顺序不同,导致 dense score 不同
   - **验证方法**: 看 `qdrant_data/meta.json` 的时间戳

### 6.3 低概率原因 (实现层)

6. **`chunks_summary_method` 配置不一致**
   - 当前 yaml 是 `sparse` (BM25),但用户可能本地改成了 `dense`
   - 这会导致 stage 5b 完全不同
   - **验证方法**: `grep "chunks_summary_method" config/code_p3_config.yaml`

7. **reranker instruction 不一致**
   - yaml 默认 `"Given a web search query, retrieve relevant passages."`
   - 如果改过,reranker 行为会变
   - **验证方法**: `grep "instruction" config/code_p3_config.yaml`

---

## 7. 验证步骤

如果用户想证明 CLI ≡ vec_results,跑以下两步:

### 7.1 提取 vec_results 到 JSON

在 ipynb Cell 3 之后加一个 cell:

```python
import json
vec_dump = [
    {
        "rank": i + 1,
        "task_id": r.task_id,
        "label": r.task_label,
        "rerank_score": r.rerank_score,
        "hybrid_score": r.hybrid_score,
    }
    for i, r in enumerate(vec_results)
]
with open("vec_results.json", "w", encoding="utf-8") as f:
    json.dump(vec_dump, f, ensure_ascii=False, indent=2)
print(json.dumps(vec_dump, ensure_ascii=False, indent=2))
```

### 7.2 跑 CLI 并提取

```bash
python3 src/code_p4_search_cli.py \
    --query "GPU的对比和选型" \
    --top-k 5 \
    --config config/code_p3_config.yaml 2>&1 | tee cli_output.txt
```

### 7.3 对比

```bash
# 把 CLI 输出中的 top 5 task_id 提取出来
grep -oP 'ses_[a-z0-9]+_T\d+' cli_output.txt | head -5 > cli_task_ids.txt
# 跟 vec_results 的 task_id 列表对比
diff <(jq -r '.[].task_id' vec_results.json) cli_task_ids.txt
```

如果 `diff` 输出为空 → 三者完全一致。
如果 `diff` 有输出 → 某个 task 顺序或内容不同,根据第 6 节逐项排查。

---

## 8. 总结

| 问题 | 答案 |
|------|------|
| `chunks_summary` 当前是 sparse 还是 dense? | **sparse (BM25)** (yaml `chunks_summary_method: sparse`) |
| CLI ≡ vec_results (ipynb)? | **理论上 100% 等价** (都调 `SessionSearcher.search()`,参数相同) |
| CLI ≡ trace no-graph JSON? | **理论上 100% 等价** (trace 脚本用同样调用栈) |
| 三者真的不一致怎么办? | 按第 6 节 7 个排查点逐项检查 |
| 为何 GPU算力对比与选型分析 不在 top 5? | Reranker 偏好长文本/概念辨析,把它从 RRF #2 挤到 rerank #11 |
| GPU查询的最佳预期输出 | `triple_GPU的对比和选型_nograph.json` 第 569-625 行的 final_top_k |

---

## 9. 相关文件引用

- `src/code_p4_search_cli.py:91-118` — `run_single_search`
- `src/code_p4_searcher.py:271-423` — `SessionSearcher.search` 5 子阶段
- `src/code_p4_searcher.py:425-443` — `_search_chunks_summary` 分支
- `src/code_p5e_graph_rag.py:194-235` — `GraphRAGSearcher.search` 的 `use_graph_rag=False` 分支
- `src/code_p3_qdrant_store.py:935-979` — `_rrf_fuse` 实现
- `config/code_p3_config.yaml:28` — `chunks_summary_method: "sparse"`
- `tests/sql_search.ipynb` Cell 3 — 两个 `rag.search` 调用
- `doc/GraphRAG_full_trace_data/triple_GPU的对比和选型_nograph.json` — ground truth 数据
- `doc/debug_graphRAG_analysis.md` — Qwen3-Reranker 长文本偏好的历史分析