"""最终验证: 同一 prompt + 不同 batch 环境下, reranker score 是否一致。

设计:
- Run A: 1 doc GPU算力 (单 prompt) + BGE-prefixed
- Run B: 25 docs (GPU算力 + 24 其它) + BGE-prefixed
- 对比 GPU算力 的 score
"""
import os, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_p4_searcher import SessionSearcher

searcher = SessionSearcher("../config/code_p3_config.yaml")

# All 25 candidate documents (vec_results 实际顺序)
candidates = [
    ("ses_053c0ac72ffe883RpIvqU1LUEw_T3", "AllReduce"),
    ("ses_053c0ac72ffe883RpIvqU1LUEw_T2", "ZeRO"),
    ("ses_0c4845647ffea08jfnkY5ArLEG_T1", "GPU算力对比与选型分析"),  # ← target
    ("ses_0ca8b3533ffeRk0KWh2cSqkKOa_T1", "浮点数格式"),
    ("ses_0c5b6171bfferME7He2F8G6UuJ_T2", "Roofline"),
    ("ses_12c6bd8dfffevT51PypMW2v5Mx_T2", "NVIDIA SP Ulysses"),
    ("ses_04d77c8caffexlaTbA0HPB87oY_T2", "OMO 配置"),
    ("ses_0870d73a7ffehev1FO7HA2HSze_T6", "Dense Embedding"),
    ("ses_0c99b81bcffeAwdzQ0s8zld82L_T2", "A800 NCCL 调优"),
    ("ses_04b987deaffeKfDUR5b1Y99vcx_T6", "增量处理"),
    ("ses_0870d73a7ffehev1FO7HA2HSze_T8", "Phase 5 实体对齐"),
    ("ses_12c6bd8dfffevT51PypMW2v5Mx_T1", "Megatron SP"),
    ("ses_0870d73a7ffehev1FO7HA2HSze_T7", "Sparse 向量"),
    ("ses_053c0ac72ffe883RpIvqU1LUEw_T1", "DDP"),
    ("ses_0870d73a7ffehev1FO7HA2HSze_T1", "Phase 5 test 对比"),
    ("ses_04b86396effeR5frMxXdiRtezt_T1", "优化器基础"),
    ("ses_0870d73a7ffehev1FO7HA2HSze_T4", "Phase 5 LLM 调用"),
    ("ses_04e4b0b9cffe8OApx3qp7sRHb3_T1", "Copilot 模型"),
    ("ses_0c5b6171bfferME7He2F8G6UuJ_T3", "FA状态变量Shape"),
    ("ses_04d77c8caffexlaTbA0HPB87oY_T1", "OMO 多 Agent"),
    ("ses_10772d114fferNiLEcTQ8YNd5k_T2", "omo 插件配置"),
    ("ses_04b987deaffeKfDUR5b1Y99vcx_T2", "端到端测试"),
    ("ses_04e4b0b9cffe8OApx3qp7sRHb3_T1", "P4/P5 架构"),  # placeholder
    ("ses_04d77c8caffexlaTbA0HPB87oY_T1", "P1 文档对齐"),  # placeholder
    ("ses_0870d73a7ffehev1FO7HA2HSze_T14", "专利交底书"),
]

Q = "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"

target_idx = 2  # GPU算力
target_doc = searcher.task_map[candidates[target_idx][0]].task_summary

print(f"target task_summary:\n{target_doc}\n")

# Run 1: 单文档
print("=== Run A: 单文档 (GPU算力) + BGE-prefixed ===")
r1 = searcher.reranker.rank(Q, [target_doc], top_k=1)
print(f"  score = {r1[0].score:.6f}")

# Run 2: 25 文档 (使用真实 order)
print("\n=== Run B: 25 文档 + BGE-prefixed ===")
docs_25 = [searcher.task_map[tid].task_summary for tid, _ in candidates]
r2 = searcher.reranker.rank(Q, docs_25, top_k=len(docs_25))
score_25 = None
for rr in r2:
    if rr.index == target_idx:
        score_25 = rr.score
        break
print(f"  GPU算力 (index={target_idx}) score = {score_25:.6f}")
print(f"  全 25 排序前 5:")
for rr in r2[:5]:
    tid, label = candidates[rr.index]
    print(f"    {rr.score:.6f}  [{rr.index}] {tid[:40]:40s} {label}")

# Run 3: 同样的 5 文档 (跟 verify_rerank_query_sensitivity 顺序一致)
print("\n=== Run C: 5 文档 (GPU算力 在 index 0) + BGE-prefixed ===")
five_task_ids = [
    "ses_0c4845647ffea08jfnkY5ArLEG_T1",
    "ses_12c6bd8dfffevT51PypMW2v5Mx_T2",
    "ses_0c5b6171bfferME7He2F8G6UuJ_T3",
    "ses_053c0ac72ffe883RpIvqU1LUEw_T3",
    "ses_04b987deaffeKfDUR5b1Y99vcx_T2",
]
five_docs = [searcher.task_map[tid].task_summary for tid in five_task_ids]
r3 = searcher.reranker.rank(Q, five_docs, top_k=len(five_docs))
for rr in r3:
    tid = five_task_ids[rr.index]
    if tid == "ses_0c4845647ffea08jfnkY5ArLEG_T1":
        print(f"  GPU算力 (index={rr.index}) score = {rr.score:.6f}")