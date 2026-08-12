"""验证 reranker 是否确定性: 同 prompt 跑 N 次, score 是否稳定?"""
import os, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_p4_searcher import SessionSearcher

searcher = SessionSearcher("../config/code_p3_config.yaml")
target_doc = searcher.task_map["ses_0c4845647ffea08jfnkY5ArLEG_T1"].task_summary
Q = "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"

print("=== Test 1: 同 prompt, 同 query, 跑 5 次 (1 doc batch) ===")
for i in range(5):
    r = searcher.reranker.rank(Q, [target_doc], top_k=1)
    print(f"  Run {i+1}: score = {r[0].score:.10f}")

print("\n=== Test 2: 同 prompt, 25 doc batch (含其它 24 个 distractor) ===")
# 取 25 个不同的 task_summaries (5 个固定 + 20 个随机填充)
other_ids = [
    "ses_053c0ac72ffe883RpIvqU1LUEw_T3",  # AllReduce
    "ses_12c6bd8dfffevT51PypMW2v5Mx_T2",  # NVIDIA SP
    "ses_0c5b6171bfferME7He2F8G6UuJ_T3",  # FA状态变量
    "ses_10772d114fferNiLEcTQ8YNd5k_T2",  # omo 配置
    "ses_04b987deaffeKfDUR5b1Y99vcx_T2",  # 端到端测试
] + [f"ses_fake_{i}" for i in range(20)]

# 用真实 task_summary 或 placeholder
import random
random.seed(42)
real_ids = list(searcher.task_map.keys())[:20]
docs_25 = [target_doc]
for tid in real_ids[:24]:
    if tid != "ses_0c4845647ffea08jfnkY5ArLEG_T1":
        docs_25.append(searcher.task_map[tid].task_summary)
# 保持 target 在 index 2
docs_25.insert(2, target_doc)
docs_25 = docs_25[:25]

print(f"  doc count: {len(docs_25)}, target at index 2")
for i in range(3):
    r = searcher.reranker.rank(Q, docs_25, top_k=len(docs_25))
    for rr in r:
        if rr.index == 2:
            print(f"  Run {i+1}: GPU算力 score = {rr.score:.10f}")
            break

print("\n=== Test 3: 同一个 GPU算力 doc 但放在 batch 不同位置 (5 doc batch) ===")
five_with_target_at_2 = [
    searcher.task_map["ses_053c0ac72ffe883RpIvqU1LUEw_T3"].task_summary,
    searcher.task_map["ses_053c0ac72ffe883RpIvqU1LUEw_T2"].task_summary,
    target_doc,  # index 2
    searcher.task_map["ses_0c5b6171bfferME7He2F8G6UuJ_T3"].task_summary,
    searcher.task_map["ses_04b987deaffeKfDUR5b1Y99vcx_T2"].task_summary,
]
r = searcher.reranker.rank(Q, five_with_target_at_2, top_k=5)
for rr in r:
    if rr.index == 2:
        print(f"  GPU算力 at index 2: score = {rr.score:.10f}")

# 把 target_doc 移到 index 0
five_with_target_at_0 = [
    target_doc,  # index 0
    searcher.task_map["ses_053c0ac72ffe883RpIvqU1LUEw_T3"].task_summary,
    searcher.task_map["ses_053c0ac72ffe883RpIvqU1LUEw_T2"].task_summary,
    searcher.task_map["ses_0c5b6171bfferME7He2F8G6UuJ_T3"].task_summary,
    searcher.task_map["ses_04b987deaffeKfDUR5b1Y99vcx_T2"].task_summary,
]
r = searcher.reranker.rank(Q, five_with_target_at_0, top_k=5)
for rr in r:
    if rr.index == 0:
        print(f"  GPU算力 at index 0: score = {rr.score:.10f}")