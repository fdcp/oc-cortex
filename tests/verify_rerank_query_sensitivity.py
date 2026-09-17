"""验证 reranker 对 query 是否敏感 (raw vs instruction-prefixed)。

理论假设:
- vec_results / CLI: query = "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"
- trace JSON: query = "GPU的对比和选型"
如果 reranker 对 query 文本敏感, 两次跑会得到完全不同的分数排序。
"""
import os, sys, json
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_p1_utils import Config
from code_p4_searcher import SessionSearcher

cfg = Config.load("../config/code_p3_config.yaml")
searcher = SessionSearcher("../config/code_p3_config.yaml")

# 用同一组 documents, 不同 query 跑两次 rerank
task_ids = [
    "ses_0c4845647ffea08jfnkY5ArLEG_T1",   # GPU算力对比与选型分析
    "ses_12c6bd8dfffevT51PypMW2v5Mx_T2",   # NVIDIA SP与Ulysses
    "ses_0c5b6171bfferME7He2F8G6UuJ_T3",   # FA状态变量Shape
    "ses_053c0ac72ffe883RpIvqU1LUEw_T3",   # AllReduce
    "ses_04b987deaffeKfDUR5b1Y99vcx_T2",   # 端到端测试
]
texts = [searcher.task_map[tid].task_summary for tid in task_ids]

for label, q in [
    ("RAW query", "GPU的对比和选型"),
    ("BGE-prefixed", "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"),
]:
    print(f"\n{'='*60}\n{label}: {q!r}\n{'='*60}")
    results = searcher.reranker.rank(q, texts, top_k=len(texts))
    for rr in results:
        tid = task_ids[rr.index]
        label_short = searcher.task_map[tid].task_label[:25]
        print(f"  {rr.score:.6f}  [{rr.index}] {tid[:50]:50s} {label_short}")