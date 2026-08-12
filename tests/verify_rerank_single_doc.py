"""精确对比: 同一文档 + 不同 query 的 reranker 分数。

消除一切变量:
- 单文档, 不分批
- 同 Python 进程
- 详细 dump 输入文本与 token 数
"""
import os, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_p4_searcher import SessionSearcher

searcher = SessionSearcher("../config/code_p3_config.yaml")

tid = "ses_0c4845647ffea08jfnkY5ArLEG_T1"
doc_text = searcher.task_map[tid].task_summary

print(f"task_id: {tid}")
print(f"task_label: {searcher.task_map[tid].task_label}")
print(f"task_summary (full, {len(doc_text)} chars):")
print(doc_text)
print()

for q_label, q in [
    ("RAW", "GPU的对比和选型"),
    ("BGE-prefixed", "为这个句子生成表示以用于检索相关文章：GPU的对比和选型"),
]:
    prompt = searcher.reranker._build_prompt(q, doc_text)
    tokens = searcher.reranker.tokenizer(prompt, return_tensors="pt")
    n_tok = tokens["input_ids"].shape[1]
    print(f"\n{'='*60}")
    print(f"{q_label} query: {q!r}")
    print(f"prompt token count: {n_tok}")
    print(f"prompt last 100 chars: ...{prompt[-100:]!r}")
    print(f"{'='*60}")

    # 单文档 rerank
    results = searcher.reranker.rank(q, [doc_text], top_k=1)
    print(f"  -> score={results[0].score:.6f}")