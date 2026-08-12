"""完整复现 vec_results rerank 路径, 拿 GPU算力 在 vec_results 25-doc 实际 rerank 时, score 到底是什么。

排除一切变量:
- 直接 monkey-patch searcher.reranker.rank, 拿原始 documents 列表和 score
- 用与 vec_results 完全相同的 SessionSearcher + 25-doc 候选 + BGE-prefixed query
"""
import os, sys, time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_p1_utils import Config
from code_p4_searcher import SessionSearcher
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher

cfg = Config.load("../config/code_p3_config.yaml")
searcher = SessionSearcher("../config/code_p3_config.yaml")
kg_db = KGDatabase("../output/triple/knowledge_graph.db")
rag = GraphRAGSearcher(searcher, kg_db, bfs_depth=1, graph_channel_weight=0.1,
                        query_instruction=cfg.get("query_instruction_for_retrieval", ""),
                        rerank_multiplier=1)

# Monkey-patch to capture exact rerank input
captured = {}
orig_rank = searcher.reranker.rank

def patched_rank(query, documents, top_k, **kwargs):
    captured["query"] = query
    captured["documents"] = list(documents)  # snapshot
    captured["top_k_requested"] = top_k
    result = orig_rank(query, documents, top_k=top_k, **kwargs)
    captured["scores"] = [r.score for r in result]
    return result

searcher.reranker.rank = patched_rank

results, debug = rag.search("GPU的对比和选型", top_k=5, use_graph_rag=False, use_reranker=True)

# Find GPU算力 in documents and report score
target = "ses_0c4845647ffea08jfnkY5ArLEG_T1"
for i, doc in enumerate(captured["documents"]):
    # doc is task_summary string
    task = None
    for tid, t in searcher.task_map.items():
        if t.task_summary == doc:
            task = t
            break
    if task and task.task_id == target:
        print(f"\n>>> {target} is at documents[{i}]")
        print(f"    task_summary (first 100 chars): {doc[:100]}")
        print(f"    full task_summary ({len(doc)} chars):")
        print(f"    {doc}")
        print(f"    rerank_score = {captured['scores'][i]:.6f}")
        break
else:
    print(f"!!! {target} NOT FOUND in documents")

print(f"\nquery passed to rerank: {captured['query']!r}")
print(f"n_documents: {len(captured['documents'])}")
print(f"top_k requested: {captured['top_k_requested']}")
print(f"\nTop 5 returned (final vec_results):")
for i, r in enumerate(results):
    print(f"  [{i+1}] {r.task_id} rerank={r.rerank_score:.6f} - {r.task_label[:30]}")