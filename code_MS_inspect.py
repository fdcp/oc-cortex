#!/usr/bin/env python3
"""
里程碑检视工具: Qdrant 数据浏览 + 检索质量评估
替代 Qdrant Dashboard (项目用嵌入式 Qdrant, 无 Web UI)

功能:
1. 集合统计 (点数, 维度, 状态)
2. 样本浏览 (随机抽取 payload)
3. 数据分析 (session/task/chunk 分布, 压缩率)
4. 向量近邻分析 (embedding 聚类质量)
5. 检索质量测试 (预设查询 + 命中率)
6. Chunk Summary 质量抽样
"""
import json
import sys
import random
import hashlib
from pathlib import Path
from collections import Counter

# ── 离线模式 ──
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from qdrant_client import QdrantClient


# ============================================================
# 工具函数
# ============================================================

def _stable_uuid(text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ============================================================
# 1. 集合统计
# ============================================================

def show_collection_stats(client: QdrantClient) -> None:
    section("1. 集合统计")
    collections = client.get_collections().collections
    print(f"集合总数: {len(collections)}")
    for c in collections:
        info = client.get_collection(c.name)
        points = getattr(info, "points_count", "?")
        vectors = getattr(info, "vectors_count", None)
        if vectors is None:
            vectors = points
        status = getattr(info, "status", "?")
        config = info.config
        vc = config.params.vectors
        if isinstance(vc, dict):
            dim_info = ", ".join(f"{k}: {v.size}维 {v.distance}" for k, v in vc.items())
        else:
            dim_info = f"{vc.size}维 {vc.distance}"

        print(f"\n  [{c.name}]")
        print(f"    状态: {status}")
        print(f"    点数: {points}")
        print(f"    向量: {vectors}")
        print(f"    维度: {dim_info}")


# ============================================================
# 2. 样本浏览
# ============================================================

def show_sample_payloads(client: QdrantClient, n: int = 5) -> None:
    section("2. 样本浏览")
    for cname in ["tasks", "chunks"]:
        subsection(f"集合: {cname}")
        try:
            info = client.get_collection(cname)
            total = getattr(info, "points_count", 0) or 0
        except Exception:
            print(f"  集合 {cname} 不存在")
            continue

        if total == 0:
            print("  (空集合)")
            continue

        all_points, _ = client.scroll(
            collection_name=cname,
            limit=total,
            with_payload=True,
            with_vectors=False,
        )

        samples = random.sample(all_points, min(n, len(all_points)))
        for i, pt in enumerate(samples, 1):
            pl = pt.payload or {}
            print(f"\n  [{i}] id={pt.id}")
            if cname == "tasks":
                print(f"      task_id:    {pl.get('task_id', '?')}")
                print(f"      session_id: {pl.get('session_id', '?')}")
                print(f"      task_label: {pl.get('task_label', '?')}")
                summary = pl.get("task_summary", "")
                print(f"      task_summary: {summary[:120]}{'...' if len(summary) > 120 else ''}")
                chunk_ids = pl.get("chunk_ids", [])
                print(f"      chunk_ids:  {len(chunk_ids)} 个 {chunk_ids[:3]}{'...' if len(chunk_ids) > 3 else ''}")
            else:
                print(f"      chunk_id:   {pl.get('chunk_id', '?')}")
                print(f"      session_id: {pl.get('session_id', '?')}")
                print(f"      turn_index: {pl.get('turn_index', '?')}")
                print(f"      task_id:    {pl.get('task_id', '?')}")
                summary = pl.get("summary", "")
                print(f"      summary:    {summary[:120]}{'...' if len(summary) > 120 else ''}")
                raw = pl.get("raw_size_tokens", 0)
                cleaned = pl.get("cleaned_size_tokens", 0)
                ratio = f"{cleaned/raw:.1%}" if raw > 0 else "N/A"
                print(f"      tokens:     raw={raw}, cleaned={cleaned}, 压缩率={ratio}")


# ============================================================
# 3. 数据分析
# ============================================================

def show_data_analysis(client: QdrantClient) -> None:
    section("3. 数据分析")

    # --- Tasks 分析 ---
    subsection("Tasks 分布")
    task_points, _ = client.scroll("tasks", limit=1000, with_payload=True, with_vectors=False)
    tasks_data = [p.payload for p in task_points if p.payload]

    session_counts = Counter(t.get("session_id", "?") for t in tasks_data)
    print(f"  总 task 数: {len(tasks_data)}")
    print(f"  涉及 session 数: {len(session_counts)}")
    print(f"  每 session task 数分布:")
    for cnt, freq in sorted(Counter(session_counts.values()).items()):
        print(f"    {cnt} 个 task: {freq} 个 session")

    chunk_counts = [len(t.get("chunk_ids", [])) for t in tasks_data]
    if chunk_counts:
        avg_chunks = sum(chunk_counts) / len(chunk_counts)
        max_chunks = max(chunk_counts)
        min_chunks = min(chunk_counts)
        print(f"\n  每 task 关联 chunk 数:")
        print(f"    平均: {avg_chunks:.1f}, 最小: {min_chunks}, 最大: {max_chunks}")
        for cnt, freq in sorted(Counter(chunk_counts).items()):
            print(f"    {cnt} 个 chunk: {freq} 个 task")

    summary_lens = [len(t.get("task_summary", "")) for t in tasks_data]
    if summary_lens:
        avg_len = sum(summary_lens) / len(summary_lens)
        print(f"\n  task_summary 长度 (字符):")
        print(f"    平均: {avg_len:.0f}, 最小: {min(summary_lens)}, 最大: {max(summary_lens)}")
        short = [t for t in tasks_data if len(t.get("task_summary", "")) < 50]
        if short:
            print(f"    过短 (<50字): {len(short)} 个")
            for s in short:
                print(f"      - {s.get('task_label', '?')}: {s.get('task_summary', '')}")

    # --- Chunks 分析 ---
    subsection("Chunks 分布")
    chunk_points, _ = client.scroll("chunks", limit=1000, with_payload=True, with_vectors=False)
    chunks_data = [p.payload for p in chunk_points if p.payload]

    print(f"  总 chunk 数: {len(chunks_data)}")

    raws = [c.get("raw_size_tokens", 0) for c in chunks_data]
    cleans = [c.get("cleaned_size_tokens", 0) for c in chunks_data]
    total_raw = sum(raws)
    total_clean = sum(cleans)
    if total_raw > 0:
        print(f"  Token 总量: raw={total_raw:,}, cleaned={total_clean:,}")
        print(f"  整体压缩率: {total_clean/total_raw:.1%}")

    has_summary = sum(1 for c in chunks_data if c.get("summary"))
    print(f"  Summary 覆盖率: {has_summary}/{len(chunks_data)} ({has_summary/len(chunks_data):.0%})")

    no_task = [c for c in chunks_data if not c.get("task_id")]
    if no_task:
        print(f"  未关联 task 的 chunk: {len(no_task)} 个")


# ============================================================
# 4. 向量近邻分析 (检查 embedding 质量)
# ============================================================

def show_neighbor_analysis(client: QdrantClient) -> None:
    section("4. 向量近邻分析 (检查 embedding 语义聚类)")

    task_points, _ = client.scroll("tasks", limit=1000, with_payload=True, with_vectors=True)
    if len(task_points) < 3:
        print("  Task 数量不足, 跳过")
        return

    print(f"  分析 {len(task_points)} 个 task 向量的近邻关系\n")

    interesting_pairs = []
    for pt in task_points[:10]:
        pl = pt.payload or {}
        query_label = pl.get("task_label", "?")

        vec = pt.vector
        if isinstance(vec, dict):
            vec = vec.get("dense", vec)

        hits = client.query_points(
            collection_name="tasks",
            query=vec,
            limit=4,
            with_payload=True,
        )

        print(f"  {query_label}")
        for h in hits.points:
            hpl = h.payload or {}
            hlabel = hpl.get("task_label", "?")
            if str(h.id) == str(pt.id):
                continue
            print(f"      -> {hlabel} (cosine={h.score:.4f})")
            if h.score > 0.85:
                interesting_pairs.append((query_label, hlabel, h.score))
        print()

    if interesting_pairs:
        subsection("高相似度 task 对 (cosine > 0.85)")
        for a, b, s in interesting_pairs:
            print(f"  [{s:.4f}] {a} <-> {b}")
        print(f"\n  共 {len(interesting_pairs)} 对高相似 task")
        print("  可能是同一 session 的连续 task, 或内容高度相关的 task")
    else:
        print("  无高相似度 task 对 (cosine > 0.85), embedding 区分度良好")


# ============================================================
# 5. 检索质量测试
# ============================================================

TEST_QUERIES = [
    # (query, expected_keywords) - 任一 keyword 出现在 label 或 summary 中即命中
    ("GPU对比分析", ["GPU", "算力", "A800", "H100"]),
    ("优化器学习率调度", ["优化器", "Adam", "SGD", "Lion"]),
    ("FlashAttention原理", ["FlashAttention", "tiling", "算术强度", "roofline"]),
    ("session管理与fork", ["fork", "/fork", "会话分支", "命令误解"]),
    ("浮点数格式转换", ["浮点", "FP16", "BF16", "float", "转换器"]),
]


def show_search_quality(client: QdrantClient) -> None:
    section("5. 检索质量测试")
    print("  使用 Dense 检索 (Cosine) 对 tasks 集合执行测试查询")
    print("  (基于关键词匹配, 适应 LLM 标签变化)\n")

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5",
            device="cpu",
            cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),
        )
    except Exception as e:
        print(f"  无法加载 embedding 模型: {e}")
        return

    correct = 0
    total = 0

    for query, keywords in TEST_QUERIES:
        qvec = encoder.encode([query], normalize_embeddings=True)[0].tolist()
        hits = client.query_points(
            collection_name="tasks",
            query=qvec,
            limit=5,
            with_payload=True,
        )

        top_labels = []
        for h in hits.points:
            label = (h.payload or {}).get("task_label", "?")
            top_labels.append((label, h.score))

        # 关键词匹配: 检查 top-5 中是否有 label 或 summary 包含任一关键词
        hit_any = False
        hit_details = []
        for h in hits.points:
            pl = h.payload or {}
            text = pl.get("task_label", "") + " " + pl.get("task_summary", "")
            matched_kw = [kw for kw in keywords if kw.lower() in text.lower()]
            if matched_kw:
                hit_any = True
                hit_details.append((pl.get("task_label", "?"), h.score, matched_kw))

        if hit_any:
            correct += 1
        total += 1

        mark = "OK" if hit_any else "MISS"
        print(f"  [{mark}] Query: \"{query}\"")
        print(f"     关键词: {keywords}")
        print(f"     Top-5:")
        for i, (label, score) in enumerate(top_labels, 1):
            matched = [kw for kw in keywords if kw.lower() in label.lower()]
            hit_mark = f" <-- 命中 {matched}" if matched else ""
            print(f"       {i}. [{score:.4f}] {label}{hit_mark}")
        print()

    print(f"  命中率: {correct}/{total} ({correct/total:.0%})")


# ============================================================
# 6. Chunk Summary 质量抽样
# ============================================================

def show_summary_quality() -> None:
    section("6. Chunk Summary 质量抽样")

    summaries_file = Path("./output/chunks_summary_p2.jsonl")
    chunks_file = Path("./output/chunks.jsonl")
    tasks_file = Path("./output/tasks.jsonl")

    if not summaries_file.exists():
        print("  chunks_summary_p2.jsonl 不存在, 跳过")
        return

    summaries = {}
    with open(summaries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                summaries[d["chunk_id"]] = d["summary"]

    chunks = {}
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    chunks[d["chunk_id"]] = d

    tasks = []
    if tasks_file.exists():
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))

    lens = [len(s) for s in summaries.values()]
    if not lens:
        print("  无 summary 数据")
        return

    avg_len = sum(lens) / len(lens)
    print(f"  总 chunk summary 数: {len(summaries)}")
    print(f"  长度 (字符): 平均={avg_len:.0f}, 最小={min(lens)}, 最大={max(lens)}")

    short_ids = [cid for cid, s in summaries.items() if len(s) < 30]
    if short_ids:
        print(f"\n  过短 summary (<30字): {len(short_ids)} 个")
        for cid in short_ids[:5]:
            print(f"    - {cid}: \"{summaries[cid]}\"")

    subsection("Task summary vs Chunk summary 对比 (抽样)")
    for task in random.sample(tasks, min(3, len(tasks))):
        t_label = task.get("task_label", "?")
        t_summary = task.get("task_summary", "")
        chunk_ids = task.get("chunk_ids", [])
        print(f"\n  Task: {t_label}")
        print(f"  task_summary: {t_summary[:100]}{'...' if len(t_summary) > 100 else ''}")
        for cid in chunk_ids[:2]:
            cs = summaries.get(cid, "(无)")
            print(f"    chunk {cid[-6:]}: {cs[:100]}{'...' if len(cs) > 100 else ''}")


# ============================================================
# Main
# ============================================================

def main():
    print("+" + "-"*54 + "+")
    print("|  Qdrant 数据检视 -- 里程碑第5步                    |")
    print("|  嵌入式 Qdrant (./qdrant_data)                     |")
    print("+" + "-"*54 + "+")

    qdrant_path = "./qdrant_data"
    if not Path(qdrant_path).exists():
        print(f"\nQdrant 数据目录不存在: {qdrant_path}")
        print("请先运行 code_p3_main.py 写入数据")
        sys.exit(1)

    client = QdrantClient(path=qdrant_path)

    show_collection_stats(client)
    show_sample_payloads(client, n=5)
    show_data_analysis(client)
    show_neighbor_analysis(client)
    show_search_quality(client)
    show_summary_quality()

    section("检视完成")
    print("  后续建议:")
    print("  - 检索命中率低 -> 换更强 embedding 模型 (Qwen3-Embedding)")
    print("  - 高相似 task 对 -> 合并或优化 task 抽取 prompt")
    print("  - summary 过短 -> 调整 Phase 2 prompt 最小长度要求")
    print("  - 完整搜索体验 -> code_p4_search_cli.py --interactive")


if __name__ == "__main__":
    main()
