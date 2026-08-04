#!/usr/bin/env python3
"""
里程碑检视工具: Qdrant 数据浏览 + 检索质量评估
替代 Qdrant Dashboard (项目用嵌入式 Qdrant, 无 Web UI)

功能:
1. 集合统计 (点数, 维度, 状态)
2. 样本浏览 (随机抽取 payload)
3. 数据分析 (session/task/chunk 分布, 压缩率)
4. 向量近邻分析 (embedding 聚类质量)
5. 检索质量测试 (预设查询 + 名义/有效双口径命中率)
6. Chunk Summary 质量抽样
7. 检视结论 (触发式事实汇总, 无固定建议)
"""
import json
import sys
import random
import hashlib
import argparse
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

# 复用 Phase 1 的全局配置加载器 (点号路径 + 脚本同目录回退)
from code_p1_utils import Config

# 仓库根目录 (src/ 的上一级), 用于解析相对路径, 使脚本可从任意 CWD 运行
REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p: str) -> Path:
    """将配置中的相对路径解析为相对仓库根目录的绝对路径。"""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


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

def show_sample_payloads(client: QdrantClient, collections: dict, n: int = 5) -> None:
    section("2. 样本浏览")
    tasks_col = collections["tasks"]
    summary_col = collections["chunks_summary"]
    for cname in [tasks_col, summary_col, collections["chunks_cleaned_text"]]:
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
            if cname == tasks_col:
                print(f"      task_id:    {pl.get('task_id', '?')}")
                print(f"      session_id: {pl.get('session_id', '?')}")
                print(f"      task_label: {pl.get('task_label', '?')}")
                summary = pl.get("task_summary", "")
                print(f"      task_summary: {summary[:120]}{'...' if len(summary) > 120 else ''}")
                chunk_ids = pl.get("chunk_ids", [])
                print(f"      chunk_ids:  {len(chunk_ids)} 个 {chunk_ids[:3]}{'...' if len(chunk_ids) > 3 else ''}")
            else:
                # chunks_summary 或 chunks_cleaned_text
                print(f"      chunk_id:   {pl.get('chunk_id', '?')}")
                print(f"      session_id: {pl.get('session_id', '?')}")
                print(f"      turn_index: {pl.get('turn_index', '?')}")
                print(f"      task_id:    {pl.get('task_id', '?')}")
                if cname == summary_col:
                    summary = pl.get("summary", "")
                raw = pl.get("raw_size_tokens", 0)
                cleaned = pl.get("cleaned_size_tokens", 0)
                ratio = f"{cleaned/raw:.1%}" if raw > 0 else "N/A"
                print(f"      tokens:     raw={raw}, cleaned={cleaned}, 压缩率={ratio}")


# ============================================================
# 3. 数据分析
# ============================================================

def show_data_analysis(client: QdrantClient, collections: dict) -> None:
    section("3. 数据分析")

    tasks_col = collections["tasks"]
    summary_col = collections["chunks_summary"]

    # --- Tasks 分析 ---
    subsection("Tasks 分布")
    task_points, _ = client.scroll(tasks_col, limit=1000, with_payload=True, with_vectors=False)
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
    chunk_points, _ = client.scroll(summary_col, limit=1000, with_payload=True, with_vectors=False)
    chunks_data = [p.payload for p in chunk_points if p.payload]

    print(f"  总 chunk 数: {len(chunks_data)} (来源: {summary_col})")

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

    return len(no_task)


# ============================================================
# 4. 向量近邻分析 (检查 embedding 质量)
# ============================================================

def show_neighbor_analysis(client: QdrantClient, collections: dict) -> list:
    section("4. 向量近邻分析 (检查 embedding 语义聚类)")

    tasks_col = collections["tasks"]
    task_points, _ = client.scroll(tasks_col, limit=1000, with_payload=True, with_vectors=True)
    if len(task_points) < 3:
        print("  Task 数量不足, 跳过")
        return []

    print(f"  分析 {len(task_points)} 个 task 向量的近邻关系\n")

    interesting_pairs = []
    for pt in task_points[:10]:
        pl = pt.payload or {}
        query_label = pl.get("task_label", "?")

        vec = pt.vector
        if isinstance(vec, dict):
            vec = vec.get("dense", vec)

        hits = client.query_points(
            collection_name=tasks_col,
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

    return interesting_pairs


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


def show_search_quality(client: QdrantClient, collections: dict,
                        model_name: str, device: str,
                        cache_folder: str | None) -> dict | None:
    section("5. 检索质量测试")
    tasks_col = collections["tasks"]
    print(f"  使用 Dense 检索 (Cosine) 对 {tasks_col} 集合执行测试查询")
    print(f"  (模型: {model_name})")
    print("  名义命中 = top-5 内任一条的 label+summary 含关键词")
    print("  有效命中 = 全排名中首个 label 含关键词的 task 进入 top-5\n")

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder or os.path.expanduser("~/.cache/huggingface/hub"),
        )
    except Exception as e:
        print(f"  无法加载 embedding 模型: {e}")
        return None

    try:
        total_points = getattr(client.get_collection(tasks_col), "points_count", 0) or 0
    except Exception:
        total_points = 0
    fetch_limit = max(total_points, 5)

    results = []

    for query, keywords in TEST_QUERIES:
        qvec = encoder.encode([query], normalize_embeddings=True)[0].tolist()
        hits = client.query_points(
            collection_name=tasks_col,
            query=qvec,
            limit=fetch_limit,
            with_payload=True,
        ).points

        top5 = hits[:5]
        top_labels = [
            ((h.payload or {}).get("task_label", "?"), h.score) for h in top5
        ]

        # 名义命中: top-5 中任一条的 label 或 summary 包含任一关键词
        hit_any = False
        hit_details = []
        for h in top5:
            pl = h.payload or {}
            text = pl.get("task_label", "") + " " + pl.get("task_summary", "")
            matched_kw = [kw for kw in keywords if kw.lower() in text.lower()]
            if matched_kw:
                hit_any = True
                hit_details.append((pl.get("task_label", "?"), h.score, matched_kw))

        # 有效命中: 全排名中首个 label 含关键词的位置
        label_first_rank = None
        label_first_label = None
        label_first_score = None
        for i, h in enumerate(hits, 1):
            label = (h.payload or {}).get("task_label", "")
            if any(kw.lower() in label.lower() for kw in keywords):
                label_first_rank, label_first_label, label_first_score = i, label, h.score
                break

        effective_hit = label_first_rank is not None and label_first_rank <= 5
        drift = hit_any and not effective_hit

        mark = "OK" if effective_hit else ("DRIFT" if drift else "MISS")
        print(f"  [{mark}] Query: \"{query}\"")
        print(f"     关键词: {keywords}")
        print(f"     Top-5:")
        for i, (label, score) in enumerate(top_labels, 1):
            matched = [kw for kw in keywords if kw.lower() in label.lower()]
            hit_mark = f" <-- label 命中 {matched}" if matched else ""
            print(f"       {i}. [{score:.4f}] {label}{hit_mark}")
        if label_first_rank is not None:
            print(
                f"     label 首个命中: rank {label_first_rank} "
                f"\"{label_first_label}\" (cosine={label_first_score:.4f})"
            )
        else:
            print(f"     label 首个命中: 无 (前 {len(hits)} 名均未在 label 命中)")
        print()

        results.append({
            "query": query,
            "nominal_hit": hit_any,
            "effective_hit": effective_hit,
            "drift": drift,
            "label_first_rank": label_first_rank,
            "label_first_label": label_first_label,
            "label_first_score": label_first_score,
            "top1_label": top_labels[0][0] if top_labels else "(空集合)",
            "top1_score": top_labels[0][1] if top_labels else None,
            "nominal_matched_kw": sorted({kw for _, _, kws in hit_details for kw in kws}),
        })

    total = len(results)
    denom = total or 1
    nominal = sum(1 for r in results if r["nominal_hit"])
    effective = sum(1 for r in results if r["effective_hit"])
    drift_count = sum(1 for r in results if r["drift"])
    print(f"  名义命中率: {nominal}/{total} ({nominal/denom:.0%})")
    print(f"  有效命中率: {effective}/{total} ({effective/denom:.0%})")
    if drift_count:
        print(f"  其中 DRIFT (名义命中但 label 目标跌出 top-5): {drift_count} 条, 详见检视结论")

    return {"queries": results, "nominal": nominal, "effective": effective, "total": total}


# ============================================================
# 6. Chunk Summary 质量抽样
# ============================================================

def show_summary_quality(summaries_file: Path, chunks_file: Path, tasks_file: Path) -> list:
    section("6. Chunk Summary 质量抽样")

    if not summaries_file.exists():
        print(f"  {summaries_file} 不存在, 跳过")
        return []

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
        return []

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

    return short_ids


# ============================================================
# 7. 检视结论 (触发式事实汇总)
# ============================================================

def print_conclusion(search_result: dict | None, high_sim_pairs: list,
                     short_ids: list, no_task_count: int) -> None:
    section("7. 检视结论 (自动生成, 仅陈述事实)")

    triggered = []

    if search_result is None:
        triggered.append("检索质量测试未执行 (embedding 模型加载失败), 无检索结论")
    else:
        s = search_result
        print(f"  检索命中率: 名义 {s['nominal']}/{s['total']}, 有效 {s['effective']}/{s['total']}")
        print("  (名义 = top-5 内 label+summary 含关键词; 有效 = label 首个命中进入 top-5)")
        for r in s["queries"]:
            if r["drift"]:
                kws = "/".join(r["nominal_matched_kw"]) if r["nominal_matched_kw"] else "?"
                triggered.append(
                    f"查询 \"{r['query']}\" 名义命中但 label 首个命中仅 rank {r['label_first_rank']}: "
                    f"\"{r['label_first_label']}\" (cosine={r['label_first_score']:.4f}); "
                    f"top-1 为 \"{r['top1_label']}\" (cosine={r['top1_score']:.4f}), "
                    f"名义命中由 summary 关键词 [{kws}] 贡献"
                )
            elif not r["nominal_hit"]:
                triggered.append(
                    f"查询 \"{r['query']}\" 全量排名中无任何关键词命中 (label+summary)"
                )

    if high_sim_pairs:
        pair_desc = "; ".join(
            f"{a} <-> {b} ({score:.4f})" for a, b, score in high_sim_pairs
        )
        triggered.append(f"高相似 task 对 (cosine > 0.85) {len(high_sim_pairs)} 对: {pair_desc}")

    if short_ids:
        shown = ", ".join(short_ids[:5]) + (" ..." if len(short_ids) > 5 else "")
        triggered.append(f"过短 chunk summary (<30字) {len(short_ids)} 条: {shown}")

    if no_task_count:
        triggered.append(f"未关联 task 的 chunk: {no_task_count} 个")

    if not triggered:
        print("  无触发项: 全部预设查询有效命中 top-5, 无高相似 task 对, 无过短 summary")
    else:
        print(f"\n  触发项 {len(triggered)} 条:")
        for i, item in enumerate(triggered, 1):
            print(f"  [{i}] {item}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qdrant 数据检视 / 检索质量评估 (里程碑工具)"
    )
    parser.add_argument(
        "--config", default="config/code_p3_config.yaml",
        help="Phase 3 配置文件 (默认 config/code_p3_config.yaml, 复用集合/模型/路径设置)",
    )
    parser.add_argument(
        "-n", "--samples", type=int, default=5,
        help="样本浏览每个集合抽取的点数 (默认 5)",
    )
    args = parser.parse_args()

    # 解析配置路径: 支持从任意 CWD 运行 (相对路径回退到仓库根目录)
    config_path = Path(args.config)
    if not config_path.exists() and not config_path.is_absolute():
        config_path = REPO_ROOT / args.config
    config = Config.load(str(config_path))

    # 从配置读取集合名 / 路径 / 模型, 不再硬编码
    collections = {
        "tasks": config.get("qdrant.collections.tasks", "tasks"),
        "chunks_summary": config.get("qdrant.collections.chunks_summary", "chunks_summary"),
        "chunks_cleaned_text": config.get(
            "qdrant.collections.chunks_cleaned_text", "chunks_cleaned_text"
        ),
    }
    qdrant_path = _resolve_path(config.get("qdrant.path", "./qdrant_data"))
    model_name = config.get("embedding.model", "BAAI/bge-small-zh-v1.5")
    device = config.get("embedding.device", "cpu")
    cache_folder = config.get("embedding.cache_folder", None)

    summaries_file = _resolve_path(
        config.get("phase2.chunk_summaries_file", "./output/chunks_summary_p2.jsonl")
    )
    chunks_file = _resolve_path(config.get("phase1.chunks_file", "./output/chunks.jsonl"))
    tasks_file = _resolve_path(config.get("phase2.tasks_file", "./output/tasks.jsonl"))

    print("+" + "-"*54 + "+")
    print("|  Qdrant 数据检视 -- 里程碑检视工具                 |")
    print(f"|  嵌入式 Qdrant ({qdrant_path})")
    print("+" + "-"*54 + "+")

    if not qdrant_path.exists():
        print(f"\nQdrant 数据目录不存在: {qdrant_path}")
        print("请先运行 code_p3_main.py 写入数据")
        sys.exit(1)

    client = QdrantClient(path=str(qdrant_path))

    show_collection_stats(client)
    show_sample_payloads(client, collections, n=args.samples)
    no_task_count = show_data_analysis(client, collections)
    high_sim_pairs = show_neighbor_analysis(client, collections)
    search_result = show_search_quality(client, collections, model_name, device, cache_folder)
    short_ids = show_summary_quality(summaries_file, chunks_file, tasks_file)

    client.close()

    print_conclusion(search_result, high_sim_pairs, short_ids, no_task_count)


if __name__ == "__main__":
    main()
