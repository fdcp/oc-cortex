"""
Phase 5 多模型 × 多模式 完整流水线对比测试

测试维度:
  - 模型: nemotron-3-ultra-free, deepseek-v4-flash-free, mimo-v2.5-free, hy3-free
  - 模式: triple (三元组关系图谱), entity (实体共现图谱)
  - 完整流水线: 抽取 → 实体收集 → 实体对齐 → 图谱构建

评估指标:
  - 抽取阶段: 成功率, 三元组/实体数, 耗时, content_empty 次数
  - 对齐阶段: 候选合并对数, LLM 确认合并对数, 耗时
  - 图谱阶段: 节点数, 边数, 构建耗时
  - 总体: 全流水线耗时

用法:
  # 设置 API key
  export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

  # 全量测试 (4 模型 × 2 模式, 全部 task)
  python3 tests/test_p5/test_p5.py

  # 指定模型
  python3 tests/test_p5/test_p5.py --models nemotron-3-ultra-free deepseek-v4-flash-free

  # 仅 triple 模式
  python3 tests/test_p5/test_p5.py --modes triple

  # 快速验证 (前 5 个 task)
  python3 tests/test_p5/test_p5.py --limit 5

  # 高并发
  python3 tests/test_p5/test_p5.py --concurrency 8
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# 添加 src 到路径 (tests/test_p5/ -> tests/ -> project root)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from loguru import logger

# ── 离线模式 (必须在 import sentence_transformers 之前) ──
import code_p3_hf_config
code_p3_hf_config.setup_hf_env(offline_mode=True)

from code_p1_utils import Config
from code_p2_models import Task
from code_p5_models import Triple, Entity, KGStats
from code_p5_kg_builder import KGBuilder


# ============================================================
# 配置
# ============================================================

DEFAULT_MODELS = [
    "nemotron-3-ultra-free",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "hy3-free",
]

DEFAULT_MODES = ["triple", "entity"]

CONFIG_FILE = str(_project_root / "config" / "code_p5_config.yaml")
TASKS_FILE = str(_project_root / "output" / "tasks.jsonl")


# ============================================================
# 数据加载
# ============================================================

def load_tasks(path: str) -> list[Task]:
    """从 JSONL 加载 Task 列表"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append(Task.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Task 解析失败 (行 {line_num}): {e}")
    return tasks


# ============================================================
# 单次 benchmark
# ============================================================

def benchmark_run(
    model: str,
    mode: str,
    tasks: list[Task],
    config: Config,
    concurrency: int,
    output_dir: str,
) -> dict:
    """
    运行单个 model×mode 的完整流水线

    每个 run 使用隔离的 Qdrant 临时目录, 避免跨 run 的 entities 集合污染。
    """
    tag = f"{model} × {mode}"
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Benchmark: {tag}")
    logger.info(f"{'=' * 70}")

    # 隔离 Qdrant 路径
    qdrant_tmp = tempfile.mkdtemp(prefix=f"p5bench_{model}_{mode}_")
    run_dir = Path(output_dir) / f"{model}__{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model,
        "mode": mode,
        "tasks_count": len(tasks),
        "success": False,
        "extraction": {},
        "alignment": {},
        "graph": {},
        "errors": [],
    }

    try:
        # ── 初始化 KGBuilder ──
        t_init = time.time()
        # 测试要遍历指定 model, 所以在 post_init 之前先改 config
        # 让 triple_model/entity_model 都指向本轮要测的 model
        config.set("llm.triple_model", model)
        config.set("llm.entity_model", model)
        # 同步确保 extraction_mode 跟 mode 参数一致 (post_init 会再次确认)
        config.set("knowledge_graph.extraction_mode", mode)

        builder = KGBuilder(
            extraction_mode=mode,
            max_retries=config.get("llm.max_retries", 3),
            content_retries=config.get("llm.content_retries", 2),
            timeout=config.get("llm.timeout", 120),
            max_tokens=config.get("llm.max_tokens", 8000),
            merge_max_tokens=config.get("llm.merge_max_tokens", 200),
            merge_batch_size=config.get("llm.merge_batch_size", 20),
            merge_batch_enable=config.get("llm.merge_batch_enable", False),
            concurrency=concurrency,
            embedding_model=config.get("embedding.model", "BAAI/bge-small-zh-v1.5"),
            embedding_dim=config.get("embedding.dim", 512),
            embedding_batch_size=config.get("embedding.batch_size", 32),
            embedding_device=config.get("embedding.device", "cpu"),
            embedding_cache_folder=config.get("embedding.cache_folder"),
            embedding_offline_mode=True,
            qdrant_path=qdrant_tmp,
            entities_collection=config.get("qdrant.entities_collection", "entities"),
            alignment_threshold=config.get("knowledge_graph.entity_alignment_threshold", 0.92),
        )
        builder.post_init(config)
        result["init_time"] = round(time.time() - t_init, 1)
        logger.info(
            f"  初始化耗时: {result['init_time']:.1f}s (Qdrant: {qdrant_tmp}, "
            f"model={builder.model}, base_url={builder.base_url})"
        )

        t_pipeline = time.time()
        triples_file = str(run_dir / "triples.jsonl")
        entity_extract_file = str(run_dir / "entity_extract.jsonl")
        entities_file = str(run_dir / "entities.jsonl")
        graph_path = str(run_dir / "knowledge_graph.gpickle")
        graph_json = str(run_dir / "knowledge_graph.json")
        inverted_index_file = str(run_dir / "inverted_index.json")

        # ================================================================
        # 阶段 1: 抽取
        # ================================================================
        t_extract = time.time()
        if mode == "triple":
            triples = builder.extract_all_triples(tasks, output_file=triples_file)
            result["extraction"] = {
                "triples_count": len(triples),
                "success": len(triples) > 0,
                "avg_per_task": round(len(triples) / len(tasks), 1) if tasks else 0,
            }
            logger.info(
                f"  抽取完成: {len(triples)} 三元组, "
                f"{len(triples)/len(tasks):.1f}/task"
            )
        else:
            inverted_index = builder.extract_all_entities(
                tasks, output_file=entity_extract_file,
            )
            total_entities_in_index = len(inverted_index)
            result["extraction"] = {
                "unique_entities": total_entities_in_index,
                "success": total_entities_in_index > 0,
                "avg_per_task": round(total_entities_in_index / len(tasks), 1) if tasks else 0,
            }
            logger.info(
                f"  抽取完成: {total_entities_in_index} 个唯一实体, "
                f"{total_entities_in_index/len(tasks):.1f}/task"
            )

        result["extraction"]["time"] = round(time.time() - t_extract, 1)
        logger.info(f"  抽取耗时: {result['extraction']['time']:.1f}s")

        if not result["extraction"]["success"]:
            result["errors"].append("抽取阶段未产出任何结果")
            return result

        # ================================================================
        # 阶段 2: 实体收集 + 对齐
        # ================================================================
        t_align = time.time()
        if mode == "triple":
            entity_map = builder.collect_entities(triples)
        else:
            entity_map = builder.collect_entities_from_index(inverted_index)

        result["alignment"]["raw_entities"] = len(entity_map)

        canonical_map = builder.align_entities(entity_map)
        merged_pairs = sum(
            1 for name, canon in canonical_map.items() if name != canon
        )
        unique_entities = len(set(canonical_map.values()))

        result["alignment"].update({
            "merged_pairs": merged_pairs,
            "unique_entities": unique_entities,
            "time": round(time.time() - t_align, 1),
        })
        logger.info(
            f"  对齐完成: {len(entity_map)} 原始 → {unique_entities} 唯一, "
            f"{merged_pairs} 合并对"
        )
        logger.info(f"  对齐耗时: {result['alignment']['time']:.1f}s")

        # ================================================================
        # 阶段 3: 图谱构建
        # ================================================================
        t_graph = time.time()
        if mode == "triple":
            G = builder.build_graph(triples, canonical_map, entity_map)
            builder.save_graph(G, graph_path, graph_json, extraction_mode="triple")
        else:
            G = builder.build_cooccurrence_graph(
                inverted_index, canonical_map, entity_map,
            )
            builder.save_graph(G, graph_path, graph_json, extraction_mode="entity")
            builder.save_inverted_index(
                inverted_index, canonical_map, inverted_index_file,
            )

        result["graph"] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "time": round(time.time() - t_graph, 1),
        }
        logger.info(
            f"  图谱构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边"
        )
        logger.info(f"  图谱构建耗时: {result['graph']['time']:.1f}s")

        # 图谱补充统计
        if G.number_of_nodes() > 0:
            from collections import Counter
            degrees = dict(G.degree()).values()
            degree_dist = Counter(degrees)
            result["graph"]["avg_degree"] = round(
                sum(degrees) / len(degrees), 1
            ) if degrees else 0
            result["graph"]["max_degree"] = max(degrees) if degrees else 0
            # Top-5 高度节点
            top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
            result["graph"]["top_nodes"] = [
                {"name": n, "degree": d, "type": G.nodes[n].get("entity_type", "?")}
                for n, d in top_nodes
            ]

        result["total_time"] = round(time.time() - t_pipeline, 1)
        result["success"] = True
        logger.info(f"  流水线总耗时: {result['total_time']:.1f}s")

    except Exception as e:
        result["errors"].append(f"{type(e).__name__}: {str(e)}")
        logger.error(f"  {tag} 失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时 Qdrant
        try:
            shutil.rmtree(qdrant_tmp, ignore_errors=True)
        except Exception:
            pass

    return result


# ============================================================
# 对比表
# ============================================================

def print_comparison_table(all_results: list[dict]):
    """打印对比表"""
    # 按模式分组
    for mode in ["triple", "entity"]:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue

        print(f"\n{'=' * 100}")
        print(f"模式: {mode}")
        print(f"{'=' * 100}")

        if mode == "triple":
            header = (
                f"{'模型':<28} {'状态':>4} {'三元组':>7} {'平均/t':>6} "
                f"{'原始实体':>8} {'唯一实体':>8} {'合并对':>6} "
                f"{'节点':>5} {'边':>5} "
                f"{'抽取s':>7} {'对齐s':>7} {'图谱s':>7} {'总计s':>7}"
            )
        else:
            header = (
                f"{'模型':<28} {'状态':>4} {'实体数':>7} {'平均/t':>6} "
                f"{'原始实体':>8} {'唯一实体':>8} {'合并对':>6} "
                f"{'节点':>5} {'边':>5} "
                f"{'抽取s':>7} {'对齐s':>7} {'图谱s':>7} {'总计s':>7}"
            )
        print(header)
        print("-" * 100)

        for r in mode_results:
            status = "OK" if r["success"] else "FAIL"
            ext = r.get("extraction", {})
            align = r.get("alignment", {})
            graph = r.get("graph", {})

            if mode == "triple":
                count = ext.get("triples_count", 0)
            else:
                count = ext.get("unique_entities", 0)

            print(
                f"{r['model']:<28} "
                f"{status:>4} "
                f"{count:>7d} "
                f"{ext.get('avg_per_task', 0):>6.1f} "
                f"{align.get('raw_entities', 0):>8d} "
                f"{align.get('unique_entities', 0):>8d} "
                f"{align.get('merged_pairs', 0):>6d} "
                f"{graph.get('nodes', 0):>5d} "
                f"{graph.get('edges', 0):>5d} "
                f"{ext.get('time', 0):>7.1f} "
                f"{align.get('time', 0):>7.1f} "
                f"{graph.get('time', 0):>7.1f} "
                f"{r.get('total_time', 0):>7.1f}"
            )

            # 打印错误
            for err in r.get("errors", []):
                print(f"  ⚠ {err[:80]}")

    # 跨模式汇总
    print(f"\n{'=' * 100}")
    print("汇总 (全部模式)")
    print(f"{'=' * 100}")
    header = (
        f"{'模型':<28} {'模式':<8} {'状态':>4} {'节点':>5} {'边':>5} "
        f"{'总耗时s':>8} {'错误'}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        graph = r.get("graph", {})
        errs = "; ".join(r.get("errors", []))[:40]
        print(
            f"{r['model']:<28} {r['mode']:<8} "
            f"{'OK' if r['success'] else 'FAIL':>4} "
            f"{graph.get('nodes', 0):>5d} "
            f"{graph.get('edges', 0):>5d} "
            f"{r.get('total_time', 0):>8.1f} "
            f"{errs}"
        )


# ============================================================
# 保存结果
# ============================================================

def save_results(all_results: list[dict], output_dir: str):
    """保存 JSON + Markdown 报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 详细结果
    json_path = output_dir / "test_p5_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {json_path}")

    # Markdown 报告
    md_path = output_dir / "test_p5_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase 5 多模型 × 多模式 Benchmark 报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"测试 task 数: {all_results[0]['tasks_count'] if all_results else 0}\n\n")

        f.write("## 对比表\n\n")

        for mode in ["triple", "entity"]:
            mode_results = [r for r in all_results if r["mode"] == mode]
            if not mode_results:
                continue

            f.write(f"### {mode} 模式\n\n")
            f.write("| 模型 | 状态 | 抽取量 | 平均/t | 原始实体 | 唯一实体 | 合并对 | 节点 | 边 | 抽取(s) | 对齐(s) | 图谱(s) | 总计(s) |\n")
            f.write("|------|------|--------|--------|----------|----------|--------|------|-----|---------|---------|---------|----------|\n")

            for r in mode_results:
                ext = r.get("extraction", {})
                align = r.get("alignment", {})
                graph = r.get("graph", {})
                count = ext.get("triples_count", 0) if mode == "triple" else ext.get("unique_entities", 0)

                f.write(
                    f"| {r['model']} "
                    f"| {'OK' if r['success'] else 'FAIL'} "
                    f"| {count} "
                    f"| {ext.get('avg_per_task', 0):.1f} "
                    f"| {align.get('raw_entities', 0)} "
                    f"| {align.get('unique_entities', 0)} "
                    f"| {align.get('merged_pairs', 0)} "
                    f"| {graph.get('nodes', 0)} "
                    f"| {graph.get('edges', 0)} "
                    f"| {ext.get('time', 0):.1f} "
                    f"| {align.get('time', 0):.1f} "
                    f"| {graph.get('time', 0):.1f} "
                    f"| {r.get('total_time', 0):.1f} |\n"
                )
            f.write("\n")

        # 图谱 Top 节点
        f.write("## 图谱 Top-5 高度节点\n\n")
        for r in all_results:
            if not r["success"]:
                continue
            top = r.get("graph", {}).get("top_nodes", [])
            if top:
                f.write(f"### {r['model']} × {r['mode']}\n\n")
                f.write("| 实体 | 度 | 类型 |\n")
                f.write("|------|-----|------|\n")
                for n in top:
                    f.write(f"| {n['name']} | {n['degree']} | {n['type']} |\n")
                f.write("\n")

        # 错误汇总
        errors = [(r["model"], r["mode"], r["errors"]) for r in all_results if r["errors"]]
        if errors:
            f.write("## 错误汇总\n\n")
            for model, mode, errs in errors:
                f.write(f"- **{model} × {mode}**: {'; '.join(errs)}\n")
            f.write("\n")

    logger.info(f"Markdown 报告已保存: {md_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5 多模型 × 多模式 完整流水线对比测试")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"模型列表 (默认: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help=f"模式列表 (默认: {' '.join(DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="只处理前 N 个 task (0=全量)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="LLM 并发数 (默认 4)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录 (默认: tests/test_p5/)",
    )
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE,
        help="配置文件路径",
    )
    args = parser.parse_args()

    output_dir = args.output or str(_project_root / "tests" / "test_p5")

    # 1. 加载配置
    config = Config.load(args.config)

    # 2. 加载 tasks
    tasks_file = config.get("phase2.tasks_file", TASKS_FILE)
    logger.info(f"加载 tasks: {tasks_file}")
    tasks = load_tasks(tasks_file)
    logger.info(f"加载 {len(tasks)} 个 task")

    if not tasks:
        logger.error("没有加载到任何 task")
        sys.exit(1)

    if args.limit > 0:
        tasks = tasks[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个 task")

    # 3. 运行组合
    models = args.models
    modes = args.modes
    total_runs = len(models) * len(modes)

    logger.info(f"\n{'#' * 70}")
    logger.info(f"Phase 5 Benchmark: {len(models)} 模型 × {len(modes)} 模式 = {total_runs} 次运行")
    logger.info(f"模型: {', '.join(models)}")
    logger.info(f"模式: {', '.join(modes)}")
    logger.info(f"Tasks: {len(tasks)} 个")
    logger.info(f"并发: {args.concurrency}")
    logger.info(f"{'#' * 70}")

    t_start = time.time()
    all_results = []

    for model in models:
        for mode in modes:
            result = benchmark_run(
                model=model,
                mode=mode,
                tasks=tasks,
                config=config,
                concurrency=args.concurrency,
                output_dir=output_dir,
            )
            all_results.append(result)

    total_time = time.time() - t_start
    logger.info(f"\n全部 {total_runs} 次运行完成, 总耗时: {total_time:.1f}s")

    # 4. 打印对比表
    print_comparison_table(all_results)

    # 5. 保存结果
    save_results(all_results, output_dir)

    print(f"\nBenchmark 总耗时: {total_time:.1f}s")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
