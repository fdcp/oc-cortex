"""
Phase 5 主入口: 知识图谱构建
流程: 加载 tasks → 抽取(triple/entity) → 实体对齐 → 图谱构建 → 可视化

两种抽取模式:
  triple: LLM 抽取 (head, relation, tail) 三元组 → 关系图谱
  entity: LLM 直接抽取关键实体 → 倒排索引 + 共现图谱

用法:
  # 默认从 opencode_models.yaml 读 base_url, 从 auth.json 读 api_key
  python code_p5_main.py --config code_p5_config.yaml
  python code_p5_main.py --config code_p5_config.yaml --extraction_mode entity
  python code_p5_main.py --config code_p5_config.yaml --concurrency 8
  python code_p5_main.py --config code_p5_config.yaml --limit 5
  python code_p5_main.py --config code_p5_config.yaml --skip-alignment
  python code_p5_main.py --config code_p5_config.yaml --visualize
"""
import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

# ── 离线模式 (必须在 import sentence_transformers 之前) ──
import code_p3_hf_config
code_p3_hf_config.setup_hf_env(offline_mode=True)

from code_p1_utils import Config, setup_logger
from code_p2_models import Task
from code_p5_models import Triple, Entity, KGStats
from code_p5_kg_builder import KGBuilder


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


def load_existing_triples(path: str) -> list[Triple]:
    """从 JSONL 加载已有三元组 (断点续传)"""
    triples = []
    if not Path(path).exists():
        return triples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(Triple.from_dict(json.loads(line)))
    return triples


def load_existing_entities(path: str) -> dict[str, list[str]]:
    """从 JSONL 加载已有实体提取结果 (断点续传), 返回倒排索引"""
    inverted_index: dict[str, list[str]] = {}
    if not Path(path).exists():
        return inverted_index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    task_id = record.get("task_id", "")
                    entities = record.get("entities", [])
                    for entity in entities:
                        if entity not in inverted_index:
                            inverted_index[entity] = []
                        if task_id not in inverted_index[entity]:
                            inverted_index[entity].append(task_id)
                except json.JSONDecodeError:
                    continue
    return inverted_index


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: 知识图谱构建")
    parser.add_argument(
        "--config", default="config/code_p5_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="并发线程数 (覆盖配置文件)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 个 task (快速验证)",
    )
    parser.add_argument(
        "--skip-alignment", action="store_true",
        help="跳过实体对齐 (每个实体保持原名)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="构建完成后自动运行 pyvis 可视化",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="跳过抽取, 直接加载已有 triples/entities 文件",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新抽取 (删除已有 triples/entities 文件后从零开始), "
             "与 --skip-extraction 互斥",
    )
    parser.add_argument(
        "--extraction_mode", type=str, default=None,
        choices=["triple", "entity"],
        help="抽取模式 (覆盖配置文件中的 knowledge_graph.extraction_mode, "
             "同时决定使用 llm.triple_model 还是 llm.entity_model)",
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = Config.load(args.config)
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("Phase 5: 知识图谱构建")
    logger.info("=" * 60)

    # 2. 加载 tasks
    tasks_file = config.get("phase2.tasks_file", "./output/tasks.jsonl")
    logger.info(f"加载 tasks: {tasks_file}")
    tasks = load_tasks(tasks_file)
    logger.info(f"加载 {len(tasks)} 个 task")

    if not tasks:
        logger.error("没有加载到任何 task")
        sys.exit(1)

    # 限制处理数量
    if args.limit:
        tasks = tasks[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个 task")

    # 3. 初始化 KGBuilder
    concurrency = args.concurrency or config.get("llm.concurrency", 4)

    # 设置 HF 离线模式
    offline_mode = config.get("embedding.offline_mode", True)
    if offline_mode:
        code_p3_hf_config.setup_hf_env(
            offline_mode=True,
            cache_folder=config.get("embedding.cache_folder"),
        )

    builder = KGBuilder(
        extraction_mode=args.extraction_mode,
        max_retries=config.get("llm.max_retries", 3),
        content_retries=config.get("llm.content_retries", 2),
        timeout=config.get("llm.timeout", 120),
        max_tokens=config.get("llm.max_tokens", 4000),
        merge_max_tokens=config.get("llm.merge_max_tokens", 200),
        merge_batch_size=config.get("llm.merge_batch_size", 20),
        merge_batch_enable=config.get("llm.merge_batch_enable", False),
        concurrency=concurrency,
        rate_limit_per_sec=config.get("llm.rate_limit_per_sec", 2.0),
        embedding_model=config.get("embedding.model", "BAAI/bge-small-zh-v1.5"),
        embedding_dim=config.get("embedding.dim", 512),
        embedding_batch_size=config.get("embedding.batch_size", 32),
        embedding_device=config.get("embedding.device", "cpu"),
        embedding_cache_folder=config.get("embedding.cache_folder"),
        embedding_offline_mode=offline_mode,
        qdrant_path=config.get("qdrant.path", "./qdrant_data"),
        qdrant_url=os.environ.get("QDRANT_URL") or config.get("qdrant.url"),
        entities_collection=config.get("qdrant.entities_collection", "entities"),
        alignment_threshold=config.get("knowledge_graph.entity_alignment_threshold", 0.92),
    )

    # LLM 后初始化: 覆盖 extraction_mode → 选 model (triple/entity) → 加载 opencode_models.yaml → 初始化 client
    builder.post_init(config)

    # 读取最终的抽取模式 (可能已被 args.extraction_mode 覆盖)
    extraction_mode = config.get("knowledge_graph.extraction_mode", "triple")
    logger.info(f"抽取模式: {extraction_mode} (model={builder.model})")

    if extraction_mode == "entity":
        output_dir = config.get("knowledge_graph.entity_output_dir", "./output/entity")
    else:
        output_dir = config.get("knowledge_graph.triple_output_dir", "./output/triple")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 所有输出文件从 output_dir 派生 (两种模式互不干扰)
    triples_file = str(Path(output_dir) / "triples.jsonl")
    entities_file = str(Path(output_dir) / "entities.jsonl")
    graph_path = str(Path(output_dir) / "knowledge_graph.gpickle")
    graph_json = str(Path(output_dir) / "knowledge_graph.json")
    html_output = str(Path(output_dir) / "knowledge_graph.html")
    entity_extract_file = str(Path(output_dir) / "entity_extract.jsonl")
    inverted_index_file = str(Path(output_dir) / "inverted_index.json")
    db_path = None  # SQLite db path (set in each branch below)

    logger.info(f"输出目录: {output_dir}")

    t0 = time.time()
    triples = []          # triple 模式使用
    inverted_index = {}   # entity 模式使用

    # ================================================================
    # 4. 抽取 (根据模式分流)
    # ================================================================
    if extraction_mode == "entity":
        # ---- entity 模式: 直接实体提取 + 倒排索引 ----

        if args.skip_extraction:
            logger.info("跳过实体提取, 加载已有文件")
            inverted_index = load_existing_entities(entity_extract_file)
            logger.info(f"加载 {len(inverted_index)} 个已有实体")
        else:
            inverted_index = builder.extract_all_entities(
                tasks, output_file=entity_extract_file, force=args.force,
            )

        t_extraction = time.time() - t0
        logger.info(f"实体提取耗时: {t_extraction:.1f}s")

        if not inverted_index:
            logger.error("没有提取到任何实体")
            sys.exit(1)

        # 5. 实体对齐
        t1 = time.time()
        entity_map = builder.collect_entities_from_index(inverted_index)

        if args.skip_alignment:
            logger.info("跳过实体对齐")
            canonical_map = {name: name for name in entity_map}
            merged_pairs = 0
        else:
            canonical_map = builder.align_entities(entity_map)
            merged_pairs = sum(
                1 for name, canon in canonical_map.items() if name != canon
            )
            builder.save_entities(entity_map, canonical_map, entities_file)

        t_alignment = time.time() - t1
        logger.info(f"实体对齐耗时: {t_alignment:.1f}s")

        # 6. 构建共现图谱
        t2 = time.time()
        G = builder.build_cooccurrence_graph(inverted_index, canonical_map, entity_map)
        db_path = str(Path(output_dir) / "knowledge_graph.db") if config.get("sqlite.enabled", True) else None
        builder.save_graph(G, graph_path, graph_json, db_path=db_path, extraction_mode="entity")
        t_graph = time.time() - t2
        logger.info(f"共现图谱构建耗时: {t_graph:.1f}s")

        # 保存倒排索引 (按 canonical 聚合)
        builder.save_inverted_index(inverted_index, canonical_map, inverted_index_file)

        # 7. 统计
        unique_entities = len(set(canonical_map.values()))
        stats = KGStats(
            total_tasks=len(tasks),
            total_triples=0,
            total_entities=len(entity_map),
            unique_entities=unique_entities,
            merged_pairs=merged_pairs if not args.skip_alignment else 0,
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            extraction_mode="entity",
            cooccurrence_edges=G.number_of_edges(),
        )

    else:
        # ---- triple 模式: 三元组抽取 (现有流程) ----
        if args.skip_extraction:
            logger.info("跳过三元组抽取, 加载已有文件")
            triples = load_existing_triples(triples_file)
            logger.info(f"加载 {len(triples)} 个已有三元组")
        else:
            triples = builder.extract_all_triples(
                tasks, output_file=triples_file, force=args.force,
            )

        t_extraction = time.time() - t0
        logger.info(f"三元组抽取耗时: {t_extraction:.1f}s")

        if not triples:
            logger.error("没有抽取到任何三元组")
            sys.exit(1)

        # 5. 实体对齐
        t1 = time.time()
        entity_map = builder.collect_entities(triples)

        if args.skip_alignment:
            logger.info("跳过实体对齐")
            canonical_map = {name: name for name in entity_map}
            merged_pairs = 0
        else:
            canonical_map = builder.align_entities(entity_map)
            merged_pairs = sum(
                1 for name, canon in canonical_map.items() if name != canon
            )
            builder.save_entities(entity_map, canonical_map, entities_file)

        t_alignment = time.time() - t1
        logger.info(f"实体对齐耗时: {t_alignment:.1f}s")

        # 6. 构建图谱
        t2 = time.time()
        G = builder.build_graph(triples, canonical_map, entity_map)
        db_path = str(Path(output_dir) / "knowledge_graph.db") if config.get("sqlite.enabled", True) else None
        builder.save_graph(G, graph_path, graph_json, db_path=db_path, extraction_mode="triple")
        t_graph = time.time() - t2
        logger.info(f"图谱构建耗时: {t_graph:.1f}s")

        # 7. 统计
        unique_entities = len(set(canonical_map.values()))
        stats = KGStats(
            total_tasks=len(tasks),
            total_triples=len(triples),
            total_entities=len(entity_map),
            unique_entities=unique_entities,
            merged_pairs=merged_pairs if not args.skip_alignment else 0,
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            extraction_mode="triple",
        )

    builder.save_stats(stats)

    total_time = time.time() - t0
    logger.info(f"Phase 5 总耗时: {total_time:.1f}s")

    # 8. 可选可视化
    if args.visualize:
        logger.info("启动 pyvis 可视化 ...")
        try:
            from code_p5_visualize import visualize_graph
            visualize_graph(
                G,
                output_path=html_output,
                height=config.get("visualization.height", 800),
                width=config.get("visualization.width", 1200),
                physics=config.get("visualization.physics_solver", "forceAtlas2Based"),
                max_nodes=config.get("visualization.max_nodes", 500),
                drop_isolated=config.get("visualization.drop_isolated_nodes", False),
            )
            logger.info(f"可视化已生成: {html_output}")
        except ImportError as e:
            logger.error(f"pyvis 未安装, 跳过可视化: {e}")
            logger.info("请运行: pip install pyvis")

    # 9. 打印示例
    if extraction_mode == "triple" and triples:
        logger.info("-" * 40)
        logger.info("示例三元组 (前 5 个):")
        for t in triples[:5]:
            logger.info(f"  ({t.head}) --[{t.relation}]--> ({t.tail}) "
                       f"[conf={t.confidence:.2f}, task={t.source_task_id[-8:]}]")
    elif extraction_mode == "entity" and inverted_index:
        logger.info("-" * 40)
        logger.info("高频实体 Top-10 (按出现 task 数):")
        sorted_entities = sorted(
            inverted_index.items(), key=lambda x: len(x[1]), reverse=True,
        )
        for name, task_ids in sorted_entities[:10]:
            canonical = canonical_map.get(name, name)
            suffix = f" → {canonical}" if canonical != name else ""
            logger.info(f"  {name}{suffix} ({len(task_ids)} 个 task)")

    # 9.5 SQLite 数据库摘要
    if db_path and Path(db_path).exists():
        from code_p5e_db import KGDatabase
        kg_db = KGDatabase(db_path)
        db_stats = kg_db.get_stats()
        logger.info(f"\n{'=' * 40}")
        logger.info(f"SQLite 数据库: {db_stats['db_path']}")
        logger.info(f"  节点数: {db_stats['nodes']}, 边数: {db_stats['edges']}")
        logger.info(f"  平均每个实体关联 task 数: {db_stats['avg_task_count_per_entity']}")

    # 10. 图谱摘要
    if G.number_of_nodes() > 0:
        from collections import Counter
        degree_dist = Counter(dict(G.degree()).values())
        logger.info(f"节点度分布: {dict(sorted(degree_dist.items()))}")

        # 高度节点
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top-10 高度节点:")
        for node, degree in top_nodes:
            node_type = G.nodes[node].get("entity_type", "?")
            logger.info(f"  {node} (type={node_type}, degree={degree})")


if __name__ == "__main__":
    main()
