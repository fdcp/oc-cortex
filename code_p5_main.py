"""
Phase 5 主入口: 知识图谱构建
流程: 加载 tasks → 三元组抽取 → 实体对齐 → 图谱构建 → 可视化

用法:
  export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
  python code_p5_main.py --config code_p5_config.yaml
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: 知识图谱构建")
    parser.add_argument(
        "--config", default="code_p5_config.yaml",
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
        help="跳过三元组抽取, 直接加载已有 triples 文件",
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
        model=config.get("llm.model", "deepseek-v4-flash-free"),
        api_key_env=config.get("llm.api_key_env", "OPENCODE_ZEN_API_KEY"),
        base_url=config.get("llm.base_url", "https://opencode.ai/zen/v1"),
        max_retries=config.get("llm.max_retries", 3),
        content_retries=config.get("llm.content_retries", 2),
        timeout=config.get("llm.timeout", 120),
        max_tokens=config.get("llm.max_tokens", 4000),
        merge_max_tokens=config.get("llm.merge_max_tokens", 200),
        merge_batch_size=config.get("llm.merge_batch_size", 20),
        merge_batch_enable=config.get("llm.merge_batch_enable", False),
        concurrency=concurrency,
        embedding_model=config.get("embedding.model", "BAAI/bge-small-zh-v1.5"),
        embedding_dim=config.get("embedding.dim", 512),
        embedding_batch_size=config.get("embedding.batch_size", 32),
        embedding_device=config.get("embedding.device", "cpu"),
        embedding_cache_folder=config.get("embedding.cache_folder"),
        embedding_offline_mode=offline_mode,
        qdrant_path=config.get("qdrant.path", "./qdrant_data"),
        entities_collection=config.get("qdrant.entities_collection", "entities"),
        alignment_threshold=config.get("knowledge_graph.entity_alignment_threshold", 0.92),
    )

    triples_file = config.get("knowledge_graph.triples_file", "./output/triples_p5.jsonl")
    entities_file = config.get("knowledge_graph.entities_file", "./output/entities_p5.jsonl")
    graph_path = config.get("knowledge_graph.graph_path", "./output/knowledge_graph.gpickle")
    graph_json = config.get("knowledge_graph.graph_json", "./output/knowledge_graph.json")

    # 4. 三元组抽取
    t0 = time.time()

    if args.skip_extraction:
        logger.info("跳过三元组抽取, 加载已有文件")
        triples = load_existing_triples(triples_file)
        logger.info(f"加载 {len(triples)} 个已有三元组")
    else:
        triples = builder.extract_all_triples(
            tasks, output_file=triples_file,
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
        # 统计合并对数
        merged_pairs = sum(
            1 for name, canon in canonical_map.items() if name != canon
        )
        builder.save_entities(entity_map, canonical_map, entities_file)

    t_alignment = time.time() - t1
    logger.info(f"实体对齐耗时: {t_alignment:.1f}s")

    # 6. 构建图谱
    t2 = time.time()
    G = builder.build_graph(triples, canonical_map, entity_map)
    builder.save_graph(G, graph_path, graph_json)
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
    )
    builder.save_stats(stats)

    total_time = time.time() - t0
    logger.info(f"Phase 5 总耗时: {total_time:.1f}s")

    # 8. 可选可视化
    if args.visualize:
        logger.info("启动 pyvis 可视化 ...")
        try:
            from code_p5_visualize import visualize_graph
            html_path = config.get("visualization.html_output",
                                   "./output/knowledge_graph.html")
            visualize_graph(
                G,
                output_path=html_path,
                height=config.get("visualization.height", 800),
                width=config.get("visualization.width", 1200),
                physics=config.get("visualization.physics_solver", "forceAtlas2Based"),
                max_nodes=config.get("visualization.max_nodes", 500),
                drop_isolated=config.get("visualization.drop_isolated_nodes", False),
            )
            logger.info(f"可视化已生成: {html_path}")
        except ImportError as e:
            logger.error(f"pyvis 未安装, 跳过可视化: {e}")
            logger.info("请运行: pip install pyvis")

    # 9. 打印示例三元组
    if triples:
        logger.info("-" * 40)
        logger.info("示例三元组 (前 5 个):")
        for t in triples[:5]:
            logger.info(f"  ({t.head}) --[{t.relation}]--> ({t.tail}) "
                       f"[conf={t.confidence:.2f}, task={t.source_task_id[-8:]}]")

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
