"""
MCP Server: 知识图谱工具服务（标准 MCP 协议）

将 Phase 5 知识图谱 (SQLite) 封装为标准 MCP Server，
通过 stdio 传输供 Claude Code / Codex / OpenCode / QoderWork 直接加载。

工具:
  - query_kg:         BFS 扩散查询，从实体出发获取关联 task（跨会话记忆核心）
  - search_entities:  模糊搜索实体
  - get_entity_info:  获取实体详情（节点 + 边）
  - graph_rag_search: Graph-RAG 增强搜索（向量 + 图谱 + Reranker）
  - get_kg_stats:     图谱统计信息

启动 (stdio, 供各客户端配置使用):
  python3 code_mcp_server.py

启动 (streamable-http, 供调试):
  python3 code_mcp_server.py --http
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from code_p5e_db import KGDatabase

# ============================================================
# 配置 & 初始化
# ============================================================

# 日志重定向到 stderr（stdout 留给 MCP stdio 协议）
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# 默认数据库路径
DEFAULT_DB_PATH = "output/triple/knowledge_graph.db"


def _load_config() -> dict:
    """加载 code_mcp_config.yaml（可选）"""
    config_path = Path(__file__).parent / "code_mcp_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_config = _load_config()
_db_path = _config.get("server", {}).get("db_path", DEFAULT_DB_PATH)
_tasks_file = _config.get("server", {}).get("tasks_file", "output/tasks.jsonl")

# 延迟初始化的全局实例
_db: Optional[KGDatabase] = None
_task_cache: Optional[dict] = None


def get_db() -> KGDatabase:
    """延迟加载 KGDatabase"""
    global _db
    if _db is None:
        if not Path(_db_path).exists():
            raise FileNotFoundError(
                f"知识图谱数据库不存在: {_db_path}。请先运行 code_p5_main.py 生成。"
            )
        _db = KGDatabase(_db_path)
        logger.info(f"KGDatabase 加载: {_db_path}")
    return _db


def get_task_cache() -> dict:
    """延迟加载 task 缓存"""
    global _task_cache
    if _task_cache is None:
        _task_cache = {}
        p = Path(_tasks_file)
        if p.exists():
            for line in p.read_text().strip().splitlines():
                try:
                    t = json.loads(line)
                    _task_cache[t["task_id"]] = t
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Task 缓存加载: {len(_task_cache)} tasks")
    return _task_cache


def _task_to_dict(task_id: str) -> Optional[dict]:
    """从缓存中获取 task 详情"""
    t = get_task_cache().get(task_id)
    if t:
        return {
            "task_id": t["task_id"],
            "session_id": t["session_id"],
            "task_label": t.get("task_label", ""),
            "task_summary": t.get("task_summary", ""),
        }
    return None


# ============================================================
# MCP Server 定义
# ============================================================

mcp = FastMCP(
    name="knowledge-graph",
    instructions=(
        "知识图谱 MCP Server：提供历史 session 的跨会话记忆能力。"
        "使用 query_kg 从实体出发 BFS 扩散获取关联 task；"
        "使用 search_entities 模糊搜索实体名；"
        "使用 graph_rag_search 做完整的图谱增强搜索。"
        "当用户说'继续''接着''上次'等延续性词语时，优先调用 query_kg 注入历史上下文。"
    ),
)


# ============================================================
# MCP 工具
# ============================================================

@mcp.tool()
def query_kg(entity: str, depth: int = 2, max_nodes: int = 30) -> dict:
    """从知识图谱中查询与指定实体关联的历史任务。

    通过 BFS 扩散从种子实体出发，找到所有相关实体及其关联的 task。
    适用于跨会话记忆注入：当用户说"继续搞 XX 模块"时，自动获取相关上下文。

    Args:
        entity: 种子实体名称（如 'OpenCode', 'RoPE位置编码', 'session管理'）
        depth: BFS 扩散深度 (1=一跳邻居, 2=二跳, 3=三跳)，默认 2
        max_nodes: 最大扩散节点数，默认 30
    """
    t0 = time.time()
    db = get_db()

    # 检查实体是否存在
    node = db.get_node(entity)
    if not node:
        matches = db.search_entities(entity, limit=3)
        if matches:
            entity = matches[0]["name"]
        else:
            return {
                "seed_entity": entity,
                "error": f"实体不存在且无法模糊匹配: {entity}",
                "suggestion": "使用 search_entities 搜索正确的实体名",
                "expanded_entities": [],
                "related_tasks": [],
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

    # BFS 扩散
    expanded = db.bfs_expand([entity], depth=depth, max_nodes=max_nodes)

    # 收集关联 task（去重）
    task_ids = set()
    for ent_name in expanded:
        task_ids.update(db.get_tasks_by_entity(ent_name))

    related_tasks = []
    for tid in sorted(task_ids):
        tr = _task_to_dict(tid)
        if tr:
            related_tasks.append(tr)

    return {
        "seed_entity": entity,
        "depth": depth,
        "expanded_entities": sorted(expanded),
        "related_tasks": related_tasks,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@mcp.tool()
def search_entities(query: str, limit: int = 10) -> dict:
    """模糊搜索知识图谱中的实体名称。

    用于找到精确的实体名后再调用 query_kg 做扩散查询。

    Args:
        query: 搜索关键词（支持部分匹配）
        limit: 返回结果数上限，默认 10
    """
    db = get_db()
    results = db.search_entities(query, limit=limit)
    return {
        "query": query,
        "results": results,
        "count": len(results),
    }


@mcp.tool()
def get_entity_info(entity: str) -> dict:
    """获取知识图谱中某个实体的详细信息。

    返回实体类型、别名、关联 task 列表，以及出入边关系。

    Args:
        entity: 实体名称
    """
    db = get_db()
    node = db.get_node(entity)
    if not node:
        return {"error": f"实体不存在: {entity}"}

    edges_out = db.get_edges(entity, direction="out")
    edges_in = db.get_edges(entity, direction="in")

    return {
        "name": node["name"],
        "entity_type": node["entity_type"],
        "aliases": node["aliases"],
        "source_tasks": node["source_tasks"],
        "task_count": node["task_count"],
        "edges_out": edges_out,
        "edges_in": edges_in,
    }


@mcp.tool()
def graph_rag_search(query: str, top_k: int = 5, use_graph: bool = True) -> dict:
    """Graph-RAG 增强搜索：结合向量检索和知识图谱扩散，返回最相关的历史 task。

    比纯向量检索多了图谱关联发现，适合查找跨 session 的相关任务。
    首次调用会初始化 SessionSearcher（较慢），后续调用复用。

    Args:
        query: 自然语言搜索查询
        top_k: 返回结果数，默认 5
        use_graph: 是否启用图谱扩散（false 则退化为纯向量检索），默认 true
    """
    t0 = time.time()

    # 延迟导入（避免 stdio 启动时加载重型依赖）
    try:
        from code_p4_searcher import SessionSearcher
        from code_p5e_graph_rag import GraphRAGSearcher
    except ImportError as e:
        return {"error": f"Graph-RAG 依赖缺失: {e}"}

    if not hasattr(graph_rag_search, "_rag"):
        searcher = SessionSearcher("code_p3_config.yaml")
        kg_db = get_db()
        graph_rag_search._rag = GraphRAGSearcher(searcher, kg_db)
        logger.info("GraphRAGSearcher 初始化完成")

    rag = graph_rag_search._rag
    results, debug = rag.search(query, top_k=top_k, use_graph_rag=use_graph)

    return {
        "query": query,
        "results": [
            {
                "task_id": r.task_id,
                "session_id": r.session_id,
                "task_label": r.task_label,
                "task_summary": r.task_summary,
            }
            for r in results
        ],
        "source_distribution": debug.get("source_distribution", {}),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@mcp.tool()
def get_kg_stats() -> dict:
    """获取知识图谱的统计信息：节点数、边数、平均每实体关联 task 数。"""
    db = get_db()
    stats = db.get_stats()
    return {
        "db_path": stats["db_path"],
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_task_count_per_entity": round(stats["avg_task_count_per_entity"], 2),
    }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"
        logger.info("以 streamable-http 模式启动")
    elif "--sse" in sys.argv:
        transport = "sse"
        logger.info("以 SSE 模式启动")

    mcp.run(transport=transport)
