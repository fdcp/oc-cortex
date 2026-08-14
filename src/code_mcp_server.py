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

# 强制离线模式：阻止 huggingface_hub / transformers 发起任何 HTTP 请求。
# 根因：在 ThreadPoolExecutor 线程中，huggingface_hub 内部共享的 httpx.Client
# 会因 GC 被提前关闭，导致 "Cannot send a request, as the client has been closed"。
# 模型已在本地缓存，无需联网检查更新。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from loguru import logger
from mcp.server.fastmcp import FastMCP

from code_p5e_db import KGDatabase

# ============================================================
# 配置 & 初始化
# ============================================================

# 日志重定向到 stderr（stdout 留给 MCP stdio 协议）
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# 默认数据库路径（仅在 config/code_mcp_config.yaml 缺失时使用，与该配置文件的默认值保持一致）
DEFAULT_DB_PATH = "output/triple/knowledge_graph.db"


def _load_config() -> dict:
    """加载 config/code_mcp_config.yaml（可选）"""
    config_path = Path(__file__).resolve().parent.parent / "config" / "code_mcp_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_config = _load_config()
_project_root = Path(__file__).resolve().parent.parent  # src/ → project root

_raw_db_path = _config.get("server", {}).get("db_path", DEFAULT_DB_PATH)
_db_path = str(_project_root / _raw_db_path) if not Path(_raw_db_path).is_absolute() else _raw_db_path

_raw_tasks_file = _config.get("server", {}).get("tasks_file", "output/tasks.jsonl")
_tasks_file = str(_project_root / _raw_tasks_file) if not Path(_raw_tasks_file).is_absolute() else _raw_tasks_file

# 延迟初始化的全局实例
_db: Optional[KGDatabase] = None
_task_cache: Optional[dict] = None

# 持久化线程池：将重型初始化（embedding 模型 + Qdrant + BM25）和搜索
# 放到独立线程执行，避免阻塞 FastMCP 的 asyncio 事件循环。
# 同时保证 Qdrant SQLite 的 check_same_thread 约束（同一线程内操作）。
import concurrent.futures
_rag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


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


_session_index: Optional[dict] = None


def _build_session_index() -> dict:
    """延迟构建 session_id 反向索引"""
    global _session_index
    if _session_index is None:
        idx: dict = {}
        for t in get_task_cache().values():
            sid = t.get("session_id")
            if not sid:
                continue
            entry = idx.setdefault(sid, {"tasks": [], "first_at": "", "last_at": ""})
            entry["tasks"].append(t)
            ca = t.get("created_at") or ""
            if not entry["first_at"] or (ca and ca < entry["first_at"]):
                entry["first_at"] = ca
            if not entry["last_at"] or (ca and ca > entry["last_at"]):
                entry["last_at"] = ca
        _session_index = idx
        logger.info(f"Session 索引构建: {len(_session_index)} sessions")
    return _session_index


_tracer = None


def get_tracer():
    """延迟初始化 DecisionTracer"""
    global _tracer
    if _tracer is None:
        from code_p6b_skeleton import DecisionTracer
        _tracer = DecisionTracer(get_db(), max_hops=3)
        logger.info("DecisionTracer 初始化完成")
    return _tracer


_shared_searcher = None


def get_shared_searcher():
    """同步初始化共享 SessionSearcher(调用方决定是否放到 executor 内)"""
    global _shared_searcher
    if _shared_searcher is None:
        os.chdir(_project_root)
        from code_p4_searcher import SessionSearcher
        _shared_searcher = SessionSearcher(str(_project_root / "config" / "code_p3_config.yaml"))
        logger.info("共享 SessionSearcher 初始化完成")
    return _shared_searcher


_skeleton = None


def get_skeleton():
    """延迟初始化 SummarySkeleton(整段 init 放到 executor,避免阻塞 async 循环)"""
    global _skeleton
    if _skeleton is None:
        def _init():
            from code_p6b_skeleton import SummarySkeleton
            from code_p6_summarizer import SessionSummarizer
            summarizer = SessionSummarizer(
                str(_project_root / "config" / "code_p3_config.yaml"),
                searcher=get_shared_searcher(),
            )
            return SummarySkeleton(get_db(), summarizer, get_tracer())
        _skeleton = _rag_executor.submit(_init).result()
        logger.info("SummarySkeleton 初始化完成")
    return _skeleton


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
        "使用 list_sessions / get_session_tasks 浏览和下钻 session。"
        "使用 trace_decision 追踪实体的决策链（选用/选型/排除/替换/依赖）。"
        "使用 skeleton_summarize 生成带决策链注入的图谱骨架主题总结。"
        "当用户说'继续''接着''上次'等延续性词语时，优先调用 query_kg 注入历史上下文。"
        "当用户说'为什么选 XX''XX 的选型对比'时，优先调用 trace_decision。"
        "当用户说'总结 XX'时，优先调用 skeleton_summarize。"
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

    try:
        from code_p5e_graph_rag import GraphRAGSearcher
    except ImportError as e:
        return {"error": f"Graph-RAG 依赖缺失: {e}"}

    def _init_rag():
        if hasattr(graph_rag_search, "_rag"):
            return graph_rag_search._rag
        os.chdir(_project_root)
        searcher = get_shared_searcher()
        kg_db = get_db()
        return GraphRAGSearcher(searcher, kg_db)

    def _do_search(rag, q, k, use_g):
        return rag.search(q, top_k=k, use_graph_rag=use_g)

    try:
        if not hasattr(graph_rag_search, "_rag"):
            graph_rag_search._rag = _rag_executor.submit(_init_rag).result()
            logger.info("GraphRAGSearcher 初始化完成")

        results, debug = _rag_executor.submit(
            _do_search, graph_rag_search._rag, query, top_k, use_graph
        ).result()
    except Exception as e:
        logger.error(f"graph_rag_search 异常: {e}")
        return {"error": f"{type(e).__name__}: {e}", "elapsed_ms": int((time.time() - t0) * 1000)}

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
        "debug": {
            "vector_results": debug.get("vector_results"),
            "vector_time_ms": debug.get("vector_time_ms"),
            "query_entities": debug.get("query_entities"),
            "graph_candidates": debug.get("graph_candidates"),
            "graph_time_ms": debug.get("graph_time_ms"),
            "merged_candidates": debug.get("merged_candidates"),
            "rerank_time_ms": debug.get("rerank_time_ms"),
            "total_time_ms": debug.get("total_time_ms"),
        },
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


@mcp.tool()
def list_sessions(limit: int = 10) -> dict:
    """列出最近活跃的 session(按最后任务时间倒序)。

    用于浏览跨 session 历史,通常配合 get_session_tasks 下钻单个 session。

    Args:
        limit: 返回数量,默认 10
    """
    idx = _build_session_index()
    sessions = []
    for sid, entry in idx.items():
        tasks_sorted = sorted(entry["tasks"], key=lambda t: t.get("created_at", ""))
        sessions.append({
            "session_id": sid,
            "task_count": len(tasks_sorted),
            "first_task_at": entry["first_at"],
            "last_task_at": entry["last_at"],
            "task_labels": [t.get("task_label", "") for t in tasks_sorted],
        })
    sessions.sort(key=lambda s: s["last_task_at"], reverse=True)
    return {
        "session_count": len(sessions),
        "sessions": sessions[:limit],
    }


@mcp.tool()
def get_session_tasks(session_id: str) -> dict:
    """按 session_id 拿该 session 的完整 task 列表(按时间排序)。

    通常由 list_sessions 找到 session_id 后调用,下钻看完整上下文。

    Args:
        session_id: 如 'ses_12c6bd8dfffevT51PypMW2v5Mx'
    """
    cache = get_task_cache()
    tasks = [t for t in cache.values() if t.get("session_id") == session_id]
    tasks.sort(key=lambda t: t.get("created_at", ""))
    if not tasks:
        return {
            "error": f"未找到 session: {session_id}",
            "session_id": session_id,
            "task_count": 0,
            "tasks": [],
        }
    return {
        "session_id": session_id,
        "task_count": len(tasks),
        "tasks": [
            {
                "task_id": t["task_id"],
                "task_label": t.get("task_label", ""),
                "task_summary": t.get("task_summary", ""),
                "created_at": t.get("created_at", ""),
            }
            for t in tasks
        ],
    }


@mcp.tool()
def trace_decision(entity: str, max_hops: int = 2) -> dict:
    """从知识图谱中追踪实体的决策链(不调 LLM,纯 KG BFS)。

    按 5 类决策关系(选用/选型/排除/替换/依赖)做 BFS 遍历,返回决策步骤及来源 task。
    适用于"为什么选 XX""A 和 B 的选型对比"类问题。

    Args:
        entity: 起始实体名称(如 'FlashAttention', 'OpenCode')
        max_hops: 最大追溯跳数,默认 2
    """
    t0 = time.time()
    tracer = get_tracer()
    chain = tracer.trace_decision(entity, max_hops=max_hops)
    return {
        "root_entity": chain.root_entity,
        "hop_count": chain.hop_count,
        "visited_entity_count": len(chain.visited_entities),
        "visited_entities": chain.visited_entities,
        "steps": [
            {
                "entity": s.entity,
                "related_entity": s.related_entity,
                "relation": s.relation,
                "category": s.category,
                "hop": s.hop,
                "direction": s.direction,
                "source_task": s.source_task,
                "weight": s.weight,
            }
            for s in chain.steps
        ],
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@mcp.tool()
def skeleton_summarize(
    topic: str,
    max_tasks: int = 5,
    use_decision_trace: bool = True,
) -> dict:
    """带决策链注入的图谱骨架主题总结(P6b)。

    流程:定位主题实体 → BFS 扩散 → 收集 task → 决策链追踪 → 骨架构建 → LLM 生成。
    适用于"总结 XX"类查询,输出结构化"决策逻辑→选型对比→技术细节→评价建议"。
    首次调用会加载 SessionSummarizer(embedding + Reranker + LLM client),约 30-50s。

    Args:
        topic: 主题关键词,如 'FlashAttention', '序列并行'
        max_tasks: 最多纳入的 task 数,默认 5
        use_decision_trace: 是否注入决策链上下文,默认 True
    """
    t0 = time.time()
    try:
        skeleton = get_skeleton()
        result = _rag_executor.submit(
            skeleton.generate_summary,
            topic,
            use_decision_trace,
            max_tasks,
        ).result()
    except Exception as e:
        logger.error(f"skeleton_summarize 异常: {e}")
        return {
            "error": f"{type(e).__name__}: {e}",
            "topic": topic,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    return {
        "topic": result.topic,
        "summary": result.summary,
        "skeleton": result.skeleton,
        "decision_chain": {
            "root_entity": result.decision_chain.root_entity,
            "hop_count": result.decision_chain.hop_count,
            "step_count": len(result.decision_chain.steps),
            "steps": [
                {
                    "entity": s.entity,
                    "related_entity": s.related_entity,
                    "relation": s.relation,
                    "category": s.category,
                    "hop": s.hop,
                    "direction": s.direction,
                    "source_task": s.source_task,
                    "weight": s.weight,
                }
                for s in result.decision_chain.steps
            ],
        },
        "task_summaries": result.task_summaries,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "debug": result.debug,
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

    # 后台预热 GraphRAGSearcher（加载 embedding + Qdrant + BM25，约 30-50s）
    def _preheat_rag():
        try:
            os.chdir(_project_root)
            from code_p5e_graph_rag import GraphRAGSearcher
            searcher = get_shared_searcher()
            kg_db = get_db()
            graph_rag_search._rag = GraphRAGSearcher(searcher, kg_db)
            logger.info("GraphRAGSearcher 预热完成")
        except Exception as e:
            logger.warning(f"GraphRAGSearcher 预热失败: {e}")

    _rag_executor.submit(_preheat_rag)

    mcp.run(transport=transport)
