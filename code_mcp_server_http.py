"""
MCP Server: 知识图谱工具服务

将 Phase 5 知识图谱 (SQLite) 封装为 HTTP API，
提供 MCP 兼容的工具发现 (/mcp/tools/list) 和调用 (/mcp/tools/call) 接口，
供 OpenCode 或其他 MCP Client 直接集成。

工具列表:
  - query_kg:         BFS 扩散查询，从实体出发获取关联 task
  - search_entities:  模糊搜索实体
  - get_entity_info:  获取实体详情（节点 + 边）
  - graph_rag_search: Graph-RAG 增强搜索（需 SessionSearcher）
  - get_stats:        图谱统计信息

启动:
  uvicorn code_mcp_server:app --host 0.0.0.0 --port 8000
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from code_p5e_db import KGDatabase

# ============================================================
# 应用初始化
# ============================================================

app = FastAPI(
    title="Knowledge Graph MCP Server",
    description="将知识图谱封装为 MCP 工具，供 OpenCode 直接调用实现跨会话记忆",
    version="1.0.0",
)

# 默认数据库路径 (Phase 5 triple 模式产出)
DEFAULT_DB_PATH = "output/triple/knowledge_graph.db"


def _load_config() -> dict:
    """加载 code_mcp_config.yaml（可选，缺失时用默认值）"""
    config_path = Path("code_mcp_config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_config = _load_config()
_db_path = _config.get("server", {}).get("db_path", DEFAULT_DB_PATH)

# 初始化 KGDatabase（全局单例，线程安全由 SQLite 保证）
_db: Optional[KGDatabase] = None


def get_db() -> KGDatabase:
    """延迟加载 KGDatabase 实例"""
    global _db
    if _db is None:
        if not Path(_db_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"知识图谱数据库不存在: {_db_path}。请先运行 code_p5_main.py 生成。",
            )
        _db = KGDatabase(_db_path)
        logger.info(f"KGDatabase 加载: {_db_path}")
    return _db


# ============================================================
# 响应模型
# ============================================================

class TaskResult(BaseModel):
    task_id: str
    session_id: str
    task_label: str
    task_summary: str


class EntityInfo(BaseModel):
    name: str
    entity_type: str
    aliases: list[str]
    source_tasks: list[str]
    task_count: int
    edges_out: list[dict] = []
    edges_in: list[dict] = []


class QueryKGResponse(BaseModel):
    seed_entity: str
    depth: int
    expanded_entities: list[str]
    related_tasks: list[TaskResult]
    elapsed_ms: int


class SearchEntitiesResponse(BaseModel):
    query: str
    results: list[dict]
    count: int


class GraphRAGSearchResponse(BaseModel):
    query: str
    results: list[TaskResult]
    source_distribution: dict
    elapsed_ms: int


class StatsResponse(BaseModel):
    db_path: str
    nodes: int
    edges: int
    avg_task_count_per_entity: float
    server_uptime_ms: int


class MCPToolDef(BaseModel):
    name: str
    description: str
    inputSchema: dict


class MCPToolListResponse(BaseModel):
    tools: list[MCPToolDef]


class MCPCallRequest(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)


class MCPCallResponse(BaseModel):
    tool: str
    result: dict
    elapsed_ms: int


# ============================================================
# 辅助: 从 task_id 查 task 详情
# ============================================================

def _get_task_detail(task_id: str) -> Optional[TaskResult]:
    """从 tasks.jsonl 加载的 task map 中查找 task 详情"""
    tasks_file = _config.get("server", {}).get("tasks_file", "output/tasks.jsonl")
    if not hasattr(_get_task_detail, "_cache"):
        cache = {}
        p = Path(tasks_file)
        if p.exists():
            for line in p.read_text().strip().splitlines():
                try:
                    t = json.loads(line)
                    cache[t["task_id"]] = t
                except (json.JSONDecodeError, KeyError):
                    continue
        _get_task_detail._cache = cache
        logger.info(f"Task 缓存加载: {len(cache)} tasks from {tasks_file}")

    t = _get_task_detail._cache.get(task_id)
    if t:
        return TaskResult(
            task_id=t["task_id"],
            session_id=t["session_id"],
            task_label=t.get("task_label", ""),
            task_summary=t.get("task_summary", ""),
        )
    return None


# ============================================================
# REST API 端点
# ============================================================

@app.get("/query_kg", response_model=QueryKGResponse, summary="BFS 扩散查询")
def query_kg(
    entity: str = Query(..., description="种子实体名称"),
    depth: int = Query(default=2, ge=1, le=3, description="BFS 扩散深度"),
    max_nodes: int = Query(default=30, ge=1, le=200, description="最大扩散节点数"),
):
    """
    从种子实体出发，BFS 扩散到邻居实体，收集所有关联 task。
    这是 MCP 的核心工具，用于跨会话记忆注入。
    """
    t0 = time.time()
    db = get_db()

    # 检查实体是否存在
    node = db.get_node(entity)
    if not node:
        # 尝试模糊匹配
        matches = db.search_entities(entity, limit=3)
        if matches:
            entity = matches[0]["name"]
            node = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"实体不存在且无法模糊匹配: {entity}")

    # BFS 扩散
    expanded = db.bfs_expand([entity], depth=depth, max_nodes=max_nodes)

    # 收集关联 task（去重）
    task_ids = set()
    for ent_name in expanded:
        task_ids.update(db.get_tasks_by_entity(ent_name))

    # 构建 task 结果
    related_tasks = []
    for tid in sorted(task_ids):
        tr = _get_task_detail(tid)
        if tr:
            related_tasks.append(tr)

    return QueryKGResponse(
        seed_entity=entity,
        depth=depth,
        expanded_entities=sorted(expanded),
        related_tasks=related_tasks,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


@app.get("/search_entities", response_model=SearchEntitiesResponse)
def search_entities(
    query: str = Query(..., description="搜索关键词"),
    limit: int = Query(default=10, ge=1, le=50),
):
    """模糊搜索知识图谱中的实体名称"""
    db = get_db()
    results = db.search_entities(query, limit=limit)
    return SearchEntitiesResponse(query=query, results=results, count=len(results))


@app.get("/get_entity_info", response_model=EntityInfo)
def get_entity_info(
    entity: str = Query(..., description="实体名称"),
):
    """获取实体详情：类型、别名、关联 task、出入边"""
    db = get_db()
    node = db.get_node(entity)
    if not node:
        raise HTTPException(status_code=404, detail=f"实体不存在: {entity}")

    edges_out = db.get_edges(entity, direction="out")
    edges_in = db.get_edges(entity, direction="in")

    return EntityInfo(
        name=node["name"],
        entity_type=node["entity_type"],
        aliases=node["aliases"],
        source_tasks=node["source_tasks"],
        task_count=node["task_count"],
        edges_out=edges_out,
        edges_in=edges_in,
    )


@app.get("/graph_rag_search", response_model=GraphRAGSearchResponse)
def graph_rag_search(
    query: str = Query(..., description="搜索查询"),
    top_k: int = Query(default=5, ge=1, le=20),
    use_graph: bool = Query(default=True, description="是否启用图谱扩散"),
):
    """Graph-RAG 增强搜索：向量检索 + 图谱扩散 + Reranker"""
    t0 = time.time()
    try:
        from code_p4_searcher import SessionSearcher
        from code_p5e_graph_rag import GraphRAGSearcher

        # 延迟初始化（首次调用较慢，后续复用）
        if not hasattr(graph_rag_search, "_rag"):
            searcher = SessionSearcher("code_p3_config.yaml")
            kg_db = get_db()
            graph_rag_search._rag = GraphRAGSearcher(searcher, kg_db)
            logger.info("GraphRAGSearcher 初始化完成")

        rag = graph_rag_search._rag
        results, debug = rag.search(query, top_k=top_k, use_graph_rag=use_graph)

        task_results = [
            TaskResult(
                task_id=r.task_id,
                session_id=r.session_id,
                task_label=r.task_label,
                task_summary=r.task_summary,
            )
            for r in results
        ]

        return GraphRAGSearchResponse(
            query=query,
            results=task_results,
            source_distribution=debug.get("source_distribution", {}),
            elapsed_ms=int((time.time() - t0) * 1000),
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Graph-RAG 依赖缺失: {e}")


@app.get("/get_stats", response_model=StatsResponse)
def get_stats():
    """获取知识图谱统计信息"""
    db = get_db()
    stats = db.get_stats()
    return StatsResponse(
        db_path=stats["db_path"],
        nodes=stats["nodes"],
        edges=stats["edges"],
        avg_task_count_per_entity=stats["avg_task_count_per_entity"],
        server_uptime_ms=stats.get("server_uptime_ms", 0),
    )


# ============================================================
# MCP 协议兼容层
# ============================================================

# MCP 工具定义 (JSON Schema 格式)
MCP_TOOLS = [
    MCPToolDef(
        name="query_kg",
        description=(
            "从知识图谱中查询与指定实体关联的历史任务。"
            "通过 BFS 扩散从种子实体出发，找到所有相关实体及其关联的 task。"
            "适用于跨会话记忆注入：当用户说'继续搞 XX 模块'时，自动获取相关上下文。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "种子实体名称（如 'OpenCode', 'RoPE位置编码', 'session管理'）",
                },
                "depth": {
                    "type": "integer",
                    "description": "BFS 扩散深度 (1=一跳邻居, 2=二跳, 3=三跳)",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 3,
                },
                "max_nodes": {
                    "type": "integer",
                    "description": "最大扩散节点数",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": ["entity"],
        },
    ),
    MCPToolDef(
        name="search_entities",
        description="模糊搜索知识图谱中的实体名称，用于找到精确的实体名后再调用 query_kg。",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数上限",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    MCPToolDef(
        name="get_entity_info",
        description="获取知识图谱中某个实体的详细信息：类型、别名、关联 task、出入边关系。",
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "实体名称",
                },
            },
            "required": ["entity"],
        },
    ),
    MCPToolDef(
        name="graph_rag_search",
        description=(
            "Graph-RAG 增强搜索：结合向量检索和知识图谱扩散，返回最相关的历史 task。"
            "比纯向量检索多了图谱关联发现，适合查找跨 session 的相关任务。"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数",
                    "default": 5,
                },
                "use_graph": {
                    "type": "boolean",
                    "description": "是否启用图谱扩散（false 则退化为纯向量检索）",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    ),
    MCPToolDef(
        name="get_stats",
        description="获取知识图谱的统计信息：节点数、边数、平均每实体关联 task 数。",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


@app.get("/mcp/tools/list", response_model=MCPToolListResponse)
def mcp_tools_list():
    """MCP 工具发现：返回所有可用工具的定义（JSON Schema 格式）"""
    return MCPToolListResponse(tools=MCP_TOOLS)


@app.post("/mcp/tools/call", response_model=MCPCallResponse)
def mcp_tools_call(req: MCPCallRequest):
    """MCP 工具调用：统一入口，根据 tool 名称分发到对应处理函数"""
    t0 = time.time()
    args = req.arguments

    try:
        if req.tool == "query_kg":
            result = query_kg(
                entity=args.get("entity", ""),
                depth=args.get("depth", 2),
                max_nodes=args.get("max_nodes", 30),
            )
        elif req.tool == "search_entities":
            result = search_entities(
                query=args.get("query", ""),
                limit=args.get("limit", 10),
            )
        elif req.tool == "get_entity_info":
            result = get_entity_info(entity=args.get("entity", ""))
        elif req.tool == "graph_rag_search":
            result = graph_rag_search(
                query=args.get("query", ""),
                top_k=args.get("top_k", 5),
                use_graph=args.get("use_graph", True),
            )
        elif req.tool == "get_stats":
            result = get_stats()
        else:
            raise HTTPException(status_code=404, detail=f"未知工具: {req.tool}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 工具调用失败 [{req.tool}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return MCPCallResponse(
        tool=req.tool,
        result=result.model_dump() if hasattr(result, "model_dump") else result.dict(),
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
def health():
    """健康检查"""
    db_ok = Path(_db_path).exists()
    return {
        "status": "ok" if db_ok else "degraded",
        "db_exists": db_ok,
        "db_path": _db_path,
        "tools_count": len(MCP_TOOLS),
    }


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = _config.get("server", {}).get("host", "0.0.0.0")
    port = _config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
