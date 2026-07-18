"""
MCP Client: OpenCode 集成客户端

封装对 KG MCP Server 的调用，提供：
  - MCPClient: HTTP 客户端，调用 MCP Server 暴露的工具
  - ContextInjector: 新 session 启动时自动注入跨会话记忆
  - extract_session_entities: 从用户查询中抽取实体（复用 code_p5e_graph_rag）

使用方式:
  from code_mcp_client import MCPClient, ContextInjector

  client = MCPClient("http://localhost:8000")
  injector = ContextInjector(client)

  # 新 session 启动时
  context = injector.build_context("继续搞 RoPE 位置编码的优化")
  print(context)  # → 注入到 system prompt
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


# ============================================================
# MCP Client
# ============================================================

class MCPClient:
    """KG MCP Server 的 HTTP 客户端"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---- MCP 协议接口 ----

    def list_tools(self) -> list[dict]:
        """获取所有可用工具定义"""
        data = self._get("/mcp/tools/list")
        return data.get("tools", [])

    def call_tool(self, tool: str, arguments: dict = None) -> dict:
        """统一工具调用入口 (MCP 协议)"""
        result = self._post("/mcp/tools/call", {
            "tool": tool,
            "arguments": arguments or {},
        })
        return result.get("result", {})

    # ---- 便捷方法 ----

    def query_kg(self, entity: str, depth: int = 2, max_nodes: int = 30) -> dict:
        """BFS 扩散查询：从实体出发获取关联 task"""
        return self._get("/query_kg", {
            "entity": entity, "depth": depth, "max_nodes": max_nodes,
        })

    def search_entities(self, query: str, limit: int = 10) -> dict:
        """模糊搜索实体"""
        return self._get("/search_entities", {"query": query, "limit": limit})

    def get_entity_info(self, entity: str) -> dict:
        """获取实体详情"""
        return self._get("/get_entity_info", {"entity": entity})

    def graph_rag_search(self, query: str, top_k: int = 5, use_graph: bool = True) -> dict:
        """Graph-RAG 增强搜索"""
        return self._get("/graph_rag_search", {
            "query": query, "top_k": top_k, "use_graph": use_graph,
        })

    def get_stats(self) -> dict:
        """图谱统计"""
        return self._get("/get_stats")

    def health(self) -> dict:
        """健康检查"""
        return self._get("/health")


# ============================================================
# 实体抽取（轻量版，不依赖 GraphRAGSearcher）
# ============================================================

ENTITY_EXTRACT_PROMPT = """从以下用户查询中提取涉及的关键实体名称。
返回 JSON: {{"entities": ["实体1", "实体2"]}}

查询: {query}

返回 JSON:"""


def extract_session_entities(
    query: str,
    model: str = "nemotron-3-ultra-free",
    max_retries: int = 2,
) -> list[str]:
    """从用户查询中抽取实体名称（用于 MCP 调用）。

    复用 code_p5e_graph_rag 的抽取逻辑，如果不可用则直接调用 LLM。
    """
    # 优先复用已有模块
    try:
        from code_p5e_graph_rag import extract_query_entities
        return extract_query_entities(query, max_retries=max_retries, model=model)
    except ImportError:
        pass

    # Fallback: 直接调用
    import re
    from openai import OpenAI

    api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url="https://opencode.ai/zen/v1", timeout=30)

    for attempt in range(max_retries + 1):
        try:
            output = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ENTITY_EXTRACT_PROMPT.format(query=query)}],
                temperature=0.1,
                max_tokens=200,
            )
            raw = output.choices[0].message.content
            if not raw:
                continue
            content = re.sub(r"", "", raw, flags=re.DOTALL).strip()
            data = json.loads(content)
            entities = data.get("entities", [])
            return [e.strip() for e in entities if isinstance(e, str) and len(e.strip()) >= 2]
        except Exception as e:
            logger.warning(f"实体抽取失败 (attempt {attempt + 1}): {e}")

    return []


# ============================================================
# 上下文注入器
# ============================================================

@dataclass
class InjectedContext:
    """注入到 session 的上下文信息"""
    query: str
    entities: list[str]
    matched_entities: list[str]     # 在图谱中命中的实体
    related_tasks: list[dict]       # 关联的历史 task
    context_text: str = ""          # 格式化后的上下文文本（可直接注入 prompt）
    elapsed_ms: int = 0


class ContextInjector:
    """新 session 启动时，自动从知识图谱注入跨会话记忆。

    流程:
      1. 从用户查询中抽取实体
      2. 对每个实体调用 MCP Server 的 query_kg
      3. 汇总关联 task，格式化为上下文文本
      4. 返回 InjectedContext，可直接拼入 system prompt
    """

    def __init__(
        self,
        client: MCPClient,
        depth: int = 1,
        max_tasks: int = 10,
        max_entities: int = 3,
    ):
        self.client = client
        self.depth = depth
        self.max_tasks = max_tasks
        self.max_entities = max_entities

    def build_context(self, query: str) -> InjectedContext:
        """为给定查询构建注入上下文。

        Args:
            query: 用户的首条消息或查询

        Returns:
            InjectedContext，其中 context_text 可直接拼入 system prompt
        """
        t0 = time.time()

        # Step 1: 抽取实体
        entities = extract_session_entities(query)
        if not entities:
            return InjectedContext(
                query=query, entities=[], matched_entities=[],
                related_tasks=[], context_text="",
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        # Step 2: 对每个实体查询图谱 (最多 max_entities 个)
        all_tasks: dict[str, dict] = {}  # task_id -> task, 去重
        matched = []
        for ent in entities[:self.max_entities]:
            try:
                result = self.client.query_kg(ent, depth=self.depth)
                matched.append(result.get("seed_entity", ent))
                for task in result.get("related_tasks", []):
                    tid = task["task_id"]
                    if tid not in all_tasks:
                        all_tasks[tid] = task
            except Exception as e:
                logger.warning(f"query_kg 失败 [{ent}]: {e}")

        # Step 3: 截断 + 排序（按 task_id 稳定排序）
        tasks_list = sorted(all_tasks.values(), key=lambda t: t["task_id"])[:self.max_tasks]

        # Step 4: 格式化上下文文本
        context_text = self._format_context(matched, tasks_list)

        return InjectedContext(
            query=query,
            entities=entities,
            matched_entities=matched,
            related_tasks=tasks_list,
            context_text=context_text,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    def _format_context(self, entities: list[str], tasks: list[dict]) -> str:
        """将关联 task 格式化为可注入 prompt 的文本"""
        if not tasks:
            return ""

        lines = [
            "## 跨会话记忆（来自知识图谱）",
            f"基于查询中识别的实体 [{', '.join(entities)}]，以下是相关的历史任务上下文：",
            "",
        ]
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. **{task['task_label']}** (session: {task['session_id'][:8]}...)")
            lines.append(f"   {task['task_summary']}")
            lines.append("")

        lines.append("请基于以上历史上下文回答用户的问题。如果用户说'继续'，请从上述任务延续。")
        return "\n".join(lines)


# ============================================================
# OpenCode 集成示例
# ============================================================

def init_opencode(base_url: str = "http://localhost:8000"):
    """OpenCode 初始化钩子：注册 MCP 工具。

    在 OpenCode 的 agent 初始化阶段调用，将 KG 工具注册为可用工具。
    """
    client = MCPClient(base_url)

    # 验证连接
    try:
        health = client.health()
        logger.info(f"KG MCP Server 连接成功: {health}")
    except Exception as e:
        logger.error(f"KG MCP Server 连接失败: {e}")
        return None

    # 获取工具定义
    tools = client.list_tools()
    logger.info(f"注册 {len(tools)} 个 MCP 工具: {[t['name'] for t in tools]}")

    return tools


def on_new_session(
    query: str,
    base_url: str = "http://localhost:8000",
) -> str:
    """新 session 启动钩子：自动注入跨会话记忆。

    在 OpenCode 每次新建 session 时调用，返回要注入 system prompt 的上下文文本。

    Args:
        query: 用户的首条消息
        base_url: MCP Server 地址

    Returns:
        要注入 system prompt 的上下文文本（空字符串表示无相关上下文）
    """
    client = MCPClient(base_url)
    injector = ContextInjector(client, depth=1, max_tasks=10)

    try:
        ctx = injector.build_context(query)
        if ctx.context_text:
            logger.info(
                f"注入上下文: {len(ctx.matched_entities)} 实体 → "
                f"{len(ctx.related_tasks)} tasks ({ctx.elapsed_ms}ms)"
            )
        return ctx.context_text
    except Exception as e:
        logger.warning(f"上下文注入失败: {e}")
        return ""


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys

    base_url = os.environ.get("KG_MCP_URL", "http://localhost:8000")
    client = MCPClient(base_url)

    if len(sys.argv) < 2:
        print("用法:")
        print(f"  python3 {sys.argv[0]} health                  # 健康检查")
        print(f"  python3 {sys.argv[0]} tools                   # 列出工具")
        print(f"  python3 {sys.argv[0]} stats                   # 图谱统计")
        print(f"  python3 {sys.argv[0]} query <entity> [depth]  # BFS 查询")
        print(f"  python3 {sys.argv[0]} search <keyword>        # 模糊搜索")
        print(f"  python3 {sys.argv[0]} context <query>         # 上下文注入")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "health":
        print(json.dumps(client.health(), indent=2, ensure_ascii=False))

    elif cmd == "tools":
        tools = client.list_tools()
        for t in tools:
            print(f"  {t['name']}: {t['description'][:60]}")

    elif cmd == "stats":
        print(json.dumps(client.get_stats(), indent=2, ensure_ascii=False))

    elif cmd == "query":
        entity = sys.argv[2] if len(sys.argv) > 2 else "OpenCode"
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        result = client.query_kg(entity, depth=depth)
        print(f"种子: {result['seed_entity']} → 扩散: {len(result['expanded_entities'])} 实体")
        print(f"关联 {len(result['related_tasks'])} 个 task:")
        for t in result["related_tasks"]:
            print(f"  [{t['task_id']}] {t['task_label']}: {t['task_summary'][:80]}")
        print(f"耗时: {result['elapsed_ms']}ms")

    elif cmd == "search":
        keyword = sys.argv[2] if len(sys.argv) > 2 else "open"
        result = client.search_entities(keyword)
        print(f"搜索 '{keyword}' → {result['count']} 结果:")
        for r in result["results"]:
            print(f"  {r['name']} ({r['entity_type']}) — {r['task_count']} tasks")

    elif cmd == "context":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "继续搞 RoPE 优化"
        injector = ContextInjector(client)
        ctx = injector.build_context(query)
        print(f"查询: {ctx.query}")
        print(f"实体: {ctx.entities} → 命中: {ctx.matched_entities}")
        print(f"关联 task: {len(ctx.related_tasks)}")
        print(f"耗时: {ctx.elapsed_ms}ms")
        print()
        if ctx.context_text:
            print("=== 注入上下文 ===")
            print(ctx.context_text)
        else:
            print("(无相关上下文)")

    else:
        print(f"未知命令: {cmd}")
