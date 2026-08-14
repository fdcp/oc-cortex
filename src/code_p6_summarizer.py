"""
Phase 6: 跨 Session 总结模块

基于 Phase 4 搜索 + Phase 5 知识图谱，对多个相关 session 的内容进行主题性总结。

流程:
  Query → SessionSearcher 检索 Top-K Task
       → (可选) GraphRAGSearcher 图谱增强
       → 收集 Task 关联 Chunk 原文
       → 时间范围过滤
       → 内容裁剪（控制 token 预算）
       → LLM 生成主题总结
       → 返回结构化结果

示例:
  from code_p6_summarizer import SessionSummarizer

  summarizer = SessionSummarizer("code_p3_config.yaml")
  result = summarizer.summarize("我最近做过的 FlashAttention 相关工作")
  print(result.summary)
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger
from openai import OpenAI

from code_update_prompt_utils import load_prompt

from code_p1_utils import count_tokens, safe_truncate
from code_p4_searcher import SessionSearcher, SessionSearchResult

# ============================================================
# LLM 客户端（复用 code_p5e 模式）
# ============================================================

DEFAULT_MODEL = "nemotron-3-ultra-free"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    return OpenAI(
        api_key=api_key,
        base_url="https://opencode.ai/zen/v1",
        timeout=120,
    )


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SummarySource:
    """总结的源 Task 信息"""
    task_id: str
    session_id: str
    task_label: str
    task_summary: str
    rerank_score: float
    chunk_count: int
    created_at: str


@dataclass
class SummaryResult:
    """跨 Session 总结结果"""
    query: str
    summary: str                  # LLM 生成的主题总结
    sources: list[SummarySource]  # 参与总结的 Task 列表
    total_tokens: int             # 输入 LLM 的总 token 数
    elapsed_ms: int               # 总耗时
    debug: dict = field(default_factory=dict)


# ============================================================
# 核心总结器
# ============================================================

class SessionSummarizer:
    """跨 Session 总结器：检索相关 Task → 收集原文 → LLM 生成总结"""

    def __init__(
        self,
        config_path: str = "code_p3_config.yaml",
        model: str = DEFAULT_MODEL,
        max_context_tokens: int = 12000,
        max_tasks: int = 10,
        searcher=None,
    ):
        """
        Args:
            config_path: Phase 3 配置文件路径
            model: LLM 模型名
            max_context_tokens: LLM 输入 token 预算
            max_tasks: 最多纳入总结的 Task 数
            searcher: 外部传入的 SessionSearcher(共享 Qdrant 客户端时使用)
        """
        self.searcher = searcher if searcher is not None else SessionSearcher(config_path)
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_tasks = max_tasks
        self._client: Optional[OpenAI] = None

    def _get_llm_client(self) -> OpenAI:
        if self._client is None:
            self._client = _get_client()
        return self._client

    def summarize(
        self,
        query: str,
        top_k: int = 8,
        time_range: Optional[tuple[str, str]] = None,
        use_graph_rag: bool = False,
        include_chunks: bool = True,
        chunk_detail_level: str = "summary",
    ) -> SummaryResult:
        """生成跨 Session 主题总结。

        Args:
            query: 用户查询（如 "我最近做过的性能优化工作"）
            top_k: 检索的 Task 数
            time_range: 可选时间范围 (start_iso, end_iso)，如 ("2026-07-01", "2026-07-15")
            use_graph_rag: 是否使用 Graph-RAG 增强检索
            include_chunks: 是否在 LLM 输入中包含 chunk 原文
            chunk_detail_level: chunk 详情级别
                "summary" — 仅 chunk 摘要（轻量）
                "preview" — chunk 摘要 + 前 500 字原文
                "full" — chunk 完整原文（重，token 消耗大）

        Returns:
            SummaryResult 包含 LLM 生成的总结和元信息
        """
        t0 = time.time()
        debug = {}

        # ---- Stage 1: 检索相关 Task ----
        if use_graph_rag:
            results, graph_debug = self._search_with_graph(query, top_k)
            debug["graph"] = graph_debug
        else:
            results = self.searcher.search(query, top_k=top_k * 2, skip_rerank=False)
            results = results[:top_k]
        debug["retrieved_tasks"] = len(results)

        # ---- Stage 2: 时间过滤 ----
        if time_range:
            results = self._filter_by_time(results, time_range)
            debug["after_time_filter"] = len(results)

        # 截断到 max_tasks
        results = results[:self.max_tasks]

        # ---- Stage 3: 收集内容 ----
        task_contents = []
        sources = []
        total_tokens = 0

        for r in results:
            task = self.searcher.task_map.get(r.task_id)
            if not task:
                continue

            # 构建 chunk 内容
            chunk_text = ""
            if include_chunks:
                chunk_text = self._gather_chunk_content(
                    task.chunk_ids, chunk_detail_level
                )

            # 格式化单个 task 内容
            formatted = load_prompt("TASK_CONTENT_TEMPLATE").format(
                task_label=r.task_label,
                task_id=r.task_id,
                session_short=r.session_id[:12] + "...",
                created_at=task.created_at[:10] if task.created_at else "N/A",
                task_summary=r.task_summary,
                chunk_content=chunk_text if chunk_text else "(无详细对话内容)",
            )

            # Token 预算检查
            content_tokens = count_tokens(formatted)
            if total_tokens + content_tokens > self.max_context_tokens:
                # 裁剪
                remaining = self.max_context_tokens - total_tokens
                if remaining < 200:
                    break
                formatted = safe_truncate(formatted, remaining)
                content_tokens = remaining

            task_contents.append(formatted)
            total_tokens += content_tokens

            sources.append(SummarySource(
                task_id=r.task_id,
                session_id=r.session_id,
                task_label=r.task_label,
                task_summary=r.task_summary,
                rerank_score=r.rerank_score,
                chunk_count=len(task.chunk_ids),
                created_at=task.created_at or "",
            ))

        debug["total_input_tokens"] = total_tokens
        debug["tasks_used"] = len(task_contents)

        # ---- Stage 4: 构建 Prompt + 调用 LLM ----
        time_range_text = ""
        if time_range:
            time_range_text = f"【时间范围】{time_range[0]} 至 {time_range[1]}"

        user_prompt = load_prompt("SUMMARY_USER_PROMPT").format(
            query=query,
            time_range_text=time_range_text,
            task_count=len(task_contents),
            task_contents="\n\n".join(task_contents),
        )

        summary_text = self._call_llm(user_prompt)
        debug["output_tokens"] = count_tokens(summary_text)

        elapsed = int((time.time() - t0) * 1000)
        debug["total_time_ms"] = elapsed

        return SummaryResult(
            query=query,
            summary=summary_text,
            sources=sources,
            total_tokens=total_tokens,
            elapsed_ms=elapsed,
            debug=debug,
        )

    # ---- 内部方法 ----

    def _search_with_graph(
        self, query: str, top_k: int
    ) -> tuple[list[SessionSearchResult], dict]:
        """使用 GraphRAGSearcher 增强检索"""
        try:
            from code_p5e_db import KGDatabase
            from code_p5e_graph_rag import GraphRAGSearcher

            db_path = "output/triple/knowledge_graph.db"
            if not os.path.exists(db_path):
                db_path = "output/entity/knowledge_graph.db"

            kg_db = KGDatabase(db_path)
            rag = GraphRAGSearcher(self.searcher, kg_db)
            return rag.search(query, top_k=top_k, use_graph_rag=True)
        except ImportError as e:
            logger.warning(f"Graph-RAG 不可用，回退到纯向量检索: {e}")
            results = self.searcher.search(query, top_k=top_k * 2, skip_rerank=False)
            return results[:top_k], {"fallback": "vector_only"}

    def _filter_by_time(
        self,
        results: list[SessionSearchResult],
        time_range: tuple[str, str],
    ) -> list[SessionSearchResult]:
        """按时间范围过滤 Task"""
        start_str, end_str = time_range
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
        except ValueError:
            logger.warning(f"时间范围格式错误: {time_range}，跳过过滤")
            return results

        filtered = []
        for r in results:
            task = self.searcher.task_map.get(r.task_id)
            if not task or not task.created_at:
                continue
            try:
                task_dt = datetime.fromisoformat(task.created_at)
                if start_dt <= task_dt <= end_dt:
                    filtered.append(r)
            except ValueError:
                continue

        return filtered

    def _gather_chunk_content(
        self,
        chunk_ids: list[str],
        detail_level: str = "summary",
    ) -> str:
        """收集 chunk 内容

        Args:
            chunk_ids: chunk ID 列表
            detail_level: "summary" | "preview" | "full"
        """
        parts = []
        for cid in chunk_ids:
            chunk = self.searcher.chunk_map.get(cid)
            if not chunk:
                continue

            if detail_level == "summary":
                summary = self.searcher.summaries.get(cid, "")
                if summary:
                    parts.append(f"[Turn {chunk.turn_index}] {summary}")
            elif detail_level == "preview":
                summary = self.searcher.summaries.get(cid, "")
                preview = chunk.cleaned_text()[:500]
                parts.append(f"[Turn {chunk.turn_index}] {summary}\n{preview}...")
            elif detail_level == "full":
                parts.append(f"[Turn {chunk.turn_index}]\n{chunk.cleaned_text()}")

        return "\n\n".join(parts)

    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM 生成总结"""
        client = self._get_llm_client()

        try:
            output = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": load_prompt("SUMMARY_SYSTEM_PROMPT")},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"[LLM 调用失败: {e}]"

        raw = output.choices[0].message.content
        if not raw:
            return "[LLM 返回空内容]"

        # 去除 <think>...</think> 标签块 (部分模型会输出思考过程)
        content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return content
