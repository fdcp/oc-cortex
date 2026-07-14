"""
Phase 2: Task Extractor
从 session chunks 中用 LLM 提炼 1-5 个 task

核心流程:
  1. 构建 prompt: 将 session 所有 chunk 格式化后注入 SESSION_TASK_PROMPT
  2. 调用 LLM (OpenAI 兼容 API, 如 DashScope / SiliconFlow / OpenRouter)
  3. 解析 JSON + 校验 (任务数量 / chunk_id 覆盖 / 无重复归属)
  4. 返回 Task 列表
"""
import json
import re
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from loguru import logger

from code_p1_models import Chunk
from code_p1_utils import count_tokens, safe_truncate
from code_p2_models import Task


# ============================================================
# Prompt 模板
# ============================================================

SESSION_TASK_PROMPT = """你是一个对话历史分析助手。请阅读以下一个 opencode session 的完整对话(已按"轮次"切分,每个轮次包含 1 个 user 消息及其触发的所有操作)。

你的任务是:总结该 session 完成的所有主要工作/任务,粒度为"一件独立的事",而不是"一轮对话"。

【重要约束】
- 同一件事可能跨多个对话轮次
- 大多数 session 包含 1-2 个任务,极少超过 5 个
- 任务之间应该是相对独立的工作,不是同一件事的子步骤

【输出格式】严格 JSON,不要其他内容:
{{
  "tasks": [
    {{
      "task_id": "T1",
      "task_label": "简短中文标签(5-15 字)",
      "task_summary": "该任务的总结(50-200 字,说清楚做了什么、结果如何)",
      "chunk_ids": ["c1", "c2", "c3"]
    }}
  ]
}}

【轮次内容】
{turns_content}
"""


# ============================================================
# 辅助函数
# ============================================================

def _extract_json_from_response(response: str) -> str:
    """
    从 LLM 响应中提取 JSON 字符串
    LLM 有时会包裹在 ```json ... ``` 中,或前后有多余文字
    """
    response = response.strip()

    # 尝试直接解析
    if response.startswith("{"):
        return response

    # 尝试从 markdown 代码块中提取
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试找到第一个 { 和最后一个 }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start:end + 1]

    return response


def _format_chunks_for_prompt(
    chunks: list[Chunk], max_tokens_per_chunk: int = 2000
) -> str:
    """
    将所有 chunk 格式化为 prompt 中的轮次文本
    对单个 chunk 超限时进行截断
    """
    parts = []
    for chunk in chunks:
        text = chunk.cleaned_text()
        if count_tokens(text) > max_tokens_per_chunk:
            text = safe_truncate(text, max_tokens_per_chunk)
        # chunk_id 格式: {session_id}_c{index} -> 提取 c{index} 作为短 ID
        short_id = chunk.chunk_id.split("_c")[-1] if "_c" in chunk.chunk_id else chunk.chunk_id
        parts.append(f"--- c{short_id} ---\n{text}\n")
    return "\n".join(parts)


# ============================================================
# TaskExtractor
# ============================================================

class TaskExtractor:
    """
    LLM-based Task Extractor

    通过 OpenAI 兼容 API 调用 LLM,从 session chunks 中提炼 task 清单。
    支持:
      - 可配置的 LLM 提供商 (DashScope / SiliconFlow / OpenAI / Ollama 等)
      - 指数退避重试
      - JSON 解析与校验
      - Token 预算管理
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
        timeout: int = 120,
        max_tokens_per_chunk: int = 2000,
        max_total_prompt_tokens: int = 30000,
        concurrency: int = 1,
    ):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_total_prompt_tokens = max_total_prompt_tokens
        self.concurrency = max(1, concurrency)  # 至少为 1

        # 获取 API Key
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"未找到 API Key: 环境变量 {api_key_env} 未设置,且未通过参数传入。\n"
                f"请先设置: export {api_key_env}='your-api-key'"
            )

        self.base_url = base_url

        # 初始化 OpenAI 兼容客户端 (线程安全)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            f"TaskExtractor 初始化: model={self.model}, "
            f"base_url={self.base_url}, concurrency={self.concurrency}"
        )

    # --------------------------------------------------------
    # LLM 调用
    # --------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 获取响应"""
        logger.debug(
            f"LLM 调用: model={self.model}, "
            f"prompt_tokens~{count_tokens(prompt)}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=8000,  # reasoning 模型需要更多 token (含 thinking)
        )

        msg = response.choices[0].message
        content = msg.content

        # Reasoning 模型 (如 DeepSeek V4) 可能把 token 消耗在 thinking 上,
        # 导致 content 为空;此时检查 reasoning_content 作为 fallback
        if not content or not content.strip():
            reasoning = None
            if hasattr(msg, 'model_extra') and msg.model_extra:
                reasoning = msg.model_extra.get('reasoning_content', '')
            if reasoning and reasoning.strip():
                logger.warning(
                    "LLM content 为空,尝试从 reasoning_content 提取 JSON"
                )
                content = reasoning
            else:
                raise ValueError("LLM 返回空内容 (content 和 reasoning 均为空)")

        # 记录 token 用量
        usage = response.usage
        if usage:
            logger.debug(
                f"LLM token: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}"
            )

        return content.strip()

    def _call_llm_with_retry(self, prompt: str) -> str:
        """带指数退避重试的 LLM 调用"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._call_llm(prompt)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 ...
                    logger.warning(
                        f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, "
                        f"{wait_time}s 后重试"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"LLM 调用失败,已达最大重试 ({self.max_retries}): {e}"
                    )

        raise last_error  # type: ignore

    # --------------------------------------------------------
    # 解析与校验
    # --------------------------------------------------------

    def _parse_and_validate(
        self,
        llm_output: str,
        session_id: str,
        valid_chunk_ids: set[str],
    ) -> list[Task]:
        """
        解析 LLM 输出并校验

        校验规则:
        1. JSON 格式合法
        2. tasks 数量在 [1, 5]
        3. 所有 chunk_ids 都对应到实际 chunk
        4. 每个 chunk 至少属于一个 task (不遗漏)
        5. 不允许一个 chunk 同时属于多个 task
        """
        # 1. 提取 JSON
        json_str = _extract_json_from_response(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON 解析失败: {e}\n原始输出:\n{llm_output[:500]}"
            )

        if "tasks" not in data:
            raise ValueError(
                f"缺少 'tasks' 字段, 实际 keys: {list(data.keys())}"
            )

        tasks_data = data["tasks"]
        if not isinstance(tasks_data, list):
            raise ValueError(
                f"'tasks' 应为列表, 实际类型: {type(tasks_data)}"
            )

        # 2. 校验数量
        if not 1 <= len(tasks_data) <= 5:
            logger.warning(
                f"Session {session_id}: 任务数 {len(tasks_data)} 不在 [1,5], 仍继续处理"
            )

        # 3. 校验 chunk_ids 合法性
        assigned_chunks: set[str] = set()
        for t in tasks_data:
            for cid in t.get("chunk_ids", []):
                # LLM 可能返回 "c1", "1", 或完整 ID "{session_id}_c1"
                if cid.startswith("ses_"):
                    full_cid = cid
                elif cid.startswith("c"):
                    full_cid = f"{session_id}_{cid}"
                else:
                    full_cid = f"{session_id}_c{cid}"
                if full_cid not in valid_chunk_ids:
                    logger.warning(
                        f"Session {session_id}: chunk_id '{full_cid}' 无效, 忽略"
                    )
                assigned_chunks.add(full_cid)

        # 4. 校验覆盖率
        uncovered = valid_chunk_ids - assigned_chunks
        if uncovered:
            logger.warning(
                f"Session {session_id}: {len(uncovered)} 个 chunk 未被覆盖: "
                f"{sorted(uncovered)[:3]}..."
            )

        # 5. 检查重复归属
        chunk_task_count: dict[str, int] = {}
        for t in tasks_data:
            for cid in t.get("chunk_ids", []):
                if cid.startswith("ses_"):
                    full_cid = cid
                elif cid.startswith("c"):
                    full_cid = f"{session_id}_{cid}"
                else:
                    full_cid = f"{session_id}_c{cid}"
                chunk_task_count[full_cid] = chunk_task_count.get(full_cid, 0) + 1
        duplicates = {k: v for k, v in chunk_task_count.items() if v > 1}
        if duplicates:
            logger.warning(
                f"Session {session_id}: {len(duplicates)} 个 chunk 被多个 task 引用"
            )

        # 构建 Task 对象
        tasks = []
        for i, t in enumerate(tasks_data):
            raw_chunk_ids = t.get("chunk_ids", [])
            full_chunk_ids = []
            for cid in raw_chunk_ids:
                if cid.startswith("ses_"):
                    full_cid = cid
                elif cid.startswith("c"):
                    full_cid = f"{session_id}_{cid}"
                else:
                    full_cid = f"{session_id}_c{cid}"
                if full_cid in valid_chunk_ids:
                    full_chunk_ids.append(full_cid)

            task = Task(
                task_id=f"{session_id}_T{i + 1}",
                session_id=session_id,
                task_label=t.get("task_label", f"任务{i + 1}"),
                task_summary=t.get("task_summary", ""),
                chunk_ids=full_chunk_ids,
            )
            tasks.append(task)

        return tasks

    # --------------------------------------------------------
    # 主方法
    # --------------------------------------------------------

    def _build_prompt(self, chunks: list[Chunk]) -> str:
        """构建 prompt,必要时截断"""
        turns_content = _format_chunks_for_prompt(
            chunks, max_tokens_per_chunk=self.max_tokens_per_chunk
        )
        prompt = SESSION_TASK_PROMPT.format(turns_content=turns_content)

        total_tokens = count_tokens(prompt)
        if total_tokens > self.max_total_prompt_tokens:
            logger.warning(
                f"Prompt 超限: {total_tokens} > {self.max_total_prompt_tokens}, 截断"
            )
            reduced_per_chunk = max(
                200,
                self.max_total_prompt_tokens // len(chunks) - 100,
            )
            turns_content = _format_chunks_for_prompt(
                chunks, max_tokens_per_chunk=reduced_per_chunk
            )
            prompt = SESSION_TASK_PROMPT.format(turns_content=turns_content)
            logger.info(f"截断后 prompt tokens: {count_tokens(prompt)}")

        return prompt

    def extract_tasks(self, session_id: str, chunks: list[Chunk]) -> list[Task]:
        """
        从单个 session 的 chunks 中提炼 task
        """
        if not chunks:
            logger.warning(f"Session {session_id}: 没有 chunks, 跳过")
            return []

        valid_chunk_ids = {c.chunk_id for c in chunks}
        prompt = self._build_prompt(chunks)

        logger.info(
            f"Session {session_id}: {len(chunks)} chunks, "
            f"prompt~{count_tokens(prompt)} tokens"
        )

        llm_output = self._call_llm_with_retry(prompt)
        tasks = self._parse_and_validate(llm_output, session_id, valid_chunk_ids)

        logger.info(f"Session {session_id}: 提取到 {len(tasks)} 个 task")
        return tasks

    def extract_tasks_batch(
        self, sessions_chunks: dict[str, list[Chunk]]
    ) -> list[Task]:
        """
        批量处理多个 session

        当 concurrency > 1 时使用线程池并行调用 LLM,加速处理。
        OpenAI SDK 的 client 是线程安全的,可以安全地在多线程中使用。
        """
        total = len(sessions_chunks)

        if self.concurrency == 1 or total <= 1:
            return self._extract_tasks_serial(sessions_chunks)

        return self._extract_tasks_parallel(sessions_chunks)

    def _extract_tasks_serial(
        self, sessions_chunks: dict[str, list[Chunk]]
    ) -> list[Task]:
        """串行处理所有 session"""
        all_tasks: list[Task] = []
        total = len(sessions_chunks)
        success = 0
        failed = 0

        for idx, (session_id, chunks) in enumerate(
            sessions_chunks.items(), 1
        ):
            logger.info(f"[{idx}/{total}] 处理 session: {session_id}")
            try:
                tasks = self.extract_tasks(session_id, chunks)
                all_tasks.extend(tasks)
                success += 1
            except Exception as e:
                logger.error(f"Session {session_id} 处理失败: {e}")
                failed += 1

        logger.info(
            f"Task 提取完成 (串行): {total} sessions, "
            f"成功 {success}, 失败 {failed}, "
            f"共 {len(all_tasks)} 个 task"
        )
        return all_tasks

    def _extract_tasks_parallel(
        self, sessions_chunks: dict[str, list[Chunk]]
    ) -> list[Task]:
        """并行处理所有 session (线程池)"""
        all_tasks: list[Task] = []
        total = len(sessions_chunks)
        success = 0
        failed = 0

        # 线程安全的进度计数器
        progress_lock = threading.Lock()
        completed_count = 0

        # 将 session 列表转为有序列表,便于追踪
        session_items = list(sessions_chunks.items())

        logger.info(
            f"并行处理 {total} 个 session, 并发数: {self.concurrency}"
        )

        def _process_one(item: tuple[str, list[Chunk]]) -> tuple[str, list[Task] | None]:
            """处理单个 session, 返回 (session_id, tasks_or_None)"""
            nonlocal completed_count
            session_id, chunks = item
            try:
                tasks = self.extract_tasks(session_id, chunks)
                return session_id, tasks
            except Exception as e:
                logger.error(f"Session {session_id} 处理失败: {e}")
                return session_id, None
            finally:
                with progress_lock:
                    completed_count += 1
                    current = completed_count
                logger.info(f"进度: [{current}/{total}]")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # 提交所有任务
            future_to_session = {
                executor.submit(_process_one, item): item[0]
                for item in session_items
            }

            # 收集结果
            for future in as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    sid, tasks = future.result()
                    if tasks is not None:
                        all_tasks.extend(tasks)
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Session {session_id} 线程异常: {e}")
                    failed += 1

        logger.info(
            f"Task 提取完成 (并发={self.concurrency}): {total} sessions, "
            f"成功 {success}, 失败 {failed}, "
            f"共 {len(all_tasks)} 个 task"
        )
        return all_tasks
