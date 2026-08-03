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

SESSION_TASK_PROMPT = """你是一个资深的技术工作分析助手。请阅读以下一个 opencode session 的完整对话(已按"轮次"切分,每个轮次包含 1 个 user 消息及其触发的所有操作)。

请严格按以下三个步骤分析,最终输出 JSON。

【步骤 1: 逐轮提炼】
逐一阅读每个轮次,提炼该轮的**核心信息或关键进展**。

写作要求:
- 聚焦于"这一轮产生了什么有价值的结论/产出/决策",而非描述对话过程
- 不要写"用户询问…,助手回答…"这种流水账
- 用主语直接陈述事实,如:"确认了 X 的性能指标为 Y"、"完成了 Z 模块的实现"

好的例子: "确认 V100 Tensor Core 算力为 125 TFLOPS (FP16),适合中小模型推理场景"
坏的例子: "用户询问了V100的算力情况，助手详细列出了V100的算力参数、硬件规格和典型用途"

【步骤 2: 识别任务】
基于步骤 1 的提炼,归纳该 session 完成的主要任务:
- 粒度是"一件独立的事",不是"一轮对话"
- 同一件事可能跨多个轮次
- 不连续的轮次可能属于同一件事，如"turn1 说事A, turn2～4 说事B，turn5 说事A"，那么taskA就包含了turn1和turn5
- 大多数 session 有 1-2 个任务,极少超过 5 个
- 任务之间应相对独立,不是同一件事的子步骤
注意：不要按时间顺序生硬切分，而是全局审视所有轮次的摘要，将语义高度相关、服务于同一目标的轮次合并为一个任务。

【步骤 3: 归类轮次】
将每个轮次归入它所属的任务。**关键约束:每个轮次都必须归入至少一个任务,不允许遗漏。**

【task_summary 写作要求】
每个 task 的 summary 必须是语义自洽的知识块（彻底去除”用户”、”助手”等对话痕迹）。
根据任务复杂度选择合适的结构：

■ 深度技术任务（涉及分析、设计、实现、对比、推导）使用完整三段式：
- 背景与目标：一句话说明任务的背景和要解决的问题（为什么做）。
- 核心产出与关键决策：说明做了什么、怎么做的，必须保留关键技术实体和参数，以便于后续构建知识图谱和精准检索。
- 结论与意义：点明最终结果、业务建议或影响（结果如何）。

■ 简单操作性任务（安装工具、解释命令、回答概念、环境配置）使用简洁结构：
- 一句话说明做了什么
- 保留关键技术实体和参数
- 不需要强行添加”结论与意义”

好的例子（深度）: “为深入理解 LLM 训练硬件选型，系统对比了 V100/A100/A800/H100/H200/H800 六款 GPU 在算力、显存带宽和互联能力上的差异。结论：基于 Hopper 架构的 H100/H200 适合大规模训练；H800/A800 因 NVLink 带宽阉割至 400GB/s 不适合多机多卡场景；A800 是当前国内性价比合规选择。”
好的例子（简洁）: “通过 brew install tmux 安装终端复用器（版本 3.6b），并使用 bunx 安装 oh-my-openagent 插件配置 opencode 平台，禁用付费模型，默认 fallback 为 gpt-5-nano。”
坏的例子: “用户询问V100算力，助手回答了。用户又对比多款GPU，助手提供了对比表。用户讨论H200，助手纠正了说法。用户问显存影响，助手解释了。”

【输出格式】严格 JSON,不要其他内容:
{{
  “chunk_summaries”: [
    {{“chunk_id”: “c1”, “summary”: “该轮的核心结论或关键进展”}},
    {{“chunk_id”: “c2”, “summary”: “...”}}
  ],
  “tasks”: [
    {{
      “task_id”: “T1”,
      “task_label”: “简短中文标签(5-15 字)”,
      “task_summary”: “深度任务用三段式,简单任务用简洁式(50-200 字)”,
      “chunk_ids”: [“c1”, “c2”, “c3”]
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


def _repair_truncated_json(json_str: str) -> str:
    """
    修复被截断的 JSON (LLM 输出因 max_tokens 限制被截断)

    策略: 追踪未关闭的括号和字符串状态,在截断处正确闭合
    """
    # 状态追踪
    stack = []  # 括号栈: '{' 或 '['
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        c = json_str[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if c == '\\' and in_string:
            escape_next = True
            i += 1
            continue

        if c == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue

        if not in_string:
            if c in ('{', '['):
                stack.append(c)
            elif c == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif c == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        i += 1

    # 如果 JSON 完整 (栈为空且不在字符串中), 直接返回
    if not stack and not in_string:
        return json_str

    # 需要修复: 先处理未关闭的字符串
    result = json_str.rstrip()
    if in_string:
        result += '"'

    # 从后向前检查是否需要移除不完整的 key-value 对
    # 找到最后一个完整的元素 (以 , 或 { 或 [ 分隔)
    last_comma = result.rfind(',')
    if last_comma > 0:
        # 检查最后一个逗号后面是否有不完整的内容
        tail = result[last_comma + 1:].strip()
        # 如果尾部是不完整的 (比如只有 key 没有 value), 截掉它
        if tail and not tail.startswith('}') and not tail.startswith(']'):
            # 检查是否是 "key": "value" 形式但不完整
            if ':' in tail:
                # 尝试保留, 但如果解析仍然失败则回退
                pass
            else:
                result = result[:last_comma]

    # 按栈中剩余括号逆序关闭
    for bracket in reversed(stack):
        if bracket == '{':
            result += '}'
        elif bracket == '[':
            result += ']'

    return result


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
        content_retries: int = 3,
        timeout: int = 120,
        max_tokens_per_chunk: int = 2000,
        max_total_prompt_tokens: int = 30000,
        concurrency: int = 1,
        temperature: float = 0.3,
    ):
        self.model = model
        self.max_retries = max_retries
        self.content_retries = max(1, content_retries)
        self.timeout = timeout
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_total_prompt_tokens = max_total_prompt_tokens
        self.concurrency = max(1, concurrency)  # 至少为 1
        self.temperature = temperature

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
            f"base_url={self.base_url}, concurrency={self.concurrency}, "
            f"network_retries={self.max_retries}, content_retries={self.content_retries}, "
            f"temperature={self.temperature}"
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
            temperature=self.temperature,
            max_tokens=16000,  # CoT prompt 含 chunk_summaries, 需要更多 token
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
        chunks: list[Chunk],
    ) -> tuple[list[Task], int, int]:
        """
        解析 LLM 输出并校验

        校验规则:
        1. JSON 格式合法
        2. tasks 数量在 [1, 5]
        3. 所有 chunk_ids 都对应到实际 chunk
        4. 每个 chunk 至少属于一个 task (不遗漏, 自动兜底补全)
        5. 不允许一个 chunk 同时属于多个 task

        返回: (tasks, summaries_written, total_chunks)
        """
        valid_chunk_ids = {c.chunk_id for c in chunks}
        # chunk_id -> turn_index 映射, 用于兜底时就近分配
        chunk_turn_map = {c.chunk_id: c.turn_index for c in chunks}

        # 1. 提取 JSON
        json_str = _extract_json_from_response(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试修复被截断的 JSON
            logger.warning(
                f"Session {session_id}: JSON 解析失败, 尝试修复截断输出"
            )
            repaired = _repair_truncated_json(json_str)
            try:
                data = json.loads(repaired)
                logger.info(
                    f"Session {session_id}: JSON 修复成功"
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"JSON 解析失败且无法修复: {e}\n原始输出:\n{llm_output[:500]}"
                )

        # --- 处理 chunk_summaries (步骤 1 的输出) ---
        chunk_summaries = data.get("chunk_summaries", [])
        if chunk_summaries:
            logger.debug(
                f"Session {session_id}: LLM 返回 {len(chunk_summaries)} 个 chunk 总结"
            )
            # 构建 chunk_id -> summary 映射, 并写回 Chunk 对象
            chunk_map = {c.chunk_id: c for c in chunks}
            for cs in chunk_summaries:
                short_cid = cs.get("chunk_id", "")
                summary = cs.get("summary", "")
                logger.debug(f"  {short_cid}: {summary[:60]}")
                # 转为完整 ID 匹配
                if short_cid.startswith("ses_"):
                    full_cid = short_cid
                elif short_cid.startswith("c"):
                    full_cid = f"{session_id}_{short_cid}"
                else:
                    full_cid = f"{session_id}_c{short_cid}"
                if full_cid in chunk_map and summary:
                    chunk_map[full_cid].task_summary = summary
            written = sum(1 for c in chunks if c.task_summary)
            if written < len(chunks):
                logger.debug(
                    f"Session {session_id}: {written}/{len(chunks)} 个 chunk 获得总结"
                )
        else:
            logger.debug(f"Session {session_id}: LLM 未返回 chunk_summaries")

        # --- 处理 tasks ---
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

        # 辅助: 将 LLM 返回的短 ID 转为完整 ID
        def _to_full_id(cid: str) -> str:
            if cid.startswith("ses_"):
                return cid
            elif cid.startswith("c"):
                return f"{session_id}_{cid}"
            else:
                return f"{session_id}_c{cid}"

        # 3. 校验 chunk_ids 合法性 + 构建 Task 对象
        assigned_chunks: set[str] = set()
        tasks = []

        for i, t in enumerate(tasks_data):
            raw_chunk_ids = t.get("chunk_ids", [])
            full_chunk_ids = []
            for cid in raw_chunk_ids:
                full_cid = _to_full_id(cid)
                if full_cid not in valid_chunk_ids:
                    logger.warning(
                        f"Session {session_id}: chunk_id '{full_cid}' 无效, 忽略"
                    )
                else:
                    full_chunk_ids.append(full_cid)
                assigned_chunks.add(full_cid)

            task = Task(
                task_id=f"{session_id}_T{i + 1}",
                session_id=session_id,
                task_label=t.get("task_label", f"任务{i + 1}"),
                task_summary=t.get("task_summary", ""),
                chunk_ids=full_chunk_ids,
            )
            tasks.append(task)

        # 4. 兜底: 未覆盖的 chunk 自动归入最近的 task
        uncovered = valid_chunk_ids - assigned_chunks
        if uncovered and tasks:
            logger.warning(
                f"Session {session_id}: {len(uncovered)} 个 chunk 未被覆盖, 自动兜底分配"
            )
            for uc_id in sorted(uncovered):
                uc_turn = chunk_turn_map.get(uc_id, 0)
                # 找到 turn_index 最近的 task
                best_task_idx = 0
                best_dist = float("inf")
                for ti, task in enumerate(tasks):
                    if not task.chunk_ids:
                        continue
                    task_turns = [
                        chunk_turn_map.get(cid, 0) for cid in task.chunk_ids
                    ]
                    # 用 task 中所有 chunk 的 turn 与当前 chunk 的最小距离
                    dist = min(abs(t - uc_turn) for t in task_turns)
                    if dist < best_dist:
                        best_dist = dist
                        best_task_idx = ti
                tasks[best_task_idx].chunk_ids.append(uc_id)
                logger.info(
                    f"  {uc_id} (turn {uc_turn}) -> "
                    f"Task '{tasks[best_task_idx].task_label}'"
                )

        # 5. 检查重复归属
        chunk_task_count: dict[str, int] = {}
        for t in tasks:
            for cid in t.chunk_ids:
                chunk_task_count[cid] = chunk_task_count.get(cid, 0) + 1
        duplicates = {k: v for k, v in chunk_task_count.items() if v > 1}
        if duplicates:
            logger.warning(
                f"Session {session_id}: {len(duplicates)} 个 chunk 被多个 task 引用"
            )

        summaries_written = sum(1 for c in chunks if c.task_summary)
        return tasks, summaries_written, len(chunks)

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

    def extract_tasks(
        self, session_id: str, chunks: list[Chunk]
    ) -> dict:
        """
        从单个 session 的 chunks 中提炼 task

        包含内容校验重试: 如果 LLM 返回不完整 (chunk_summaries 缺失或 tasks 为空),
        自动重试, 最多 content_retries 次。

        返回 dict:
        {
            "tasks": list[Task],
            "summaries_written": int,
            "total_chunks": int,
            "total_attempts": int,
            "status": "success" | "incomplete" | "failed",
            "error": str | None,
        }
        """
        empty_result = {
            "tasks": [],
            "summaries_written": 0,
            "total_chunks": len(chunks),
            "total_attempts": 0,
            "status": "failed",
            "error": None,
        }

        if not chunks:
            logger.warning(f"Session {session_id}: 没有 chunks, 跳过")
            empty_result["status"] = "skipped"
            return empty_result

        prompt = self._build_prompt(chunks)

        logger.info(
            f"Session {session_id}: {len(chunks)} chunks, "
            f"prompt~{count_tokens(prompt)} tokens"
        )

        # 记录哪些 chunk 在第一次调用前就有 task_summary (正常情况: 都没有)
        pre_existing = {c.chunk_id for c in chunks if c.task_summary}
        last_error = None

        for attempt in range(1, self.content_retries + 1):
            # 重试前清除上一次 attempt 设置的 task_summary, 避免残留
            if attempt > 1:
                for c in chunks:
                    if c.chunk_id not in pre_existing:
                        c.task_summary = None

            try:
                llm_output = self._call_llm_with_retry(prompt)
                tasks, summaries_written, total_chunks = self._parse_and_validate(
                    llm_output, session_id, chunks
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Session {session_id}: LLM 调用失败 "
                    f"(内容重试 {attempt}/{self.content_retries}): {e}"
                )
                continue

            # 内容校验: chunk_summaries 是否完整 + tasks 是否非空
            issues = []
            if summaries_written < total_chunks:
                missing = total_chunks - summaries_written
                issues.append(
                    f"chunk_summaries 缺失 {missing}/{total_chunks}"
                )
            if not tasks:
                issues.append("tasks 为空")

            if not issues:
                logger.info(
                    f"Session {session_id}: 提取到 {len(tasks)} 个 task"
                    + (f" (第 {attempt} 次尝试)" if attempt > 1 else "")
                )
                return {
                    "tasks": tasks,
                    "summaries_written": summaries_written,
                    "total_chunks": total_chunks,
                    "total_attempts": attempt,
                    "status": "success",
                    "error": None,
                }

            # 内容不完整
            issue_str = "; ".join(issues)
            if attempt < self.content_retries:
                logger.warning(
                    f"Session {session_id}: {issue_str}, "
                    f"内容重试 {attempt}/{self.content_retries}"
                )
                last_error = issue_str
            else:
                logger.error(
                    f"Session {session_id}: {issue_str}, "
                    f"已达最大内容重试 ({self.content_retries}), 使用不完整结果"
                )
                return {
                    "tasks": tasks,
                    "summaries_written": summaries_written,
                    "total_chunks": total_chunks,
                    "total_attempts": attempt,
                    "status": "incomplete",
                    "error": issue_str,
                }

        # 所有尝试都异常 (每次 _call_llm_with_retry 都失败)
        logger.error(
            f"Session {session_id}: 全部 {self.content_retries} 次内容重试均失败, "
            f"最后错误: {last_error}"
        )
        empty_result["total_attempts"] = self.content_retries
        empty_result["error"] = last_error
        return empty_result

    def extract_tasks_batch(
        self, sessions_chunks: dict[str, list[Chunk]],
        on_session_done=None,
    ) -> tuple[list[Task], list[dict]]:
        """
        批量处理多个 session

        返回: (all_tasks, session_coverage)
        - all_tasks: 所有 session 的 task 列表
        - session_coverage: 每个 session 的 summary 覆盖统计列表

        当 concurrency > 1 时使用线程池并行调用 LLM,加速处理。
        OpenAI SDK 的 client 是线程安全的,可以安全地在多线程中使用。

        on_session_done(session_id, result, chunks): 每跑完一个 session 在
        主线程同步调用一次 (串行写入主线程), 适合增量落盘。result['status']
        与 coverage 同字段集。chunks 是该 session 的 Chunk 列表 (含被 worker
        写入的 task_summary)。
        """
        total = len(sessions_chunks)

        if self.concurrency == 1 or total <= 1:
            return self._extract_tasks_serial(sessions_chunks, on_session_done)

        return self._extract_tasks_parallel(sessions_chunks, on_session_done)

    def _extract_tasks_serial(
        self, sessions_chunks: dict[str, list[Chunk]], on_session_done=None,
    ) -> tuple[list[Task], list[dict]]:
        """串行处理所有 session"""
        all_tasks: list[Task] = []
        session_coverage: list[dict] = []
        total = len(sessions_chunks)

        for idx, (session_id, chunks) in enumerate(
            sessions_chunks.items(), 1
        ):
            logger.info(f"[{idx}/{total}] 处理 session: {session_id}")
            try:
                result = self.extract_tasks(session_id, chunks)
                all_tasks.extend(result["tasks"])
                cov = {
                    "session_id": session_id,
                    "total_chunks": result["total_chunks"],
                    "summaries_written": result["summaries_written"],
                    "total_attempts": result["total_attempts"],
                    "status": result["status"],
                    "error": result["error"],
                }
                session_coverage.append(cov)
                if on_session_done:
                    on_session_done(session_id, result, chunks)
            except Exception as e:
                # extract_tasks 内部已 catch 所有异常, 这里兜底意外错误
                logger.error(f"Session {session_id} 意外错误: {e}")
                session_coverage.append({
                    "session_id": session_id,
                    "total_chunks": len(chunks),
                    "summaries_written": 0,
                    "total_attempts": 0,
                    "status": "failed",
                    "error": str(e),
                })

        self._log_batch_summary("串行", session_coverage, all_tasks)
        return all_tasks, session_coverage

    def _extract_tasks_parallel(
        self, sessions_chunks: dict[str, list[Chunk]], on_session_done=None,
    ) -> tuple[list[Task], list[dict]]:
        """并行处理所有 session (线程池)"""
        all_tasks: list[Task] = []
        session_coverage: list[dict] = []
        total = len(sessions_chunks)

        # 线程安全的进度计数器
        progress_lock = threading.Lock()
        completed_count = 0

        # 将 session 列表转为有序列表,便于追踪
        session_items = list(sessions_chunks.items())

        logger.info(
            f"并行处理 {total} 个 session, 并发数: {self.concurrency}"
        )

        def _process_one(
            item: tuple[str, list[Chunk]]
        ) -> tuple[str, dict]:
            """处理单个 session, 返回 (session_id, result_dict)"""
            nonlocal completed_count
            session_id, chunks = item
            try:
                result = self.extract_tasks(session_id, chunks)
                return session_id, result
            except Exception as e:
                # extract_tasks 内部已 catch 所有异常, 这里兜底意外错误
                logger.error(f"Session {session_id} 意外错误: {e}")
                return session_id, {
                    "tasks": [],
                    "summaries_written": 0,
                    "total_chunks": len(chunks),
                    "total_attempts": 0,
                    "status": "failed",
                    "error": str(e),
                }
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
                    sid, result = future.result()
                    all_tasks.extend(result["tasks"])
                    cov = {
                        "session_id": sid,
                        "total_chunks": result["total_chunks"],
                        "summaries_written": result["summaries_written"],
                        "total_attempts": result["total_attempts"],
                        "status": result["status"],
                        "error": result["error"],
                    }
                    session_coverage.append(cov)
                    if on_session_done:
                        on_session_done(
                            sid, result, sessions_chunks.get(sid, [])
                        )
                except Exception as e:
                    logger.error(f"Session {session_id} 线程异常: {e}")
                    session_coverage.append({
                        "session_id": session_id,
                        "total_chunks": 0,
                        "summaries_written": 0,
                        "total_attempts": 0,
                        "status": "thread_error",
                        "error": str(e),
                    })

        self._log_batch_summary(
            f"并发={self.concurrency}", session_coverage, all_tasks
        )
        return all_tasks, session_coverage

    @staticmethod
    def _log_batch_summary(
        mode: str, session_coverage: list[dict], all_tasks: list[Task]
    ) -> None:
        """输出批次处理汇总"""
        total = len(session_coverage)
        n_success = sum(1 for s in session_coverage if s["status"] == "success")
        n_incomplete = sum(1 for s in session_coverage if s["status"] == "incomplete")
        n_failed = sum(1 for s in session_coverage if s["status"] in ("failed", "thread_error"))
        retried = sum(
            1 for s in session_coverage
            if s.get("total_attempts", 0) > 1
        )

        parts = [
            f"Task 提取完成 ({mode}): {total} sessions",
            f"成功 {n_success}",
        ]
        if n_incomplete:
            parts.append(f"不完整 {n_incomplete}")
        if n_failed:
            parts.append(f"失败 {n_failed}")
        if retried:
            parts.append(f"重试后恢复 {retried}")
        parts.append(f"共 {len(all_tasks)} 个 task")
        logger.info(", ".join(parts))
