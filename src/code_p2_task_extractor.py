"""
Phase 2: Task Extractor
从 session chunks 中用 LLM 提炼 1-5 个 task

核心流程:
  1. 构建 prompt: 通过 load_prompt("SESSION_TASK_PROMPT") 读取并注入 session 所有 chunk
  2. 调用 LLM (OpenAI 兼容 API, 如 DashScope / SiliconFlow / OpenRouter)
  3. 解析 JSON + 校验 (任务数量 / chunk_id 覆盖 / 无重复归属)
  4. 返回 Task 列表
"""
import json
import re
import time
import os
import math
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from loguru import logger

from code_p1_models import Chunk
from code_p1_utils import count_tokens, truncate_chunk_text, chunk_content_hash
from code_p2_models import Task
from code_update_prompt_utils import load_prompt


@dataclass
class BatchPlan:
    """单个 batch 的切分计划"""
    start_idx: int
    end_idx: int
    chunks: list[Chunk]
    content_hash: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass
class BatchResult:
    """单个 batch 的执行结果"""
    batch_id: int
    start_idx: int
    end_idx: int
    content_hash: str
    status: str
    attempts: int
    tasks: list[Task] = field(default_factory=list)
    chunk_summaries: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_checkpoint_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "chunk_range": [self.start_idx, self.end_idx],
            "content_hash": self.content_hash,
            "status": self.status,
            "attempts": self.attempts,
            "tasks": [t.to_dict() for t in self.tasks],
            "chunk_summaries": self.chunk_summaries,
            "error": self.error,
        }

    @classmethod
    def from_checkpoint_dict(cls, data: dict) -> "BatchResult":
        return cls(
            batch_id=data["batch_id"],
            start_idx=data["chunk_range"][0],
            end_idx=data["chunk_range"][1],
            content_hash=data["content_hash"],
            status=data["status"],
            attempts=data.get("attempts", 1),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            chunk_summaries=dict(data.get("chunk_summaries", {})),
            error=data.get("error"),
        )


class BatchPlanner:
    """
    多 batch 切分器: 顺序切 m 个 chunk/批, 每批独立校验 token 是否超
    max_total_prompt_tokens, 超了则按 ceil(m/2) 递归, 直到 len(batch)==1 兜底。
    """

    def __init__(
        self,
        max_total_tokens: int,
        m: int = 16,
        per_chunk_output_tokens: int = 250,
    ):
        if m < 1:
            raise ValueError(f"batch_size m must be >= 1, got {m}")
        self.max_total_tokens = max_total_tokens
        self.m = m
        self.per_chunk_output_tokens = per_chunk_output_tokens

    def _chunk_input_tokens(self, c: Chunk) -> int:
        if c.cleaned_size_tokens and c.cleaned_size_tokens > 0:
            return c.cleaned_size_tokens
        return count_tokens(c.cleaned_text())

    def _batch_input_tokens(self, chunks: list[Chunk]) -> int:
        return sum(self._chunk_input_tokens(c) for c in chunks)

    def _batch_fits(self, chunks: list[Chunk]) -> bool:
        if not chunks:
            return True
        total = self._batch_input_tokens(chunks) + len(chunks) * self.per_chunk_output_tokens
        return total <= self.max_total_tokens

    def needs_split(self, chunks: list[Chunk]) -> bool:
        """session 总输入 + N*250 > max_total 时需要切分"""
        if not chunks:
            return False
        total = self._batch_input_tokens(chunks) + len(chunks) * self.per_chunk_output_tokens
        return total > self.max_total_tokens

    def plan(self, chunks: list[Chunk]) -> list[BatchPlan]:
        """返回顺序 batch 列表; 每个 batch 内部校验大小, 超了递归减半"""
        plans: list[BatchPlan] = []
        i = 0
        n = len(chunks)
        while i < n:
            target = self.m
            batch = self._slice_to_fit(chunks, i, target)
            start, end = i, i + len(batch)
            batch_hash = self._hash_batch(batch)
            plans.append(BatchPlan(
                start_idx=start,
                end_idx=end,
                chunks=batch,
                content_hash=batch_hash,
            ))
            i = end
        return plans

    def _slice_to_fit(
        self, chunks: list[Chunk], start: int, target: int
    ) -> list[Chunk]:
        """从 start 起取最多 target 个 chunk, 但不超过 budget;
        超了则 ceil(target/2) 递归, 1 chunk 兜底。"""
        n = len(chunks)
        while True:
            end = min(start + target, n)
            batch = chunks[start:end]
            if self._batch_fits(batch) or len(batch) == 1:
                return batch
            target = max(1, math.ceil(target / 2))

    @staticmethod
    def _hash_batch(chunks: list[Chunk]) -> str:
        """保留顺序的 batch 内容哈希 (用于增量续跑判断)"""
        acc = hashlib.sha256()
        for c in chunks:
            h = c.content_hash or chunk_content_hash(c)
            acc.update(h.encode("utf-8"))
            acc.update(b"|")
        return acc.hexdigest()



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
    单个 chunk 超限时走分级截断: 先丢工具调用 (bash -> tool_call -> mcp),
    仍超则 user_message 全量 + assistant 头70%尾30%
    """
    parts = []
    for chunk in chunks:
        text = truncate_chunk_text(chunk, max_tokens_per_chunk)
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

    @staticmethod
    def _session_headers() -> dict:
        session_id = (
            os.environ.get("OPENCODE_SESSION_ID")
            or f"cli-{os.getpid()}-{int(time.time())}"
        )
        return {"x-opencode-session": session_id}

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
        batch_size: int = 16,
        per_chunk_output_tokens: int = 250,
    ):
        self.model = model
        self.max_retries = max_retries
        self.content_retries = max(1, content_retries)
        self.timeout = timeout
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_total_prompt_tokens = max_total_prompt_tokens
        self.concurrency = max(1, concurrency)
        self.temperature = temperature
        self.batch_size = max(1, batch_size)
        self.per_chunk_output_tokens = max(1, per_chunk_output_tokens)

        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"未找到 API Key: 环境变量 {api_key_env} 未设置,且未通过参数传入。\n"
                f"请先设置: export {api_key_env}='your-api-key'"
            )

        self.base_url = base_url

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            # opencode-go 端点要求 x-opencode-session header,
            # 否则 400 MissingSessionID (同 code_p5e_graph_rag._get_client)
            default_headers=self._session_headers(),
        )

        self._planner = BatchPlanner(
            max_total_tokens=self.max_total_prompt_tokens,
            m=self.batch_size,
            per_chunk_output_tokens=self.per_chunk_output_tokens,
        )

        logger.info(
            f"TaskExtractor 初始化: model={self.model}, "
            f"base_url={self.base_url}, concurrency={self.concurrency}, "
            f"network_retries={self.max_retries}, content_retries={self.content_retries}, "
            f"temperature={self.temperature}, batch_size={self.batch_size}, "
            f"per_chunk_output_tokens={self.per_chunk_output_tokens}"
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
            max_tokens=16000,
        )

        if response is None or not response.choices:
            raise ValueError(
                f"LLM 返回空 choices, response={response!r}"
            )

        msg = response.choices[0].message
        if msg is None:
            raise ValueError(f"LLM 返回空 message, choice={response.choices[0]!r}")

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
        task_id_prefix: Optional[str] = None,
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
                task_id=f"{task_id_prefix or session_id + '_'}T{i + 1}",
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
        prompt = load_prompt("SESSION_TASK_PROMPT").format(turns_content=turns_content)

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
            prompt = load_prompt("SESSION_TASK_PROMPT").format(turns_content=turns_content)
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
        on_batch_done=None,
        prior_batches_map: Optional[dict[str, list[dict]]] = None,
    ) -> tuple[list[Task], list[dict]]:
        """
        批量处理多个 session

        返回: (all_tasks, session_coverage)
        - all_tasks: 所有 session 的 task 列表
        - session_coverage: 每个 session 的 summary 覆盖统计列表

        当 concurrency > 1 时使用线程池并行调用 LLM,加速处理。
        单 batch session 优先入池, 多 batch session 全部排在 Phase 2。

        on_session_done(session_id, result, chunks): 每跑完一个 session 在
        主线程同步调用一次 (串行写入主线程), 适合增量落盘。
        on_batch_done(session_id, batch_result): 多 batch session 每跑完
        一个 batch 调一次 (per-batch checkpoint 落盘)。
        prior_batches_map: 多 batch session 的 checkpoint 续跑数据,
        {session_id: [prior_batch_dict, ...]}, 用于跳过已成功的 batch。
        """
        total = len(sessions_chunks)

        if self.concurrency == 1 or total <= 1:
            return self._extract_tasks_serial(
                sessions_chunks, on_session_done,
                on_batch_done=on_batch_done,
                prior_batches_map=prior_batches_map,
            )

        return self._extract_tasks_parallel(
            sessions_chunks, on_session_done,
            on_batch_done=on_batch_done,
            prior_batches_map=prior_batches_map,
        )

    def _extract_tasks_serial(
        self, sessions_chunks: dict[str, list[Chunk]], on_session_done=None,
        on_batch_done=None,
        prior_batches_map: Optional[dict[str, list[dict]]] = None,
    ) -> tuple[list[Task], list[dict]]:
        """串行处理所有 session (concurrency=1)"""
        all_tasks: list[Task] = []
        session_coverage: list[dict] = []
        total = len(sessions_chunks)
        prior_batches_map = prior_batches_map or {}

        for idx, (session_id, chunks) in enumerate(
            sessions_chunks.items(), 1
        ):
            logger.info(f"[{idx}/{total}] 处理 session: {session_id}")
            try:
                if not self._needs_split(chunks):
                    result = self.extract_tasks(session_id, chunks)
                else:
                    result = self._run_multi_batches_serial(
                        session_id, chunks,
                        prior=prior_batches_map.get(session_id, []),
                        on_batch_done=on_batch_done,
                    )
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

    def _run_multi_batches_serial(
        self,
        session_id: str,
        chunks: list[Chunk],
        prior: list[dict],
        on_batch_done=None,
    ) -> dict:
        """串行跑多 batch session (用于 concurrency=1)"""
        plans = self._planner.plan(chunks)
        prior_idx = self._build_prior_index(prior)
        results: list[BatchResult] = []
        for batch_id, plan in enumerate(plans):
            pb = prior_idx.get((batch_id, plan.content_hash))
            if pb and pb.get("status") == "success":
                result = self._batch_result_from_prior(batch_id, plan, pb)
                logger.debug(f"Session {session_id} batch {batch_id}: 复用 checkpoint")
            else:
                result = self._extract_one_batch(session_id, batch_id, plan)
            results.append(result)
            if on_batch_done:
                on_batch_done(session_id, result, chunks)
        return self._merge_batch_results(session_id, chunks, results)

    def _extract_tasks_parallel(
        self, sessions_chunks: dict[str, list[Chunk]], on_session_done=None,
        on_batch_done=None,
        prior_batches_map: Optional[dict[str, list[dict]]] = None,
    ) -> tuple[list[Task], list[dict]]:
        """两阶段并行: Phase 1 单 batch session, Phase 2 多 batch session 的所有 batch。
        共享 ThreadPoolExecutor, 但多 batch session 全部排在 Phase 2。"""
        all_tasks: list[Task] = []
        session_coverage: list[dict] = []
        prior_batches_map = prior_batches_map or {}

        single_sessions: dict[str, list[Chunk]] = {}
        multi_sessions: dict[str, list[Chunk]] = {}
        for sid, chunks in sessions_chunks.items():
            (multi_sessions if self._needs_split(chunks) else single_sessions)[sid] = chunks
        logger.info(
            f"分类: 单 batch={len(single_sessions)} sessions, "
            f"多 batch={len(multi_sessions)} sessions, "
            f"并发={self.concurrency}"
        )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            phase1 = self._run_phase1(
                executor, single_sessions, on_session_done,
                total_sessions=len(sessions_chunks),
            )
            all_tasks.extend(phase1["all_tasks"])
            session_coverage.extend(phase1["coverage"])

            if multi_sessions:
                phase2 = self._run_phase2(
                    executor, multi_sessions, prior_batches_map,
                    on_session_done, on_batch_done,
                )
                all_tasks.extend(phase2["all_tasks"])
                session_coverage.extend(phase2["coverage"])

        self._log_batch_summary(
            f"并发={self.concurrency}", session_coverage, all_tasks
        )
        return all_tasks, session_coverage

    def _run_phase1(
        self, executor, single_sessions, on_session_done, total_sessions
    ) -> dict:
        """Phase 1: 单 batch session 全部入池"""
        all_tasks: list[Task] = []
        coverage: list[dict] = []
        progress_lock = threading.Lock()
        completed = 0

        def _process(sid, chunks):
            nonlocal completed
            try:
                return sid, self.extract_tasks(sid, chunks)
            except Exception as e:
                logger.error(f"Session {sid} 意外错误: {e}")
                return sid, {
                    "tasks": [], "summaries_written": 0,
                    "total_chunks": len(chunks), "total_attempts": 0,
                    "status": "failed", "error": str(e),
                }
            finally:
                with progress_lock:
                    completed += 1
                    logger.info(f"进度: [{completed}/{total_sessions}]")

        futures = {
            executor.submit(_process, sid, chs): sid
            for sid, chs in single_sessions.items()
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                _, result = fut.result()
                all_tasks.extend(result["tasks"])
                coverage.append(self._build_coverage(sid, result))
                if on_session_done:
                    on_session_done(sid, result, single_sessions.get(sid, []))
            except Exception as e:
                logger.error(f"Session {sid} 线程异常: {e}")
                coverage.append(self._build_thread_error_coverage(sid, e))
        return {"all_tasks": all_tasks, "coverage": coverage}

    def _run_phase2(
        self, executor, multi_sessions, prior_batches_map,
        on_session_done, on_batch_done,
    ) -> dict:
        """Phase 2: 多 batch session 全部 batch 入池 (含可复用 batch)"""
        all_tasks: list[Task] = []
        coverage: list[dict] = []
        progress_lock = threading.Lock()
        completed = 0
        total_batches = sum(
            len(self._planner.plan(chs)) for chs in multi_sessions.values()
        )

        reusable: dict[str, list[BatchResult]] = {}
        futures = {}

        for sid, chunks in multi_sessions.items():
            plans = self._planner.plan(chunks)
            prior_idx = self._build_prior_index(
                prior_batches_map.get(sid, [])
            )
            for batch_id, plan in enumerate(plans):
                pb = prior_idx.get((batch_id, plan.content_hash))
                if pb and pb.get("status") == "success":
                    reusable.setdefault(sid, []).append(
                        self._batch_result_from_prior(batch_id, plan, pb)
                    )
                    logger.debug(
                        f"Session {sid} batch {batch_id}: 复用 checkpoint"
                    )
                else:
                    fut = executor.submit(
                        self._extract_one_batch, sid, batch_id, plan
                    )
                    futures[fut] = (sid, batch_id)

        batch_results: dict[str, list[BatchResult]] = {
            sid: list(rs) for sid, rs in reusable.items()
        }
        for fut in as_completed(futures):
            sid, batch_id = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                logger.error(f"Session {sid} batch {batch_id} 线程异常: {e}")
                result = BatchResult(
                    batch_id=batch_id, start_idx=-1, end_idx=-1,
                    content_hash="", status="failed", attempts=0,
                    error=str(e),
                )
            batch_results.setdefault(sid, []).append(result)
            if on_batch_done:
                on_batch_done(sid, result, multi_sessions.get(sid, []))
            with progress_lock:
                completed += 1
                logger.info(f"多 batch 进度: [{completed}/{total_batches}]")

        for sid, brs in batch_results.items():
            brs.sort(key=lambda r: r.batch_id)
            chunks = multi_sessions[sid]
            merged = self._merge_batch_results(sid, chunks, brs)
            all_tasks.extend(merged["tasks"])
            coverage.append(self._build_coverage(sid, merged))
            if on_session_done:
                on_session_done(sid, merged, chunks)
        return {"all_tasks": all_tasks, "coverage": coverage}

    @staticmethod
    def _build_coverage(session_id: str, result: dict) -> dict:
        return {
            "session_id": session_id,
            "total_chunks": result["total_chunks"],
            "summaries_written": result["summaries_written"],
            "total_attempts": result["total_attempts"],
            "status": result["status"],
            "error": result["error"],
        }

    @staticmethod
    def _build_thread_error_coverage(session_id: str, e: Exception) -> dict:
        return {
            "session_id": session_id,
            "total_chunks": 0, "summaries_written": 0,
            "total_attempts": 0,
            "status": "thread_error", "error": str(e),
        }

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

    def _needs_split(self, chunks: list[Chunk]) -> bool:
        return self._planner.needs_split(chunks)

    def _extract_one_batch(
        self,
        session_id: str,
        batch_id: int,
        plan: BatchPlan,
    ) -> BatchResult:
        """单 batch 调 LLM, content_retries 内重试; 返回 BatchResult
        (status: success | incomplete | failed)"""
        batch_chunks = plan.chunks
        pre_existing = {c.chunk_id for c in batch_chunks if c.task_summary}
        last_error = None
        tasks: list[Task] = []

        for attempt in range(1, self.content_retries + 1):
            if attempt > 1:
                for c in batch_chunks:
                    if c.chunk_id not in pre_existing:
                        c.task_summary = None

            try:
                prompt = self._build_prompt(batch_chunks)
                llm_output = self._call_llm_with_retry(prompt)
                task_id_prefix = f"{session_id}_b{batch_id}_"
                tasks, summaries_written, total_chunks = self._parse_and_validate(
                    llm_output, session_id, batch_chunks,
                    task_id_prefix=task_id_prefix,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Session {session_id} batch {batch_id} "
                    f"(内容重试 {attempt}/{self.content_retries}): {e}"
                )
                continue

            chunk_summaries = {
                c.chunk_id: c.task_summary
                for c in batch_chunks if c.task_summary
            }

            issues = []
            if summaries_written < total_chunks:
                issues.append(
                    f"chunk_summaries 缺失 {total_chunks - summaries_written}/{total_chunks}"
                )
            if not tasks:
                issues.append("tasks 为空")

            if not issues:
                logger.info(
                    f"Session {session_id} batch {batch_id} "
                    f"({len(batch_chunks)} chunks): 提取 {len(tasks)} tasks"
                    + (f" (第 {attempt} 次尝试)" if attempt > 1 else "")
                )
                return BatchResult(
                    batch_id=batch_id,
                    start_idx=plan.start_idx,
                    end_idx=plan.end_idx,
                    content_hash=plan.content_hash,
                    status="success",
                    attempts=attempt,
                    tasks=tasks,
                    chunk_summaries=chunk_summaries,
                )

            issue_str = "; ".join(issues)
            last_error = issue_str
            if attempt < self.content_retries:
                logger.warning(
                    f"Session {session_id} batch {batch_id}: {issue_str}, "
                    f"内容重试 {attempt}/{self.content_retries}"
                )
            else:
                logger.error(
                    f"Session {session_id} batch {batch_id}: {issue_str}, "
                    f"已达最大内容重试"
                )
                return BatchResult(
                    batch_id=batch_id,
                    start_idx=plan.start_idx,
                    end_idx=plan.end_idx,
                    content_hash=plan.content_hash,
                    status="incomplete",
                    attempts=attempt,
                    tasks=tasks,
                    chunk_summaries=chunk_summaries,
                    error=issue_str,
                )

        return BatchResult(
            batch_id=batch_id,
            start_idx=plan.start_idx,
            end_idx=plan.end_idx,
            content_hash=plan.content_hash,
            status="failed",
            attempts=self.content_retries,
            tasks=[],
            chunk_summaries={},
            error=last_error,
        )

    def _merge_batch_results(
        self,
        session_id: str,
        chunks: list[Chunk],
        batch_results: list[BatchResult],
    ) -> dict:
        """合并 batch 结果: chunk_summaries 按 chunk_id 合并;
        tasks 全局重编号 T1..Tn, 按 batch 顺序拼接。
        返回与 extract_tasks 兼容的 dict, 额外含 batch_results / split_mode。"""
        all_summaries: dict[str, str] = {}
        ordered_tasks: list[Task] = []
        n_success = 0
        n_failed = 0
        n_incomplete = 0

        for br in sorted(batch_results, key=lambda r: r.batch_id):
            if br.status == "success":
                n_success += 1
            elif br.status == "incomplete":
                n_incomplete += 1
            else:
                n_failed += 1
                continue
            all_summaries.update(br.chunk_summaries)
            ordered_tasks.extend(br.tasks)

        for i, task in enumerate(ordered_tasks):
            task.task_id = f"{session_id}_T{i + 1}"

        if n_failed > 0:
            status = "incomplete"
            err = f"{n_failed} batches failed"
        elif n_incomplete > 0:
            status = "incomplete"
            err = f"{n_incomplete} batches incomplete"
        else:
            status = "success"
            err = None

        for c in chunks:
            if c.chunk_id in all_summaries and not c.task_summary:
                c.task_summary = all_summaries[c.chunk_id]

        summaries_written = sum(1 for c in chunks if c.task_summary)
        total_attempts = max((br.attempts for br in batch_results), default=0)

        return {
            "tasks": ordered_tasks,
            "summaries_written": summaries_written,
            "total_chunks": len(chunks),
            "total_attempts": total_attempts,
            "status": status,
            "error": err,
            "split_mode": "multi",
            "batch_results": batch_results,
        }

    @staticmethod
    def _build_prior_index(
        prior: list[dict],
    ) -> dict[tuple[int, str], dict]:
        idx: dict[tuple[int, str], dict] = {}
        for pb in prior or []:
            key = (pb.get("batch_id"), pb.get("content_hash"))
            if key[0] is not None and key[1]:
                idx[key] = pb
        return idx

    @staticmethod
    def _batch_result_from_prior(
        batch_id: int, plan: BatchPlan, prior: dict
    ) -> BatchResult:
        return BatchResult(
            batch_id=batch_id,
            start_idx=plan.start_idx,
            end_idx=plan.end_idx,
            content_hash=plan.content_hash,
            status=prior.get("status", "success"),
            attempts=prior.get("attempts", 1),
            tasks=[Task.from_dict(t) for t in prior.get("tasks", [])],
            chunk_summaries=dict(prior.get("chunk_summaries", {})),
            error=prior.get("error"),
        )
