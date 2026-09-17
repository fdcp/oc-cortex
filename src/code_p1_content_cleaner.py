"""
Content Cleaner: 内容整理(降噪 + 减体积)
规则:
  - user/assistant 文本: 完整保留
  - bash 调用:    { "cmd": "原命令", "ok": true/false, "error": ... }
  - 工具调用:     { "action": "做了什么", "ok": true/false, "error": ... }
  - MCP 调用:     { "target": "获取什么", "ok": true/false, "error": ... }
  - 失败调用:     保留 error 摘要(前 N 字符)
"""
from typing import Optional
from loguru import logger

from code_p1_models import Chunk, CleanedToolCall
from code_p1_utils import count_tokens


def _summarize_error(error: Optional[str], max_len: int) -> Optional[str]:
    """失败时,截断 error 字符串保留关键信息"""
    if not error:
        return None
    error = str(error).strip()
    if len(error) <= max_len:
        return error
    return error[:max_len] + f"...[truncated, total {len(error)} chars]"


def _clean_tool_call(raw, max_error_length: int) -> CleanedToolCall:
    """
    把原始 ToolCall 转成 CleanedToolCall
    按 type 决定 summary 字段:
      - bash:      cmd
      - tool_call: action 或 description
      - mcp_call:  target / resource
    """
    tc_type = getattr(raw, "type", "tool_call")
    if tc_type == "bash":
        summary = raw.cmd or "(empty cmd)"
    elif tc_type == "mcp_call":
        summary = raw.action or raw.target or "(empty target)"
    else:
        summary = raw.action or raw.cmd or raw.target or "(unknown)"

    return CleanedToolCall(
        type=tc_type,
        summary=summary,
        ok=raw.ok,
        error=_summarize_error(getattr(raw, "error", None), max_error_length),
    )


def clean_chunk(chunk: Chunk, max_error_length: int = 200) -> Chunk:
    """
    对单个 chunk 做内容整理
    - 计算 raw_size_tokens(原始)
    - 把 _raw_tool_calls 分类成 tool_calls / mcp_calls 并整理
    - 计算 cleaned_size_tokens(整理后)
    """
    # 1. 原始 token 数(基于原始字段,口径与 cleaned_text() 一致)
    raw_text_parts = [chunk.user_message] + list(chunk.assistant_messages)
    for tc in chunk._raw_tool_calls:
        # 包含原始 tool output(cleaned 中会被摘要化)
        raw_text_parts.append(str(getattr(tc, "output", "") or ""))
        # 包含原始 cmd/action/target/error(cleaned 后保留摘要形式)
        raw_text_parts.append(str(getattr(tc, "cmd", "") or ""))
        raw_text_parts.append(str(getattr(tc, "action", "") or ""))
        raw_text_parts.append(str(getattr(tc, "target", "") or ""))
        raw_text_parts.append(str(getattr(tc, "error", "") or ""))
    chunk.raw_size_tokens = count_tokens("\n".join(raw_text_parts))

    # 2. 整理工具调用
    for raw_tc in chunk._raw_tool_calls:
        cleaned = _clean_tool_call(raw_tc, max_error_length)
        if cleaned.type == "mcp_call":
            chunk.mcp_calls.append(cleaned)
        else:
            chunk.tool_calls.append(cleaned)

    # 3. 整理后 token 数
    chunk.cleaned_size_tokens = count_tokens(chunk.cleaned_text())

    # 4. 清理临时字段(节省内存)
    chunk._raw_tool_calls = []

    return chunk


def clean_chunks(chunks: list[Chunk], max_error_length: int = 200) -> list[Chunk]:
    """批量整理"""
    for c in chunks:
        clean_chunk(c, max_error_length=max_error_length)
    total_raw = sum(c.raw_size_tokens for c in chunks)
    total_cleaned = sum(c.cleaned_size_tokens for c in chunks)
    logger.info(
        f"整理 {len(chunks)} 个 chunk: "
        f"raw_content={total_raw} tokens, "
        f"cleaned_formatted={total_cleaned} tokens "
        f"(cleaned 含 [User]/[Tool] 等结构化标签)"
    )
    return chunks
