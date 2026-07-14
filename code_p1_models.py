"""
数据模型定义
所有 Phase 1 涉及的 dataclass 集中在此
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ============================================================
# 原始 session 数据结构(从 OpenCode 加载)
# ============================================================

@dataclass
class ToolCall:
    """工具调用原始数据"""
    type: str                    # bash / tool_call / mcp_call
    cmd: Optional[str] = None    # bash 命令
    action: Optional[str] = None # 工具动作
    target: Optional[str] = None # MCP 目标
    ok: bool = True
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Turn:
    """单条对话(原始)"""
    role: str                    # user / assistant / tool_result
    content: str = ""
    timestamp: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # tool_result 用
    output: Optional[str] = None        # tool_result 用


@dataclass
class Session:
    """OpenCode session 原始数据"""
    id: str
    type: str                    # root / branch
    created_at: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def count_user_messages(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")


# ============================================================
# 切分 + 整理后的 Chunk
# ============================================================

@dataclass
class CleanedToolCall:
    """整理后的工具调用(摘要形式)"""
    type: str                    # bash / tool_call / mcp_call
    summary: str                 # 做什么 / 取什么
    ok: bool = True
    error: Optional[str] = None  # 失败时保留的错误摘要


@dataclass
class Chunk:
    """一个对话轮次 = 1 user + 多 assist/工具/MCP"""
    chunk_id: str                # {session_id}_c{index}, e.g. "abc_c1"
    session_id: str
    turn_index: int              # 轮次序号(从 1 开始)
    user_message: str
    assistant_messages: list[str] = field(default_factory=list)
    tool_calls: list[CleanedToolCall] = field(default_factory=list)
    mcp_calls: list[CleanedToolCall] = field(default_factory=list)
    # 内部使用: chunker 阶段暂存原始工具调用,cleaner 阶段处理
    _raw_tool_calls: list = field(default_factory=list, repr=False)
    raw_size_tokens: int = 0
    cleaned_size_tokens: int = 0
    created_at: Optional[str] = None
    task_summary: Optional[str] = None  # Phase 2 CoT 生成的 chunk 总结

    def add_raw_tool_call(self, tc) -> None:
        """chunker 阶段: 添加原始工具调用"""
        self._raw_tool_calls.append(tc)

    def cleaned_text(self) -> str:
        """整理后的可读文本(供后续 LLM 使用)"""
        parts = [f"[User]\n{self.user_message}\n"]
        for msg in self.assistant_messages:
            parts.append(f"\n[Assistant]\n{msg}\n")
        for tc in self.tool_calls:
            status = "OK" if tc.ok else f"FAIL: {tc.error or 'unknown'}"
            parts.append(f"\n[Tool:{tc.type}] {tc.summary} [{status}]\n")
        for mc in self.mcp_calls:
            status = "OK" if mc.ok else f"FAIL: {mc.error or 'unknown'}"
            parts.append(f"\n[MCP:{mc.type}] {mc.summary} [{status}]\n")
        return "".join(parts)

    def to_dict(self) -> dict:
        data = asdict(self)
        # 内部临时字段,不对外暴露
        data.pop("_raw_tool_calls", None)
        return data
