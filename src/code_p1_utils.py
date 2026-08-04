"""
工具模块: 配置加载 + 日志 + token 计数 + chunk 指纹
"""
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
import yaml
import tiktoken
from pathlib import Path
from loguru import logger
from typing import Optional, TYPE_CHECKING

from code_p1_models import render_chunk_text

if TYPE_CHECKING:
    from code_p1_models import Chunk, Turn


# ============================================================
# 配置加载
# ============================================================

class Config:
    """全局配置(单例)"""
    _instance: Optional["Config"] = None

    def __init__(self, config_path: str = "code_p1_config.yaml"):
        if not Path(config_path).exists():
            # 回退到脚本同目录
            script_dir = Path(__file__).parent
            config_path = script_dir / config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)

    @classmethod
    def load(cls, config_path: str = "code_p1_config.yaml") -> "Config":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def get(self, key: str, default=None):
        """支持点号路径,如 'opencode.session_filter.min_user_messages'"""
        keys = key.split(".")
        val = self.raw
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val


# ============================================================
# 日志
# ============================================================

def setup_logger(log_file: Optional[str] = None, level: str = "INFO"):
    """配置 loguru logger"""
    logger.remove()  # 移除默认 handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=level,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8"
        )
    return logger


# ============================================================
# Token 计数与截断
# ============================================================

# Qwen3 系列用 GPT-4 兼容的 o200k_base / cl100k_base 编码近似
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 极端情况(离线)回退到字符数 / 4
            _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """粗略统计 token 数"""
    if not text:
        return 0
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text, disallowed_special=()))
    # 回退: 中英文混合 1 字 ≈ 1.5 token
    return int(len(text) * 0.75)


def safe_truncate(text: str, max_tokens: int, head_ratio: float = 0.7) -> str:
    """
    Token 超限时截断,保留首尾,中间用省略号
    head_ratio: 头部保留比例,剩余给尾部
    """
    if count_tokens(text) <= max_tokens:
        return text

    enc = _get_tokenizer()
    if not enc:
        # 字符级截断
        head_chars = int(len(text) * head_ratio)
        tail_chars = max_tokens * 2 - head_chars  # 粗略
        return text[:head_chars] + "\n\n[...TRUNCATED...]\n\n" + text[-tail_chars:]

    tokens = enc.encode(text, disallowed_special=())
    head_len = int(max_tokens * head_ratio)
    tail_len = max_tokens - head_len - 20  # 留 token 给省略号标记
    head = enc.decode(tokens[:head_len])
    tail = enc.decode(tokens[-tail_len:]) if tail_len > 0 else ""
    return f"{head}\n\n[...TRUNCATED, original {len(tokens)} tokens...]\n\n{tail}"


# 分级截断时, 各段 token 数相加与整段实测可能有 ±few 的 tokenizer 边界误差,
# 末级拼装时预留少量余量避免超预算
_CONCAT_SLACK = 4


# 命令注入折叠: <auto-slash-command> 包裹型 (opencode 把 slash 命令模板展开注入)
_AUTO_CMD_RE = re.compile(r"^\s*<auto-slash-command>")
# 包裹文本首个标题中的命令, 如 "# /init-deep Command"
_AUTO_CMD_HEAD_RE = re.compile(r"#\s*(/[A-Za-z][\w-]*)")
# 裸 /命令: 单段 (字母开头, 仅字母/数字/下划线/连字符), 且其后必须是空白或结尾 —
# 多段绝对路径 (如 "/Users/a/b.yaml 中的...") 首段后跟 "/", 不会命中
_BARE_SLASH_CMD_RE = re.compile(r"^(/[A-Za-z][\w-]*)(?=\s|$)")
# 裸 @mention: 取首个非空白 token (如 @src/foo.py)
_BARE_AT_CMD_RE = re.compile(r"^(@\S+)")
# 兜底: 包裹文本中提取第一个命令形态 token
_ANY_SLASH_CMD_RE = re.compile(r"(/[A-Za-z][\w-]*)(?=\s|$)")


def collapse_command_injection(user_message: str) -> str:
    """
    命令注入型 user_message 折叠为命令 token 本身。

    opencode 会把 slash 命令的完整模板文本注入 user_message (可达数千 token),
    真实用户意图只有命令本身。支持两种形态:
      1. <auto-slash-command> 包裹: 提取首个 "# /xxx Command" 标题中的命令,
         兜底提取文本中第一个命令形态 token
      2. 裸命令开头: 单段 /command ("/init-deep --flag" → "/init-deep")
         或 @mention 开头 (保留首个 @token)

    多段绝对路径 ("/Users/.../x.yaml 中的 embedding...") 与普通文本
    不视为命令, 原样返回。
    """
    text = user_message.lstrip()
    if not text:
        return user_message

    if _AUTO_CMD_RE.match(text):
        m = _AUTO_CMD_HEAD_RE.search(text)
        if m:
            return m.group(1)
        m = _ANY_SLASH_CMD_RE.search(text)
        if m:
            return m.group(1)
        return user_message

    m = _BARE_SLASH_CMD_RE.match(text)
    if m:
        return m.group(1)
    m = _BARE_AT_CMD_RE.match(text)
    if m:
        return m.group(1)
    return user_message


def truncate_chunk_text(chunk: "Chunk", max_tokens: int) -> str:
    """
    分级截断单个 chunk 至 max_tokens 以内 (供 P2 prompt 拼装使用)

    前置: 命令注入折叠 — user_message 为 slash 命令/@mention 注入时
    (含 <auto-slash-command> 包裹型), 只保留命令 token (见
    collapse_command_injection); 折叠后不超预算即整体保留。

    分级策略 (每级判断是否达标, 达标即返回):
      0. 全文 (折叠后的 user + 其余部分) 不超预算 → 原样返回
      1. 丢弃 bash 类 tool_calls
      2. 再丢弃 tool_call 类 tool_calls
      3. 丢弃全部 tool/mcp calls (附一行省略数量标记)
      4. 只剩 user_message + assistant_messages 仍超预算:
         user_message 保持全量, assistant_messages 块按剩余预算的
         头 70% + 尾 30% 截断 (复用 safe_truncate)

    边界情况:
      - user_message 单独已超预算 → 直接截断 user_message, assistant 无法保留
      - assistant_messages 为空或剩余预算过小 → 仅返回 user 部分 + 省略标记
    """
    user_msg = collapse_command_injection(chunk.user_message)
    if user_msg != chunk.user_message:
        logger.info(
            f"truncate_chunk_text {chunk.chunk_id}: 命令注入折叠 "
            f"{count_tokens(chunk.user_message)} -> {count_tokens(user_msg)} "
            f"tokens: {user_msg!r}"
        )

    full = render_chunk_text(
        user_msg, chunk.assistant_messages, chunk.tool_calls, chunk.mcp_calls
    )
    if count_tokens(full) <= max_tokens:
        return full

    total_calls = len(chunk.tool_calls) + len(chunk.mcp_calls)

    def _note(dropped: int) -> str:
        if dropped <= 0:
            return ""
        return f"[... 已省略 {dropped} 条工具/MCP 调用记录 ...]"

    def _render(tcs: list, mcs: list) -> str:
        dropped = total_calls - len(tcs) - len(mcs)
        return render_chunk_text(
            user_msg, chunk.assistant_messages, tcs, mcs,
            note=_note(dropped),
        )

    # 阶段 1-3: 渐进丢弃工具调用 (先 bash, 再 tool_call, 最后 mcp_calls)
    staged = [
        ([tc for tc in chunk.tool_calls if tc.type != "bash"],
         list(chunk.mcp_calls)),
        ([tc for tc in chunk.tool_calls
          if tc.type not in ("bash", "tool_call")],
         list(chunk.mcp_calls)),
        ([], []),
    ]
    for level, (tcs, mcs) in enumerate(staged, 1):
        text = _render(tcs, mcs)
        if count_tokens(text) <= max_tokens:
            logger.debug(
                f"truncate_chunk_text {chunk.chunk_id}: "
                f"阶段 {level} 达标 ({count_tokens(text)}/{max_tokens} tokens)"
            )
            return text

    # 阶段 4: 只剩 user + assistant 仍超: user 全量, assistant 头70%+尾30%
    dropped_note = _note(total_calls)
    note_tokens = count_tokens(dropped_note) + (2 if dropped_note else 0)
    user_block = f"[User]\n{user_msg}\n"
    user_tokens = count_tokens(user_block)

    if user_tokens + note_tokens + _CONCAT_SLACK >= max_tokens:
        logger.warning(
            f"truncate_chunk_text {chunk.chunk_id}: user_message 单独已超预算 "
            f"({user_tokens} >= {max_tokens}), assistant 内容无法保留"
        )
        return safe_truncate(user_block, max_tokens)

    assistant_budget = (
        max_tokens - user_tokens - note_tokens - _CONCAT_SLACK
    )
    if not chunk.assistant_messages or assistant_budget <= 20:
        text = user_block
        if dropped_note:
            text += f"\n{dropped_note}\n"
        return text

    joined = "".join(
        f"\n[Assistant]\n{msg}\n" for msg in chunk.assistant_messages
    )
    truncated = safe_truncate(joined, assistant_budget)
    text = user_block + truncated
    if dropped_note:
        text += f"\n{dropped_note}\n"
    logger.debug(
        f"truncate_chunk_text {chunk.chunk_id}: 阶段 4 assistant 截断 "
        f"({count_tokens(text)}/{max_tokens} tokens)"
    )
    return text


# ============================================================
# 指纹 / 哈希
# ============================================================

CHUNKER_VERSION = 1


def _safe_asdict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    return asdict(obj)


def chunk_content_hash(c: "Chunk") -> str:
    """SHA-256 over Phase-1-defined fields (excludes task_summary,
    content_hash itself: 前者 P2 产物,后者由本函数生成 -> 自反馈)。"""
    payload = {
        "chunk_id": c.chunk_id,
        "session_id": c.session_id,
        "turn_index": c.turn_index,
        "user_message": c.user_message,
        "assistant_messages": c.assistant_messages,
        "tool_calls": [_safe_asdict(tc) for tc in c.tool_calls],
        "mcp_calls": [_safe_asdict(mc) for mc in c.mcp_calls],
        "raw_size_tokens": c.raw_size_tokens,
        "cleaned_size_tokens": c.cleaned_size_tokens,
        "created_at": c.created_at,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def session_fingerprint(chunks: "list[Chunk]") -> str:
    """session 级 SHA-256: 链式哈希 chunk_id 排序后的 per-chunk hashes,
    保证同 session 内 chunk 输入顺序变化不影响指纹。

    优先使用 chunk.content_hash (P1 chunker 已填), 缺失时回退重算 -
    兼容旧 chunks.jsonl 无 content_hash 字段的样本。
    """
    pairs = sorted(
        (c.chunk_id, c.content_hash or chunk_content_hash(c))
        for c in chunks
    )
    acc = hashlib.sha256()
    for _cid, h in pairs:
        acc.update(h.encode("utf-8"))
        acc.update(b"|")
    return acc.hexdigest()


def raw_turns_hash(turns: "list[Turn]") -> str:
    """对 session.turns 整体算 SHA-256, 用作 P1 增量短路信号:
    hash 不变 → 跳过 chunker, 复用旧 chunks.jsonl 的对应行。"""
    payload = []
    for t in turns:
        payload.append({
            "role": t.role,
            "content": t.content,
            "timestamp": t.timestamp,
            "tool_calls": [_safe_asdict(tc) for tc in t.tool_calls],
            "tool_call_id": t.tool_call_id,
            "output": t.output,
        })
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
