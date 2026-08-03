"""
工具模块: 配置加载 + 日志 + token 计数 + chunk 指纹
"""
import hashlib
import json
import os
import sys
from dataclasses import asdict
import yaml
import tiktoken
from pathlib import Path
from loguru import logger
from typing import Optional, TYPE_CHECKING

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
    head_len = int(len(tokens) * head_ratio)
    tail_len = max_tokens - head_len - 20  # 留 token 给省略号标记
    head = enc.decode(tokens[:head_len])
    tail = enc.decode(tokens[-tail_len:]) if tail_len > 0 else ""
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
