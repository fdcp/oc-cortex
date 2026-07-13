"""
工具模块: 配置加载 + 日志 + token 计数
"""
import os
import sys
import yaml
import tiktoken
from pathlib import Path
from loguru import logger
from typing import Optional


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
    return f"{head}\n\n[...TRUNCATED, original {len(tokens)} tokens...]\n\n{tail}"
