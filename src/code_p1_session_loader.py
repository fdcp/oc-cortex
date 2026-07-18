"""
Session 加载器
负责: 1) 读取 JSONL / JSON 数据源  2) 过滤 root + turn_count > 1
"""
import json
from pathlib import Path
from typing import Iterator
from loguru import logger

from code_p1_models import Session, Turn, ToolCall


def _parse_tool_call(raw: dict) -> ToolCall:
    """解析工具调用字段(支持 bash / tool_call / mcp_call)"""
    return ToolCall(
        type=raw.get("type", "tool_call"),
        cmd=raw.get("cmd") or raw.get("command"),
        action=raw.get("action") or raw.get("description"),
        target=raw.get("target") or raw.get("resource"),
        ok=raw.get("ok", raw.get("success", True)),
        output=raw.get("output") or raw.get("result"),
        error=raw.get("error"),
    )


def _parse_turn(raw: dict) -> Turn:
    """解析单条对话"""
    tool_calls = [_parse_tool_call(tc) for tc in raw.get("tool_calls", [])]
    return Turn(
        role=raw.get("role", "user"),
        content=raw.get("content", "") or raw.get("text", ""),
        timestamp=raw.get("timestamp") or raw.get("created_at"),
        tool_calls=tool_calls,
        tool_call_id=raw.get("tool_call_id"),
        output=raw.get("output"),
    )


def _parse_session(raw: dict) -> Session:
    """解析单个 session"""
    return Session(
        id=raw["id"],
        type=raw.get("type", "root"),
        created_at=raw.get("created_at"),
        turns=[_parse_turn(t) for t in raw.get("turns", raw.get("messages", []))],
        metadata=raw.get("metadata", {}),
    )


def load_sessions_from_jsonl(path: str) -> Iterator[Session]:
    """从 JSONL 文件逐行加载 session"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session 文件不存在: {path}")
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                yield _parse_session(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_no} 行 JSON 解析失败: {e}")


def load_sessions_from_json(path: str) -> list[Session]:
    """从 JSON 文件加载(支持 list 或 单个 session)"""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [_parse_session(item) for item in raw]
    return [_parse_session(raw)]


def filter_sessions(
    sessions,
    session_type: str = "root",
    min_user_messages: int = 2,
) -> list[Session]:
    """
    过滤条件:
    1. session.type == session_type (默认 "root")
    2. user 消息数 >= min_user_messages
    """
    filtered = []
    for s in sessions:
        if s.type != session_type:
            continue
        user_count = s.count_user_messages()
        if user_count < min_user_messages:
            logger.debug(f"跳过 session {s.id}: user 消息数 {user_count} < {min_user_messages}")
            continue
        filtered.append(s)
    logger.info(f"过滤后保留 {len(filtered)} 个 session")
    return filtered


# ============================================================
# Mock 数据(开发 / 测试用)
# ============================================================

def get_mock_sessions() -> list[Session]:
    """构造 2 个 mock session 用于本地验证"""
    return [
        Session(
            id="ses_mock_001",
            type="root",
            created_at="2026-07-10T10:00:00Z",
            turns=[
                Turn(role="user", content="帮我修一下登录 bug,登录一直失败"),
                Turn(role="assistant", content="好的,我先排查 token 验证逻辑"),
                Turn(role="assistant", content="", tool_calls=[
                    ToolCall(type="bash", cmd="grep -n 'token' src/auth.py", ok=True, output="12:def verify_token(t)...")
                ]),
                Turn(role="assistant", content="找到问题了,是 token 过期判断错了"),
                Turn(role="assistant", content="", tool_calls=[
                    ToolCall(type="tool_call", action="edit_file", target="src/auth.py", ok=True)
                ]),
                Turn(role="user", content="再帮我加个测试用例"),
                Turn(role="assistant", content="好的,补充 token 过期的单元测试"),
                Turn(role="assistant", content="", tool_calls=[
                    ToolCall(type="bash", cmd="pytest tests/test_auth.py", ok=True, output="3 passed")
                ]),
            ],
        ),
        Session(
            id="ses_mock_002",
            type="root",
            created_at="2026-07-11T14:00:00Z",
            turns=[
                Turn(role="user", content="我需要给项目加一个性能监控"),
                Turn(role="assistant", content="我推荐用 Prometheus + Grafana"),
                Turn(role="user", content="行,帮我搭一下"),
                Turn(role="assistant", content="", tool_calls=[
                    ToolCall(type="bash", cmd="docker run -d -p 9090:9090 prom/prometheus", ok=True)
                ]),
                Turn(role="assistant", content="", tool_calls=[
                    ToolCall(type="mcp_call", target="metrics_endpoint", ok=True, output="CPU 35%, Memory 60%")
                ]),
            ],
        ),
        # 这个 session 会被过滤掉: 只有 1 个 user 消息
        Session(
            id="ses_mock_skip",
            type="root",
            created_at="2026-07-11T15:00:00Z",
            turns=[
                Turn(role="user", content="你好"),
                Turn(role="assistant", content="你好,有什么可以帮你的?"),
            ],
        ),
    ]
