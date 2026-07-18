"""
SQLite Loader: 从 OpenCode 的 opencode.db 加载 session 数据
数据模型:
  session (id, parent_id, title, time_created)
    → message (id, session_id, data={role, agent, model, tokens, cost})
      → part (id, message_id, data={type: text/tool/reasoning/file/step-start/step-finish})
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional
from loguru import logger

from code_p1_models import Session, Turn, ToolCall


# ============================================================
# Part 解析
# ============================================================

def _parse_tool_part(part_data: dict) -> Optional[ToolCall]:
    """
    从 type=tool 的 part 解析出 ToolCall
    结构: {
      "type": "tool",
      "tool": "bash" | "write" | "read" | "edit" | ...,
      "callID": "...",
      "state": {
        "status": "completed",
        "input": { "command": "..." } | { "filePath": "...", "content": "..." },
        "output": "...",
        "metadata": { "exit": 0, ... }
      }
    }
    """
    state = part_data.get("state", {})
    inp = state.get("input", {})
    output = state.get("output", "")
    metadata = state.get("metadata", {})
    tool_name = part_data.get("tool", "unknown")

    # 判断成功/失败
    exit_code = metadata.get("exit")
    if exit_code is not None:
        ok = exit_code == 0
    else:
        ok = state.get("status") == "completed"

    # 按工具类型映射
    if tool_name == "bash":
        cmd = inp.get("command", "")
        description = inp.get("description") or part_data.get("title", "")
        return ToolCall(
            type="bash",
            cmd=cmd,
            action=description,
            ok=ok,
            output=output[:2000] if output else None,
            error=output[:500] if not ok and output else None,
        )
    else:
        # write / read / edit / glob / grep / 其他
        file_path = inp.get("filePath") or inp.get("path") or inp.get("file_path")
        action_desc = part_data.get("title") or tool_name
        if file_path:
            action_desc = f"{tool_name}: {file_path}"
        return ToolCall(
            type="tool_call",
            action=action_desc,
            target=file_path,
            ok=ok,
            output=output[:2000] if output else None,
            error=output[:500] if not ok and output else None,
        )


def _extract_text_from_parts(parts_data: list[dict]) -> str:
    """从 parts 中提取所有 text 类型的文本"""
    texts = []
    for p in parts_data:
        if p.get("type") == "text":
            text = p.get("text", "")
            if text:
                texts.append(text)
    return "\n".join(texts)


def _extract_tools_from_parts(parts_data: list[dict]) -> list[ToolCall]:
    """从 parts 中提取所有 tool 类型的调用"""
    tools = []
    for p in parts_data:
        if p.get("type") == "tool":
            tc = _parse_tool_part(p)
            if tc:
                tools.append(tc)
    return tools


# ============================================================
# Session 加载
# ============================================================

def _load_parts_for_messages(conn: sqlite3.Connection, message_ids: list[str]) -> dict[str, list[dict]]:
    """批量加载多条 message 的 parts,返回 {message_id: [part_data, ...]}"""
    if not message_ids:
        return {}

    result: dict[str, list[dict]] = {mid: [] for mid in message_ids}

    # SQLite IN 查询,分批避免参数过多
    batch_size = 500
    for i in range(0, len(message_ids), batch_size):
        batch = message_ids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cursor = conn.execute(
            f"SELECT message_id, data FROM part WHERE message_id IN ({placeholders}) ORDER BY time_created",
            batch,
        )
        for row in cursor:
            msg_id = row[0]
            try:
                part_data = json.loads(row[1])
                result[msg_id].append(part_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Part JSON 解析失败 (message={msg_id}): {e}")

    return result


def _load_messages_with_parts(
    conn: sqlite3.Connection, session_id: str
) -> list[tuple[dict, list[dict]]]:
    """加载一个 session 的所有 message 及其 parts,按时间排序"""
    cursor = conn.execute(
        "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created, id",
        (session_id,),
    )
    messages = []
    msg_ids = []
    for row in cursor:
        try:
            msg_data = json.loads(row[1])
            msg_data["_id"] = row[0]
            messages.append(msg_data)
            msg_ids.append(row[0])
        except json.JSONDecodeError as e:
            logger.warning(f"Message JSON 解析失败 (session={session_id}): {e}")

    # 批量加载 parts
    parts_map = _load_parts_for_messages(conn, msg_ids)

    return [(msg, parts_map.get(msg["_id"], [])) for msg in messages]


def _clean_user_text(text: str) -> str:
    """
    清理 user 消息中的系统注入内容
    OpenCode 会在 user 消息前注入 system-reminder,需要去掉
    """
    if not text:
        return ""

    # 移除 <system-reminder>...</system-reminder> 块
    cleaned = re.sub(
        r'<system-reminder>.*?</system-reminder>',
        '',
        text,
        flags=re.DOTALL,
    ).strip()

    return cleaned


def load_sessions_from_sqlite(
    db_path: str,
    session_type: str = "root",
    min_user_messages: int = 2,
    limit: Optional[int] = None,
) -> list[Session]:
    """
    从 OpenCode SQLite 数据库加载 session

    Args:
        db_path: 数据库文件路径
        session_type: "root" (parent_id IS NULL) 或 "branch" 或 "all"
        min_user_messages: 最少 user 消息数
        limit: 最多加载 session 数(None=全部)
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    logger.info(f"连接 SQLite 数据库: {db_path}")
    conn = sqlite3.connect(str(p))

    try:
        # 查询 session
        if session_type == "root":
            sql = "SELECT id, title, time_created, parent_id, metadata FROM session WHERE parent_id IS NULL ORDER BY time_created DESC"
        elif session_type == "branch":
            sql = "SELECT id, title, time_created, parent_id, metadata FROM session WHERE parent_id IS NOT NULL ORDER BY time_created DESC"
        else:
            sql = "SELECT id, title, time_created, parent_id, metadata FROM session ORDER BY time_created DESC"

        if limit:
            sql += f" LIMIT {limit}"

        cursor = conn.execute(sql)
        session_rows = cursor.fetchall()
        logger.info(f"查询到 {len(session_rows)} 个 session")

        sessions = []
        skipped = 0

        for row in session_rows:
            sid = row[0]
            title = row[1]
            time_created = row[2]
            parent_id = row[3]
            metadata_raw = row[4]

            # 加载 message + parts
            messages_with_parts = _load_messages_with_parts(conn, sid)

            # 转成 Turn
            turns: list[Turn] = []
            user_count = 0

            for msg_data, parts in messages_with_parts:
                role = msg_data.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                # 提取文本
                text = _extract_text_from_parts(parts)
                # 提取工具调用
                tool_calls = _extract_tools_from_parts(parts)

                # 时间戳
                ts = msg_data.get("time", {}).get("created")
                timestamp = str(ts) if ts else None

                if role == "user":
                    user_count += 1
                    # user 消息: system-reminder 内容过滤掉
                    user_text = _clean_user_text(text)
                    if not user_text.strip():
                        continue
                    turns.append(Turn(
                        role="user",
                        content=user_text,
                        timestamp=timestamp,
                    ))
                else:
                    # assistant 消息
                    turns.append(Turn(
                        role="assistant",
                        content=text,
                        timestamp=timestamp,
                        tool_calls=tool_calls,
                    ))

            # 过滤: 至少 min_user_messages 个 user 消息
            if user_count < min_user_messages:
                skipped += 1
                logger.debug(
                    f"跳过 session {sid} ({title}): user 消息 {user_count} < {min_user_messages}"
                )
                continue

            # 解析 metadata
            metadata = {}
            if metadata_raw:
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    pass

            session = Session(
                id=sid,
                type="root" if parent_id is None else "branch",
                created_at=str(time_created),
                turns=turns,
                metadata={**metadata, "title": title},
            )
            sessions.append(session)

        logger.info(
            f"加载完成: {len(sessions)} 个 session (跳过 {skipped} 个)"
        )
        return sessions

    finally:
        conn.close()
