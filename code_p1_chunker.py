"""
Chunker: 按 user 消息切分轮次
规则: 轮次边界 = user 消息出现位置
     每个 chunk = 1 个 user + 后续所有非 user 直到下一个 user
"""
from loguru import logger

from code_p1_models import Session, Chunk, Turn


def _new_chunk(session_id: str, index: int, first_turn: Turn) -> Chunk:
    """从第一个 user 消息初始化一个 chunk"""
    return Chunk(
        chunk_id=f"{session_id}_c{index}",
        session_id=session_id,
        turn_index=index,
        user_message=first_turn.content,
        created_at=first_turn.timestamp,
    )


def _accumulate(chunk: Chunk, turn: Turn) -> None:
    """把 turn 累加到 chunk"""
    if turn.role == "assistant":
        if turn.content:
            chunk.assistant_messages.append(turn.content)
        # 工具调用先暂存到 chunk._raw_tool_calls,cleaner 阶段再处理
        for tc in turn.tool_calls:
            chunk.add_raw_tool_call(tc)
    elif turn.role == "tool_result":
        # 工具结果作为 assistant 消息的补充
        if turn.output:
            chunk.assistant_messages.append(f"[Tool Result] {turn.output}")


def chunk_session(session: Session) -> list[Chunk]:
    """把 session 切分成 chunk 列表"""
    chunks: list[Chunk] = []
    current: Chunk | None = None
    chunk_index = 0

    for turn in session.turns:
        if turn.role == "user":
            # 遇到新 user,闭合上一个 chunk
            if current is not None:
                chunks.append(current)
            chunk_index += 1
            current = _new_chunk(session.id, chunk_index, turn)
        else:
            if current is None:
                # 异常情况: session 以非 user 开头,跳过
                logger.warning(
                    f"Session {session.id} 以非 user 消息开头,跳过首条 {turn.role}"
                )
                continue
            _accumulate(current, turn)

    if current is not None:
        chunks.append(current)

    logger.info(
        f"Session {session.id}: 切分出 {len(chunks)} 个 chunk"
    )
    return chunks
