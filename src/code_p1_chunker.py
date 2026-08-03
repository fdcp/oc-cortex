"""
Chunker: 按 user 消息切分轮次
规则: 轮次边界 = user 消息出现位置
     每个 chunk = 1 个 user + 后续所有非 user 直到下一个 user
后处理:
     1a) 无 assistant 回复的 chunk 合并到下一个 chunk
     1b) 连续相同 user_message (无 assistant 间隔) 只保留最后一个
     1c) 切分+后处理完成后,为每个 chunk 填充 SHA-256 content_hash (Phase 1 专属字段)
"""
from loguru import logger

from code_p1_models import Session, Chunk, Turn
from code_p1_utils import chunk_content_hash


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


def _is_empty_chunk(chunk: Chunk) -> bool:
    """判断 chunk 是否无任何 assistant 交互 (无回复、无工具调用)"""
    return (
        not chunk.assistant_messages
        and not chunk._raw_tool_calls
    )


def _postprocess_chunks(raw_chunks: list[Chunk]) -> list[Chunk]:
    """
    后处理:
    1a) 无 assistant 回复的 chunk, 其 user_message 合并到下一个 chunk
    1b) 连续相同 user_message (中间无 assistant 交互) 只保留最后一个
    """
    if len(raw_chunks) <= 1:
        return raw_chunks

    result: list[Chunk] = []
    # 累积的空 chunk 的 user_messages (已去重)
    deferred_msgs: list[str] = []
    n_merged = 0
    n_deduped = 0

    for chunk in raw_chunks:
        if _is_empty_chunk(chunk):
            user_msg = chunk.user_message.strip()
            # 1b) 去重: 如果和已累积的最后一条消息相同, 替换 (等于只保留最新的)
            if deferred_msgs and deferred_msgs[-1] == user_msg:
                n_deduped += 1
                # 已在 deferred_msgs 中, 不需要再添加
            else:
                deferred_msgs.append(user_msg)
            n_merged += 1
            # 不放入 result, 等下一个非空 chunk 吸收
        else:
            # 非空 chunk: 将累积的 user_messages 合并进来
            if deferred_msgs:
                user_msg = chunk.user_message.strip()
                # 最终去重: 累积的最后一条和当前 chunk 的 user_message 相同
                if deferred_msgs[-1] == user_msg:
                    n_deduped += 1
                    # 移除 deferred 中的重复, 保留当前 chunk 的
                    all_msgs = deferred_msgs[:-1] + [user_msg]
                else:
                    all_msgs = deferred_msgs + [user_msg]
                chunk.user_message = "\n\n".join(all_msgs)
                deferred_msgs = []
            result.append(chunk)

    # 处理末尾的空 chunk (没有后续非空 chunk 可合并)
    if deferred_msgs:
        last = raw_chunks[-1]
        last.user_message = "\n\n".join(deferred_msgs)
        result.append(last)

    if n_merged > 0 or n_deduped > 0:
        logger.info(
            f"Session {raw_chunks[0].session_id}: "
            f"后处理合并 {n_merged} 个空 chunk, 去重 {n_deduped} 条重复 user_message, "
            f"{len(raw_chunks)} -> {len(result)} chunks"
        )

    return result


def chunk_session(session: Session) -> list[Chunk]:
    """把 session 切分成 chunk 列表, 并后处理合并空 chunk"""
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

    # 后处理: 合并空 chunk + 去重
    chunks = _postprocess_chunks(chunks)

    # 重新编号 chunk_id 和 turn_index, 保证连续
    for i, chunk in enumerate(chunks, 1):
        chunk.chunk_id = f"{session.id}_c{i}"
        chunk.turn_index = i

    # 填充 content_hash (chunk_id/turn_index 此刻已确定, 可参与哈希)
    for chunk in chunks:
        chunk.content_hash = chunk_content_hash(chunk)

    logger.info(
        f"Session {session.id}: 切分出 {len(chunks)} 个 chunk"
    )
    return chunks
