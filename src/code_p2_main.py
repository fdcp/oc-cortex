"""
Phase 2 主入口
流程: 加载 Phase 1 chunks -> 按 session 分组 -> LLM 提炼 task -> 输出

用法:
  export DASHSCOPE_API_KEY='your-api-key'
  python code_p2_main.py
  python code_p2_main.py --chunks ./output/chunks.jsonl
  python code_p2_main.py --config code_p2_config.yaml --output ./output/tasks.jsonl
  python code_p2_main.py --limit 3    # 只处理前 3 个 session (快速验证)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from loguru import logger

from code_p1_utils import Config, setup_logger, count_tokens, session_fingerprint
from code_p1_models import Chunk, CleanedToolCall
from code_p2_task_extractor import TaskExtractor
from code_p2_models import Task


def load_chunks_from_jsonl(path: str) -> list[Chunk]:
    """从 JSONL 文件加载 Chunk 对象"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = Chunk(
                    chunk_id=data["chunk_id"],
                    session_id=data["session_id"],
                    turn_index=data["turn_index"],
                    user_message=data["user_message"],
                    assistant_messages=data.get("assistant_messages", []),
                    tool_calls=[
                        CleanedToolCall(**tc)
                        for tc in data.get("tool_calls", [])
                    ],
                    mcp_calls=[
                        CleanedToolCall(**mc)
                        for mc in data.get("mcp_calls", [])
                    ],
                    raw_size_tokens=data.get("raw_size_tokens", 0),
                    cleaned_size_tokens=data.get("cleaned_size_tokens", 0),
                    created_at=data.get("created_at"),
                    task_summary=data.get("task_summary"),
                    content_hash=data.get("content_hash"),
                )
                chunks.append(chunk)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Chunk 解析失败 (行 {line_num}): {e}")
    return chunks


def group_chunks_by_session(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """按 session_id 分组"""
    groups: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        groups[c.session_id].append(c)
    # 按 turn_index 排序
    for sid in groups:
        groups[sid].sort(key=lambda x: x.turn_index)
    return dict(groups)


def save_tasks(tasks: list[Task], output_path: str):
    """保存 tasks 到 JSONL (原子 temp+rename)"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    logger.info(f"已写入 {len(tasks)} 个 task 到 {output_path}")


def save_chunk_summaries(chunks: list[Chunk], output_path: str):
    """将 chunk 总结写入独立的 JSONL 文件 (仅含 chunk_id + summary)"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", encoding="utf-8") as f:
        for c in chunks:
            if c.task_summary:
                f.write(json.dumps(
                    {"chunk_id": c.chunk_id, "summary": c.task_summary},
                    ensure_ascii=False,
                ) + "\n")
                count += 1
    logger.info(f"已写入 {count} 条 chunk 总结到 {output_path}")


def save_summary_map(summary_map: dict[str, str], output_path: str) -> None:
    """字典 {chunk_id: summary} 写为 JSONL (整体原子 temp+rename)。"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for cid, smry in sorted(summary_map.items()):
            f.write(json.dumps(
                {"chunk_id": cid, "summary": smry},
                ensure_ascii=False,
            ) + "\n")
    os.replace(tmp, p)
    logger.info(f"已写入 {len(summary_map)} 条 chunk 总结到 {output_path}")


# --------------------------------------------------------
# 增量运行 (resume) 辅助
# --------------------------------------------------------

DEFAULT_CHECKPOINT = "./output/.p2_checkpoint.json"


def load_checkpoint(path: str) -> dict:
    """加载已成功完成的 session 记录。

    返回: {session_id: {"content_hash": ..., "chunk_ids": [...], "chunk_count": int, "completed_at": str}}
    若 checkpoint 存在但 content_hash 与当前 P1 输出不一致,调用方负责把该
    session 挪到 pending 集合重新处理 (见 main 中的比对逻辑)。
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"checkpoint 格式异常 (非 dict),忽略: {path}")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"checkpoint 读取失败,忽略: {path} ({e})")
        return {}


def save_checkpoint(path: str, done: dict) -> None:
    """原子写入 checkpoint (temp + rename,避免半写损坏)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    logger.info(f"checkpoint 已更新: {path} (已完成 {len(done)} session)")


def load_existing_tasks_by_session(path: str) -> dict[str, list[Task]]:
    """读取磁盘上已有的 tasks.jsonl,按 session_id 分组。

    用于增量场景: 已完成 session 的旧 task 直接复用, 不再调 LLM。
    反序列化为 Task 对象, 以便 save_tasks / merged_tasks 的属性访问一致。
    """
    by_session: dict[str, list[Task]] = defaultdict(list)
    if not os.path.exists(path):
        return by_session
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = t.get("session_id")
            if not sid:
                continue
            try:
                by_session[sid].append(Task.from_dict(t))
            except (KeyError, TypeError) as e:
                logger.warning(
                    f"task 反序列化失败 (行 {line_num}, session={sid}): {e}"
                )
    return by_session


def load_existing_summaries(path: str) -> dict[str, str]:
    """读取磁盘上已有的 chunks_summary_p2.jsonl -> {chunk_id: summary}。"""
    m: dict[str, str] = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = r.get("chunk_id")
            if cid and r.get("summary"):
                m[cid] = r["summary"]
    return m


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Task 清单生成")
    parser.add_argument(
        "--config", default="config/code_p2_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--chunks", default=None,
        help="Phase 1 chunks JSONL 文件路径 (覆盖配置文件)"
    )
    parser.add_argument(
        "--output", default="./output/tasks.jsonl",
        help="task 输出路径"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 个 session (快速验证)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="并发线程数 (覆盖配置文件, 1=串行, 默认从 config 读取)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="忽略 checkpoint, 全量重新处理所有 session (成功后会覆盖 checkpoint)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint 文件路径 (默认 output/.p2_checkpoint.json, 可在 config output.checkpoint 配置)"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = Config.load(args.config)
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("Phase 2: Task 清单生成")
    logger.info("=" * 60)

    # 2. 加载 Phase 1 chunks
    chunks_file = args.chunks or config.get(
        "phase1.chunks_file", "./output/chunks.jsonl"
    )
    logger.info(f"加载 chunks: {chunks_file}")

    chunks = load_chunks_from_jsonl(chunks_file)
    logger.info(f"加载 {len(chunks)} 个 chunk")

    if not chunks:
        logger.error("没有加载到任何 chunk,请检查输入文件")
        sys.exit(1)

    # 3. 按 session 分组
    sessions_chunks = group_chunks_by_session(chunks)
    logger.info(f"共 {len(sessions_chunks)} 个 session")

    summary_file = config.get("output.chunk_summaries", "./output/chunks_summary_p2.jsonl")
    checkpoint_path = (
        args.checkpoint
        or config.get("output.checkpoint", DEFAULT_CHECKPOINT)
    )

    # 4. 增量 (resume): 加载 checkpoint, 拆分 done / pending
    if args.force:
        done: dict = {}
        pending = dict(sessions_chunks)
        logger.info("--force: 忽略 checkpoint, 全量重新处理")
    else:
        done = load_checkpoint(checkpoint_path)
        # 只保留当前 P1 输出中仍存在的 session (P1 删掉的 session 从 done 剔除)
        done = {sid: rec for sid, rec in done.items() if sid in sessions_chunks}
        existing_summaries = load_existing_summaries(summary_file)
        pending: dict[str, list[Chunk]] = {}
        stale: list[str] = []
        for sid, scs in sessions_chunks.items():
            rec = done.get(sid)
            cur_hash = session_fingerprint(scs)
            if rec and rec.get("content_hash") == cur_hash:
                for c in scs:
                    if c.task_summary is None and c.chunk_id in existing_summaries:
                        c.task_summary = existing_summaries[c.chunk_id]
            else:
                pending[sid] = scs
                if rec:
                    stale.append(sid)
                    done.pop(sid, None)
        logger.info(
            f"增量: 已完成 {len(done)} / 待处理 {len(pending)}"
            + (f" (其中 chunk 变更重处理 {len(stale)})" if stale else "")
        )

    # 4b. 限制本次处理数量 (作用于 pending)
    if args.limit and len(pending) > args.limit:
        sids = list(pending.keys())[:args.limit]
        pending = {sid: pending[sid] for sid in sids}
        logger.info(f"--limit: 本次仅处理前 {args.limit} 个待处理 session")

    # 5. 执行 task 提取 (仅 pending; 全部已完成则跳过 LLM)
    # 每 session 跑完立即同步落盘: tasks.jsonl + chunks_summary_p2.jsonl + checkpoint
    # (原子 temp+rename), 中途断电下次自动从 checkpoint 续跑, 已成功的 session 不重调 LLM
    merged_tasks: list[Task] = []
    summary_map: dict[str, str] = {}
    session_coverage: list[dict] = []

    if not args.force:
        existing_by_session = load_existing_tasks_by_session(args.output)
        existing_summaries_full = load_existing_summaries(summary_file)
        for sid in done:
            merged_tasks.extend(existing_by_session.get(sid, []))
            for c in sessions_chunks[sid]:
                if c.chunk_id in existing_summaries_full:
                    summary_map[c.chunk_id] = existing_summaries_full[c.chunk_id]

    def _on_session_done(sid, result, scs) -> None:
        if result["status"] != "success":
            return
        merged_tasks.extend(result["tasks"])
        for c in scs:
            if c.task_summary:
                summary_map[c.chunk_id] = c.task_summary
        save_tasks(merged_tasks, args.output)
        save_summary_map(summary_map, summary_file)
        done[sid] = {
            "chunk_ids": sorted(c.chunk_id for c in scs),
            "chunk_count": len(scs),
            "content_hash": session_fingerprint(scs),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_checkpoint(checkpoint_path, done)
        logger.info(
            f"  → session {sid} 已落盘 "
            f"({len(result['tasks'])} task, 累计 {len(merged_tasks)}, "
            f"checkpoint {len(done)})"
        )

    if pending:
        concurrency = args.concurrency or config.get("llm.concurrency", 1)
        extractor = TaskExtractor(
            model=config.get("llm.model", "qwen-plus"),
            api_key_env=config.get("llm.api_key_env", "DASHSCOPE_API_KEY"),
            base_url=config.get(
                "llm.base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            max_retries=config.get("llm.max_retries", 3),
            content_retries=config.get("llm.content_retries", 3),
            timeout=config.get("llm.timeout", 120),
            max_tokens_per_chunk=config.get("llm.max_tokens_per_chunk", 2000),
            max_total_prompt_tokens=config.get(
                "llm.max_total_prompt_tokens", 30000
            ),
            concurrency=concurrency,
            temperature=config.get("llm.temperature", 0.3),
        )
        t0 = time.time()
        _new_tasks_unused, session_coverage = extractor.extract_tasks_batch(
            pending, on_session_done=_on_session_done
        )
        elapsed = time.time() - t0
        logger.info(f"Task 提取耗时: {elapsed:.1f}s")

        # 5b. 本次处理的 session 级报告 (incomplete / failed / retried)
        incomplete = [s for s in session_coverage if s["status"] == "incomplete"]
        failed_sessions = [
            s for s in session_coverage if s["status"] in ("failed", "thread_error")
        ]
        retried_sessions = [
            s for s in session_coverage if s.get("total_attempts", 0) > 1
        ]
        if incomplete or failed_sessions or retried_sessions:
            logger.warning("=" * 60)
            logger.warning("Session 处理详情:")
            for s in retried_sessions:
                if s["status"] == "success":
                    logger.info(
                        f"  {s['session_id']}: 重试后成功 "
                        f"(尝试 {s['total_attempts']} 次)"
                    )
            for s in incomplete:
                missing = s["total_chunks"] - s["summaries_written"]
                logger.warning(
                    f"  {s['session_id']}: INCOMPLETE "
                    f"- chunk_summaries {s['summaries_written']}/{s['total_chunks']} "
                    f"(缺失 {missing}), "
                    f"尝试 {s['total_attempts']} 次, "
                    f"原因: {s.get('error', 'unknown')}"
                )
            for s in failed_sessions:
                logger.warning(
                    f"  {s['session_id']}: FAILED "
                    f"- 尝试 {s.get('total_attempts', 0)} 次, "
                    f"原因: {s.get('error', 'unknown')}"
                )
            logger.warning("=" * 60)

        # 5c. 兜底再写一次 (callback 已经按 session 写过; 此处保证最终一致)
        save_tasks(merged_tasks, args.output)
        save_summary_map(summary_map, summary_file)
        save_checkpoint(checkpoint_path, done)
    else:
        logger.info("全部 session 已在 checkpoint 中完成, 跳过 LLM 调用")
        save_tasks(merged_tasks, args.output)
        save_summary_map(summary_map, summary_file)

    # 8. 打印摘要
    new_task_count = sum(
        1 for t in merged_tasks if t.session_id in pending
    ) if pending else 0
    chunks_with_summary = len(summary_map)
    logger.info("=" * 60)
    logger.info("Phase 2 完成")
    logger.info(f"  输入 session: {len(sessions_chunks)}")
    logger.info(f"  本次处理: {len(pending)}  (新增 task {new_task_count})")
    logger.info(f"  累计 task: {len(merged_tasks)}")
    logger.info(f"  chunk 含总结: {chunks_with_summary}/{len(chunks)}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"  chunk 总结: {summary_file}")
    logger.info(f"  checkpoint: {checkpoint_path} ({len(done)} 已完成)")
    logger.info("=" * 60)

    # 9. 打印示例 (本次新增)
    pending_success_sids = {cov["session_id"] for cov in session_coverage if cov["status"] == "success"}
    recent_tasks = [t for t in merged_tasks if t.session_id in pending_success_sids]
    if recent_tasks:
        for t in recent_tasks[:3]:
            logger.info("-" * 40)
            logger.info(f"Task ID:    {t.task_id}")
            logger.info(f"Session:    {t.session_id}")
            logger.info(f"Label:      {t.task_label}")
            logger.info(f"Summary:    {t.task_summary}")
            logger.info(f"Chunks:     {t.chunk_ids}")

    # 10. 统计 (累计)
    task_counts = defaultdict(int)
    for t in merged_tasks:
        task_counts[t.session_id] += 1
    if task_counts:
        avg = sum(task_counts.values()) / len(task_counts)
        logger.info(f"平均每 session task 数: {avg:.1f}")


if __name__ == "__main__":
    main()
