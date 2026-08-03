"""
Phase 1 主入口
流程: 加载 session → 过滤 → 切分 → 整理 → 输出

用法:
  python code_p1_main.py
  python code_p1_main.py --source ./data/my_sessions.jsonl
  python code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db
  python code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3
  python code_p1_main.py --mock     # 用 mock 数据
  python code_p1_main.py --incremental       # 跳过 turns 未变的 session (复用旧 chunks 行)
  python code_p1_main.py --force             # 忽略 checkpoint 全量重跑
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from loguru import logger

from code_p1_utils import (
    Config, setup_logger,
    CHUNKER_VERSION, raw_turns_hash, session_fingerprint,
)
from code_p1_models import Session, Chunk, CleanedToolCall
from code_p1_session_loader import (
    load_sessions_from_jsonl,
    load_sessions_from_json,
    filter_sessions,
    get_mock_sessions,
)
from code_p1_sqlite_loader import load_sessions_from_sqlite
from code_p1_chunker import chunk_session
from code_p1_content_cleaner import clean_chunks


DEFAULT_CHECKPOINT = "./output/.p1_checkpoint.json"


def process_session(session: Session) -> list[Chunk]:
    """处理单个 session: 切分 + 整理"""
    chunks = chunk_session(session)
    chunks = clean_chunks(chunks)
    return chunks


def save_chunks(chunks: list[Chunk], output_path: str):
    """保存 chunks 到 JSONL"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"已写入 {len(chunks)} 个 chunk 到 {output_path}")


def save_chunks_rows(rows: list[dict], output_path: str):
    """直接写 dict 行到 JSONL (增量模式: 未变 session 复用旧 dict 行)。"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    logger.info(f"已写入 {len(rows)} 个 chunk 到 {output_path}")


def load_existing_chunk_rows(path: str) -> dict[str, list[dict]]:
    """读旧 chunks.jsonl -> {session_id: [row_dict, ...]}。增量复用。"""
    by_session: dict[str, list[dict]] = defaultdict(list)
    if not os.path.exists(path):
        return by_session
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("session_id")
            if sid:
                by_session[sid].append(row)
    return by_session


def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"checkpoint 格式异常,忽略: {path}")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"checkpoint 读取失败,忽略: {path} ({e})")
        return {}


def save_checkpoint(path: str, done: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    logger.info(f"checkpoint 已更新: {path} ({len(done)} session)")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: 数据预处理")
    parser.add_argument(
        "--config", default="config/code_p1_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--source", default=None,
        help="session 数据源(JSONL 或 JSON 文件),覆盖配置文件"
    )
    parser.add_argument(
        "--sqlite", default=None,
        help="OpenCode SQLite 数据库路径(如 ~/.local/share/opencode/opencode.db)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多加载 N 个 session(用于快速验证)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="使用 mock 数据(开发验证)"
    )
    parser.add_argument(
        "--output", default="./output/chunks.jsonl",
        help="chunk 输出路径"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="跳过 turns 未变更的 session (复用旧 chunks.jsonl 的对应行), 需 checkpoint"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="忽略 checkpoint, 全量重新处理 (--incremental 时有效)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint 路径 (默认 output/.p1_checkpoint.json)"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = Config.load(args.config)
    setup_logger(
        log_file=config.get("logging.file"),
        level=config.get("logging.level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("Phase 1: 数据预处理启动")
    logger.info("=" * 60)

    # 2. 加载 session
    # 优先级: --sqlite > --source > --mock > config.mock_data > config.session_source
    min_user = config.get("opencode.session_filter.min_user_messages", 2)

    if args.sqlite:
        db_path = Path(args.sqlite).expanduser()
        logger.info(f"从 SQLite 数据库加载: {db_path}")
        raw_sessions = load_sessions_from_sqlite(
            str(db_path),
            session_type=config.get("opencode.session_filter.type", "root"),
            min_user_messages=min_user,
            limit=args.limit,
        )
    elif args.mock:
        logger.info("使用 mock 数据")
        raw_sessions = get_mock_sessions()
    elif args.source:
        logger.info(f"从 {args.source} 加载 session")
        if args.source.endswith(".jsonl"):
            raw_sessions = list(load_sessions_from_jsonl(args.source))
        elif args.source.endswith(".json"):
            raw_sessions = load_sessions_from_json(args.source)
        else:
            logger.error(f"不支持的格式: {args.source}")
            sys.exit(1)
    elif config.get("project.mock_data"):
        logger.info("使用 mock 数据(config 配置)")
        raw_sessions = get_mock_sessions()
    else:
        source = config.get("opencode.session_source")
        logger.info(f"从 {source} 加载 session")
        if source.endswith(".jsonl"):
            raw_sessions = list(load_sessions_from_jsonl(source))
        elif source.endswith(".json"):
            raw_sessions = load_sessions_from_json(source)
        else:
            logger.error(f"不支持的格式: {source}")
            sys.exit(1)

    logger.info(f"原始加载 {len(raw_sessions)} 个 session")

    # 3. 过滤(SQLite 加载时已在 loader 内过滤,这里再过滤一次兼容 JSONL/mock 来源)
    if not args.sqlite:
        filtered = filter_sessions(
            raw_sessions,
            session_type=config.get("opencode.session_filter.type", "root"),
            min_user_messages=min_user,
        )
    else:
        filtered = raw_sessions

    if not filtered:
        logger.warning("过滤后没有 session,退出")
        return

    # 4. 切分 + 整理 (+ 增量短路)
    incremental = args.incremental and not args.force
    checkpoint_path = (
        args.checkpoint
        or config.get("incremental.checkpoint", DEFAULT_CHECKPOINT)
    )
    done: dict[str, dict] = {}
    old_rows_by_session: dict[str, list[dict]] = {}
    if incremental:
        done = load_checkpoint(checkpoint_path)
        old_rows_by_session = load_existing_chunk_rows(args.output)
        logger.info(
            f"增量模式: checkpoint {len(done)} session, "
            f"旧 chunks.jsonl 含 {len(old_rows_by_session)} session"
        )

    output_rows: list[dict] = []
    n_reused = n_rechunked = n_new = 0
    for s in filtered:
        sid = s.id
        rows: list[dict]
        if incremental:
            rec = done.get(sid)
            cur_turns_hash = raw_turns_hash(s.turns)
            if (rec
                    and rec.get("turns_hash") == cur_turns_hash
                    and rec.get("chunker_version") == CHUNKER_VERSION
                    and sid in old_rows_by_session):
                rows = old_rows_by_session[sid]
                done[sid]["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                n_reused += 1
            else:
                new_chunks = process_session(s)
                new_fp = session_fingerprint(new_chunks)
                prev_fp = rec.get("content_hash") if rec else None
                done[sid] = {
                    "turns_hash": cur_turns_hash,
                    "chunk_count": len(new_chunks),
                    "chunk_ids": [c.chunk_id for c in new_chunks],
                    "content_hash": new_fp,
                    "chunker_version": CHUNKER_VERSION,
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                if prev_fp and prev_fp != new_fp:
                    n_rechunked += 1
                else:
                    n_new += 1
                rows = [c.to_dict() for c in new_chunks]
        else:
            new_chunks = process_session(s)
            rows = [c.to_dict() for c in new_chunks]
        output_rows.extend(rows)

    # 5. 输出
    save_chunks_rows(output_rows, args.output)

    if incremental:
        save_checkpoint(checkpoint_path, done)

    # 6. 打印摘要
    logger.info("=" * 60)
    logger.info("Phase 1 完成")
    logger.info(f"  输入 session: {len(raw_sessions)}")
    logger.info(f"  过滤后 session: {len(filtered)}")
    logger.info(f"  输出 chunk: {len(output_rows)}")
    if incremental:
        logger.info(
            f"  增量: 复用 {n_reused} / 重切 {n_rechunked} / 新增 {n_new}"
        )
    logger.info(f"  输出文件: {args.output}")
    if incremental:
        logger.info(f"  checkpoint: {checkpoint_path} ({len(done)} session)")
    logger.info("=" * 60)

    # 7. 打印前 1 个 chunk 示例
    if output_rows:
        row = output_rows[0]
        logger.info("=" * 60)
        logger.info("示例 chunk 整理后内容:")
        logger.info("=" * 60)
        rebuilt = Chunk(
            chunk_id=row["chunk_id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            user_message=row["user_message"],
            assistant_messages=row.get("assistant_messages", []),
            tool_calls=[CleanedToolCall(**tc) for tc in row.get("tool_calls", [])],
            mcp_calls=[CleanedToolCall(**mc) for mc in row.get("mcp_calls", [])],
            raw_size_tokens=row.get("raw_size_tokens", 0),
            cleaned_size_tokens=row.get("cleaned_size_tokens", 0),
            created_at=row.get("created_at"),
            task_summary=row.get("task_summary"),
            content_hash=row.get("content_hash"),
        )
        print(rebuilt.cleaned_text())
        logger.info("=" * 60)
        logger.info(
            f"token 占用: raw={rebuilt.raw_size_tokens} → "
            f"cleaned={rebuilt.cleaned_size_tokens}  "
            f"content_hash={rebuilt.content_hash[:12] if rebuilt.content_hash else 'NONE'}..."
        )


if __name__ == "__main__":
    main()
