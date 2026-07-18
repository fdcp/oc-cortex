"""
Phase 1 主入口
流程: 加载 session → 过滤 → 切分 → 整理 → 输出

用法:
  python code_p1_main.py
  python code_p1_main.py --source ./data/my_sessions.jsonl
  python code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db
  python code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3
  python code_p1_main.py --mock     # 用 mock 数据
"""
import argparse
import json
import sys
from pathlib import Path
from loguru import logger

from code_p1_utils import Config, setup_logger
from code_p1_models import Session, Chunk
from code_p1_session_loader import (
    load_sessions_from_jsonl,
    load_sessions_from_json,
    filter_sessions,
    get_mock_sessions,
)
from code_p1_sqlite_loader import load_sessions_from_sqlite
from code_p1_chunker import chunk_session
from code_p1_content_cleaner import clean_chunks


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

    # 4. 切分 + 整理
    all_chunks: list[Chunk] = []
    for s in filtered:
        chunks = process_session(s)
        all_chunks.extend(chunks)

    # 5. 输出
    save_chunks(all_chunks, args.output)

    # 6. 打印摘要
    logger.info("=" * 60)
    logger.info("Phase 1 完成")
    logger.info(f"  输入 session: {len(raw_sessions)}")
    logger.info(f"  过滤后 session: {len(filtered)}")
    logger.info(f"  生成 chunk: {len(all_chunks)}")
    logger.info(f"  输出文件: {args.output}")
    logger.info("=" * 60)

    # 7. 打印前 1 个 chunk 示例
    if all_chunks:
        example = all_chunks[0]
        logger.info("=" * 60)
        logger.info("示例 chunk 整理后内容:")
        logger.info("=" * 60)
        print(example.cleaned_text())
        logger.info("=" * 60)
        logger.info(
            f"token 占用: raw={example.raw_size_tokens} → "
            f"cleaned={example.cleaned_size_tokens}"
        )


if __name__ == "__main__":
    main()
