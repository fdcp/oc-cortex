#!/usr/bin/env python3
"""
把本地 embedded Qdrant (qdrant_data/) 的数据迁移到 Docker Qdrant (http://localhost:6333)

用法:
  python scripts/migrate_qdrant_to_server.py
  python scripts/migrate_qdrant_to_server.py --src ./qdrant_data --dst http://localhost:6333
  python scripts/migrate_qdrant_to_server.py --collections tasks chunks_summary chunks_cleaned_text entities
  python scripts/migrate_qdrant_to_server.py --dry-run

注意:
  - 单进程运行 (本地 embedded 是单进程独占,会创建 .lock)
  - 跑完即关,不会污染
  - 不会影响 MCP server (40121 当前没在用 embedded)
"""
import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


# ----------------------------------------------------------
# Schema 探查与重建
# ----------------------------------------------------------

def dump_schema(client: QdrantClient, collection: str) -> dict:
    """读 collection 完整 schema (vectors_config / sparse_vectors_config / hnsw 等)"""
    info = client.get_collection(collection_name=collection)
    return {
        "vectors_config": info.config.params.vectors,
        "sparse_vectors_config": info.config.params.sparse_vectors,
        "shard_number": info.config.params.shard_number,
        "replication_factor": info.config.params.replication_factor,
        "write_consistency_factor": info.config.params.write_consistency_factor,
        "on_disk_payload": info.config.params.on_disk_payload,
    }


def recreate_collection(dst: QdrantClient, collection: str, schema: dict) -> None:
    """用同样的 schema 在目标 server 上重建 collection (drop if exists)"""
    existing = {c.name for c in dst.get_collections().collections}
    if collection in existing:
        logger.info(f"  dst 集合 {collection!r} 已存在, 先 drop")
        dst.delete_collection(collection_name=collection)

    # vectors_config 和 sparse_vectors_config 可能是 dict 或单个 VectorParams
    vectors_config = schema["vectors_config"]
    sparse_config = schema["sparse_vectors_config"]

    create_kwargs = dict(
        collection_name=collection,
        vectors_config=vectors_config,
    )
    # shard / replication / consistency 字段不强制传, 用 server 默认 (1 / 1 / 1)
    # 如果 src 显式设了, 才传过去 (避免和 server 集群拓扑冲突)
    if sparse_config is not None:
        create_kwargs["sparse_vectors_config"] = sparse_config
    logger.info(
        f"  创建集合 {collection!r} "
        f"(vectors={type(vectors_config).__name__}, sparse={'yes' if sparse_config else 'no'})"
    )
    dst.create_collection(**create_kwargs)


# ----------------------------------------------------------
# 数据迁移 (scroll + upsert 分批)
# ----------------------------------------------------------

def migrate_collection(
    src: QdrantClient,
    dst: QdrantClient,
    collection: str,
    batch_size: int = 200,
    recreate: bool = True,
) -> int:
    """迁移单个 collection, 返回写入的点数"""
    if recreate:
        schema = dump_schema(src, collection)
        recreate_collection(dst, collection, schema)
    else:
        # 仅追加 (不重建)
        existing_dst = {c.name for c in dst.get_collections().collections}
        if collection not in existing_dst:
            logger.error(f"  dst 集合 {collection!r} 不存在, 跳过 (use --recreate)")
            return 0

    logger.info(f"  scroll {collection!r} (batch={batch_size}) ...")
    offset = None
    total = 0
    t0 = time.time()
    while True:
        points, next_offset = src.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        # Qdrant 1.19+ upsert 需要 PointStruct 或 dict, scroll 返回的是 Record
        from qdrant_client.http.models import PointStruct
        point_structs = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload=p.payload or {},
            )
            for p in points
        ]
        dst.upsert(
            collection_name=collection,
            points=point_structs,
            wait=True,
        )
        total += len(points)
        logger.info(f"    进度: {total} points")
        if next_offset is None:
            break
        offset = next_offset

    elapsed = time.time() - t0
    logger.info(f"  ✅ {collection!r}: {total} points, 耗时 {elapsed:.1f}s")
    return total


# ----------------------------------------------------------
# 主流程
# ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./qdrant_data", help="本地 embedded Qdrant 路径")
    ap.add_argument("--dst", default="http://localhost:6333", help="目标 server URL")
    ap.add_argument(
        "--collections", nargs="+", default=None,
        help="要迁移的集合 (默认: 全部)",
    )
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--no-recreate", action="store_true", help="不重建, 只追加")
    ap.add_argument("--dry-run", action="store_true", help="只打印 schema, 不写数据")
    args = ap.parse_args()

    src_path = Path(args.src).resolve()
    if not (src_path / "meta.json").exists():
        logger.error(f"本地 qdrant 数据目录不存在或无效: {src_path}")
        sys.exit(1)

    logger.info(f"src: {src_path}")
    logger.info(f"dst: {args.dst}")

    # 只读 src (避免影响其他进程)
    src = QdrantClient(path=str(src_path))
    dst = QdrantClient(url=args.dst)

    # 探查所有集合
    src_collections = sorted(c.name for c in src.get_collections().collections)
    logger.info(f"src 集合: {src_collections}")

    targets = args.collections or src_collections
    missing = [c for c in targets if c not in src_collections]
    if missing:
        logger.error(f"src 缺少集合: {missing}")
        sys.exit(1)

    # dry-run: 只打印 schema
    if args.dry_run:
        for c in targets:
            schema = dump_schema(src, c)
            print(f"\n=== {c} ===")
            for k, v in schema.items():
                print(f"  {k}: {v}")
        return

    # 迁移
    grand_total = 0
    t_all = time.time()
    for c in targets:
        logger.info(f"\n--- {c} ---")
        n = migrate_collection(
            src, dst, c,
            batch_size=args.batch_size,
            recreate=not args.no_recreate,
        )
        grand_total += n

    elapsed = time.time() - t_all
    logger.info(f"\n🎉 迁移完成: {grand_total} points, 总耗时 {elapsed:.1f}s")

    # 验证
    logger.info("\n=== 验证 ===")
    for c in targets:
        try:
            info = dst.get_collection(collection_name=c)
            logger.info(f"  dst {c}: points={info.points_count}, indexed={info.indexed_vectors_count}")
        except Exception as e:
            logger.warning(f"  dst {c}: 验证失败 {e}")


if __name__ == "__main__":
    main()
