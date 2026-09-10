#!/usr/bin/env python3
"""
Qdrant snapshot 管理工具: backup / restore / list / download

用法:
  # 1. 备份所有 collections 到 host
  python3 utils/qdrant_snapshot.py backup --out ./qdrant_snapshots/20260910

  # 2. 备份指定 collections
  python3 utils/qdrant_snapshot.py backup --out ./backups --collections tasks entities

  # 3. 列出 server 上已有 snapshots
  python3 utils/qdrant_snapshot.py list
  python3 utils/qdrant_snapshot.py list --collection tasks

  # 4. 从 host 恢复到 server (容器重建后场景)
  python3 utils/qdrant_snapshot.py restore --in ./qdrant_snapshots/20260910
  python3 utils/qdrant_snapshot.py restore --in ./backups --collection tasks

  # 5. 下载单个 snapshot (容器内已存在 -> host)
  python3 utils/qdrant_snapshot.py download --collection tasks --name tasks-2026-09-10.snapshot --out ./backups/

参数:
  --host <url>         Qdrant server URL (默认 http://localhost:6333)
  --out <dir|file>     backup/download 输出路径 (backup 必须是 dir, download 可以是 dir 或 file)
  --in <dir>           restore 输入目录
  --collections ...    backup 子集 (默认全部)
  --wait / --no-wait   restore 后是否等待 collection ready (默认 --wait)

文件命名约定:
  下载后一律按 {collection}__{original_name}.snapshot 命名
  restore 时按此规则反解析 collection

注意:
  - snapshot 是 Qdrant 私有二进制格式, 不能 cat/grep
  - 跨 Qdrant 版本不保证兼容 (v1.19 snapshot 不能在 v1.18 恢复)
  - 没 -v 挂载时, snapshot 必须在 host 留一份 (否则容器销毁 = 数据丢失)
"""
import argparse
import sys
import time
from pathlib import Path

import requests

HOST_DEFAULT = "http://localhost:6333"
TIMEOUT = 60
CHUNK = 64 * 1024

_NO_PROXY_SESSION = requests.Session()
_NO_PROXY_SESSION.trust_env = False


def _url(host: str, path: str) -> str:
    return f"{host.rstrip('/')}{path}"


def _list_collections(host: str) -> list[str]:
    r = _NO_PROXY_SESSION.get(_url(host, "/collections"), timeout=TIMEOUT)
    r.raise_for_status()
    return [c["name"] for c in r.json()["result"]["collections"]]


def cmd_backup(args) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.collections:
        collections = args.collections
    else:
        collections = _list_collections(args.host)
        print(f"[auto] {len(collections)} collections: {collections}")

    summary = []
    for cname in collections:
        print(f"\n[{cname}]")
        t0 = time.time()
        try:
            r = _NO_PROXY_SESSION.post(
                _url(args.host, f"/collections/{cname}/snapshots"),
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            snap_name = r.json()["result"]["name"]
            print(f"  create  {snap_name} ({time.time() - t0:.2f}s)")

            t1 = time.time()
            r = _NO_PROXY_SESSION.get(
                _url(args.host, f"/collections/{cname}/snapshots/{snap_name}"),
                timeout=TIMEOUT,
                stream=True,
            )
            r.raise_for_status()
            out_path = out_dir / f"{cname}__{snap_name}"
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"  download {out_path.name} ({size_mb:.2f} MB, {time.time() - t1:.2f}s)")
            summary.append((cname, snap_name, size_mb))
        except Exception as e:
            print(f"  FAIL: {e}", file=sys.stderr)
            summary.append((cname, None, 0))

    ok = sum(1 for s in summary if s[1])
    print(f"\n=== backup 完成 ({ok}/{len(summary)} ok) ===")
    print(f"输出目录: {out_dir}")
    for cname, snap_name, size_mb in summary:
        if snap_name:
            print(f"  OK  {cname}: {snap_name} ({size_mb:.2f} MB)")
        else:
            print(f"  FAIL {cname}")
    return 0 if ok == len(summary) else 1


def cmd_restore(args) -> int:
    in_dir = Path(args.in_).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"FAIL: 目录不存在: {in_dir}", file=sys.stderr)
        return 1

    snap_files = sorted(in_dir.glob("*__*.snapshot"))
    if args.collection:
        snap_files = [f for f in snap_files if f.name.startswith(f"{args.collection}__")]

    if not snap_files:
        print(f"FAIL: 未在 {in_dir} 找到 *__*.snapshot 文件", file=sys.stderr)
        return 1

    print(f"[restore] {len(snap_files)} snapshots:")
    for f in snap_files:
        print(f"  {f.name}")

    failed = []
    for snap_path in snap_files:
        parts = snap_path.name.split("__", 1)
        if len(parts) != 2:
            print(f"WARN 跳过(无法解析 collection): {snap_path.name}", file=sys.stderr)
            continue
        cname, snap_name = parts

        print(f"\n[{cname} <- {snap_name}]")
        t0 = time.time()
        try:
            with open(snap_path, "rb") as f:
                files = {"snapshot": (snap_path.name, f, "application/octet-stream")}
                params = {"priority": "snapshot", "wait": str(args.wait).lower()}
                r = _NO_PROXY_SESSION.put(
                    _url(args.host, f"/collections/{cname}/snapshots/recover"),
                    files=files,
                    params=params,
                    timeout=TIMEOUT * 5,
                )
            r.raise_for_status()
            print(f"  OK ({time.time() - t0:.2f}s, wait={args.wait})")
        except Exception as e:
            print(f"  FAIL: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"    response: {e.response.text[:300]}", file=sys.stderr)
            failed.append(cname)

    print(f"\n=== restore 完成 ({len(snap_files) - len(failed)}/{len(snap_files)} ok) ===")
    return 0 if not failed else 1


def cmd_list(args) -> int:
    collections = [args.collection] if args.collection else _list_collections(args.host)

    for cname in collections:
        r = _NO_PROXY_SESSION.get(
            _url(args.host, f"/collections/{cname}/snapshots"),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        snaps = r.json()["result"]
        print(f"\n{cname}: {len(snaps)} snapshot(s)")
        for s in snaps:
            size_mb = (s.get("size") or 0) / 1024 / 1024
            print(f"  {s['name']:50}  {size_mb:>8.2f} MB  {s['creation_time']}")
    return 0


def cmd_download(args) -> int:
    out_path = Path(args.out).expanduser().resolve()
    if out_path.is_dir():
        out_path = out_path / f"{args.collection}__{args.name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"download {args.collection}/{args.name} -> {out_path}")
    r = _NO_PROXY_SESSION.get(
        _url(args.host, f"/collections/{args.collection}/snapshots/{args.name}"),
        timeout=TIMEOUT,
        stream=True,
    )
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK):
            if chunk:
                f.write(chunk)
    print(f"  OK {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Qdrant snapshot 管理 (backup / restore / list / download)",
    )
    p.add_argument(
        "--host",
        default=HOST_DEFAULT,
        help=f"Qdrant server URL (默认 {HOST_DEFAULT})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bk = sub.add_parser("backup", help="创建 + 下载 snapshot 到 host")
    p_bk.add_argument("--out", required=True, help="host 输出目录")
    p_bk.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="指定子集, 默认全部",
    )

    p_rs = sub.add_parser("restore", help="从 host snapshot 恢复")
    p_rs.add_argument("--in", dest="in_", required=True, help="snapshot 目录")
    p_rs.add_argument("--collection", default=None, help="只恢复指定 collection")
    p_rs.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="等待 collection ready (默认 True)",
    )

    p_ls = sub.add_parser("list", help="列出 server 上 snapshots")
    p_ls.add_argument("--collection", default=None, help="只列指定 collection")

    p_dl = sub.add_parser("download", help="下载单个 snapshot")
    p_dl.add_argument("--collection", required=True)
    p_dl.add_argument("--name", required=True)
    p_dl.add_argument("--out", required=True, help="输出路径 (dir 或 file)")

    args = p.parse_args()

    return {
        "backup": cmd_backup,
        "restore": cmd_restore,
        "list": cmd_list,
        "download": cmd_download,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)