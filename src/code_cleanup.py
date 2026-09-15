"""
输出清理工具: 按 phase 选择性清理 output/ qdrant_data/ logs/ 下的 pipeline 产物

用法:
  python3 src/code_cleanup.py list                  # 列出所有 phase 产物
  python3 src/code_cleanup.py clean --p2            # 默认 dry-run, 只打印删除计划
  python3 src/code_cleanup.py clean --p2 --dry-run  # 显式 dry-run (干跑不删)
  python3 src/code_cleanup.py clean --p2 --yes      # 真删 (跳过确认)
  python3 src/code_cleanup.py clean --p1 --cascade --yes  # 级联删除下游产物

安全机制:
  - clean 默认 dry-run, 不加 --yes 不删任何文件
  - checkpoint 与 phase 强绑: 清 P1/P2 必带对应 checkpoint
  - --cascade 默认关闭, 关闭时仅 WARNING 列出下游孤儿产物
  - 不触碰 tests/ 下的 benchmark 副产物, 无 backup/undo
"""
import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from code_p1_utils import Config

REPO_ROOT = Path(__file__).resolve().parent.parent

_PHASE_ORDER = ["p1", "p2", "p3", "p5", "logs"]

# phase -> 上游依赖 (p2 运行依赖 p1 产物, 以此类推)
_PHASE_DEPS = {
    "p1": [],
    "p2": ["p1"],
    "p3": ["p1", "p2"],
    "p5": ["p2"],
    "logs": [],
}


@dataclass
class OutputTarget:
    phase: str
    kind: str  # "file" | "dir"
    path: Path
    description: str


def _load_phase_config(name: str):
    cfg_path = REPO_ROOT / "config" / name
    if not cfg_path.exists():
        logger.warning(f"配置文件不存在, 使用内置默认路径: {cfg_path}")
        return None
    return Config(str(cfg_path))


def _resolve(raw, default: str) -> Path:
    p = Path(raw or default)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _detect_qdrant_mode() -> tuple:
    """Returns (mode, url). mode is 'server' or 'embedded'. url 是解析后的 QDRANT_URL 或 None。"""
    env_url = os.environ.get("QDRANT_URL", "").strip()
    if env_url:
        return "server", env_url

    c3 = _load_phase_config("code_p3_config.yaml")
    yaml_url = ""
    if c3:
        yaml_url = (c3.get("qdrant.url", "") or "").strip()
    if yaml_url:
        return "server", yaml_url
    return "embedded", None


def _phase_to_server_collections(phase: str, c3, c5) -> list:
    """phase → server 上对应的 collection 名列表。空列表表示该 phase 无 server 数据。"""

    def g(cfg, key: str, default: str):
        return cfg.get(key, default) if cfg else default

    if phase == "p3":
        return [
            g(c3, "qdrant.collections.tasks", "tasks"),
            g(c3, "qdrant.collections.chunks_summary", "chunks_summary"),
            g(c3, "qdrant.collections.chunks_cleaned_text", "chunks_cleaned_text"),
        ]
    if phase == "p5":
        return [g(c5, "qdrant.entities_collection", "entities")]
    return []


def _build_target_registry(mode: str = "embedded") -> list:
    c1 = _load_phase_config("code_p1_config.yaml")
    c2 = _load_phase_config("code_p2_config.yaml")
    c3 = _load_phase_config("code_p3_config.yaml")
    c5 = _load_phase_config("code_p5_config.yaml")

    def g(cfg, key: str, default: str):
        return cfg.get(key, default) if cfg else default

    targets = [
        OutputTarget("p1", "file",
                     _resolve(g(c2, "phase1.chunks_file", "./output/chunks.jsonl"), "./output/chunks.jsonl"),
                     "P1 chunks 产物"),
        OutputTarget("p1", "file",
                     _resolve(g(c1, "incremental.checkpoint", "./output/.p1_checkpoint.json"), "./output/.p1_checkpoint.json"),
                     "P1 增量 checkpoint"),
        OutputTarget("p2", "file",
                     _resolve(g(c3, "phase2.tasks_file", "./output/tasks.jsonl"), "./output/tasks.jsonl"),
                     "P2 task 产物"),
        OutputTarget("p2", "file",
                     _resolve(g(c2, "output.chunk_summaries", "./output/chunks_summary_p2.jsonl"), "./output/chunks_summary_p2.jsonl"),
                     "P2 chunk 总结"),
        OutputTarget("p2", "file",
                     _resolve(g(c2, "output.checkpoint", "./output/.p2_checkpoint.json"), "./output/.p2_checkpoint.json"),
                     "P2 增量 checkpoint"),
    ]

    if mode == "embedded":
        targets.append(OutputTarget("p3", "dir",
                     _resolve(g(c3, "qdrant.path", "./qdrant_data"), "./qdrant_data"),
                     "P3 Qdrant 向量库 (embedded)"))
        targets.append(OutputTarget("p5", "dir",
                     _resolve(g(c5, "qdrant.path", "./qdrant_data"), "./qdrant_data")
                     / "collection" / g(c5, "qdrant.entities_collection", "entities"),
                     "P5 实体对齐 Qdrant 集合 (embedded 复用 P3 存储)"))

    targets.extend([
        OutputTarget("p5", "dir",
                     _resolve(g(c5, "knowledge_graph.triple_output_dir", "./output/triple"), "./output/triple"),
                     "P5 triple 模式产物"),
        OutputTarget("p5", "dir",
                     _resolve(g(c5, "knowledge_graph.entity_output_dir", "./output/entity"), "./output/entity"),
                     "P5 entity 模式产物"),
        OutputTarget("logs", "file",
                     _resolve(g(c1, "logging.file", "./logs/phase1.log"), "./logs/phase1.log"),
                     "P1 日志"),
        OutputTarget("logs", "file",
                     _resolve(g(c2, "logging.file", "./logs/phase2.log"), "./logs/phase2.log"),
                     "P2 日志"),
        OutputTarget("logs", "file",
                     _resolve(g(c3, "logging.file", "./logs/phase3.log"), "./logs/phase3.log"),
                     "P3 日志"),
        OutputTarget("logs", "file",
                     _resolve(g(c5, "logging.file", "./logs/phase5.log"), "./logs/phase5.log"),
                     "P5 日志"),
        OutputTarget("logs", "file",
                     _resolve(None, "./logs/phase5_entity.log"),
                     "P5 entity 模式日志"),
    ])

    return targets


def _format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _target_size(t: OutputTarget) -> int:
    if not t.path.exists():
        return 0
    if t.kind == "file":
        return t.path.stat().st_size
    # 跨平台递归统计 (macOS BSD du 无 -b, 不依赖外部命令)
    total = 0
    for root, _dirs, files in os.walk(t.path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _display_path(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _list_server_state(qdrant_url: str) -> None:
    """查询并打印 server 上 collections + snapshots 详情.

    网络失败 → 抛 RuntimeError, 由 caller 捕获降级为 warning.
    """
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from qdrant_snapshot import (
        _list_collections,
        _get_collection_info,
        _list_server_snapshots_full,
    )

    collections = sorted(_list_collections(qdrant_url))

    print(f"\n[server collections @ {qdrant_url}]", flush=True)
    if not collections:
        print("  (server 上无 collection)", flush=True)
    else:
        print(f"  {'COLLECTION':<28} {'POINTS':>8} {'VECTORS':>8}  STATUS", flush=True)
        print("  " + "-" * 64, flush=True)
        for cname in collections:
            info = _get_collection_info(qdrant_url, cname)
            if info is None:
                print(f"  {cname:<28} {'-':>8} {'-':>8}  (race: get 时已消失)",
                      flush=True)
                continue
            status = info.get("status", "?")
            pc = info.get("points_count", 0)
            vc = info.get("vectors_count", 0)
            print(f"  {cname:<28} {pc:>8} {vc:>8}  {status}", flush=True)

    print(f"\n[server snapshots @ {qdrant_url}]", flush=True)
    has_any = False
    for cname in collections:
        snaps = _list_server_snapshots_full(qdrant_url, cname)
        if not snaps:
            continue
        if not has_any:
            print(f"  {'COLLECTION':<28} {'SNAPSHOT':<35} {'SIZE':>9}  CREATION",
                  flush=True)
            print("  " + "-" * 92, flush=True)
            has_any = True
        for s in snaps:
            name = s.get("name", "?")
            size = s.get("size", 0)
            size_str = _format_size(size) if size else "-"
            ctime = s.get("creation_time", "")
            if ctime:
                ctime = ctime.replace("T", " ").replace("Z", "")[:19]
            print(f"  {cname:<28} {name:<35} {size_str:>9}  {ctime}", flush=True)
    if not has_any:
        print("  (server 上无 snapshot)", flush=True)


def cmd_list(args, registry: list, mode: str, qdrant_url: str | None = None) -> int:
    print(f"{'PHASE':<6} {'EXISTS':<7} {'SIZE':>9}  PATH", flush=True)
    print("-" * 72, flush=True)
    for phase in _PHASE_ORDER:
        for t in registry:
            if t.phase != phase:
                continue
            exists = t.path.exists()
            size = _format_size(_target_size(t)) if exists else "-"
            print(f"{t.phase:<6} {'yes' if exists else 'no':<7} {size:>9}  {_display_path(t.path)}",
                  flush=True)
    if mode == "server" and qdrant_url:
        try:
            _list_server_state(qdrant_url)
        except Exception as e:
            print(f"\n[server 状态] 查询失败: {e}", file=sys.stderr)
            logger.warning(f"无法列出 server collections/snapshots: {e}")
    if mode == "server":
        print(f"\n[server 模式] QDRANT_URL={qdrant_url}", flush=True)
        print("  清理时 P3 / P5 entities 集合由 `code_cleanup.py clean --p3/--p5/--all --yes`", flush=True)
        print("  调 drop_collections() 管理 (含删前 auto-backup)。", flush=True)
    return 0


def _parse_selection(args) -> set:
    selected = set()
    for phase in ("p1", "p2", "p3", "p5", "logs"):
        if getattr(args, phase):
            selected.add(phase)
    if args.all:
        selected.update(_PHASE_ORDER)
    return selected


def _build_delete_plan(registry: list, selected: set) -> list:
    return [t for t in registry if t.phase in selected]


def _downstream_closure(phases: set) -> set:
    # 反向依赖图: phase -> 依赖它的下游 phase; 沿下游传递闭包 (不向上)
    dependents = {p: set() for p in _PHASE_ORDER}
    for phase, deps in _PHASE_DEPS.items():
        for up in deps:
            if up in dependents:
                dependents[up].add(phase)
    closure = set(phases)
    frontier = set(phases)
    while frontier:
        nxt = set()
        for p in frontier:
            nxt |= {q for q in dependents.get(p, ()) if q not in closure}
        closure |= nxt
        frontier = nxt
    return closure


def _find_orphans(registry: list, deleted_phases: set) -> list:
    downstream = _downstream_closure(deleted_phases) - deleted_phases
    return [t for t in registry if t.phase in downstream and t.path.exists()]


def _assert_within_repo(plan: list):
    # 安全兜底: 拒绝删除 repo 之外的路径 (防止配置被改成外部绝对路径)
    for t in plan:
        try:
            t.path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            logger.error(f"路径不在 repo 内, 拒绝删除: {t.path}")
            sys.exit(1)


def _print_unified_plan(
    local_plan: list,
    server_plan: list,
    server_result: dict | None,
    qdrant_url: str | None,
    dry_run: bool,
) -> int:
    """统一 plan 输出, 覆盖本地文件 + server collections.

    Args:
        local_plan: OutputTarget list (来自 registry)
        server_plan: [(phase, collection_name), ...] 只在 server mode 有值
        server_result: drop_collections 返回的 dict (server mode) 或 None (embedded)
        qdrant_url: server URL 或 None
        dry_run: 是否 dry-run

    Returns:
        累计 size (bytes), 供 caller 打 INFO 日志
    """
    mode_str = "dry-run: 只打印计划, 不删除" if dry_run else "--yes: 将真实删除"
    print(f"\n删除计划 ({mode_str})", flush=True)

    total_count = 0
    total_size = 0

    if local_plan:
        print(f"\n[本地文件]", flush=True)
        for t in local_plan:
            exists = t.path.exists()
            size = _target_size(t) if exists else 0
            total_count += 1
            total_size += size
            flag = "delete" if exists else "skip"
            size_str = _format_size(size) if exists else "-"
            print(f"  [{flag:<14}] {t.phase:<5} {_display_path(t.path):<42} "
                  f"{size_str:>10}  {t.description}", flush=True)

    if server_plan and server_result is not None:
        print(f"\n[server collections @ {qdrant_url}]", flush=True)
        for phase, cname in server_plan:
            if cname in server_result.get("existing", []):
                pc = server_result["points_count"].get(cname, 0)
                total_count += 1
                print(f"  [{'drop':<14}] {phase:<5} {cname:<28} {pc:>8} points",
                      flush=True)
            elif cname in server_result.get("missing", []):
                print(f"  [{'skip':<14}] {phase:<5} {cname:<28} {'-':>8}  (server 上不存在)",
                      flush=True)

    size_suffix = f", {_format_size(total_size)}" if total_size > 0 else ""
    print(f"\n合计: {total_count} 个目标{size_suffix}", flush=True)

    if dry_run:
        print("[dry-run] 未删除任何文件 / collection, 加 --yes 真实删除", flush=True)

    return total_size


def _execute_delete(plan: list) -> list:
    deleted = []
    for t in plan:
        if not t.path.exists():
            logger.info(f"跳过 (不存在): {_display_path(t.path)}")
            continue
        try:
            if t.kind == "file":
                t.path.unlink(missing_ok=True)
            else:
                shutil.rmtree(t.path, ignore_errors=True)
            deleted.append(t)
            logger.info(f"已删除: {_display_path(t.path)}")
        except OSError as e:
            logger.error(f"删除失败: {_display_path(t.path)}: {e}")
    return deleted


def cmd_clean(args, registry: list, mode: str, qdrant_url: str | None) -> int:
    if args.dry_run and args.yes:
        logger.error("--dry-run 与 --yes 互斥")
        return 2
    selected = _parse_selection(args)
    if not selected:
        logger.error("请至少选择一个 phase: --p1/--p2/--p3/--p5/--logs/--all")
        return 2
    if args.cascade:
        expanded = _downstream_closure(selected)
        added = expanded - selected
        if added:
            logger.info(f"cascade: 扩展清理下游 phase: {', '.join(sorted(added))}")
        selected = expanded
    dry_run = not args.yes

    if mode == "server":
        return _cmd_clean_server(args, registry, selected, dry_run, qdrant_url)
    return _cmd_clean_embedded(args, registry, selected, dry_run)


def _cmd_clean_embedded(args, registry: list, selected: set, dry_run: bool) -> int:
    plan = _build_delete_plan(registry, selected)
    _assert_within_repo(plan)
    total = _print_unified_plan(plan, [], None, None, dry_run)
    if dry_run:
        return 0
    deleted = _execute_delete(plan)
    orphans = _find_orphans(registry, selected)
    for t in orphans:
        logger.warning(
            f"orphan detected: {_display_path(t.path)} "
            f"(上游 phase 已删, 产物仍存在; 加 --cascade 一并清理)"
        )
    logger.info(f"完成: 删除 {len(deleted)}/{len(plan)} 个目标, 释放 {_format_size(total)}")
    return 0


def _cmd_clean_server(args, registry: list, selected: set, dry_run: bool, qdrant_url: str | None) -> int:
    local_phases = {p for p in selected if p != "p3"}
    local_plan = [t for t in registry if t.phase in local_phases]

    server_phases = {p for p in selected if p in ("p3", "p5")}
    server_plan: list = []
    server_collections: list = []
    if server_phases and qdrant_url:
        c3 = _load_phase_config("code_p3_config.yaml")
        c5 = _load_phase_config("code_p5_config.yaml")
        for phase in sorted(server_phases):
            for cname in _phase_to_server_collections(phase, c3, c5):
                server_plan.append((phase, cname))
                server_collections.append(cname)

    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from qdrant_snapshot import drop_collections

    backup_out = (
        Path(args.backup_out).expanduser().resolve() if args.backup_out else None
    )

    plan_result: dict | None = None
    if server_collections:
        try:
            plan_result = drop_collections(
                qdrant_url=qdrant_url or "",
                collections=server_collections,
                dry_run=True,
            )
        except RuntimeError as e:
            logger.error(f"server 清理失败: {e}")
            return 1

    if local_plan:
        _assert_within_repo(local_plan)
    _print_unified_plan(local_plan, server_plan, plan_result, qdrant_url, dry_run)

    if dry_run:
        return 0

    if local_plan:
        deleted = _execute_delete(local_plan)
        logger.info(f"本地清理: 删除 {len(deleted)}/{len(local_plan)} 个目标")

    if server_collections:
        try:
            result = drop_collections(
                qdrant_url=qdrant_url,
                collections=server_collections,
                auto_backup=not args.no_backup,
                backup_out=backup_out,
                keep_host_backups=args.keep_host_backups,
                dry_run=False,
                force=args.force,
            )
            logger.info(
                f"server 清理: 实际删除 {len(result['deleted'])}/"
                f"{len(result['existing'])} collections, "
                f"跳过不存在 {len(result['missing'])}"
            )
            if result.get("backup_path"):
                logger.info(f"  backup: {result['backup_path']}")
        except RuntimeError as e:
            logger.error(f"server 清理失败: {e}")
            return 1

    orphans = _find_orphans(registry, selected)
    for t in orphans:
        logger.warning(
            f"orphan detected: {_display_path(t.path)} "
            f"(上游 phase 已删, 产物仍存在; 加 --cascade 一并清理)"
        )

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="按 phase 清理 pipeline 输出产物 (output/ qdrant_data/ logs/)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="列出所有 phase 产物 (path/exists/size)")

    p_clean = sub.add_parser("clean", help="按 phase 删除产物 (默认 dry-run)")
    for phase in ("p1", "p2", "p3", "p5", "logs"):
        p_clean.add_argument(f"--{phase}", action="store_true",
                             help=f"清理 {phase.upper()} 产物")
    p_clean.add_argument("--all", action="store_true",
                         help="清理所有 phase 产物")
    p_clean.add_argument("-n", "--dry-run", action="store_true",
                         help="显式干跑: 只打印删除计划, 不删除 (与默认行为一致)")
    p_clean.add_argument("-y", "--yes", action="store_true",
                         help="真实删除 (跳过确认); 不加此 flag 一律 dry-run")
    p_clean.add_argument("--cascade", action="store_true",
                         help="级联删除下游依赖产物 (默认关闭; 关闭时仅 WARNING 列出孤儿)")
    p_clean.add_argument("--no-backup", action="store_true",
                         help="(server mode) 跳过删前自动 backup, CI 用")
    p_clean.add_argument("--backup-out", default=None,
                         help="(server mode) 自定义 backup 输出目录 "
                              "(默认 ./qdrant_snapshots/<timestamp>/)")
    p_clean.add_argument("--keep-host-backups", type=int, default=3,
                         help="(server mode) host 端 backup 目录保留最近 N 个 "
                              "(默认 3, 0 = 全保留)")
    p_clean.add_argument("--force", action="store_true",
                         help="(server mode) 跳过 'DELETE' 二次确认 (用于自动化)")
    args = parser.parse_args(argv)

    mode, qdrant_url = _detect_qdrant_mode()
    registry = _build_target_registry(mode)
    if args.command == "list":
        return cmd_list(args, registry, mode, qdrant_url)
    if args.command == "clean":
        return cmd_clean(args, registry, mode, qdrant_url)
    return 2


if __name__ == "__main__":
    sys.exit(main())
