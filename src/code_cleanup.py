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


def _build_target_registry() -> list:
    c1 = _load_phase_config("code_p1_config.yaml")
    c2 = _load_phase_config("code_p2_config.yaml")
    c3 = _load_phase_config("code_p3_config.yaml")
    c5 = _load_phase_config("code_p5_config.yaml")

    def g(cfg, key: str, default: str):
        return cfg.get(key, default) if cfg else default

    return [
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
        OutputTarget("p3", "dir",
                     _resolve(g(c3, "qdrant.path", "./qdrant_data"), "./qdrant_data"),
                     "P3 Qdrant 向量库"),
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
    ]


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


def cmd_list(args, registry: list) -> int:
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


def _print_plan(plan: list, dry_run: bool):
    mode = "dry-run: 只打印计划, 不删除" if dry_run else "--yes: 将真实删除"
    print(f"删除计划 ({mode})", flush=True)
    total = 0
    for t in plan:
        exists = t.path.exists()
        size = _target_size(t) if exists else 0
        total += size
        flag = "delete" if exists else "skip (不存在)"
        print(f"  [{flag:<14}] {t.phase:<5} {_display_path(t.path):<35} "
              f"{_format_size(size) if exists else '-':>9}  {t.description}",
              flush=True)
    print(f"合计: {len(plan)} 个目标, {_format_size(total)}", flush=True)
    return total


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


def cmd_clean(args, registry: list) -> int:
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
            logger.info(f"cascade: 扩展删除下游 phase: {', '.join(sorted(added))}")
        selected = expanded
    plan = _build_delete_plan(registry, selected)
    _assert_within_repo(plan)
    dry_run = not args.yes
    total = _print_plan(plan, dry_run)
    if dry_run:
        print("[dry-run] 未删除任何文件; 加 --yes 真实删除", flush=True)
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
    args = parser.parse_args(argv)

    registry = _build_target_registry()
    if args.command == "list":
        return cmd_list(args, registry)
    if args.command == "clean":
        return cmd_clean(args, registry)
    return 2


if __name__ == "__main__":
    sys.exit(main())
