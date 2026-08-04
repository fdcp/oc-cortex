#!/usr/bin/env python3
"""
code_p2_rerun.py — P2 定点重跑工具 (checkpoint 手术 + 续跑 + 自动验证)

适用场景:
  某些 chunk 的 summary/task 提取结果有问题 (例如 "无实质交互内容"
  这类坏样本), 需要只对受影响的 session 重调 LLM, 而不触碰其余
  session 的结果。

原理:
  P2 续跑以 output/.p2_checkpoint.json 为准: 仍在其中且 content_hash
  一致的 session 视为已完成 (结果从输出文件承接), 缺失的落入 pending
  重新调用 LLM。本脚本先备份三件套, 再做 checkpoint 手术:

    * 整 session 重跑 (--session / --auto-bad):
        从 checkpoint 移除该 session key, 全部 chunk 重新提取。
        单 batch session 的最小重跑单位就是整个 session。
    * batch 级重跑 (--chunk, 仅多 batch session):
        多 batch session 的 checkpoint 记录里逐 batch 存 status /
        content_hash / 结果; 续跑按 (batch_id, content_hash) 复用
        status=success 的 batch。把含目标 chunk 的 batch status 改为
        非 success, P2 自身续跑逻辑即只重跑该 batch、复用其余。
        若目标 chunk 属于单 batch session, 自动升级为整 session 重跑。

  然后 subprocess 调用 code_p2_main.py (不带 --force), 完成后自动
  校验输出完整性并给出新旧对比。

安全设计:
  - 默认 dry-run, 只输出计划, 不写任何文件; --yes 才真正执行
  - 手术前必备份 checkpoint / tasks / summaries 三件套
  - 检测到其他 code_p2_main.py 进程时拒绝执行
  - checkpoint 原子写入 (temp + os.replace), 写后复读校验
  - 运行失败不自动回滚 (per-session 原子落盘保证状态一致),
    只提示 --rollback 命令

用法:
  python3 src/code_p2_rerun.py --auto-bad                 # 干跑: 查看坏 summary 分布
  python3 src/code_p2_rerun.py --auto-bad --yes           # 真正修复全部坏 session
  python3 src/code_p2_rerun.py --session ses_xxx --yes    # 整 session 重跑
  python3 src/code_p2_rerun.py --chunk ses_xxx_c5 --yes   # 只重跑含该 chunk 的 batch
  python3 src/code_p2_rerun.py --rollback <backup_dir>    # 从备份恢复三件套

环境变量:
  OPENCODE_ZEN_API_KEY  优先使用; 未设置时自动读取
                        ~/.local/share/opencode/auth.json 的 opencode-go key
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BAD_PATTERN = "无实质"
CKPT_REL = "output/.p2_checkpoint.json"
TASKS_REL = "output/tasks.jsonl"
SUMMARY_REL = "output/chunks_summary_p2.jsonl"
CHUNKS_REL = "output/chunks.jsonl"
P2_MAIN_REL = "src/code_p2_main.py"


def die(msg: str, code: int = 1) -> None:
    logger.error(msg)
    sys.exit(code)


# ============================================================
# 路径 / 读取
# ============================================================

def resolve_output_paths() -> dict:
    """从 config/code_p2_config.yaml 读输出路径 (缺失用默认), 相对路径按仓库根解析。"""
    paths = {
        "checkpoint": CKPT_REL,
        "tasks": TASKS_REL,
        "summaries": SUMMARY_REL,
    }
    cfg_path = REPO_ROOT / "config" / "code_p2_config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            out = cfg.get("output", {}) or {}
            paths["checkpoint"] = out.get("checkpoint", CKPT_REL)
            paths["tasks"] = out.get("tasks", TASKS_REL)
            paths["summaries"] = out.get("chunk_summaries", SUMMARY_REL)
        except Exception as e:
            logger.warning(f"config 解析失败, 使用默认路径: {e}")
    resolved = {}
    for k, v in paths.items():
        p = Path(v)
        resolved[k] = p if p.is_absolute() else REPO_ROOT / p
    resolved["chunks"] = REPO_ROOT / CHUNKS_REL
    resolved["p2_main"] = REPO_ROOT / P2_MAIN_REL
    return resolved


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"JSON 解析失败 {path}: {e}")


def load_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    die(f"JSONL 第 {i} 行解析失败 {path}: {e}")
    except FileNotFoundError:
        die(f"文件不存在: {path}")
    return rows


def atomic_write_json(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ============================================================
# 预检
# ============================================================

def check_no_concurrent_p2() -> None:
    """存在其他 code_p2_main.py 进程时拒绝执行 (防止两个进程交替写输出)。"""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "code_p2_main.py"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        logger.warning("pgrep 不可用, 跳过并发检查")
        return
    if r.returncode == 0 and r.stdout.strip():
        die(
            "检测到正在运行的 code_p2_main.py 进程 "
            f"(pid: {', '.join(r.stdout.split())}), 拒绝执行。"
            "请等它结束或确认无并发 pipeline 后重试。"
        )


def resolve_api_key() -> str:
    env_name = "OPENCODE_ZEN_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if key:
        logger.info("API key: 使用环境变量 OPENCODE_ZEN_API_KEY")
        return key
    auth = Path.home() / ".local/share/opencode/auth.json"
    try:
        data = json.loads(auth.read_text(encoding="utf-8"))
        key = (data.get("opencode-go", {}) or {}).get("key", "").strip()
    except Exception:
        key = ""
    if not key:
        die(
            "无法获取 API key: 环境变量 OPENCODE_ZEN_API_KEY 未设置, "
            f"且 {auth} 中无 opencode-go key。"
        )
    logger.info(f"API key: 从 {auth} 读取 opencode-go key")
    return key


def chunk_to_session(chunk_id: str) -> str:
    """ses_xxx_c12 -> ses_xxx (chunk_id 以 _c<数字> 结尾)"""
    if "_c" in chunk_id:
        head, _, tail = chunk_id.rpartition("_c")
        if tail.isdigit():
            return head
    return chunk_id


# ============================================================
# 计划构建
# ============================================================

def build_plan(args, paths: dict) -> dict:
    """根据 --session/--chunk/--auto-bad 构建手术计划。返回:
    {
      "session_mode": {sid: {...}},   # 整 session 移除 checkpoint
      "chunk_mode":   {sid: {"chunks": [...], "batches": [batch_id...]}},
      "escalated":    {sid: [chunk_ids]},  # chunk 目标但单 batch, 升级为整 session
      "bad_chunks":   {chunk_id: summary},
      "missing_sessions": [...],      # 不在 checkpoint (本就 pending)
    }
    """
    ckpt = load_json(paths["checkpoint"])
    if not isinstance(ckpt, dict):
        die("checkpoint 顶层不是 dict, 格式异常")
    summaries = {r["chunk_id"]: r.get("summary", "")
                 for r in load_jsonl(paths["summaries"])}
    chunks_rows = load_jsonl(paths["chunks"])
    known_chunks = {c["chunk_id"] for c in chunks_rows}
    known_sessions = {c["session_id"] for c in chunks_rows}

    plan = {
        "session_mode": {},
        "chunk_mode": {},
        "escalated": {},
        "bad_chunks": {},
        "missing_sessions": [],
    }

    target_sessions = set(args.session or [])
    target_chunks = list(args.chunk or [])

    if args.auto_bad:
        for cid, s in summaries.items():
            if args.pattern in (s or ""):
                plan["bad_chunks"][cid] = s
                target_sessions.add(chunk_to_session(cid))
        logger.info(
            f"--auto-bad: 命中 {len(plan['bad_chunks'])} 条坏 summary "
            f"(pattern={args.pattern!r}), 涉及 {len(target_sessions)} 个 session"
        )

    # 校验目标 session 存在性
    for sid in sorted(target_sessions):
        if sid not in known_sessions:
            die(f"session 不存在于 chunks.jsonl: {sid}")

    # 校验目标 chunk 存在性并归类
    chunk_targets_by_session: dict[str, list] = {}
    for cid in target_chunks:
        if cid not in known_chunks:
            die(f"chunk 不存在于 chunks.jsonl: {cid}")
        chunk_targets_by_session.setdefault(
            chunk_to_session(cid), []
        ).append(cid)

    for sid in sorted(target_sessions):
        plan["session_mode"][sid] = {"in_checkpoint": sid in ckpt}
        if sid not in ckpt:
            plan["missing_sessions"].append(sid)

    for sid, cids in sorted(chunk_targets_by_session.items()):
        if sid in plan["session_mode"]:
            logger.info(f"{sid}: 已按 --session 整体重跑, 忽略其 --chunk 目标")
            continue
        rec = ckpt.get(sid)
        if not rec or rec.get("split_mode") != "multi":
            plan["escalated"][sid] = cids
            plan["session_mode"][sid] = {"in_checkpoint": sid in ckpt}
            if sid not in ckpt:
                plan["missing_sessions"].append(sid)
            logger.info(
                f"{sid}: 单 batch session, --chunk 目标 {cids} "
                "自动升级为整 session 重跑"
            )
            continue
        # 多 batch: 找出含目标 chunk 的 batch (chunk_summaries key 即 batch 成员)
        hit_batches = []
        for b in rec.get("batches", []):
            members = set((b.get("chunk_summaries") or {}).keys())
            if members & set(cids):
                hit_batches.append(b.get("batch_id"))
        unmatched = [
            c for c in cids
            if not any(
                c in (b.get("chunk_summaries") or {})
                for b in rec.get("batches", [])
            )
        ]
        if unmatched:
            die(f"{sid}: chunk 未出现在任何 checkpoint batch 中: {unmatched}")
        plan["chunk_mode"][sid] = {
            "chunks": cids,
            "batches": sorted(hit_batches),
            "batch_count": len(rec.get("batches", [])),
        }

    total = (
        len(plan["session_mode"]) + len(plan["chunk_mode"])
    )
    if total == 0:
        logger.info("无需处理: 未命中任何目标")
        if args.auto_bad:
            logger.info("没有发现坏 summary, 数据是干净的")
        sys.exit(0)
    return plan


def print_plan(plan: dict, paths: dict, backup_dir: Path) -> None:
    logger.info("=" * 60)
    logger.info("手术计划 (dry-run 不写文件; 加 --yes 执行)")
    logger.info("=" * 60)
    for sid, info in sorted(plan["session_mode"].items()):
        state = "在 checkpoint 中 → 将移除 key" if info["in_checkpoint"] \
            else "本就不在 checkpoint (已 pending, 无需手术)"
        logger.info(f"  [整 session 重跑] {sid}: {state}")
    for sid, info in sorted(plan["chunk_mode"].items()):
        logger.info(
            f"  [batch 级重跑] {sid}: 翻转 batch {info['batches']} "
            f"(共 {info['batch_count']} 个 batch), "
            f"目标 chunk: {info['chunks']}"
        )
    for sid, cids in sorted(plan["escalated"].items()):
        logger.info(f"  [升级说明] {sid}: 单 batch, --chunk {cids} → 整 session")
    if plan["bad_chunks"]:
        logger.info(f"  坏 summary {len(plan['bad_chunks'])} 条:")
        for cid, s in sorted(plan["bad_chunks"].items()):
            logger.info(f"    {cid}: {s[:48]}")
    logger.info(f"  备份目录 (--yes 时创建): {backup_dir}")
    logger.info(
        "  执行流程: 备份三件套 → checkpoint 手术 (原子写) → "
        "python3 src/code_p2_main.py 续跑 → 自动验证 + 新旧对比"
    )


# ============================================================
# 手术 + 运行 + 验证
# ============================================================

def backup_triple(paths: dict, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint", "tasks", "summaries"):
        src = paths[name]
        if not src.exists():
            die(f"备份失败, 文件不存在: {src}")
        shutil.copy2(src, backup_dir / src.name)
    logger.info(f"三件套已备份 → {backup_dir}")


def save_baseline(paths: dict, plan: dict, backup_dir: Path) -> dict:
    """存档目标 session 的旧 tasks/summaries, 供事后对比。"""
    target_sessions = set(plan["session_mode"]) | set(plan["chunk_mode"])
    tasks = [t for t in load_jsonl(paths["tasks"])
             if t.get("session_id") in target_sessions]
    summaries = {r["chunk_id"]: r.get("summary", "")
                 for r in load_jsonl(paths["summaries"])
                 if chunk_to_session(r["chunk_id"]) in target_sessions}
    baseline = {"tasks": tasks, "summaries": summaries}
    (backup_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return baseline


def apply_surgery(paths: dict, plan: dict) -> None:
    ckpt = load_json(paths["checkpoint"])
    removed, flipped = [], []
    for sid in plan["session_mode"]:
        if ckpt.pop(sid, None) is not None:
            removed.append(sid)
    for sid, info in plan["chunk_mode"].items():
        rec = ckpt[sid]
        for b in rec.get("batches", []):
            if b.get("batch_id") in info["batches"]:
                if b.get("status") == "success":
                    b["status"] = "rerun_requested"
                    flipped.append(f"{sid}#batch{b.get('batch_id')}")
    atomic_write_json(paths["checkpoint"], ckpt)
    # 复读校验
    verify = load_json(paths["checkpoint"])
    for sid in removed:
        if sid in verify:
            die(f"手术后校验失败: {sid} 仍在 checkpoint 中")
    logger.info(
        f"checkpoint 手术完成: 移除 {len(removed)} session, "
        f"翻转 {len(flipped)} batch {flipped if flipped else ''}"
    )


def run_p2(paths: dict, api_key: str, backup_dir: Path) -> None:
    log_path = backup_dir / "p2_run.log"
    cmd = [sys.executable, str(paths["p2_main"])]
    env = dict(os.environ)
    env["OPENCODE_ZEN_API_KEY"] = api_key
    logger.info(f"启动 P2 重跑: {' '.join(cmd)} (日志: {log_path})")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            lf.write(line)
            logger.info(line.rstrip())
        proc.wait()
    elapsed = time.time() - start
    if proc.returncode != 0:
        die(
            f"code_p2_main.py 退出码 {proc.returncode} (耗时 {elapsed:.1f}s)。"
            f" 日志: {log_path}。如需还原: "
            f"python3 src/code_p2_rerun.py --rollback {backup_dir}",
            code=2,
        )
    logger.info(f"P2 重跑完成, 耗时 {elapsed:.1f}s")


def verify_after(paths: dict, plan: dict, baseline: dict,
                 backup_dir: Path, pattern: str = DEFAULT_BAD_PATTERN) -> None:
    problems = []
    ckpt = load_json(paths["checkpoint"])
    summaries = {r["chunk_id"]: r.get("summary", "")
                 for r in load_jsonl(paths["summaries"])}
    tasks = load_jsonl(paths["tasks"])
    chunks_rows = load_jsonl(paths["chunks"])

    # 1. checkpoint: 目标 session 重新落回且状态完整
    for sid in plan["session_mode"]:
        if sid not in ckpt:
            problems.append(f"{sid}: 重跑后仍不在 checkpoint")
    for sid, info in plan["chunk_mode"].items():
        rec = ckpt.get(sid)
        if not rec:
            problems.append(f"{sid}: 重跑后不在 checkpoint")
            continue
        bad = [b.get("batch_id") for b in rec.get("batches", [])
               if b.get("status") != "success"]
        if bad:
            problems.append(f"{sid}: batch {bad} 状态非 success")

    # 2. 唯一性
    sids = [r["chunk_id"] for r in load_jsonl(paths["summaries"])]
    if len(sids) != len(set(sids)):
        problems.append("summaries 存在重复 chunk_id")
    tids = [t.get("task_id") for t in tasks]
    if len(tids) != len(set(tids)):
        problems.append("tasks 存在重复 task_id")

    # 3. 覆盖完整性: 每个 chunk 都有 summary
    known = {c["chunk_id"] for c in chunks_rows}
    if set(summaries) != known:
        diff = known - set(summaries)
        problems.append(f"{len(diff)} 个 chunk 缺 summary, 如 {sorted(diff)[:3]}")

    # 4. 坏样本清零 (目标范围内)
    target_sessions = set(plan["session_mode"]) | set(plan["chunk_mode"])
    remain = [c for c, s in summaries.items()
              if chunk_to_session(c) in target_sessions
              and pattern in (s or "")]
    if remain:
        problems.append(f"目标 session 仍有坏 summary: {remain}")

    # 5. 新旧对比
    changed, unchanged = [], []
    for cid, old in baseline["summaries"].items():
        new = summaries.get(cid)
        if new is None:
            problems.append(f"{cid}: 重跑后 summary 丢失")
        elif new != old:
            changed.append(cid)
        else:
            unchanged.append(cid)
    logger.info(
        f"对比基线: {len(changed)} 条 summary 变化, "
        f"{len(unchanged)} 条未变 (chunk 级模式下未变=所属 batch 被复用)"
    )
    for cid in changed[:10]:
        logger.info(
            f"  {cid}: {baseline['summaries'][cid][:36]} → "
            f"{summaries[cid][:36]}"
        )
    if len(changed) > 10:
        logger.info(f"  ... 其余 {len(changed) - 10} 条略")

    global_bad = [c for c, s in summaries.items()
                  if pattern in (s or "")]
    logger.info(f"全局坏 summary (pattern={pattern!r}): {len(global_bad)}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": {
            "session_mode": sorted(plan["session_mode"]),
            "chunk_mode": {k: v["batches"] for k, v in plan["chunk_mode"].items()},
        },
        "changed_summaries": changed,
        "unchanged_summaries": unchanged,
        "global_bad_remaining": global_bad,
        "problems": problems,
    }
    (backup_dir / "rerun_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if problems:
        die(
            "验证发现问题:\n  - " + "\n  - ".join(problems)
            + f"\n报告: {backup_dir / 'rerun_report.json'}",
            code=2,
        )
    logger.info(f"全部验证通过, 报告: {backup_dir / 'rerun_report.json'}")


# ============================================================
# 回滚
# ============================================================

def do_rollback(backup_dir: Path, paths: dict) -> None:
    check_no_concurrent_p2()
    need = {"checkpoint": ".p2_checkpoint.json",
            "tasks": "tasks.jsonl",
            "summaries": "chunks_summary_p2.jsonl"}
    for key, name in need.items():
        f = backup_dir / name
        if not f.exists():
            die(f"备份目录缺少 {name}: {backup_dir}")
        if key == "checkpoint":
            load_json(f)
        else:
            load_jsonl(f)
    # 先备份当前状态再覆盖 (回滚也是破坏性操作)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pre_rollback = backup_dir.parent / f"pre_rollback_{ts}"
    backup_triple(paths, pre_rollback)
    for key, name in need.items():
        shutil.copy2(backup_dir / name, paths[key])
    # 复读校验
    load_json(paths["checkpoint"])
    load_jsonl(paths["tasks"])
    load_jsonl(paths["summaries"])
    logger.info(f"回滚完成: 三件套已恢复自 {backup_dir}")
    logger.info(f"回滚前的状态已备份至 {pre_rollback}")


# ============================================================
# main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P2 定点重跑: checkpoint 手术 + 续跑 + 自动验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 src/code_p2_rerun.py --auto-bad           # 干跑查看坏样本\n"
            "  python3 src/code_p2_rerun.py --auto-bad --yes     # 真正修复\n"
            "  python3 src/code_p2_rerun.py --session ses_x --yes\n"
            "  python3 src/code_p2_rerun.py --chunk ses_x_c5 --yes\n"
            "  python3 src/code_p2_rerun.py --rollback <backup_dir>\n"
        ),
    )
    parser.add_argument("--session", action="append",
                        help="整 session 重跑 (可重复)")
    parser.add_argument("--chunk", action="append",
                        help="只重跑含该 chunk 的 batch; 单 batch session "
                             "自动升级为整 session (可重复)")
    parser.add_argument("--auto-bad", action="store_true",
                        help="自动选择 summary 含坏模式的 session")
    parser.add_argument("--pattern", default=DEFAULT_BAD_PATTERN,
                        help=f"坏 summary 匹配串 (默认 {DEFAULT_BAD_PATTERN!r})")
    parser.add_argument("--yes", action="store_true",
                        help="真正执行 (默认 dry-run 只输出计划)")
    parser.add_argument("--backup-dir", default=None,
                        help="备份目录 (默认 output/p2_rerun_backup_<时间戳>)")
    parser.add_argument("--rollback", default=None, metavar="BACKUP_DIR",
                        help="从指定备份目录恢复三件套后退出")
    args = parser.parse_args()

    paths = resolve_output_paths()
    for key in ("checkpoint", "tasks", "summaries", "chunks"):
        if not paths[key].exists():
            die(f"文件不存在: {paths[key]} (请先跑完 P1/P2)")

    if args.rollback:
        do_rollback(Path(args.rollback).expanduser().resolve(), paths)
        return

    if not (args.session or args.chunk or args.auto_bad):
        parser.print_help()
        die("请指定 --session / --chunk / --auto-bad 之一", code=1)

    check_no_concurrent_p2()
    plan = build_plan(args, paths)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        Path(args.backup_dir).expanduser().resolve()
        if args.backup_dir
        else REPO_ROOT / "output" / f"p2_rerun_backup_{ts}"
    )
    print_plan(plan, paths, backup_dir)

    if not args.yes:
        logger.info("dry-run 结束, 未修改任何文件。确认无误后加 --yes 执行。")
        return

    api_key = resolve_api_key()
    backup_triple(paths, backup_dir)
    baseline = save_baseline(paths, plan, backup_dir)
    apply_surgery(paths, plan)
    run_p2(paths, api_key, backup_dir)
    verify_after(paths, plan, baseline, backup_dir, pattern=args.pattern)
    logger.info("定点重跑完成。")


if __name__ == "__main__":
    main()
