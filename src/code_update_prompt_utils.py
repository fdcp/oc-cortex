"""
Prompt 工具集:从 md 文件按 yaml 对应表加载 prompt,以及更新对应表。

设计:
- 所有 prompt 常量都定义在 prompts/*.md 里(外置为单一真源)
- config/prompts_table.yaml 维护 name → (file, start_line, end_line) 映射
- 代码通过 load_prompt(name) 加载,模块级缓存
- update_prompt_table() 扫描所有 prompts/*.md,自动生成/更新 yaml

md 格式约束(由更新脚本强制):
- 每个 prompt 块必须用 ``` 围栏包裹
- 块前需要有 (#+) xxx (`CONSTANT_NAME`) 标题,或单块文件用 **Source** 行
- start_line / end_line 指 ``` 围栏的行号(1-indexed,含两端)

CLI:
  python3 src/code_update_prompt_utils.py update [--dry-run]
  python3 src/code_update_prompt_utils.py list
  python3 src/code_update_prompt_utils.py load <NAME>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

# ============================================================
# 路径常量(基于 __file__ 自定位仓库根,兼容任意 cwd)
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = _REPO_ROOT / "config" / "prompts_table.yaml"
PROMPTS_DIR = _REPO_ROOT / "prompts"

# ============================================================
# 内部状态(模块级缓存)
# ============================================================
_TABLE: Optional[dict] = None
_FILE_CACHE: Dict[str, List[str]] = {}
_PROMPT_CACHE: Dict[str, str] = {}


# ============================================================
# 加载器
# ============================================================
def _load_table() -> dict:
    """加载 yaml 对应表(模块级缓存;reload_prompts 时清)。"""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 prompt 对应表: {TABLE_PATH.resolve()}\n"
            f"请先运行: python3 src/code_update_prompt_utils.py update"
        )
    with TABLE_PATH.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {"version": 1, "prompts": []}
    # 索引: name → entry(避免每次 list 查找)
    loaded["_index"] = {p["name"]: p for p in loaded.get("prompts", [])}
    _TABLE = loaded
    return _TABLE


def _read_file_lines(file_path: str) -> List[str]:
    """按文件路径缓存读取整文件行列表。

    yaml 里 file 字段为相对仓库根的路径(如 prompts/xxx.md);
    若已经是绝对路径则原样使用(便于跨环境调试)。
    """
    if file_path not in _FILE_CACHE:
        p = Path(file_path)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"prompt 文件不存在: {p}")
        with p.open("r", encoding="utf-8") as f:
            _FILE_CACHE[file_path] = f.readlines()
    return _FILE_CACHE[file_path]


def load_prompt(name: str) -> str:
    """
    从 md 文件加载指定 prompt,自动去掉外层 ``` 围栏,模块级缓存。

    Args:
        name: prompt 常量名(全局唯一,如 SESSION_TASK_PROMPT)

    Returns:
        prompt 文本(末尾保留单个 \\n)

    Raises:
        KeyError: name 不在对应表里
        FileNotFoundError: yaml 或 md 文件不存在
        ValueError: 行号越界或围栏缺失
    """
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    table = _load_table()
    if name not in table["_index"]:
        raise KeyError(
            f"prompt '{name}' 不在对应表 {TABLE_PATH} 中。\n"
            f"已知: {sorted(table['_index'].keys())}"
        )
    entry = table["_index"][name]
    lines = _read_file_lines(entry["file"])

    start = entry["start_line"] - 1  # 1-indexed → 0-indexed 切片下界
    end = entry["end_line"]          # 切片上界(不含)
    if start < 0 or end > len(lines) or start >= end:
        raise ValueError(
            f"prompt '{name}': 行号范围 [{entry['start_line']}, {entry['end_line']}] "
            f"无效(文件共 {len(lines)} 行)"
        )

    block = lines[start:end]
    # 去掉首尾 ``` 围栏
    if not block or block[0].strip() != "```":
        raise ValueError(
            f"prompt '{name}': 第 {entry['start_line']} 行应为 ``` 围栏起始,实际: {block[0]!r}"
        )
    if block[-1].strip() != "```":
        raise ValueError(
            f"prompt '{name}': 第 {entry['end_line']} 行应为 ``` 围栏结束,实际: {block[-1]!r}"
        )
    content = "".join(block[1:-1]).rstrip() + "\n"

    _PROMPT_CACHE[name] = content
    return content


def reload_prompts() -> None:
    """清空所有缓存,下次 load_prompt 重新读盘(开发调试用)。"""
    global _TABLE
    _TABLE = None
    _FILE_CACHE.clear()
    _PROMPT_CACHE.clear()


# ============================================================
# 更新器
# ============================================================
# 标题: #/##/###/#### ... (`CONSTANT_NAME`)
_HEADING_RE = re.compile(r"^(#{1,6})\s+.*?\(`([A-Z][A-Z0-9_]+)`\)")
# Source 行: **Source**: ... (`CONSTANT_NAME`)
_SOURCE_RE = re.compile(r"\*\*Source\*\*:.*?\(`([A-Z][A-Z0-9_]+)`\)")
# ``` 围栏(允许前导空白)
_FENCE_RE = re.compile(r"^\s*```\s*$")


def _scan_prompts_in_md(md_path: Path) -> List[dict]:
    """
    扫描单个 md,返回 [{name, file, start_line, end_line}, ...]。

    规则:
    - 所有 prompt 块必须用 ``` 围栏包裹
    - 多块文件:每个块前有 (#+) xxx (`NAME`) 标题,常量名取自标题
    - 单块文件:常量名取自 **Source**: ... (`NAME`) 行
    """
    with md_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # 找所有 ``` 围栏的起止行(1-indexed)
    fence_lines = [i + 1 for i, line in enumerate(lines) if _FENCE_RE.match(line)]
    if not fence_lines:
        return []
    if len(fence_lines) % 2 != 0:
        logger.warning(f"{md_path}: ``` 围栏不成对,跳过")
        return []

    blocks = []
    for i in range(0, len(fence_lines), 2):
        start_line = fence_lines[i]      # ``` 开始行
        end_line = fence_lines[i + 1]    # ``` 结束行

        # 向上找常量名:优先标题,其次 Source 行
        name = None
        for j in range(start_line - 2, -1, -1):  # 从围栏上一行往上
            heading = _HEADING_RE.match(lines[j])
            if heading:
                name = heading.group(2)
                break
        if name is None:
            for j in range(start_line - 2, -1, -1):
                source = _SOURCE_RE.search(lines[j])
                if source:
                    name = source.group(1)
                    break
        if name is None:
            logger.warning(
                f"{md_path}: 第 {start_line}-{end_line} 行围栏块未找到常量名"
                f"(上方既无 `(#+) xxx (`NAME`)` 标题也无 `**Source**: ... (`NAME`)` 行),跳过"
            )
            continue

        blocks.append({
            "name": name,
            "file": str(md_path.relative_to(_REPO_ROOT)),
            "start_line": start_line,
            "end_line": end_line,
        })
    return blocks


def _scan_all() -> List[dict]:
    """扫描 prompts/ 下所有 .md,返回所有 prompt 条目(按 name 排序)。"""
    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(f"prompts 目录不存在: {PROMPTS_DIR}")
    all_blocks: List[dict] = []
    for md_path in sorted(PROMPTS_DIR.glob("*.md")):
        all_blocks.extend(_scan_prompts_in_md(md_path))
    all_blocks.sort(key=lambda b: b["name"])
    return all_blocks


def update_prompt_table(dry_run: bool = False) -> dict:
    """
    重新扫描 prompts/*.md,更新 config/prompts_table.yaml。

    Returns:
        diff 统计: {"added": [...], "updated": [...], "missing": [...], "total": int}
        - added: 新出现在 md 中
        - updated: 已在 yaml 中但行号变了
        - missing: 在 yaml 中但 md 中已找不到(会保留在 yaml 里但告警)
    """
    new_blocks = _scan_all()
    new_index = {b["name"]: b for b in new_blocks}

    if TABLE_PATH.exists():
        with TABLE_PATH.open("r", encoding="utf-8") as f:
            old = yaml.safe_load(f) or {"version": 1, "prompts": []}
    else:
        old = {"version": 1, "prompts": []}
    old_index = {p["name"]: p for p in old.get("prompts", [])}

    added = [n for n in new_index if n not in old_index]
    updated = [
        n for n in new_index
        if n in old_index and (
            old_index[n].get("file") != new_index[n]["file"]
            or old_index[n].get("start_line") != new_index[n]["start_line"]
            or old_index[n].get("end_line") != new_index[n]["end_line"]
        )
    ]
    missing = [n for n in old_index if n not in new_index]

    diff = {
        "added": added,
        "updated": updated,
        "missing": missing,
        "total": len(new_blocks),
    }

    if dry_run:
        return diff

    out = {"version": old.get("version", 1), "prompts": new_blocks}
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            out, f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    # 清缓存,确保下次 load_prompt 用新行号
    reload_prompts()
    return diff


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Prompt 工具:加载和更新对应表")
    sub = parser.add_subparsers(dest="cmd")

    p_update = sub.add_parser("update", help="扫描 prompts/*.md 并更新 yaml")
    p_update.add_argument("--dry-run", action="store_true", help="只看不写")

    p_load = sub.add_parser("load", help="加载并打印指定 prompt(用于调试)")
    p_load.add_argument("name")

    p_list = sub.add_parser("list", help="列出当前 yaml 中所有 prompt")

    args = parser.parse_args()

    if args.cmd == "update":
        diff = update_prompt_table(dry_run=args.dry_run)
        print(f"扫描结果: 共 {diff['total']} 个 prompt")
        if diff["added"]:
            print(f"  + 新增 ({len(diff['added'])}): {diff['added']}")
        if diff["updated"]:
            print(f"  ~ 更新 ({len(diff['updated'])}): {diff['updated']}")
        if diff["missing"]:
            print(f"  ! 警告(在 yaml 中但 md 中已找不到) ({len(diff['missing'])}): {diff['missing']}")
        if not any([diff["added"], diff["updated"], diff["missing"]]):
            print("  无变化")
        if args.dry_run:
            print("(dry-run,未写入)")

    elif args.cmd == "load":
        try:
            print(load_prompt(args.name), end="")
        except (KeyError, FileNotFoundError, ValueError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "list":
        if not TABLE_PATH.exists():
            print(f"对应表不存在: {TABLE_PATH},请先跑: python3 src/code_update_prompt_utils.py update")
            return
        with TABLE_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name_w = max((len(p["name"]) for p in data.get("prompts", [])), default=20)
        file_w = max((len(p["file"]) for p in data.get("prompts", [])), default=20)
        for p in data.get("prompts", []):
            print(
                f"{p['name']:<{name_w}}  {p['file']:<{file_w}}  "
                f"lines {p['start_line']:>4}-{p['end_line']:<4}"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
