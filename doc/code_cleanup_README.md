# code_cleanup: Pipeline 输出清理工具

## 概述

`src/code_cleanup.py` 是一个独立的命令行工具,按 phase 选择性删除 `output/`、`qdrant_data/`、`logs/` 下的 pipeline 产物。核心安全机制是**默认 dry-run**:不加 `--yes` 一律只打印删除计划,不动任何文件。

设计目标:重跑 pipeline 前快速清空指定阶段的产物,避免手动 `rm` 漏删 checkpoint 或误删上游数据。

## 文件结构

```
src/
  code_cleanup.py             # 本文档对应的工具 (list + clean 两个子命令)
config/
  code_p1_config.yaml         # 路径来源: incremental.checkpoint, logging.file
  code_p2_config.yaml         # 路径来源: phase1.chunks_file, output.*, logging.file
  code_p3_config.yaml         # 路径来源: phase2.tasks_file, qdrant.path, logging.file
  code_p5_config.yaml         # 路径来源: knowledge_graph.*_output_dir, logging.file
doc/
  code_cleanup_README.md      # 本文档
```

不新增任何配置文件;配置缺失时使用内置默认路径兜底。

## 用法

### list: 列出所有产物

```bash
python3 src/code_cleanup.py list
```

输出示例(按 P1→P2→P3→P5→logs 排序):

```
PHASE  EXISTS       SIZE  PATH
------------------------------------------------------------------------
p1     yes        1.1 MB  output/chunks.jsonl
p1     yes       19.3 KB  output/.p1_checkpoint.json
p2     yes       72.8 KB  output/tasks.jsonl
p2     yes       62.3 KB  output/chunks_summary_p2.jsonl
p2     yes       79.0 KB  output/.p2_checkpoint.json
p3     yes        3.0 MB  qdrant_data
p5     no              -  output/triple
p5     no              -  output/entity
p5     no              -  qdrant_data/collection/entities
logs   yes        3.7 KB  logs/phase1.log
...
```

### clean: 按 phase 删除

```bash
# 默认 dry-run: 只打印计划, 不删除
python3 src/code_cleanup.py clean --p2

# 显式 dry-run (与默认行为一致, 仅作明示)
python3 src/code_cleanup.py clean --p2 --dry-run

# 真删 (跳过确认)
python3 src/code_cleanup.py clean --p2 --yes

# 组合选择
python3 src/code_cleanup.py clean --p1 --p2 --yes

# 全清 (所有 phase, 含 checkpoint 与 Qdrant 向量库)
python3 src/code_cleanup.py clean --all --yes

# 级联删除下游
python3 src/code_cleanup.py clean --p1 --cascade --yes
```

flags 一览:

| flag | 作用 |
|------|------|
| `--p1` / `--p2` / `--p3` / `--p5` / `--logs` | 选择要清理的 phase, 可组合 |
| `--all` | 选择全部 phase |
| `-n` / `--dry-run` | 显式干跑: 只打印计划不删除 (与默认行为一致) |
| `-y` / `--yes` | 真实删除; **不加此 flag 一律 dry-run** |
| `--cascade` | 级联删除下游依赖产物 (默认关闭) |

`--dry-run` 与 `--yes` 互斥, 同时给出报错退出 (exit 2)。

## Phase 产物清单

| phase | 目标 | 类型 | 路径来源 |
|-------|------|------|----------|
| p1 | chunks 产物 | file | `code_p2_config.yaml: phase1.chunks_file` → `output/chunks.jsonl` |
| p1 | 增量 checkpoint | file | `code_p1_config.yaml: incremental.checkpoint` → `output/.p1_checkpoint.json` |
| p2 | task 产物 | file | `code_p3_config.yaml: phase2.tasks_file` → `output/tasks.jsonl` |
| p2 | chunk 总结 | file | `code_p2_config.yaml: output.chunk_summaries` → `output/chunks_summary_p2.jsonl` |
| p2 | 增量 checkpoint | file | `code_p2_config.yaml: output.checkpoint` → `output/.p2_checkpoint.json` |
| p3 | Qdrant 向量库 | dir | `code_p3_config.yaml: qdrant.path` → `qdrant_data/` |
| p5 | triple 模式产物 | dir | `code_p5_config.yaml: knowledge_graph.triple_output_dir` → `output/triple/` |
| p5 | entity 模式产物 | dir | `code_p5_config.yaml: knowledge_graph.entity_output_dir` → `output/entity/` |
| p5 | 实体对齐 Qdrant 集合 | dir | `code_p5_config.yaml: qdrant.path` + `qdrant.entities_collection` → `qdrant_data/collection/entities/` |
| logs | phase1/2/3/5 日志 | file | 各 config 的 `logging.file` |
| logs | entity 模式日志 | file | 内置默认 `logs/phase5_entity.log` |

说明:

- checkpoint 与 phase **强绑**: checkpoint 注册为对应 phase 的目标,清 P1 必带 `.p1_checkpoint.json`,清 P2 必带 `.p2_checkpoint.json`,`--all` 一并清。
- `--p5` 同时清 triple 与 entity 两个子目录,以及 P5 实体对齐复用的 `qdrant_data/collection/entities` 集合(不碰 P3 的 tasks/chunks_summary/chunks_cleaned_text 三个集合),与 `knowledge_graph.extraction_mode` 无关。
- 相对路径统一按 repo root 解析,工具可在任意 CWD 下运行。
- `tests/` 下的 benchmark 副产物不在清理范围内。

## cascade 行为

依赖图(`_PHASE_DEPS`,方向为"上游依赖"):

```
p1 ← p2 ← p3
      └── p5
logs (独立, 无依赖关系)
```

- `--cascade` 开启: 选中 phase 沿依赖图**向下游**做传递闭包扩展。只向下,不向上。
  - `clean --p1 --cascade --yes` → 删 p1+p2+p3+p5
  - `clean --p2 --cascade --yes` → 删 p2+p3+p5 (不碰 p1)
  - `clean --logs --cascade --yes` → 仅删 logs (无下游)
- `--cascade` 关闭(默认): 只删选中 phase;真删完成后检测仍存在的下游产物,逐个 `WARNING: orphan detected`,并提示加 `--cascade`。

## 安全说明

- **默认 dry-run**: 只有 `--yes` 才真删;`--dry-run` 与 `--yes` 互斥。
- **路径断言**: 删除前断言所有目标路径位于 repo root 之内,配置被改成外部绝对路径时拒绝执行 (exit 1)。
- **幂等**: 重复执行不报错,不存在的目标跳过 (`unlink(missing_ok=True)` / `rmtree(ignore_errors=True)`)。
- **不可逆**: 无 backup/undo/回收站。删除前先跑 dry-run 确认计划。
- 目录大小用 `os.walk` 递归统计,跨平台 (不依赖 GNU `du -b`)。

## 验证记录

实现过程中的 4 种模式验证快照保存在 `.omo/evidence/`(gitignored):

- `task-1-output-cleanup.txt` — list 表格 + `--bogus` 报错
- `task-2-output-cleanup.txt` — dry-run / `--all` / 互斥 / 空选择
- `task-3-output-cleanup.txt` — cascade 扩展 + orphan 检测
- `final-{list,dry-run,yes,cascade}-output-cleanup.txt` — 端到端最终验证

真删验证在备份 (`output/` + `qdrant_data/` + `logs/` 全量副本) 保护下执行,验证后已从备份恢复原数据。
