# code_cleanup: Pipeline 输出清理工具

## 概述

`src/code_cleanup.py` 是一个独立的命令行工具,按 phase 选择性删除 pipeline 产物。**同时支持 embedded 与 server 两种 Qdrant 模式**:

| 模式 | 数据来源 | 触发条件 |
|---|---|---|
| **embedded** | 本地 `qdrant_data/` | 无 `QDRANT_URL` 且 yaml `qdrant.url` 为空 |
| **server** | Qdrant HTTP API (`delete_collection`) | `QDRANT_URL` 环境变量或 yaml `qdrant.url` 非空 |

核心安全机制是**默认 dry-run**:不加 `--yes` 一律只打印删除计划,不动任何文件/collection。

设计目标:重跑 pipeline 前快速清空指定阶段的产物,避免手动 `rm` 漏删 checkpoint 或误删上游数据。

## 文件结构

```
src/
  code_cleanup.py             # 本文档对应的工具 (list + clean 两个子命令)
utils/
  qdrant_snapshot.py          # server mode 调用的 drop_collections() 库函数所在文件
config/
  code_p1_config.yaml         # 路径来源: incremental.checkpoint, logging.file
  code_p2_config.yaml         # 路径来源: phase1.chunks_file, output.*, logging.file
  code_p3_config.yaml         # 路径来源: phase2.tasks_file, qdrant.path/collections, logging.file
  code_p5_config.yaml         # 路径来源: knowledge_graph.*_output_dir, qdrant.entities_collection, logging.file
doc/
  code_cleanup_README.md      # 本文档
```

不新增任何配置文件;配置缺失时使用内置默认路径兜底。

## 用法

### list: 列出所有产物

```bash
python3 src/code_cleanup.py list
```

输出按 P1→P2→P3→P5→logs 排序。embedded mode 只列本地文件;**server mode 末尾额外列出 server 上的 collections + snapshots**(直接查 Qdrant HTTP API,需要 server 可达)。

embedded mode 输出示例(本地 14 个目标):

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

server mode 额外输出(`export QDRANT_URL=http://localhost:6333` 后):

```
[server collections @ http://localhost:6333]
  COLLECTION                    POINTS  VECTORS  STATUS
  ----------------------------------------------------------------
  chunks_cleaned_text             234      234  green
  chunks_summary                  234      234  green
  entities                       1447     1447  green
  tasks                            76       76  green

[server snapshots @ http://localhost:6333]
  COLLECTION                    SNAPSHOT                              SIZE  CREATION
  ------------------------------------------------------------------------------------------
  chunks_cleaned_text            chunks_cleaned_text-2026-09-13-...snapshot  3.1 MB  2026-09-13 14:22:01
  chunks_summary                 chunks_summary-2026-09-13-...snapshot       3.0 MB  2026-09-13 14:22:01
  entities                       entities-2026-09-13-...snapshot            11.2 MB  2026-09-13 14:22:01
  tasks                          tasks-2026-09-13-...snapshot                2.3 MB  2026-09-13 14:22:01

[server 模式] QDRANT_URL=http://localhost:6333
  清理时 P3 / P5 entities 集合由 `code_cleanup.py clean --p3/--p5/--all --yes`
  调 drop_collections() 管理 (含删前 auto-backup)。
```

server 查询失败降级为 warning,不退出(本地列表仍可看)。

### clean: 按 phase 删除

embedded 模式:

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

server 模式(export `QDRANT_URL` 后):

```bash
# dry-run: 列出本地 + server collections 待删除清单 (不删、不 backup)
python3 src/code_cleanup.py clean --p3
python3 src/code_cleanup.py clean --p1 --cascade

# 真删: 删本地文件 + drop_collections() 调 server API 删 collection (删前 auto-backup)
python3 src/code_cleanup.py clean --p3 --yes

# 跳过删前 backup (CI 用)
python3 src/code_cleanup.py clean --p3 --yes --no-backup

# 自定义 backup 输出目录 + 保留历史数
python3 src/code_cleanup.py clean --p3 --yes --backup-out /tmp/qdrant_backups --keep-host-backups 5

# 跳过 'DELETE' 二次确认 (CI 用; 不推荐生产)
python3 src/code_cleanup.py clean --p3 --yes --force
```

flags 一览:

| flag | 作用 |
|------|------|
| `--p1` / `--p2` / `--p3` / `--p5` / `--logs` | 选择要清理的 phase, 可组合 |
| `--all` | 选择全部 phase |
| `-n` / `--dry-run` | 显式干跑: 只打印计划不删除 (与默认行为一致) |
| `-y` / `--yes` | 真实删除; **不加此 flag 一律 dry-run** |
| `--cascade` | 级联删除下游依赖产物 (默认关闭) |
| `--no-backup` | **(server mode)** 跳过删前自动 backup;CI 用 |
| `--backup-out <path>` | **(server mode)** 自定义 backup 输出目录;默认 `./qdrant_snapshots/<timestamp>/` |
| `--keep-host-backups <N>` | **(server mode)** host 端 backup 目录保留最近 N 个;默认 3,0 = 全保留 |
| `--force` | **(server mode)** 跳过 `'DELETE'` 二次确认;用于自动化 |

`--dry-run` 与 `--yes` 互斥, 同时给出报错退出 (exit 2)。

#### server mode 行为细节

- **dry-run 阶段**: 调 `drop_collections(dry_run=True)` 只查 server (`list_collections` + `get_collection_info`),**不创建 snapshot、不删除任何东西**
- **真删阶段**: `--yes` 调 `drop_collections(dry_run=False)` 依次: ① auto-backup ② `_prune_host_backups` 保留最近 N 个 ③ `delete_collection` 调 server API 删除
- **多客户端可见**: server mode 删除影响所有连接此 server 的进程 (多 IDE / MCP / opencode 进程)。**默认要求用户在交互中输入 `DELETE` 二次确认**;`--force` 跳过此确认
- **Phase→Collection 映射**:
  - `--p3` → server collections: `tasks` + `chunks_summary` + `chunks_cleaned_text`
  - `--p5` → server collections: `entities`
  - collection 名从 yaml `qdrant.collections.{tasks,chunks_summary,chunks_cleaned_text}` 与 `qdrant.entities_collection` 读取
- **dry-run 输出格式统一**: 本地文件段(`[本地文件]`) + server collections 段(`[server collections @ url]`) + 单个 dry-run 底部标识,两种模式一致
- **删除原子性**: Qdrant server `delete_collection(name)` 一次 API call 原子清理 storage 子目录 + SQLite 元数据 + 对应 snapshot 子目录,无需手动 rmtree

#### container `/qdrant/` 目录结构

server mode 只动 collection data + snapshot,其他目录(`.qdrant-initialized` sentinel、`storage/audit/` 审计日志) **不碰**:

```
/qdrant/
├── storage/                      # ← delete_collection 自动清 collection 子目录
│   ├── collections.db           # ← delete_collection 同时删该 collection 元数据行
│   └── <collection>/             # ← 每个 collection 一个子目录
├── snapshots/                    # ← delete_collection 自动清 collection 子目录
│   └── <collection>/             # ← 每个 collection 的 snapshot 文件
└── .qdrant-initialized           # ← 永远不删 (删了下次启动重新初始化)
```

`audit/` 子目录(若 config `audit.enabled: true`)由 Qdrant config 控制,本工具不管理。

## Phase 产物清单

不同模式下,P3/P5 entities 的存储位置不同,清理路径也不同:

### embedded mode (本地 `qdrant_data/`)

| phase | 目标 | 类型 | 路径来源 |
|-------|------|------|----------|
| p1 | chunks 产物 | file | `code_p2_config.yaml: phase1.chunks_file` → `output/chunks.jsonl` |
| p1 | 增量 checkpoint | file | `code_p1_config.yaml: incremental.checkpoint` → `output/.p1_checkpoint.json` |
| p2 | task 产物 | file | `code_p3_config.yaml: phase2.tasks_file` → `output/tasks.jsonl` |
| p2 | chunk 总结 | file | `code_p2_config.yaml: output.chunk_summaries` → `output/chunks_summary_p2.jsonl` |
| p2 | 增量 checkpoint | file | `code_p2_config.yaml: output.checkpoint` → `output/.p2_checkpoint.json` |
| p3 | Qdrant 向量库 (tasks/chunks_summary/chunks_cleaned_text) | dir | `code_p3_config.yaml: qdrant.path` → `qdrant_data/` |
| p5 | triple 模式产物 | dir | `code_p5_config.yaml: knowledge_graph.triple_output_dir` → `output/triple/` |
| p5 | entity 模式产物 | dir | `code_p5_config.yaml: knowledge_graph.entity_output_dir` → `output/entity/` |
| p5 | 实体对齐 Qdrant 集合 (entities) | dir | `code_p5_config.yaml: qdrant.path` + `qdrant.entities_collection` → `qdrant_data/collection/entities/` |
| logs | phase1/2/3/5 日志 | file | 各 config 的 `logging.file` |
| logs | entity 模式日志 | file | 内置默认 `logs/phase5_entity.log` |

### server mode (Qdrant HTTP API)

| phase | 目标 | 类型 | 路径/API |
|-------|------|------|----------|
| p1/p2/p5 | 同 embedded | file/dir | 同 embedded,只列本地文件输出 |
| p3 | tasks + chunks_summary + chunks_cleaned_text | HTTP collection | `DELETE /collections/{name}`,name 来自 `code_p3_config.yaml: qdrant.collections.*` |
| p5 | entities | HTTP collection | `DELETE /collections/entities`,name 来自 `code_p5_config.yaml: qdrant.entities_collection` |
| logs | 同 embedded | file | 各 config 的 `logging.file` |

server mode 下 `--p3` **不**清本地任何文件(无本地 `qdrant_data/`),只调 server API 删 3 个 collection;`--p5` 删本地 `output/triple/` + `output/entity/` + server 端 `entities` collection。

说明:

- checkpoint 与 phase **强绑**: checkpoint 注册为对应 phase 的目标,清 P1 必带 `.p1_checkpoint.json`,清 P2 必带 `.p2_checkpoint.json`,`--all` 一并清。
- `--p5` 同时清 triple 与 entity 两个子目录,以及 server 端 `entities` 集合(不碰 P3 的 tasks/chunks_summary/chunks_cleaned_text 三个集合),与 `knowledge_graph.extraction_mode` 无关。
- 相对路径统一按 repo root 解析,工具可在任意 CWD 下运行。
- `tests/` 下的 benchmark 副产物不在清理范围内。
- server mode 删除**不可逆**(同 embedded),删前务必 dry-run。删前 auto-backup(除非 `--no-backup`)提供一层保险,但 backup 恢复需用 `utils/qdrant_snapshot.py restore`。

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
- server mode 下 cascade 跨两种介质: `--p1 --cascade --yes` → 本地清 p1/p2/p5 文件 + server 删 4 个 collections(tasks/chunks_summary/chunks_cleaned_text/entities)

## 安全说明

- **默认 dry-run**: 只有 `--yes` 才真删;`--dry-run` 与 `--yes` 互斥。
- **路径断言**: 删除前断言所有目标路径位于 repo root 之内,配置被改成外部绝对路径时拒绝执行 (exit 1)。
- **幂等**: 重复执行不报错,不存在的目标跳过 (`unlink(missing_ok=True)` / `rmtree(ignore_errors=True)`);server collections 调 API 删不存在的会标 `skip (server 上不存在)`,不报错。
- **不可逆**: 无 backup/undo/回收站。删除前先跑 dry-run 确认计划。
  - embedded mode: 真删前无自动 backup
  - server mode: 真删前默认 auto-backup(`--no-backup` 可关),backup 存 host `./qdrant_snapshots/<timestamp>/`,用 `utils/qdrant_snapshot.py restore --in <dir>` 恢复
- **server mode 二次确认**: 默认要求交互输入 `DELETE` 确认 (防止误删影响多客户端);`--force` 跳过
- **dry-run 不 backup**: dry-run 阶段不调 `delete_collection` 也不创建 server snapshot,只是查询 plan
- 目录大小用 `os.walk` 递归统计,跨平台 (不依赖 GNU `du -b`)。

## 验证记录

实现过程中的 6 种模式验证快照保存在 `.omo/evidence/`(gitignored):

- `task-1-output-cleanup.txt` — embedded list 表格 + `--bogus` 报错
- `task-2-output-cleanup.txt` — embedded dry-run / `--all` / 互斥 / 空选择
- `task-3-output-cleanup.txt` — embedded cascade 扩展 + orphan 检测
- `final-{list,dry-run,yes,cascade}-output-cleanup.txt` — embedded 端到端最终验证
- `final-{list,dry-run,cascade}-cleanup-server.txt` — server mode list + dry-run + cascade 验证

真删验证在备份 (`output/` + `qdrant_data/` + `logs/` 全量副本) 保护下执行,验证后已从备份恢复原数据;server 真删验证使用 `--force` 跳过交互确认。
