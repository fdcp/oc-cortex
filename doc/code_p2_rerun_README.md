# code_p2_rerun.py — P2 定点重跑工具

## 概述

`src/code_p2_rerun.py` 用于对个别 session / chunk 定点重跑 Phase 2（LLM task 提取 + chunk summary），不触碰其余 session 的结果。典型场景：某些 chunk 的 summary 质量有问题（如历史上 `safe_truncate` 丢 return 造成的 "无实质交互内容" 坏样本），或某 session 需要用新 prompt/模型重新提取。

原理：P2 续跑以 `output/.p2_checkpoint.json` 为准，仍在其中且 `content_hash` 一致的 session 视为已完成。本脚本先备份三件套（checkpoint / tasks.jsonl / chunks_summary_p2.jsonl），再做 checkpoint 手术，然后调用 `code_p2_main.py` 普通续跑（不带 `--force`），最后自动验证 + 新旧对比。

## 重跑粒度

| 模式 | 触发 | 手术动作 | 最小重跑单位 |
|---|---|---|---|
| 整 session | `--session SID` / `--auto-bad` | 从 checkpoint 移除该 session key | 整个 session（单 batch session 没有更细粒度） |
| batch 级 | `--chunk CHUNK_ID` | 把含该 chunk 的 batch `status` 翻转为 `rerun_requested` | 单个 batch（仅多 batch session） |

batch 级依赖 P2 自带续跑机制：多 batch session 的 checkpoint 逐 batch 存 `status` / `content_hash` / 结果，续跑按 `(batch_id, content_hash)` 匹配，`status=success` 的 batch 直接复用旧结果，只重跑非 success 的。`--chunk` 指向单 batch session 的 chunk 时自动升级为整 session 重跑（日志有提示）。

`--auto-bad`：扫描 `chunks_summary_p2.jsonl`，把 summary 含 `--pattern`（默认 `无实质`）的 chunk 归到所属 session 一并重跑。**命中为 0 时直接退出，不做任何重跑**。`--session` 则是无条件重跑指定 session，不判断其 chunk 坏没坏。

## 用法

```bash
cd oc_sess_graph

# 干跑（默认）：只输出手术计划，不写任何文件
python3 src/code_p2_rerun.py --auto-bad
python3 src/code_p2_rerun.py --session ses_xxx
python3 src/code_p2_rerun.py --chunk ses_xxx_c51

# 真正执行（备份 → 手术 → 续跑 → 验证）
export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
python3 src/code_p2_rerun.py --auto-bad --yes
python3 src/code_p2_rerun.py --session ses_xxx --yes
python3 src/code_p2_rerun.py --chunk ses_xxx_c51 --yes

# 从备份回滚三件套（回滚前会先备份当前状态）
python3 src/code_p2_rerun.py --rollback output/p2_rerun_backup_<时间戳>
```

API key 未设环境变量时自动读取 `~/.local/share/opencode/auth.json` 的 `opencode-go` key。`--backup-dir` 可自定义备份位置，默认 `output/p2_rerun_backup_<时间戳>/`。

## 执行流程（--yes）

1. 预检：无并发 `code_p2_main.py` 进程；目标 session/chunk 存在于 `chunks.jsonl`；三个输出文件可解析；API key 可获取。
2. 备份三件套 + 存档目标 session 旧 tasks/summaries 基线（`baseline.json`）。
3. checkpoint 手术：temp + `os.replace` 原子写，写后复读校验。
4. subprocess 调 `python3 src/code_p2_main.py`（cwd=仓库根），日志 tee 到备份目录 `p2_run.log`。
5. 验证：checkpoint 目标落回且 batch 全 success；task_id / chunk_id 无重复；全部 chunk 有 summary；目标范围内坏 pattern 清零；与基线对比列出变化 summary。结果写 `rerun_report.json`。
6. 验证失败时退出码 2，打印 `--rollback` 提示；不自动回滚（P2 per-session 原子落盘保证中间状态一致，自动回滚可能丢弃有效部分进展）。

退出码：0 成功 / 1 预检或用法错误 / 2 运行或验证失败。

## 安全设计

- 默认 dry-run，`--yes` 才写文件；干跑经 md5 校验零写入。
- 手术前强制备份；`--rollback` 恢复前也会先备份当前状态（`output/pre_rollback_<时间戳>/`）。
- checkpoint 手术前后都完整解析 JSON，写后复读断言。
- 检测到其他 `code_p2_main.py` 进程即拒绝执行（两个进程交替写输出会互相覆盖）。

## 测试记录（2026-08-04，qwen3.7-plus，实测值）

| 测试 | 内容 | 结果 |
|---|---|---|
| 干跑 | `--auto-bad`（0 坏样本）、`--session`、`--chunk` 两种粒度 | 通过，md5 校验零写入 |
| 失败路径 | 不存在的 session / chunk、无参数 | 均退出码 1，报错清晰 |
| 并发守卫 | 伪造运行中的 code_p2_main.py 进程 | 拒绝执行，退出码 1 |
| E2E 整 session | ses_0c9e4a359（3 chunks）`--yes` | 通过，P2 重跑实测 30.5s，3 条 summary 更新，验证全绿 |
| E2E batch 级 | ses_0870d73a（58 chunks，6 batch）`--chunk ..._c51 --yes` | 首次运行暴露 P2 续跑 bug（见下）；修复后通过，实测 47.6s，仅 batch 5 重跑：14 条 summary 变化 / 44 条逐字节未变（复用），task 19→21，无重复 |
| 回滚 | 用 `--rollback` 恢复被 bug 污染的输出 | 三件套与备份逐字节一致，74 task / 234 summary / 无重复 |

**发现并修复的 P2 bug**（`src/code_p2_main.py`）：多 batch session 部分完成后续跑时，session 进 `pending` 但未从 `done` 移除，carry-over 循环把旧 tasks 承接进 `merged_tasks`，与重跑新 tasks 叠加造成 `tasks.jsonl` 重复 task_id。修复：部分完成分支补 `done.pop(sid, None)`。修复前 E2E batch 级测试产生 93 条 task（19 条重复），修复后同场景 76 条、零重复。

## 注意

- 重跑会重新提取整个目标范围的 summary/task，非坏 chunk 的 summary 也可能措辞变化（整 session 模式）——属正常，语义一致。
- batch 级模式下被复用 batch 的 summary 逐字节不变。
- P2 输出更新后，受影响 chunk 若已入 P3/Qdrant 索引，需重跑 P3 刷新（summary 参与入库）。
- 备份目录在 gitignored 的 `output/` 下，确认无需回滚后可自行清理。
