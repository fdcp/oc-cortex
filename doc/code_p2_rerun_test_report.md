# code_p2_rerun.py 测试报告

日期: 2026-08-04 · 模型: qwen3.7-plus (opencode.ai/zen/go/v1) · 状态: 全部通过，过程中发现并修复 1 个 P2 真 bug

## 1. 被测对象

`src/code_p2_rerun.py`（P2 定点重跑工具）：默认 dry-run、`--yes` 执行、`--rollback` 回滚；支持 `--session`（整 session）、`--chunk`（多 batch session 的 batch 级）、`--auto-bad`（按坏 pattern 自动选，0 命中直接退出）三种目标选择。设计文档见 `doc/code_p2_rerun_README.md`。

## 2. 测试环境

- macOS (arm64)，Python 3.10，仓库 oc_sess_graph @ cedc0c5（测试开始时的 HEAD）
- 数据规模: 26 sessions / 234 chunks / 74 tasks（修复完成态）
- LLM: qwen3.7-plus, concurrency=4, batch_size=16
- 所有耗时均为实际测量值

## 3. 测试矩阵与结果

### 3.1 干跑与失败路径（不调 LLM）

| 用例 | 操作 | 结果 |
|---|---|---|
| T2 auto-bad 空命中 | `--auto-bad`（数据已无坏样本） | 报 "命中 0 条…数据是干净的"，exit=0，不重跑 |
| T3 整 session 计划 | `--session ses_0c9e4a359…` | 正确显示"在 checkpoint 中 → 将移除 key" |
| T4a 不存在 session | `--session ses_nonexistent_123` | exit=1，报错清晰 |
| T4b 不存在 chunk | `--chunk ses_nonexistent_c9` | exit=1 |
| T4c 无参数 | 裸跑 | exit=1，打印 help |
| T5 单 batch 升级 | `--chunk`（目标属单 batch session） | 正确升级为整 session 并打提示 |
| T6 多 batch 定位 | `--chunk ses_0870d73a…_c51` | 正确定位 batch [5]（共 6 batch） |
| 零写入校验 | 全部干跑前后 md5 三件套 | 逐字节一致 |

### 3.2 并发守卫

伪造运行中的 `code_p2_main.py` 进程（`python3 -c "sleep 20" code_p2_main.py`）→ 脚本检测到 pid 并拒绝执行，exit=1。

### 3.3 E2E-1 整 session 模式（真实 LLM）

目标 `ses_0c9e4a359ffeGDSrVSG2jNVB6n`（3 chunks）。流程完整走通：备份三件套+baseline → 手术移除 key → P2 续跑（增量: 已完成 25 / 待处理 1）→ 验证。实测 P2 重跑 **30.5s**（脚本全程 31.6s）。结果：3 条 summary 更新（语义一致、措辞变化），task 1→1，全局坏样本 0，exit=0。

### 3.4 E2E-2 batch 级模式（真实 LLM）——首次运行暴露 P2 bug

目标 `ses_0870d73a7ffehev1FO7HA2HSze`（58 chunks，6 batch）的 c51（属 batch 5）。

**首次运行（修复前）**：机制本身正确——只重跑 batch 5，实测 **60.3s**，14 条 summary 变化 / 44 条逐字节未变（5 个 batch 被复用）。但验证器发现 `tasks.jsonl` 出现 19 条重复 task_id（74 → 93 条，该 session 的 T1..T19 完整出现两份），脚本报 exit=2 阻止了成功上报。

**根因**：`code_p2_main.py` 续跑逻辑中，多 batch session 存在非 success batch 时进入 pending 重跑，但**没有从 done 中移除**。随后 carry-over 循环把该 session 的旧 tasks 从磁盘承接进 merged_tasks，`_save_multi_session` 又追加重跑合并的新 tasks → 重复。日志证据："增量: 已完成 26 / 待处理 1"——同一 session 同时在 done 和 pending。此 bug 此前从未触发：以往的 batch 失败都在同一次运行内被内容重试消化，从未形成跨运行的部分完成态。

**修复**：部分完成分支补 `done.pop(sid, None)`（`src/code_p2_main.py`）。

**回滚验证**：用脚本自身 `--rollback output/p2_rerun_backup_20260804_110043` 恢复污染前状态，三件套与备份逐字节一致（74 task / 234 summary / 无重复 / 6 batch 全 success）；回滚前的污染态自动快照至 `pre_rollback_20260804_112554/`。

**修复后复跑**：增量日志变为"已完成 25 / 待处理 1"，实测 **47.6s**，14 条 summary 变化 / 44 条未变，task 74 → 76（旧 19 被新 21 替换，重跑 batch 划分略变细），重复 0，坏样本 0，exit=0。

## 4. 分析

- batch 级"44 条逐字节未变"证明 checkpoint 复用路径按设计工作，手术只影响目标 batch。
- 验证器的 task_id/chunk_id 唯一性检查是捕获该 bug 的关键——污染发生在 P2 主程序内部，若无独立验证步骤会被静默写盘。
- 回滚链路在真实污染场景下验证有效（不是只测了 happy path）。
- E2E-2 两次运行的耗时差（60.3s vs 47.6s）来自 LLM 生成波动，机制无关。

## 5. 结论与遗留

结论：`code_p2_rerun.py` 两种粒度、干跑、执行、回滚全部验证通过；连带修复 P2 部分续跑 task 重复 bug 一处。

遗留：无功能性遗留。`output/` 下留存测试期备份目录（`p2_rerun_backup_20260804_105954`、`p2_rerun_backup_20260804_110043`、`p2_rerun_backup_20260804_112617`、`pre_rollback_20260804_112554`），确认无需回滚后可清理。
