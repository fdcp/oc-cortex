# P2 "无实质交互内容" 问题修复与真实 LLM 验证报告

日期: 2026-08-04 · 模型: qwen3.7-plus (opencode.ai/zen/go/v1) · 状态: 28/28 坏 summary 全部修复

## 1. 背景

`chunks_summary_p2.jsonl` 中部分 chunk 的 summary 为 "无实质交互内容。"，但对应 chunk 实际内容丰富。全局扫描共 28 条坏 summary，与 `cleaned_size_tokens > max_tokens_per_chunk` 的 chunk 集合 100% 重合（28/28）。

## 2. 根因

`code_p1_utils.safe_truncate()` 在 fed7dc7 重构搬迁 src/ 时丢失了函数尾部 `return` 语句（用 `git log -L` 对比重构前后确认），超预算路径返回 `None`。`None` 被拼进 P2 prompt，LLM 看到字面 "None" 或残缺内容，判定"无实质交互"。另有一处隐患：`head_len` 按原文长度 70% 计算而非按预算，12639 token 的文本会返回 ~8800 token，超出预算。

触发面扩大因素：`config/code_p2_config.yaml` 的 `max_tokens_per_chunk` 由 5000 降为 3000（会话前遗留的未提交改动），使 3000–5000 token 的 chunk 也开始触发截断路径。

## 3. 修复内容（均未提交）

`src/code_p1_utils.py`：

- `safe_truncate()`：恢复 return；`head_len = int(max_tokens * 0.7)` 按预算计算，尾部留 20 token 给省略标记。
- 新增 `collapse_command_injection()`：`<auto-slash-command>` 包裹的数千 token slash 命令模板折叠为 `/命令`（从 `# /xxx Command` 标题提取）；裸 `/命令`、`@xxx` 只保留命令 token；多段文件路径（`/Users/...`）与普通文本不受影响（`(?=\s|$)` 前瞻守卫）。
- 新增 `truncate_chunk_text()` 分级截断：阶段 0 折叠命令注入 → 阶段 1 丢 bash tool_calls → 阶段 2 再丢 tool_call 类 → 阶段 3 丢全部工具/MCP 调用（附 "已省略 N 条" 标记）→ 阶段 4 user_message 全量 + assistant 按剩余预算头 70% + 尾 30%。user 单独超预算时只截 user 并打 WARNING。

`src/code_p1_models.py`：提取模块级 `render_chunk_text()` 作为唯一渲染源，`Chunk.cleaned_text()` 委托之（保证与分级截断输出格式一致）。

`src/code_p2_task_extractor.py`：`_format_chunks_for_prompt` 改用 `truncate_chunk_text(chunk, max_tokens_per_chunk)`。

文档同步：`doc/code_p1_README.md`、`doc/code_p2_README.md`。

## 4. 离线验证（真实 LLM 之前）

工作区 `test_truncate_staged.py`，49 项检查全部通过：

- cleaned_text 回归：234 个 chunk 渲染结果与改动前逐字节一致。
- 预算 3000 全量重放：阶段分布 {0: 206, 1: 11, 2: 2, 3: 0, 4: 15}，全部达标（≤ 预算）。
- 命令折叠：`ses_04ba8451..._c1` 的 user_message 6137 token → `/init-deep`，整 chunk 1415 token（阶段 0 即达标）；`/Users/...` 多段路径样本确认不误伤。
- 边界：user 单独超预算、空 assistant、极小预算均按设计分支处理。

## 5. 真实 LLM 验证第一轮（2 个 session）

### 过程

1. 确认无并发 P2/pipeline 进程。
2. 备份 `output/.p2_checkpoint.json`、`output/tasks.jsonl`、`output/chunks_summary_p2.jsonl` 至工作区 `backup_p2_rerun_20260804_101425/`，并存档两 session 旧输出为基线 `p2_old_baseline.json`。
3. checkpoint 手术：原子写移除 2 个重灾 session，其余 24 个不动（`python3 src/code_p2_main.py` 无需 `--force` 即只重跑这 2 个）。
4. 执行：仓库根目录 `export OPENCODE_ZEN_API_KEY=... && python3 src/code_p2_main.py`。

实测耗时 **98.7s**（2 sessions，其中 ses_072ebc268 走 multi-batch 2/2，并发=4）。结束后 26/26 session 完成，234/234 chunk 含总结，checkpoint 一致性完好。

### 选定对象

- `ses_04ba8451cffeeg4eVFN8BencC8`（2 chunks）：c1 是 /init-deep `<auto-slash-command>` 模板注入（user 单独 6137 token），旧 summary "会话初始化占位，无实质交互内容。"
- `ses_072ebc268ffe227bGpbCtWbM97`（12 chunks，multi-batch）：5 条旧 "无实质交互内容。"

### 结果：6 条坏样本全部修复

| chunk | 旧 summary | 新 summary（节选） |
|---|---|---|
| ses_04ba…_c1 ★ | 会话初始化占位，无实质交互内容。 | 触发 /init-deep 工作流，评估项目规模（12682行代码），识别出 doc/ 目录需独立文档化，并启动 9 个后台代理… |
| ses_072e…_c1 ★ | 无实质交互内容。 | 编写 Python 脚本解析 fa3.json，过滤空消息并处理 7 类 parts 与 12 种工具调用… |
| ses_072e…_c3 ★ | 无实质交互内容。 | 排查并修复 HTML 渲染不完整问题。根因为原始数据含未转义的 \<template\> 标签…html.escape() 修复 |
| ses_072e…_c4 ★ | 无实质交互内容。 | 重构 HTML 渲染逻辑：长文本/长代码自动折叠、工具输出深色主题、内嵌代码高亮、TOC 导航… |
| ses_072e…_c8 ★ | 无实质交互内容。 | 排查 TOC 100+ 编号显示为 00，定位 CSS padding-left: 24px 过小导致三位数首位被截断 |
| ses_072e…_c9 ★ | 无实质交互内容。 | padding-left 24px → 3em 自适应修复编号截断 Bug，截图验证通过 |

### 保真度抽查（新 summary vs chunk 原文）

- c3：user 原文 "显示有问题，只能看到一部分"，13 条 assistant + 40 条 tool_calls → 新 summary 精确对应 "\<template\> 未转义 → html.escape() 修复"。
- c8：user 原文 "你是把…目录标号限制在99了嘛？为什么99之后是00" → 新 summary 精确对应 CSS padding-left 定位。
- c1：user 是 /init-deep 模板，assistant 首句 "run the /init-deep workflow to refresh/generate hierarchical AGENTS.md" → 新 summary 与任务内容（12682 行、9 个后台代理、doc/ 独立文档化）一致。

### 任务级产出

两 session 共重提取 5 个 task（ses_04ba 1 个 + ses_072e 4 个，与旧数量相同但内容完整）。ses_04ba 的 T1 标签 "执行init-deep生成层级化文档" 现在覆盖了 c1——旧版因 c1 被判"无实质"而信息缺失。

## 6. 第二轮修复（其余 6 个 session）

### 过程

方法与第一轮相同：确认无并发进程 → 备份三件套至工作区 `backup_p2_rerun2_20260804_102912/` → 存档旧基线 `p2_old_baseline_round2.json`（31 task、118 summary、23 坏样本）→ checkpoint 原子写移除 6 个 session（其余 20 个不动）→ `python3 src/code_p2_main.py`。

| session | chunks | 模式 | 旧坏样本数 |
|---|---|---|---|
| ses_04b987deaffeKfDUR5b1Y99vcx | 30 | multi | 3 |
| ses_04d77c8caffexlaTbA0HPB87oY | 17 | multi | 4 |
| ses_04e4b0b9cffe8OApx3qp7sRHb3 | 5 | single | 2 |
| ses_0870d73a7ffehev1FO7HA2HSze | 58 | multi | 12 |
| ses_0c9e4a359ffeGDSrVSG2jNVB6n | 3 | single | 1 |
| ses_0ca8a24d9ffeHoz0wYkhYDDO4w | 5 | single | 1 |

实测耗时 **295.8s**（6 sessions / 118 chunks，并发=4）。运行期 2 条 WARNING 均被既有机制消化：ses_0c9e4a359 的 3 个 chunk 未被 task 覆盖 → `_parse_and_validate` 自动兜底分配；ses_0870d73a batch 2 JSON 解析失败 → 内容重试 1/3 后成功。结束后 26/26 session 完成，234/234 chunk 含总结。

### 结果：23 条坏样本全部修复，全局零残留

修复后全局扫描 `chunks_summary_p2.jsonl`，含 "无实质" 的 summary 为 **0**（28/28 全部修复，含第一轮 6 条）。新 summary 抽样：

| chunk | 新 summary（节选） |
|---|---|
| ses_04b987…_c5 | 对照源码深度校验 test_all.md，修正 MCP 协议模式、P6b 配置读取等 11 处事实错误 |
| ses_04b987…_c19 | 完成代码实施：将 hash 逻辑从 P2 迁移至 P1（CHUNKER_VERSION=1 及 hash 辅助函数） |
| ses_04d77c…_c13 | 将 librarian 从 opencode-go/hy3 切换至 github-copilot/gpt-5.4-mini |
| ses_04e4b0…_c4 | 执行系统级网络与客户端探查，确认国内环境、系统代理（7890端口）及 API 连通性正常 |
| ses_0870d7…_c51 | 完成 Ver5.docx 的 4 个公式段落渲染：LaTeX 符号转 Unicode、\\frac 栈式解析、PUA 占位符保护下划线 |

保真抽查 ses_0870d73a_c51：user 原文 "docx中的公式显示不正常，处理一下，保存成 ver5"，assistant 尾部给出 4 段公式处理结果表 → 新 summary 与之一一对应。

### 一致性核对与 task 变化

- chunks_summary_p2.jsonl：234 → 234 行，chunk_id 集合一致，无重复（定向覆盖替换，非追加）。
- tasks.jsonl：73 → 74 行，无重复。6 session task 数 31 → 32（ses_04b987 8→7、ses_04d77c 3→2、ses_0870d7 16→19，其余不变）——重新提取的任务划分粒度有正常波动，净增 1。

## 7. 分析

- 根因链条完整闭环：`safe_truncate` 返回 None → prompt 注入字面 "None" → LLM 判"无实质"。修复后同一批 chunk 重跑即产出忠实 summary，无需改 prompt。
- 命令折叠把 c1 从"阶段 4 user 截断 + assistant 全丢"变为"阶段 0 全文保留"（1415/3000 token），这是该 chunk summary 质量逆转的直接原因。
- 非坏样本的 chunk 重跑后 summary 有措辞变化属正常（整 session 重新提取），语义一致，未观察到质量回退。

## 8. 遗留事项

- 受影响 chunk 需重跑 P3 重建向量索引（summary 参与 P3 入库）：涉及 8 个 session 的 28 个原坏样本 chunk 及其同 session 重提取部分。
- 所有代码改动未提交，待用户审阅。备份位于工作区 `backup_p2_rerun_20260804_101425/`（第一轮前）与 `backup_p2_rerun2_20260804_102912/`（第二轮前），两轮旧输出基线分别为 `p2_old_baseline.json`、`p2_old_baseline_round2.json`，均可回滚。
