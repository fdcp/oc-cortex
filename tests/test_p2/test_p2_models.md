# Phase 2 多模型对比测试报告

> **注意 (2026-07 更新)**: 本报告为历史基线, 使用的 `hy3-free` 已从 OpenCode Zen 免费层下线,
> 现应使用 OpenCode Go 端点的 `hy3` (见 `tests/test_p2/config.yaml`)。
> 新版测试脚本会将结果写入带日期的子文件夹 `tests/test_p2/run_<YYYY-MM-DD_HH-MM>/`,
> 内含 `test_p2_results.json` 与自动生成的 `test_p2_models.md`。
> 复现命令: `python3 tests/test_p2/test_p2.py --sessions 100 --concurrency 8`
> (模型 / 端点 / 认证均由 `config.yaml` 顺序加载, 无需再手动 export API Key)。

## 测试概要

- **测试日期**: 2026-07-19
- **测试目标**: 对比 4 个 OpenCode Zen 免费模型在 Phase 2 任务提取 (CoT 三步法) 上的效果
- **测试样本**: 16 个 session（全量），覆盖 2~13 个 chunk 的不同规模
- **测试模型**: nemotron-3-ultra-free, deepseek-v4-flash-free, mimo-v2.5-free, hy3-free
- **测试脚本**: `tests/test_p2.py`

## 复现方法

```bash
# 1. 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 2. 进入项目目录
cd ~/Desktop/oc_sess_graph

# 3. 运行测试（全量 session，串行）
python3 tests/test_p2.py

# 4. 指定 session 数量（快速验证）
python3 tests/test_p2.py --sessions 3

# 5. 指定并发数（加速，注意 free tier 可能限流）
python3 tests/test_p2.py --sessions 100 --concurrency 8

# 6. 指定输出路径
python3 tests/test_p2.py --output tests/my_results.json
```

## 汇总对比表

| 指标 | nemotron-3-ultra-free | deepseek-v4-flash-free | mimo-v2.5-free | hy3-free |
|------|----------------------|----------------------|----------------|----------|
| **成功率** | 100% (16/16) | 100% (16/16) | 100% (16/16) | 100% (16/16) |
| **平均耗时** | 34.9s | 61.6s | 35.1s | **15.5s** |
| **总耗时** | 558s | 985s | 562s | **248s** |
| **P50 延迟** | 14.5s | 51.8s | 31.8s | **15.8s** |
| **P90 延迟** | 105.6s | 109.1s | 58.6s | **28.5s** |
| **最大延迟** | 120.0s | 214.9s | 65.8s | **33.0s** |
| **总 Task 数** | **35** | 29 | 28 | 29 |
| **平均 Task/session** | **2.2** | 1.8 | 1.8 | 1.8 |
| **Task 数范围** | 1~4 | 1~3 | 1~5 | 1~4 |
| **Label 均长(字)** | **20** | 15 | 17 | 14 |
| **Summary 均长(字)** | 336 | 235 | 298 | **366** |
| **Summary P50(字)** | 352 | 265 | 296 | **352** |
| **Chunk 覆盖率** | 100% | 100% | 100% | 100% |
| **Summary 覆盖率** | 100% | 100% | 100% | 100% |
| **需要重试** | 0 session | 1 session | 0 session | 0 session |

## 逐 Session 延迟对比 (秒)

| Session (chunks) | nemotron | deepseek | mimo | hy3 |
|-------------------|----------|----------|------|-----|
| ses_0c4e03... (2c) | 12.3 | 28.2 | 31.8 | 7.9 |
| ses_0ec02c... (2c) | 6.8 | 18.7 | 27.7 | 6.1 |
| ses_16a4e1... (2c) | 9.5 | 28.0 | 21.8 | 7.9 |
| ses_0a53b6... (3c) | 50.9 | 12.4 | 32.0 | 10.8 |
| ses_0c9e4a... (3c) | 37.2 | 34.8 | 29.5 | 16.9 |
| ses_0ca8b3... (3c) | 32.8 | 51.8 | 34.6 | 15.8 |
| ses_0e8898... (3c) | 14.5 | 80.5 | 28.8 | 12.8 |
| ses_0ee881... (3c) | 9.3 | 18.5 | 30.5 | 8.4 |
| ses_14a9ad... (3c) | 9.7 | 42.8 | 28.5 | 10.0 |
| ses_0c99b8... (4c) | 11.2 | 88.5 | 28.0 | 17.7 |
| ses_0a16b2... (5c) | 16.0 | 50.8 | 36.1 | 20.6 |
| ses_0ca8a2... (5c) | 10.5 | 60.8 | 27.4 | 19.7 |
| ses_10772d... (8c) | 12.9 | 78.1 | 36.9 | 17.1 |
| ses_12c6bd... (11c) | 120.0 | 67.5 | 44.4 | 28.5 |
| ses_0c4845... (13c) | 99.2 | 109.1 | 65.8 | 15.1 |
| ses_0c5b61... (13c) | 105.6 | 214.9 | 58.6 | 33.0 |

## Task 提取质量对比（以 5-chunk session 为例）

以 `ses_0a16b26dcffe` (5 chunks) 为例，展示各模型提取的 task 差异：

### nemotron-3-ultra-free (16.0s, 2 tasks)

1. **检索并总结指定 session 工作内容** (352字)
   — 三段式：背景→核心产出（SP 并行文档迭代）→结论
2. **查询 opencode 免费模型列表** (310字)
   — 三段式：分类梳理免费模型来源→具体列表→建议使用方式

### deepseek-v4-flash-free (50.8s, 2 tasks)

1. **查询历史session并行训练文档** (346字)
   — 详细还原了 SP 文档三轮迭代过程，包含技术细节纠正
2. **查询opencode zen免费模型** (282字)
   — 逐一列出 5 个免费模型及其限制说明

### mimo-v2.5-free (36.1s, 2 tasks)

1. **检索并获取指定Session的历史记录** (320字)
   — 三段式，强调了跨工作区检索的排查过程
2. **梳理opencode免费模型** (250字)
   — 简洁的三段式分类总结

### hy3-free (20.6s, 2 tasks)

1. **查询异地session工作内容** (639字)
   — 最详尽：完整还原了 SQLite 查询过程、session 归属、文档版本迭代细节
2. **梳理opencode免费模型** (417字)
   — 区分三类免费来源 + 专门核实 Zen 平台 5 个模型

## 分析结论

### 速度排名
1. **hy3-free** — 平均 15.5s，最快。P90 仅 28.5s，即使 13-chunk 大 session 也只需 15~33s
2. **nemotron-3-ultra-free** — 平均 34.9s，小 session 很快（6~12s），但大 session 偶尔飙到 100~120s
3. **mimo-v2.5-free** — 平均 35.1s，波动最小（P50=31.8s, P90=58.6s），表现最稳定
4. **deepseek-v4-flash-free** — 平均 61.6s，最慢。reasoning 模型开销大，13-chunk session 需 109~215s

### 质量排名
1. **hy3-free** — Summary 均长 366 字（最长），三段式结构最完整，信息密度高
2. **nemotron-3-ultra-free** — Summary 均长 336 字，三段式执行好，但偶尔过于详细
3. **mimo-v2.5-free** — Summary 均长 298 字，质量适中，结构规范
4. **deepseek-v4-flash-free** — Summary 均长 235 字，偏简洁但信息完整

### 任务粒度
- **nemotron** 倾向多拆（平均 2.2 tasks/session），最大 session 拆出 4 个 task
- **deepseek / hy3** 倾向合并（平均 1.8），粒度更保守
- **mimo** 大部分 session 只提 1 个 task，但对大 session 也能拆到 5 个

### 稳定性
- 所有模型均 100% 成功率，无失败
- **deepseek** 有 1 次 JSON 截断需要重试（reasoning tokens 占满 budget）
- **mimo** 在小 session (2~3 chunks) 上 chunk 归属有时遗漏（触发自动兜底）

### chunk_id 准确性
- **nemotron / hy3** — 0 个 task 的 chunk_ids 为空，ID 格式完全正确
- **deepseek** — 2 个 task 的 chunk_ids 为空（11-chunk session 拆出 3 个语义任务，但 T2/T3 的 chunk_id 格式不匹配被过滤，兜底机制已自动覆盖）
- **mimo** — 1 个 task 的 chunk_ids 为空（3-chunk session 中 T2 的 chunk_id 不合法）

这反映 **nemotron 和 hy3 对 chunk_id 格式的遵从性更好**，deepseek 和 mimo 偶尔会用错误的 ID 格式（如省略前缀）。

### 综合推荐

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **日常使用 (平衡)** | **hy3-free** | 最快 + 最详细 summary + 100% 覆盖，性价比最高 |
| **高质量模式** | nemotron-3-ultra-free | 任务拆分更细致，summary 三段式执行最好 |
| **大 batch 处理** | hy3-free | 总耗时仅 248s（vs deepseek 985s），适合全量重跑 |
| **稳定性优先** | mimo-v2.5-free | 延迟波动最小，P50/P90 差距小 |

## 附录：原始日志

完整运行日志保存在 `tests/test_p2_run_full.log`，详细 JSON 结果在 `tests/test_p2_results.json`。
