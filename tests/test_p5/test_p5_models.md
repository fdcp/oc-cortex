# Phase 5 多模型 × 多模式 Benchmark 报告

## 测试概要

- **测试日期**: 2026-07-19
- **测试目标**: 对比 4 个 OpenCode Zen 免费模型在 Phase 5 知识图谱构建完整流水线上的效果
- **测试样本**: 31 个 task（全量），来自 Phase 2 产出
- **测试模型**: nemotron-3-ultra-free, deepseek-v4-flash-free, mimo-v2.5-free, hy3-free
- **测试模式**: triple（三元组关系图谱）、entity（实体共现图谱）
- **测试脚本**: `tests/test_p5/test_p5.py`
- **测试范围**: 完整流水线 — 抽取 → 实体收集 → 实体对齐（Qdrant + LLM） → 图谱构建
- **并发数**: 8
- **总耗时**: triple 1007.1s + entity 453.8s ≈ 24.3 分钟

## 复现方法

```bash
# 1. 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 2. 进入项目目录
cd ~/Desktop/oc_sess_graph

# 3. 全量测试 (4 模型 × 2 模式)
python3 tests/test_p5/test_p5.py --concurrency 8

# 4. 仅 triple 模式
python3 tests/test_p5/test_p5.py --modes triple --concurrency 8

# 5. 仅 entity 模式
python3 tests/test_p5/test_p5.py --modes entity --concurrency 8

# 6. 指定模型
python3 tests/test_p5/test_p5.py --models nemotron-3-ultra-free hy3-free --concurrency 8

# 7. 快速验证 (前 5 个 task)
python3 tests/test_p5/test_p5.py --limit 5

# 8. 指定输出目录
python3 tests/test_p5/test_p5.py --output tests/test_p5/my_results
```

## Triple 模式结果

### 对比表

| 指标 | nemotron-3-ultra-free | deepseek-v4-flash-free | mimo-v2.5-free | hy3-free |
|------|----------------------|----------------------|----------------|----------|
| **三元组总数** | **294** | 43 | 208 | 189 |
| **平均每 task** | **9.5** | 1.4 | 6.7 | 6.1 |
| **原始实体数** | 374 | 51 | 257 | 204 |
| **对齐后唯一实体** | 363 | 50 | 256 | 201 |
| **合并实体对** | **11** | 1 | 1 | 3 |
| **图谱节点数** | 363 | 50 | 256 | 201 |
| **图谱边数** | **294** | 43 | 208 | 189 |
| **平均度** | 1.6 | 1.7 | 1.6 | **1.9** |
| **最大度** | 10 | 8 | 12 | **18** |
| **抽取耗时** | 127.1s | 312.8s | 419.8s | **20.2s** |
| **对齐耗时** | 51.3s | **2.9s** | 64.2s | 2.8s |
| **总耗时** | 178.4s | 315.6s | 484.1s | **23.0s** |
| **抽取错误数** | 2 | ~28 | 4 | **0** |

### 三元组质量示例（以 "GPU算力与选型对比分析" task 为例）

**nemotron** (8 个三元组): 实体精准，关系明确（V100/A800/H100 对比选型、NVLink 互联带宽等）

**hy3-free** (6 个三元组): 同样高质量，使用中文关系动词（对比了、依赖、新增了）

**mimo-v2.5-free** (8 个三元组): 质量稳定，偶有 JSON 截断但修复成功

**deepseek** (多数 task 0 个三元组): reasoning_content 中无法提取有效 JSON，仅有少量 task 成功

### Top-5 高度节点

| 模型 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
|------|-------|-------|-------|-------|-------|
| nemotron | OpenCode(10) | FlashAttention(10) | fdcp.github.io(8) | bf16_training_guide_v4(7) | Sequence Parallel(7) |
| deepseek | OpenCode(8) | precompute_freqs_cis(6) | SDPA(5) | oh-my-openagent(5) | FlashAttention(4) |
| mimo | OpenCode(12) | NeMo Megatron Bridge(10) | optimizers_guide_v2(8) | FlashAttention-2(8) | optimizers_guide.md(6) |
| hy3 | OpenCode(18) | 序列并行(9) | ses_12c6...session ID(8) | fdcp.github.io(8) | bf16_training_guide_v4.html(7) |

注意: hy3 将 session ID (`ses_12c6bd8dfffe...`) 作为实体抽取，违反了 prompt 的排除规则。

## Entity 模式结果

### 对比表

| 指标 | nemotron-3-ultra-free | deepseek-v4-flash-free | mimo-v2.5-free | hy3-free |
|------|----------------------|----------------------|----------------|----------|
| **唯一实体数** | **258** | 246 | 223 | 258 |
| **平均每 task** | **8.3** | 7.9 | 7.2 | 8.3 |
| **对齐后唯一实体** | 246 | 237 | 223 | 252 |
| **合并实体对** | **12** | 9 | 0 | 6 |
| **图谱节点数** | 246 | 237 | 223 | 252 |
| **图谱边数（共现）** | **1216** | 1148 | 960 | 1206 |
| **平均度** | **9.9** | 9.7 | 8.6 | 9.6 |
| **最大度** | 45 | 49 | 28 | **50** |
| **抽取耗时** | 40.0s | 131.0s | 134.2s | **10.8s** |
| **对齐耗时** | 60.3s | 6.8s | 60.5s | **4.6s** |
| **总耗时** | 100.3s | 137.8s | 194.7s | **15.3s** |
| **抽取错误数** | 0 | 1 | 0 | **0** |

### Top-5 高度节点

| 模型 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
|------|-------|-------|-------|-------|-------|
| nemotron | OpenCode(45) | FlashAttention(36) | Sequence Parallel(31) | Ulysses(25) | Tensor Parallel(23) |
| deepseek | OpenCode(49) | FlashAttention(42) | 序列并行(27) | Sequence Parallel(24) | FP16(23) |
| mimo | opencode(28) | OpenCode(23) | FlashAttention(19) | opencode.db(18) | Sequence Parallel(18) |
| hy3 | OpenCode(50) | FlashAttention(35) | gpt-5-nano(20) | A100(18) | NVLink(18) |

注意: deepseek 未将 "序列并行" 和 "Sequence Parallel" 合并（对齐阶段仅合并 9 对），mimo 未将 "opencode" 和 "OpenCode" 合并（0 合并对）。nemotron 和 hy3 的合并效果更好。

## 跨模式对比分析

### 耗时对比（秒）

| 模型 | Triple 抽取 | Triple 对齐 | Triple 总 | Entity 抽取 | Entity 对齐 | Entity 总 |
|------|-------------|-------------|-----------|-------------|-------------|-----------|
| nemotron | 127.1 | 51.3 | 178.4 | 40.0 | 60.3 | 100.3 |
| deepseek | 312.8 | 2.9 | 315.6 | 131.0 | 6.8 | 137.8 |
| mimo | 419.8 | 64.2 | 484.1 | 134.2 | 60.5 | 194.7 |
| hy3 | **20.2** | **2.8** | **23.0** | **10.8** | **4.6** | **15.3** |

Entity 模式整体比 Triple 模式快 1.5~13x，主要因为:
1. Entity 的 prompt 更简单（只需输出实体列表，无需构造关系三元组）
2. Entity 每 task 产出更少（~8 实体 vs ~7 三元组，但三元组的 head+tail 产生更多实体）

### 图谱规模对比

| 模型 | Triple 节点 | Triple 边 | Entity 节点 | Entity 边 |
|------|-------------|-----------|-------------|-----------|
| nemotron | 363 | 294 | 246 | 1216 |
| deepseek | 50 | 43 | 237 | 1148 |
| mimo | 256 | 208 | 223 | 960 |
| hy3 | 201 | 189 | 252 | 1206 |

Entity 模式的边数远多于 Triple 模式（共现 C(n,2) 两两配对），这是设计使然。

## 模型逐项分析

### nemotron-3-ultra-free

**优势**: Triple 抽取数量最多（294 个，9.5/task），实体对齐合并最积极（11 对），图谱最丰富（363 节点）。Entity 模式同样表现优秀（258 实体，1216 共现边）。

**劣势**: 有 2 次 `'list' object has no attribute 'get'` 错误（LLM 返回 JSON 数组而非对象）。抽取速度中等（127s/31task ≈ 4.1s/task）。

**适用场景**: 需要最丰富知识图谱的场景，是 Triple 模式的最佳选择。

### deepseek-v4-flash-free

**优势**: Entity 模式表现尚可（246 实体，1148 边），与 nemotron 接近。

**劣势**: **Triple 模式严重失败** — 仅 43 个三元组（1.4/task），大量 reasoning_content 中的 JSON 无法解析。几乎每次 LLM 调用都返回空 content，且 reasoning_content 的 JSON 提取失败。这是 deepseek reasoning 模型 + 结构化输出（三元组 JSON）的经典兼容性问题。Entity 模式因 JSON 更简单（只需 `{"entities": [...]}`）而表现正常。

**错误细节**: 约 28 次 JSON 解析失败（`Expecting value: line 1 column 1 (char 0)` 或类似），reasoning_content 中提取到的内容不是有效 JSON。

**适用场景**: 仅适合 Entity 模式，不推荐用于 Triple 模式。

### mimo-v2.5-free

**优势**: Triple 模式产出质量不错（208 三元组，6.7/task），接近 nemotron。

**劣势**: 最慢的模型 — Triple 484.1s（约 15.6s/task），Entity 194.7s（约 6.3s/task）。频繁出现 `LLM 返回空内容` 重试。Entity 模式对齐阶段 0 合并对（对齐 LLM 调用全部返回空内容，无法确认合并）。

**错误细节**: Triple 模式 4 次抽取失败（3 次空内容 + 1 次 list 错误），Entity 对齐阶段 LLM 调用全部失败导致 0 合并。

**适用场景**: 不推荐 — 速度最慢且对齐阶段不稳定。

### hy3-free

**优势**: **速度绝对第一** — Triple 23.0s（0.74s/task），Entity 15.3s（0.49s/task），比第二名快 4~20x。0 错误率。Entity 模式实体数最多（258），图谱边数接近 nemotron。

**劣势**: Triple 模式三元组数偏少（189，6.1/task），比 nemotron 少 36%。有 1 次将 session ID 作为实体（违反 prompt 规则）。最大度 50 说明有少量超高频实体未对齐（如 "OpenCode" 出现在 50 个共现关系中）。

**适用场景**: 对速度敏感或需要快速迭代的场景。Entity 模式综合最优（速度 + 实体数 + 图谱规模）。

## 综合推荐

| 维度 | 推荐模型 | 理由 |
|------|----------|------|
| **Triple 模式** | nemotron-3-ultra-free | 三元组最多（294），实体对齐最积极（11 对合并），图谱最丰富 |
| **Entity 模式** | hy3-free | 速度最快（15.3s），实体数最多（258），图谱边数接近 nemotron |
| **速度优先** | hy3-free | 所有模式下均 < 25s，比 nemotron 快 4~8x |
| **稳定性** | hy3-free | 0 错误率，无 content_empty 问题 |
| **不推荐** | deepseek（Triple）、mimo（全模式） | deepseek Triple JSON 解析大面积失败；mimo 最慢且对齐不稳定 |

## 已知问题

1. **deepseek Triple JSON 解析**: reasoning 模型将 JSON 放在 reasoning_content 中，但 `_extract_json_from_response` 无法从中提取有效的三元组 JSON。Entity 模式的简单 JSON 可以正常提取。
2. **mimo 空内容**: 频繁返回空 content，即使有 reasoning_content fallback 也经常失败。对齐阶段（短文本 MERGE/KEEP 回复）尤其不稳定。
3. **hy3 实体规范违反**: 偶有将 session ID 作为实体抽取的情况，违反 prompt 中的排除规则。
4. **nemotron list 格式错误**: 偶有返回 JSON 数组而非对象（2 次/31 task），导致 `'list' object has no attribute 'get'`。
5. **对齐 LLM 失败**: 当 LLM 无法产生 MERGE/KEEP 响应时，该实体对保持不合并，导致不同模型的合并对数差异较大（0~12 对）。

## 文件清单

```
tests/test_p5/
├── test_p5.py                              # Benchmark 脚本
├── test_p5_models.md                       # 本报告
├── test_p5_results.json                    # 完整 JSON 结果
├── test_p5_report.md                       # 自动生成的摘要报告
├── test_p5_triple_run.log                  # Triple 模式运行日志
├── test_p5_entity_run.log                  # Entity 模式运行日志
├── nemotron-3-ultra-free__triple/          # Triple 模式输出 (per model)
│   ├── triples.jsonl
│   ├── knowledge_graph.gpickle
│   └── knowledge_graph.json
├── deepseek-v4-flash-free__triple/
├── mimo-v2.5-free__triple/
├── hy3-free__triple/
├── nemotron-3-ultra-free__entity/          # Entity 模式输出 (per model)
│   ├── entity_extract.jsonl
│   ├── inverted_index.json
│   ├── knowledge_graph.gpickle
│   └── knowledge_graph.json
├── deepseek-v4-flash-free__entity/
├── mimo-v2.5-free__entity/
└── hy3-free__entity/
```
