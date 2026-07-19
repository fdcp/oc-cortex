# Phase 5 多模型 × 多模式 Benchmark 报告

生成时间: 2026-07-19 04:39:16

测试 task 数: 31

## 对比表

### entity 模式

| 模型 | 状态 | 抽取量 | 平均/t | 原始实体 | 唯一实体 | 合并对 | 节点 | 边 | 抽取(s) | 对齐(s) | 图谱(s) | 总计(s) |
|------|------|--------|--------|----------|----------|--------|------|-----|---------|---------|---------|----------|
| nemotron-3-ultra-free | OK | 258 | 8.3 | 258 | 246 | 12 | 246 | 1216 | 40.0 | 60.3 | 0.0 | 100.3 |
| deepseek-v4-flash-free | OK | 246 | 7.9 | 246 | 237 | 9 | 237 | 1148 | 131.0 | 6.8 | 0.0 | 137.8 |
| mimo-v2.5-free | OK | 223 | 7.2 | 223 | 223 | 0 | 223 | 960 | 134.2 | 60.5 | 0.0 | 194.7 |
| hy3-free | OK | 258 | 8.3 | 258 | 252 | 6 | 252 | 1206 | 10.8 | 4.6 | 0.0 | 15.3 |

## 图谱 Top-5 高度节点

### nemotron-3-ultra-free × entity

| 实体 | 度 | 类型 |
|------|-----|------|
| OpenCode | 45 | concept |
| FlashAttention | 36 | concept |
| Sequence Parallel | 31 | concept |
| Ulysses | 25 | concept |
| Tensor Parallel | 23 | concept |

### deepseek-v4-flash-free × entity

| 实体 | 度 | 类型 |
|------|-----|------|
| OpenCode | 49 | concept |
| FlashAttention | 42 | concept |
| 序列并行 | 27 | concept |
| Sequence Parallel | 24 | concept |
| FP16 | 23 | concept |

### mimo-v2.5-free × entity

| 实体 | 度 | 类型 |
|------|-----|------|
| opencode | 28 | concept |
| OpenCode | 23 | concept |
| FlashAttention | 19 | concept |
| opencode.db | 18 | concept |
| Sequence Parallel | 18 | concept |

### hy3-free × entity

| 实体 | 度 | 类型 |
|------|-----|------|
| OpenCode | 50 | concept |
| FlashAttention | 35 | concept |
| gpt-5-nano | 20 | concept |
| A100 | 18 | concept |
| NVLink | 18 | concept |

