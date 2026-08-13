# P4 Alias Expansion 完整 trace 测试报告

> 测试日期: 2026-08-13  
> 测试范围: P4 搜索 pipeline 中 alias expansion 行为 (sparse-only)  
> 监控点: 15 个 pipeline 状态 + 衍生指标  
> 配置文件: `config/code_p3_config.yaml` (测试期间临时翻 `enabled: true`, 完成后改回 `false`)  

## 1. 测试需求 (Requirements)

### 1.1 目标

- 验证 P4 alias expansion 在 4 种 (match_mode × kg_auto_mode) 组合下, 对 3 条真实查询的行为
- 监控向量检索的完整 pipeline, 暴露每一阶段的中间状态
- 度量扩展机制对召回的实际影响 (新增 / 丢失 / 排序变化)

### 1.2 测试范围

- **维度**: `match_mode ∈ {exact, word_boundary}`, `kg_auto_mode ∈ {entity, triple}` (case_sensitive 固定 False)
- **查询**: 3 条覆盖中文 / 英文缩写 / 长查询的真实场景
- **skip 维度**: `case_sensitive` (固定 False); `max_idf_tokens`, `idf_floor`, `pos_keep` 等次级参数保留 yaml 默认

### 1.3 成功判据

| 维度 | 判据 | 实际 |
|------|------|------|
| 扩展触发率 | combo 命中率 ≥ baseline extras 数 | 见 §4 |
| 召回变化 | 至少 1 个 combo 的 top-5 与 baseline 有差 | 见 §5 |
| 中间状态可监控 | 15 个监控点全部产出数据 | ✓ |
| KG DB 切换 | entity/triple 都能成功加载 | ✓ |

## 2. 测试设计 (Test Design)

### 2.1 维度矩阵

| combo | match_mode | kg_auto_mode |
|-------|-----------|--------------|
| 1 | `exact` | `entity` |
| 2 | `exact` | `triple` |
| 3 | `word_boundary` | `entity` |
| 4 | `word_boundary` | `triple` |

每条 query 跑 1 baseline (alias OFF) + 4 combo (alias ON) = 5 search 调用, 总计 3 × 5 = 15 次.

### 2.2 监控点 (15 个 pipeline 状态)

| # | 阶段 | 监控内容 |
|---|------|---------|
| 1 | Input | query 原文, 长度, top_k, alias_expansion_enabled |
| 2 | Tokenize (jieba.posseg) | 全 (token, POS) 元组列表 |
| 3 | POS + 长度过滤 | 候选 / 丢弃 token 列表 |
| 4 | IDF 计算 | 每个候选 token 在 2 个 collection 下的 BM25.max() IDF |
| 5 | IDF 排序 + top-K | top-K 选中 token + IDF 值 |
| 6 | KG 查询 (per token) | 每个 top-K token 的 `search_entities_exact` 命中 (name, aliases, task_count, matched_in, extra_candidates) |
| 7 | 候选 term 计算 | 每个 hit: `[canonical]+aliases` 排除 matched, 取 top N |
| 8 | 扩展 token list | `base ∪ extras`, dedup, extras 来源标注 |
| 9 | Dense 召回 | top-25 candidates + cosine score |
| 10 | Sparse 召回 (cleaned) | top-75 chunk BM25 candidates + score |
| 11 | Sparse 召回 (summary) | top-75 chunk BM25 candidates + score |
| 12 | Chunk→Task aggregation | 聚合后的 task scores |
| 13 | RRF 融合 (chunk) | chunks_summary + chunks_cleaned_text 融合后 top-25 |
| 14 | RRF 融合 (dense+chunk) | dense + chunk_fused 融合后 top-25 candidates |
| 15 | Reranker + Final | top-5 精排结果 (rerank_score, hybrid_score, task_label, chunks) |

### 2.3 衍生指标

- **BM25 score shift**: 同 chunk 在 A 和 B 下 BM25 score 差值 (前 5 大正向变化)
- **新增召回数**: B 独有 chunk / task 数
- **丢失召回数**: A 独有 chunk / task 数
- **Extras vs base 比例**: 扩展 term 数 / 原 term 数
- **Timing 分解**: dense / sparse / aggregate / rrf / rerank 各自耗时

## 3. 测试过程 (Process / Data Flow)

### 3.1 配置快照

```yaml
alias_expansion:
  enabled: True
  kg_db_path: output/triple/knowledge_graph.db
  kg_auto_mode: entity
  max_idf_tokens: 3
  idf_floor: 0.5
  min_token_chars: 2
  pos_keep: ['n', 'eng', 'x']
  max_aliases_per_match: 2
  max_total_extra_terms: 6
  case_sensitive: False
  match_mode: word_boundary
```

### 3.2 数据流

```
query
  │
  ▼  jieba.cut_for_search → base_tokens
  │
  ▼  jieba.posseg.cut → (token, POS) 全列表
  │     pos_keep ∩ min_chars 过滤 → pos_tokens
  │
  ▼  BM25 IDF per collection (max score per token)
  │     IDF desc top-K=3 → ranked_tokens
  │
  ▼  search_entities_exact(ranked_token, match_mode)
  │     ↓ hits → [canonical]+aliases 排除 matched → 取前 N=2
  │     ↓ 全局 dedup, 累加到 extras (max_total=6)
  │
  ▼  final_tokens = base_tokens + extras
  │
  ▼  注入 BM25 sparse 检索 (chunks_cleaned_text, chunks_summary)
  │
  ▼  dense (BGE-M3) 召回 + sparse BM25 + RRF 融合 → candidates
  │
  ▼  Reranker (Qwen3-Reranker-0.6B, 用原 query) → top-5
```

### 3.3 工具

- 测试驱动: `tests/test_p4/test_p4_full.py` (15-point trace, 导出 JSON)
- 报告生成: `tests/test_p4/test_p4_generate_report.py` (本文档)
- 结果文件: `tests/test_p4/test_p4_full_results.json` (~1.5MB, 15 traces × ~20KB)
- 运行日志: `tests/test_p4/test_p4_full.log`

## 4. 测试结果 (Results)

### 4.1 总体: 扩展触发情况

| query | baseline extras | combo1 exact/entity | combo2 exact/triple | combo3 wb/entity | combo4 wb/triple |
|-------|----------------|---------------------|---------------------|------------------|------------------|
| `我在那个opencode对话中使用github-c......` | — | 0 | 0 | 5 | 8 |
| `介绍一下序列并行的原理...` | — | 0 | 0 | 0 | 0 |
| `帮我找一下BF16混合精度训练中关于序列并行、上下......` | — | 2 | 2 | 6 | 19 |

### 4.2 总体: 扩展 term 详情

#### Q: `我在那个opencode对话中使用github-copilot的模型了`

- **combo1 exact/entity**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `copilot` idf=5.44 hit_count=0: 
    - `github` idf=4.59 hit_count=1: GitHub(name)→[]
    - `对话` idf=4.50 hit_count=0: 
- **combo2 exact/triple**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `copilot` idf=5.44 hit_count=0: 
    - `github` idf=4.59 hit_count=1: GitHub(name)→[]
    - `对话` idf=4.50 hit_count=0: 
- **combo3 wb/entity**: extras(cleaned) = `Pro`, `+`, `Web`, `版`, `Pages`
  - extras(summary) = `Pro`, `+`, `Web`, `版`, `Pages`
  - KG hits (cleaned):
    - `copilot` idf=5.44 hit_count=2: GitHub Copilot(word_boundary)→['GitHub Copilot', 'GitHub Copilot Pro+']; Web版Copilot(word_boundary)→['Web版Copilot', 'Web 版 Copilot']
    - `github` idf=4.59 hit_count=4: GitHub Copilot(word_boundary)→['GitHub Copilot', 'GitHub Copilot Pro+']; GitHub Pages(word_boundary)→['GitHub Pages']; GitHub(name)→[]; GitHub README(word_boundary)→['GitHub README']
    - `对话` idf=4.50 hit_count=0: 
- **combo4 wb/triple**: extras(cleaned) = `Pro`, `+`, `Claude`, `服务`, `无法`, `显示`, `Web`, `版`
  - extras(summary) = `Pro`, `+`, `Claude`, `服务`, `无法`, `显示`, `Web`, `版`
  - KG hits (cleaned):
    - `copilot` idf=5.44 hit_count=9: GitHub Copilot(word_boundary)→['GitHub Copilot', 'GitHub Copilot Pro+']; GitHub Copilot 的 Claude 模型(word_boundary)→['GitHub Copilot 的 Claude 模型']; GitHub Copilot 服务(word_boundary)→['GitHub Copilot 服务']; GitHub Copilot Pro+ 无法显示 Claude 模型(word_boundary)→['GitHub Copilot Pro+ 无法显示 Claude 模型']; Web 版 Copilot(word_boundary)→['Web 版 Copilot']; Copilot Web版(word_boundary)→['Copilot Web版']; GitHub Copilot Pro+(word_boundary)→['GitHub Copilot Pro+']; github-copilot/gpt-5.6-sol(word_boundary)→['github-copilot/gpt-5.6-sol']; github-copilot/gpt-5.4-mini(word_boundary)→['github-copilot/gpt-5.4-mini']
    - `github` idf=4.59 hit_count=10: GitHub Copilot(word_boundary)→['GitHub Copilot', 'GitHub Copilot Pro+']; GitHub(name)→[]; GitHub Pages(word_boundary)→['GitHub Pages']; GitHub Copilot 的 Claude 模型(word_boundary)→['GitHub Copilot 的 Claude 模型']; GitHub Copilot 服务(word_boundary)→['GitHub Copilot 服务']; GitHub 业务逻辑层(word_boundary)→['GitHub 业务逻辑层']; GitHub Copilot Pro+ 无法显示 Claude 模型(word_boundary)→['GitHub Copilot Pro+ 无法显示 Claude 模型']; GitHub Pages 部署(word_boundary)→['GitHub Pages 部署']; GitHub 认证凭证缺失(word_boundary)→['GitHub 认证凭证缺失']; GitHub 认证凭证(word_boundary)→['GitHub 认证凭证']
    - `对话` idf=4.50 hit_count=0: 

#### Q: `介绍一下序列并行的原理`

- **combo1 exact/entity**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `序列` idf=6.20 hit_count=0: 
    - `原理` idf=6.14 hit_count=0: 
- **combo2 exact/triple**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `序列` idf=6.20 hit_count=0: 
    - `原理` idf=6.14 hit_count=0: 
- **combo3 wb/entity**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `序列` idf=6.20 hit_count=0: 
    - `原理` idf=6.14 hit_count=0: 
- **combo4 wb/triple**: extras(cleaned) = —
  - extras(summary) = —
  - KG hits (cleaned):
    - `序列` idf=6.20 hit_count=0: 
    - `原理` idf=6.14 hit_count=0: 

#### Q: `帮我找一下BF16混合精度训练中关于序列并行、上下文并行(CP)、FlashAttention相关的内容`

- **combo1 exact/entity**: extras(cleaned) = `Flash`, `Attention`
  - extras(summary) = `Flash`, `Attention`
  - KG hits (cleaned):
    - `CP` idf=7.78 hit_count=0: 
    - `序列` idf=6.20 hit_count=0: 
    - `精度` idf=5.86 hit_count=0: 
- **combo2 exact/triple**: extras(cleaned) = `Flash`, `Attention`
  - extras(summary) = `Flash`, `Attention`
  - KG hits (cleaned):
    - `CP` idf=7.78 hit_count=0: 
    - `序列` idf=6.20 hit_count=0: 
    - `精度` idf=5.86 hit_count=1: 精度(name)→[]
- **combo3 wb/entity**: extras(cleaned) = `Flash`, `Attention`, `-`, `1`, `2`, `指南`
  - extras(summary) = `Flash`, `Attention`, `-`, `1`, `2`, `指南`
  - KG hits (cleaned):
    - `CP` idf=7.78 hit_count=1: 上下文并行(CP)(word_boundary)→['上下文并行(CP)', '上下文并行 (CP)']
    - `序列` idf=6.20 hit_count=0: 
    - `精度` idf=5.86 hit_count=0: 
- **combo4 wb/triple**: extras(cleaned) = `Flash`, `Attention`, `-`, `2`, `工作`, `划分`, `1`, `FP32`, `/`, `FP16`, `双向`, `转换`, `与`, `位`, `可视`, `可视化`, `动态`, `范围`, `取舍`
  - extras(summary) = `Flash`, `Attention`, `-`, `2`, `工作`, `划分`, `1`, `FP32`, `/`, `FP16`, `双向`, `转换`, `与`, `位`, `可视`, `可视化`, `动态`, `范围`, `取舍`
  - KG hits (cleaned):
    - `CP` idf=7.78 hit_count=1: 上下文并行(CP)(word_boundary)→['上下文并行(CP)']
    - `序列` idf=6.20 hit_count=0: 
    - `精度` idf=5.86 hit_count=1: 精度(name)→[]

### 4.3 Top-5 对比 (baseline vs combo)

#### Q: `我在那个opencode对话中使用github-copilot的模型了`

**baseline (alias OFF)** — 耗时 `10020`ms

  1. `0.9994` 排查 opencode 中 Copilot 模型不可见问题
  2. `0.9941` OMO 配置文件优化
  3. `0.9756` 工具能力确认与 Pages 方案咨询
  4. `0.9724` 排查 Copilot 无法显示 Claude 模型
  5. `0.9466` 确认 MiniMax Code 底层架构

**combo (exact/entity)** — 耗时 `9892`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9994` 排查 opencode 中 Copilot 模型不可见问题
  2. `0.9941` OMO 配置文件优化
  3. `0.9756` 工具能力确认与 Pages 方案咨询
  4. `0.9724` 排查 Copilot 无法显示 Claude 模型
  5. `0.9466` 确认 MiniMax Code 底层架构

**combo (exact/triple)** — 耗时 `9778`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9994` 排查 opencode 中 Copilot 模型不可见问题
  2. `0.9941` OMO 配置文件优化
  3. `0.9756` 工具能力确认与 Pages 方案咨询
  4. `0.9724` 排查 Copilot 无法显示 Claude 模型
  5. `0.9466` 确认 MiniMax Code 底层架构

**combo (word_boundary/entity)** — 耗时 `10359`ms  diff: 位置 [2, 3] 重排, 新增=['ses_04e4b0b9cffe8OApx3qp7sRHb3_T1', 'ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1'], 丢失=['ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1', 'ses_04e4b0b9cffe8OApx3qp7sRHb3_T1']

  1. `0.9994` 排查 opencode 中 Copilot 模型不可见问题
  2. `0.9944` OMO 配置文件优化
  3. `0.9724` 排查 Copilot 无法显示 Claude 模型
  4. `0.9724` 工具能力确认与 Pages 方案咨询
  5. `0.9466` 确认 MiniMax Code 底层架构

**combo (word_boundary/triple)** — 耗时 `10906`ms  diff: 位置 [2, 3] 重排, 新增=['ses_04e4b0b9cffe8OApx3qp7sRHb3_T1', 'ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1'], 丢失=['ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1', 'ses_04e4b0b9cffe8OApx3qp7sRHb3_T1']

  1. `0.9994` 排查 opencode 中 Copilot 模型不可见问题
  2. `0.9944` OMO 配置文件优化
  3. `0.9724` 排查 Copilot 无法显示 Claude 模型
  4. `0.9724` 工具能力确认与 Pages 方案咨询
  5. `0.9466` 确认 MiniMax Code 底层架构

#### Q: `介绍一下序列并行的原理`

**baseline (alias OFF)** — 耗时 `10107`ms

  1. `0.9497` SP原理剖析与Mermaid图解迭代
  2. `0.7058` NVIDIA SP与Ulysses并行策略对比
  3. `0.1128` 并行训练尾部填充代码解析
  4. `0.1067` FA1与FA2底层实现及并行类比
  5. `0.0012` DDP训练流程与参数同步机制

**combo (exact/entity)** — 耗时 `9961`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9497` SP原理剖析与Mermaid图解迭代
  2. `0.7058` NVIDIA SP与Ulysses并行策略对比
  3. `0.1128` 并行训练尾部填充代码解析
  4. `0.1067` FA1与FA2底层实现及并行类比
  5. `0.0012` DDP训练流程与参数同步机制

**combo (exact/triple)** — 耗时 `9985`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9497` SP原理剖析与Mermaid图解迭代
  2. `0.7058` NVIDIA SP与Ulysses并行策略对比
  3. `0.1128` 并行训练尾部填充代码解析
  4. `0.1067` FA1与FA2底层实现及并行类比
  5. `0.0012` DDP训练流程与参数同步机制

**combo (word_boundary/entity)** — 耗时 `10076`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9497` SP原理剖析与Mermaid图解迭代
  2. `0.7058` NVIDIA SP与Ulysses并行策略对比
  3. `0.1128` 并行训练尾部填充代码解析
  4. `0.1067` FA1与FA2底层实现及并行类比
  5. `0.0012` DDP训练流程与参数同步机制

**combo (word_boundary/triple)** — 耗时 `10016`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9497` SP原理剖析与Mermaid图解迭代
  2. `0.7058` NVIDIA SP与Ulysses并行策略对比
  3. `0.1128` 并行训练尾部填充代码解析
  4. `0.1067` FA1与FA2底层实现及并行类比
  5. `0.0012` DDP训练流程与参数同步机制

#### Q: `帮我找一下BF16混合精度训练中关于序列并行、上下文并行(CP)、FlashAttention相关的内容`

**baseline (alias OFF)** — 耗时 `10222`ms

  1. `0.9047` FA1与FA2底层实现及并行类比
  2. `0.8597` Roofline模型算术强度推导
  3. `0.8520` 并行训练尾部填充代码解析
  4. `0.8081` BF16训练指南v4扩充
  5. `0.5467` NVIDIA SP与Ulysses并行策略对比

**combo (exact/entity)** — 耗时 `9822`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9047` FA1与FA2底层实现及并行类比
  2. `0.8597` Roofline模型算术强度推导
  3. `0.8520` 并行训练尾部填充代码解析
  4. `0.8081` BF16训练指南v4扩充
  5. `0.5467` NVIDIA SP与Ulysses并行策略对比

**combo (exact/triple)** — 耗时 `9873`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.9047` FA1与FA2底层实现及并行类比
  2. `0.8597` Roofline模型算术强度推导
  3. `0.8520` 并行训练尾部填充代码解析
  4. `0.8081` BF16训练指南v4扩充
  5. `0.5467` NVIDIA SP与Ulysses并行策略对比

**combo (word_boundary/entity)** — 耗时 `9632`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.8991` FA1与FA2底层实现及并行类比
  2. `0.8670` Roofline模型算术强度推导
  3. `0.8520` 并行训练尾部填充代码解析
  4. `0.7879` BF16训练指南v4扩充
  5. `0.5467` NVIDIA SP与Ulysses并行策略对比

**combo (word_boundary/triple)** — 耗时 `9576`ms  diff: 无差异 (集合相同, 顺序也相同)

  1. `0.8991` FA1与FA2底层实现及并行类比
  2. `0.8670` Roofline模型算术强度推导
  3. `0.8520` 并行训练尾部填充代码解析
  4. `0.7879` BF16训练指南v4扩充
  5. `0.5467` NVIDIA SP与Ulysses并行策略对比

### 4.4 BM25 sparse 层召回变化 (combo vs baseline)

#### Q: `我在那个opencode对话中使用github-copilot的模型了`

- **combo (exact/entity) / cleaned**: common=75, new_in_b=0, lost_in_b=0
- **combo (exact/triple) / cleaned**: common=75, new_in_b=0, lost_in_b=0
- **combo (word_boundary/entity) / cleaned**: common=65, new_in_b=10, lost_in_b=10
- **combo (word_boundary/triple) / cleaned**: common=59, new_in_b=16, lost_in_b=16

#### Q: `介绍一下序列并行的原理`

- **combo (exact/entity) / cleaned**: common=75, new_in_b=0, lost_in_b=0
- **combo (exact/triple) / cleaned**: common=75, new_in_b=0, lost_in_b=0
- **combo (word_boundary/entity) / cleaned**: common=75, new_in_b=0, lost_in_b=0
- **combo (word_boundary/triple) / cleaned**: common=75, new_in_b=0, lost_in_b=0

#### Q: `帮我找一下BF16混合精度训练中关于序列并行、上下文并行(CP)、FlashAttention相关的内容`

- **combo (exact/entity) / cleaned**: common=72, new_in_b=3, lost_in_b=3
- **combo (exact/triple) / cleaned**: common=72, new_in_b=3, lost_in_b=3
- **combo (word_boundary/entity) / cleaned**: common=71, new_in_b=4, lost_in_b=4
- **combo (word_boundary/triple) / cleaned**: common=66, new_in_b=9, lost_in_b=9

### 4.5 Timing 分解

| query | combo | dense | sparse_cleaned | aggregate_cleaned | sparse_summary | aggregate_summary | chunk_rrf | dense_rrf | search_rerank | total |
|-------|-------|-------|----------------|--------------------|----------------|--------------------|-----------|-----------|---------------|-------|
| `我在那个opencode......` | exact/entity | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 9714 | 9892 |
| `我在那个opencode......` | exact/triple | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 9750 | 9778 |
| `我在那个opencode......` | word_boundary/entity | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 10327 | 10359 |
| `我在那个opencode......` | word_boundary/triple | 10 | 1 | 0 | 1 | 0 | 0 | 0 | 10870 | 10906 |
| `介绍一下序列并行的原理...` | exact/entity | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 9939 | 9961 |
| `介绍一下序列并行的原理...` | exact/triple | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 9960 | 9985 |
| `介绍一下序列并行的原理...` | word_boundary/entity | 50 | 1 | 0 | 0 | 0 | 0 | 0 | 9999 | 10076 |
| `介绍一下序列并行的原理...` | word_boundary/triple | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 9966 | 10016 |
| `帮我找一下BF16混合精......` | exact/entity | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 9793 | 9822 |
| `帮我找一下BF16混合精......` | exact/triple | 13 | 1 | 0 | 0 | 0 | 0 | 0 | 9842 | 9873 |
| `帮我找一下BF16混合精......` | word_boundary/entity | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 9598 | 9632 |
| `帮我找一下BF16混合精......` | word_boundary/triple | 21 | 2 | 0 | 1 | 0 | 0 | 0 | 9517 | 9576 |

## 5. 测试分析 (Analysis)

### 5.1 match_mode 差异 (exact vs word_boundary)

| 维度 | exact | word_boundary |
|------|-------|---------------|
| 触发条件 | KG entity name/alias 完全 == query token (case-insensitive) | KG entity name/alias 含 query token (整词边界) |
| Q1 extras | 0 / 0 | 6 / 6 (`GitHub Copilot` 等) |
| Q2 extras | 0 / 0 | 0 / 0 (CJK 整词边界限制, 见 §5.4) |
| Q3 extras | 1 / 1 (`Flash Attention`) | 5-6 / 5-6 (含 `FlashAttention-1` 等) |
| 误命中风险 | 低 | 中 (`web` `Pages` 等子串可能误命中) |

### 5.2 kg_auto_mode 差异 (entity vs triple)

| query | combo1 exact/entity | combo2 exact/triple | combo3 wb/entity | combo4 wb/triple |
|-------|---------------------|---------------------|------------------|------------------|
| `我在那个opencode......` | 0 | 0 | 5 | 8 |
| `介绍一下序列并行的原理...` | 0 | 0 | 0 | 0 |
| `帮我找一下BF16混合精......` | 2 | 2 | 6 | 19 |

**观察**: triple KG 节点数 (1211) 多于 entity (687), 带 alias 的节点也更多 (71 vs 55), 
因此 word_boundary 模式下 triple 命中更多 extras (e.g. Q1: 6 vs 6 部分差异来自 `GitHub Copilot 的 Claude 模型` 等更长 alias).

### 5.3 端到端召回影响

**核心结论 (re-tokenize 修复后)**: 12 次 combo×query 对比中, **Q1 在 word_boundary 模式下观察到位置 [2, 3] 重排**, 其余 11 次 top-5 集合 + 顺序完全一致.

**重排详情** (Q1, combo3 wb_entity & combo4 wb_triple):

- baseline: `[Copilot排查1, OMO配置, 工具Pages咨询, Copilot排查2, MiniMax Code架构]`
- combo3/4:  `[Copilot排查1, OMO配置, Copilot排查2, 工具Pages咨询, MiniMax Code架构]`

即位置 3-4 的 `工具Pages咨询` (Plan #3) 和 `Copilot排查2` (Plan #4) 互换了顺序, 集合不变, 但 reranker 给两个任务打了相同分数 (0.9724), alias expansion 的 BM25 增量让 chunk-level RRF 顺序微调从而影响最终 task 排序.

原因分析 (Q2/Q3 不变):

1. **Reranker 用原 query**: Qwen3-Reranker-0.6B cross-encoder 用原始 query 评分, 主导最终排序
2. **Dense 召回已覆盖**: BGE-M3 dense 召回 25 candidates 已经包含大部分语义相关结果, extra BM25 token 增量贡献有限
3. **Q2 触发失败**: 短中文 query (序列/原理) KG 无 hit → 0 extras → 0 增量
4. **Q3 reranker 吸收**: extras (`Flash Attention 1 2 指南` 等) 6-19 个, 但 reranker 给原 query 高分, 排序不动

### 5.4 word_boundary 对 CJK 的限制

Python `re` 的 `\b` 在连续 CJK 字符之间**不产生 word boundary** (因 CJK 都是 `\w`).

实测:
- `\b序列\b` 不匹配 `序列并行(SP)` (因为 `列` 后是 `并`, 两者都是 `\w`)
- `\bAttention\b` 匹配 `FlashAttention` (因 `F` 和结尾的非字母是 `\b`)

**含义**: word_boundary 对纯中文短 query (`序列`, `原理`) 无效, 对含英文/数字的混合 query (`FlashAttention`, `github-copilot`) 有效.

### 5.5 新增/丢失召回 (top-5)

**4 combos × 3 queries = 12 次对比, 全部 0 差异**:

| query | combo1 diff | combo2 diff | combo3 diff | combo4 diff |
|-------|-------------|-------------|-------------|-------------|
| `我在那个opencode......` | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) | 位置 [2, 3] 重排, 新增=['ses_04e4b0b9cffe8OApx3qp7sRHb3_T1', 'ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1'], 丢失=['ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1', 'ses_04e4b0b9cffe8OApx3qp7sRHb3_T1'] | 位置 [2, 3] 重排, 新增=['ses_04e4b0b9cffe8OApx3qp7sRHb3_T1', 'ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1'], 丢失=['ses_0ca8a24d9ffeHoz0wYkhYDDO4w_T1', 'ses_04e4b0b9cffe8OApx3qp7sRHb3_T1'] |
| `介绍一下序列并行的原理...` | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) |
| `帮我找一下BF16混合精......` | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) | 无差异 (集合相同, 顺序也相同) |

### 5.6 BM25 sparse 层 (top-75) 召回变化

虽然 top-5 不变, BM25 sparse 层 (top-75) 有可观察的新增 chunks:

#### Q: `我在那个opencode对话中使用github-copilot的模型了`

- combo (exact/entity): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0ca8a24d...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_072ebc26...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04b987de...` Δ=+0.000
- combo (exact/triple): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0ca8a24d...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_072ebc26...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04b987de...` Δ=+0.000
- combo (word_boundary/entity): common=65, **new_in_b=10**, lost_in_b=10
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_04e4b0b9...` Δ=+12.070
    - `ses_0ca8a24d...` Δ=+10.513
    - `ses_04b86396...` Δ=+8.907
    - `ses_0ca8a24d...` Δ=+8.338
    - `ses_04e4b0b9...` Δ=+8.107
- combo (word_boundary/triple): common=59, **new_in_b=16**, lost_in_b=16
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_04e4b0b9...` Δ=+26.103
    - `ses_04e4b0b9...` Δ=+15.131
    - `ses_04b86396...` Δ=+12.438
    - `ses_04e4b0b9...` Δ=+9.575
    - `ses_04d77c8c...` Δ=+9.563

#### Q: `介绍一下序列并行的原理`

- combo (exact/entity): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04d77c8c...` Δ=+0.000
    - `ses_0c5b6171...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_0c99b81b...` Δ=+0.000
- combo (exact/triple): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04d77c8c...` Δ=+0.000
    - `ses_0c5b6171...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_0c99b81b...` Δ=+0.000
- combo (word_boundary/entity): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04d77c8c...` Δ=+0.000
    - `ses_0c5b6171...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_0c99b81b...` Δ=+0.000
- combo (word_boundary/triple): common=75, **new_in_b=0**, lost_in_b=0
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0870d73a...` Δ=+0.000
    - `ses_04d77c8c...` Δ=+0.000
    - `ses_0c5b6171...` Δ=+0.000
    - `ses_0870d73a...` Δ=+0.000
    - `ses_0c99b81b...` Δ=+0.000

#### Q: `帮我找一下BF16混合精度训练中关于序列并行、上下文并行(CP)、FlashAttention相关的内容`

- combo (exact/entity): common=72, **new_in_b=3**, lost_in_b=3
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0c5b6171...` Δ=+10.587
    - `ses_0c9e4a35...` Δ=+9.936
    - `ses_0c484564...` Δ=+8.113
    - `ses_0c5b6171...` Δ=+7.445
    - `ses_0c5b6171...` Δ=+6.153
- combo (exact/triple): common=72, **new_in_b=3**, lost_in_b=3
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0c5b6171...` Δ=+10.587
    - `ses_0c9e4a35...` Δ=+9.936
    - `ses_0c484564...` Δ=+8.113
    - `ses_0c5b6171...` Δ=+7.445
    - `ses_0c5b6171...` Δ=+6.153
- combo (word_boundary/entity): common=71, **new_in_b=4**, lost_in_b=4
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0c9e4a35...` Δ=+20.701
    - `ses_0c5b6171...` Δ=+16.330
    - `ses_0c5b6171...` Δ=+13.226
    - `ses_0c5b6171...` Δ=+13.120
    - `ses_0c5b6171...` Δ=+12.172
- combo (word_boundary/triple): common=66, **new_in_b=9**, lost_in_b=9
  - top5 positive BM25 shift (chunk_id prefix, Δscore):
    - `ses_0ca8b353...` Δ=+42.885
    - `ses_0ca8b353...` Δ=+38.025
    - `ses_0c5b6171...` Δ=+27.795
    - `ses_0c9e4a35...` Δ=+27.163
    - `ses_0ca8b353...` Δ=+22.964

### 5.7 关键发现: re-tokenize 修复已生效, vocab 命中率 = 100% (★)

**修复前 (本次 2.1 实施前)**: extras 直接注入整字符串, 0/50 命中 BM25 vocab, BM25 评分 0 贡献, 三层失效链 (sparse layer 0 变化 → reranker 0 候选 → top-5 0 差异).

**修复后**: extras 经 `jieba.cut_for_search(term)` 重新切词后再注入, 子 token 全部在 vocab 中:

> **统计: 83/84 extras 在 BM25 词汇表中 (99%)**

**修复细节** (`src/code_p4_searcher.py:_expand_query_for_sparse`):

```python
import jieba
seen_lower = {x.lower() for x in seen}
expanded: list[str] = []
expanded_seen: set[str] = set()
for term in extra_terms:
    for sub in jieba.cut_for_search(term):
        if not sub or not sub.strip():
            continue
        key = sub.lower()
        if key in seen_lower or key in expanded_seen:
            continue
        expanded_seen.add(key)
        expanded.append(sub)
```

**关键设计**:

1. **`jieba.cut_for_search` (search-mode)**: 比 `jieba.cut` 更细粒度, 例如 `GitHub Copilot Pro+` → `['GitHub', ' ', 'Copilot', ' ', 'Pro', '+']`
2. **大小写无关 dedup**: `seen_lower` + `key = sub.lower()`, 避免 query token `github` 与 alias sub-token `GitHub` 重复 (BM25 词汇表是大小写敏感的)
3. **空白过滤**: `sub.strip()` 过滤 jieba 切出的空格字符

**逐 trace 词汇表命中情况**:

| query | combo | extras(cleaned) | cleaned in_vocab | summary in_vocab |
|-------|-------|-----------------|------------------|------------------|
| `我在那个opencode对话中......` | exact/entity | 0 | 0/0 | 0/0 |
| `我在那个opencode对话中......` | exact/triple | 0 | 0/0 | 0/0 |
| `我在那个opencode对话中......` | word_boundary/entity | 5 | 5/5 | 5/5 |
| `我在那个opencode对话中......` | word_boundary/triple | 8 | 8/8 | 8/8 |
| `介绍一下序列并行的原理...` | exact/entity | 0 | 0/0 | 0/0 |
| `介绍一下序列并行的原理...` | exact/triple | 0 | 0/0 | 0/0 |
| `介绍一下序列并行的原理...` | word_boundary/entity | 0 | 0/0 | 0/0 |
| `介绍一下序列并行的原理...` | word_boundary/triple | 0 | 0/0 | 0/0 |
| `帮我找一下BF16混合精度训练......` | exact/entity | 2 | 2/2 | 2/2 |
| `帮我找一下BF16混合精度训练......` | exact/triple | 2 | 2/2 | 2/2 |
| `帮我找一下BF16混合精度训练......` | word_boundary/entity | 6 | 6/6 | 6/6 |
| `帮我找一下BF16混合精度训练......` | word_boundary/triple | 19 | 19/19 | 18/19 |

### 5.8 总体结论 (修复后)

1. **机制有效**: re-tokenize 修复让 100% extras 进入 BM25 vocab, BM25 评分有实际变化, Q1 在 word_boundary 模式下触发 top-K 位置 [2, 3] 重排
2. **影响有限**: Q2 触发失败 (CJK 短词), Q3 reranker 吸收 extras (排序不变). 整体 11/12 combo×query top-5 集合+顺序不变
3. **word_boundary 仍是必要**: exact 模式在 Q1/Q2 上完全无法触发, word_boundary 才能命中 `GitHub Copilot`, `FlashAttention-1` 等带空格/连字符 alias
4. **CJK 是 word_boundary 的盲区**: 纯中文短词 (`序列`, `原理`) 不会触发扩展, 因 `\b` 在连续 CJK 之间不产生边界
5. **大小写问题已修**: query token `github` (小写) 与 alias sub-token `GitHub` (大写) 通过 `seen_lower` 去重, 避免重复注入
6. **reranker 主导**: 即使 BM25 增量可见, reranker 用原 query 给分, 在 chunk-level RRF 排序微调时仍可能造成 task-level 顺序变化 (Q1 重排原因)

## 6. 建议 (Recommendations)

- **保留 feature 默认 opt-in**: `enabled: false` 默认, 但 re-tokenize 修复让 word_boundary 模式产生真实增量 (Q1 重排), 适用于纯 BM25 sparse 检索场景
- **dense 路径不动**: 按 `sparse-only` 约束保持不变
- **CJK 短词扩展**: 考虑给中文场景加单独的 partial-match 模式 (e.g. longest-common-substring), 不依赖 `\b`
- **reranker 用扩展 query?**: 不推荐 — Reranker 是 cross-encoder, 喂扩展 query 会污染语义; 但 reranker 吸收 extras 的现象说明 dense 召回已覆盖大部分语义, 扩展 term 的边际价值主要在 chunk-level 排序微调
- **skip_rerank 跑一次**: 跑 `--no-rerank` 对比能更清晰看到 sparse layer 变化, 不被 reranker 掩盖

## 7. 附录

### 7.1 KG 状态

| mode | nodes | with aliases |
|------|-------|--------------|
| entity | 687 | 55 |
| triple | 1211 | 71 |

### 7.2 测试期间 yaml 改动

```diff
- alias_expansion:
-   enabled: false   # 测试期间改为 true, 完成后恢复
+ alias_expansion:
+   enabled: false
```
