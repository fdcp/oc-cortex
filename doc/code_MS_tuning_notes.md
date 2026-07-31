## 里程碑第6步: Prompt 调优与整理规则分析

### 现状评估

基于 code_MS_inspect.py 检视结果，当前系统整体质量良好：

| 指标 | 结果 | 评价 |
|------|------|------|
| Dense 检索命中率 (Top-5) | 5/5 (100%) | 优秀 |
| Embedding 区分度 | 无 cosine > 0.85 的近重复 task | 良好 |
| Chunk summary 覆盖率 | 86/86 (100%) | 完整 |
| Task summary 平均长度 | 306 字符 | 适中 |
| Token 压缩率 | 63.1% (124K → 78K) | 有效 |
| 每 session 平均 task 数 | 1.8 | 合理 |

### 发现的问题

#### 问题 1: 低信息密度 chunk 过度切分 (Phase 1 chunker)

session ses_10772d114... (omo 插件安装) 共 13 个 chunk，其中 5 个 (c3-c7) 仅是用户重复询问"这个可以用吗"，信息量为零。类似地，ses_0ee881bdd... c1 是"用户上传图片无说明"。

这些 chunk 在 chunks_cleaned_text 集合中各占一个向量点，稀释了有效信息的检索权重。

**改进方向 (Phase 1 chunker)**:
- 新增参数 `min_user_message_length`（当前仅过滤 session 级别，未过滤 chunk 级别）
- 对 user_message 过短（< 20 字）且无 tool_calls 的 chunk，尝试与相邻 chunk 合并
- 或标记为 `trivial: true`，Phase 3 写入时可选择跳过

#### 问题 2: task_summary 模板化 (Phase 2 prompt)

当前 prompt 要求所有 task_summary 遵循"背景与目标 → 核心产出 → 结论与意义"三段式。对深度技术 task 效果好，但对简单 task（如"安装 tmux"、"解释 http.server 命令"）显得冗余。

**改进方向 (Phase 2 prompt)**:
在 `SESSION_TASK_PROMPT` 的 `【task_summary 写作要求】` 部分增加弹性：

```
对于技术深度高的任务（涉及分析、设计、实现、对比），使用完整三段式结构。
对于简单操作性任务（安装工具、解释命令、回答概念），使用简洁结构：
- 一句话说明做了什么
- 保留关键技术实体和参数
- 不需要强行添加"结论与意义"
```

#### 问题 3: chunk_summary 与 task_summary 信息重叠

当一个 task 只关联 1-2 个 chunk 时，task_summary 往往是 chunk_summary 的拼接改写。两者向量化后语义高度重合。

**改进方向**:
- 短期可接受：数据量小时冗余不影响检索
- 长期方案：Phase 3 已将 chunks 拆为 chunks_summary (dense: summary) 和 chunks_cleaned_text (dense+BM25: cleaned_text) 两个独立集合，避免 summary 被 cleaned_text 稀释

#### 问题 4: chunk cleaned_text 压缩率异常

部分 chunk 压缩率 > 100%（cleaned > raw），因为 content_cleaner 添加的结构标签（`[TOOL_CALL]`、`[OUTPUT]` 等）在短 tool output 场景下比原文更长。这在 mock 数据中已预见，但真实数据中 FlashAttention session 的多轮推导也出现了（如 c3: raw=329, cleaned=335）。

**改进方向 (Phase 1 content_cleaner)**:
- 对 cleaned_size_tokens > raw_size_tokens 的 chunk，直接使用 raw text
- 或减少结构标签的使用密度

### 是否需要重新运行

| 改进 | 影响范围 | 是否需要重跑 | 优先级 |
|------|---------|-------------|--------|
| 合并低信息 chunk | Phase 1 → 2 → 3 全链路 | 是 | 中（当前不影响检索） |
| task_summary 弹性模板 | Phase 2 → 3 | 重跑 Phase 2+3 | 低（当前质量可接受） |
| chunk embedding 策略 | Phase 3 | 重跑 Phase 3 | 低 |
| cleaned_text 回退 | Phase 1 → 2 → 3 全链路 | 是 | 低 |

**建议**: 当前数据规模（27 tasks / 86 chunks）下检索质量已达标。上述改进更适合在数据规模扩大后（>100 sessions）实施，届时低信息 chunk 和冗余向量的影响会被放大。当前阶段优先推进后续 Phase（知识图谱），待数据量增长后再回头调优。

### 后续行动

1. **立即执行**: 无（当前质量达标，无需重跑）
2. **数据规模 > 100 sessions 时**:
   - 修改 Phase 1 chunker，合并 trivial chunks
   - 修改 Phase 2 prompt，增加弹性模板
   - 修改 Phase 3 embedding 策略，减少冗余
3. **检索质量回归测试**: 每次 prompt 修改后用 `code_MS_inspect.py` 的检索测试验证命中率不降

### 检视工具本身的改进（已实施）

原 `code_MS_inspect.py` 存在与项目规范不一致的问题，已修复：

| 问题 | 原状 | 修复 |
|------|------|------|
| 硬编码相对路径 | `./qdrant_data`、`./output/*.jsonl` 写死，从 `src/` 运行会找不到文件 | 统一经 `Config`（`config/code_p3_config.yaml`）读取，相对路径回退到仓库根目录，可从任意 CWD 运行 |
| 硬编码集合名 | `"tasks"`、`"chunks_summary"` 等字符串散落各函数 | 从 `qdrant.collections.*` 配置读取并逐层传参 |
| 硬编码 embedding 模型 | 检索测试写死 `BAAI/bge-small-zh-v1.5` / `cpu` | 从 `embedding.model` / `embedding.device` / `embedding.cache_folder` 读取，与 Phase 3 保持一致 |
| 缺少命令行参数 | 无 `argparse` | 新增 `--config` / `--samples`，符合项目 CLI 约定 |

修复后行为等价（默认配置下输出不变），但避免了模型/集合名漂移，并可跨目录运行。

