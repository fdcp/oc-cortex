# Phase 2: Task 清单生成

## 概述

Phase 2 是知识图谱系统的核心环节:利用 LLM 对 Phase 1 输出的 session chunks 进行整 session 视角的分析,提炼出 1-5 个 session 级 Task。每个 Task 包含简短标签、详细总结和关联的 chunk 列表。同时,LLM 还会为每个 chunk 生成一句话总结(task_summary),输出到独立的 `chunks_summary_p2.jsonl` 文件,为后续 Phase 3 的向量化嵌入和跨 session 检索提供更丰富的语义信息。

## 文件结构

```
src/
  code_p2_models.py           # Task 数据模型 (dataclass)
  code_p2_task_extractor.py   # 核心模块: CoT Prompt + LLM 调用 + JSON 解析/修复 + 校验
  code_p2_main.py             # 入口脚本
config/
  code_p2_config.yaml         # Phase 2 配置 (LLM 参数 + 并发数 + 输出路径)
doc/
  code_p2_README.md           # 本文档
```

## 核心流程

```
Phase 1 chunks (JSONL)
        │
   [按 session 分组]
        │
   [构建 CoT Prompt] ── 所有 chunk 格式化后注入三步分析模板
        │
   [调用 LLM] ── OpenAI 兼容 API, 支持多线程并发
        │
   [解析 JSON] ── 提取 + 截断修复
        │
   [校验 + 兜底] ── 任务数量 / chunk_id 合法性 / 自动补全未覆盖 chunk
        │
   ├─→ Task 列表 (output/tasks.jsonl)
   └─→ Chunk 总结 (output/chunks_summary_p2.jsonl)
```

## CoT Prompt 设计

采用三步 Chain-of-Thought 分析流程,确保 LLM 先理解内容再归类,避免遗漏:

**步骤 1: 逐轮总结** — 逐一阅读每个轮次,用 1-2 句话概括该轮"做了什么"。忽略格式噪音(如 `<permission-response>`、`<mavis-attachments>`),关注实质内容。输出 `chunk_summaries` 数组。

**步骤 2: 识别任务** — 基于步骤 1 的总结,归纳 1-5 个独立任务。粒度是"一件独立的事"而非"一轮对话"。

**步骤 3: 归类轮次** — 将每个轮次归入所属任务。约束: 每个轮次都必须归入至少一个任务,不允许遗漏。

LLM 输出格式:

```json
{
  "chunk_summaries": [
    {"chunk_id": "c1", "summary": "该轮次的一句话总结"},
    {"chunk_id": "c2", "summary": "..."}
  ],
  "tasks": [
    {
      "task_id": "T1",
      "task_label": "简短中文标签(5-15 字)",
      "task_summary": "该任务的总结(50-200 字)",
      "chunk_ids": ["c1", "c2", "c3"]
    }
  ]
}
```

## 快速开始

### 1. 设置 API Key

```bash
# 使用 OpenCode Zen (推荐, 免费模型 hy3-free)
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
```

### 2. 运行

```bash
# 全量运行 (使用配置文件中的 concurrency, 默认 4 线程)
python3 src/code_p2_main.py --config config/code_p2_config.yaml

# 快速验证 (只处理前 3 个 session)
python3 src/code_p2_main.py --config config/code_p2_config.yaml --limit 3

# 指定并发数 (覆盖配置文件)
python3 src/code_p2_main.py --config config/code_p2_config.yaml --concurrency 8

# 指定输入/输出
python3 src/code_p2_main.py --config config/code_p2_config.yaml \
  --chunks ./output/chunks.jsonl --output ./output/tasks.jsonl
```

### 3. 输出

Phase 2 产出两个文件,**不修改** Phase 1 的 `chunks.jsonl`:

**output/tasks.jsonl** — 每行一个 Task JSON 对象:

```json
{
  "task_id": "ses_0c4e0312affeUm1WYU03ISPgU2_T1",
  "session_id": "ses_0c4e0312affeUm1WYU03ISPgU2",
  "task_label": "RoPE预计算函数解析",
  "task_summary": "背景与目标：理解 LLaMA 风格 RoPE 位置编码中 precompute_freqs_cis 函数的实现与张量形状。核心产出与关键决策：补全了截断代码，标准实现通过 torch.polar 生成复数旋转因子...",
  "chunk_ids": ["ses_0c4e0312affeUm1WYU03ISPgU2_c1", "ses_0c4e0312affeUm1WYU03ISPgU2_c2"],
  "created_at": "2026-07-19T02:20:43.283866"
}
```

字段说明:

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | `{session_id}_T{序号}`, 从 1 开始 |
| `session_id` | string | 来源 session |
| `task_label` | string | 简短中文标签 (5-15 字) |
| `task_summary` | string | 任务总结 (50-200 字) |
| `chunk_ids` | string[] | 关联的 chunk ID 列表 |
| `created_at` | string | ISO 8601 时间戳 |

**output/chunks_summary_p2.jsonl** — 每行一个 chunk 总结, 仅含 `chunk_id` 和 `summary` 两个字段:

```json
{
  "chunk_id": "ses_0a16b26dcffewGBp0akSLB5IZG_c1",
  "summary": "确认当前工作区不存在 session ID 为 ses_12c6bd8dfffevT51PypMW2v5Mx 的记录，仅列出本地 oc_sess_graph 项目下的唯一 session。"
}
```

输出路径可通过配置文件 `config/code_p2_config.yaml` 修改:

```yaml
output:
  chunk_summaries: "./output/chunks_summary_p2.jsonl"
```

`tasks.jsonl` 的输出路径通过 `--output` 命令行参数指定, 默认 `./output/tasks.jsonl`。

## 技术细节

### 并发处理

支持可配置的线程池并行(`concurrent.futures.ThreadPoolExecutor`):

- 配置文件 `config/code_p2_config.yaml` 中 `llm.concurrency` 设置默认并发数
- 命令行 `--concurrency N` 可覆盖配置文件
- `concurrency=1` 走串行路径,`>1` 走线程池并行
- OpenAI SDK 的 client 是线程安全的,无需额外加锁
- 进度计数器使用 `threading.Lock` 保证线程安全

### Token 预算管理

- 单个 chunk 在 prompt 中最多 2000 tokens (超限自动截断,首尾保留策略)
- 整个 prompt 最多 30000 tokens (超限时按比例缩减每个 chunk)
- LLM 输出 `max_tokens=16000` (CoT 输出含 chunk_summaries, 需要更多 token)
- `max_tokens` 仅控制输出 token 数, 不含输入

### JSON 截断修复

LLM 输出可能因 `max_tokens` 限制被截断,导致 JSON 不完整。修复策略:

1. 追踪未关闭的括号栈 (`{`, `[`) 和字符串状态
2. 在截断处关闭未结束的字符串
3. 移除不完整的尾部元素
4. 按栈中剩余括号逆序闭合

先尝试正常解析,失败后再尝试修复,修复仍失败才抛异常。

### 未覆盖 chunk 自动兜底

如果 LLM 漏掉了某些 chunk 未归入任何 task:

1. 检测到未覆盖的 chunk 后自动触发兜底
2. 按 `turn_index` 计算每个 chunk 与已归类 chunk 的距离
3. 将未覆盖 chunk 归入 turn_index 最近的 task

### Reasoning 模型适配

DeepSeek V4 Flash 是 reasoning 模型,会在 `reasoning_content` 中产生 thinking tokens:

- `max_tokens=16000`, 避免 thinking 耗尽 token 导致 content 为空
- Fallback: 当 content 为空时, 尝试从 `reasoning_content` 提取 JSON

### 校验规则

1. JSON 格式合法性 (失败时尝试截断修复)
2. 任务数量 [1, 5] (超范围仅警告,不阻断)
3. chunk_id 对应实际 chunk (支持短 ID "c1" 和完整 ID 自动转换)
4. chunk 覆盖率检查 + 自动兜底分配
5. 重复归属检查 (一个 chunk 不应属于多个 task)

### 失败重试

分为两层重试机制:

- **网络层重试** (`llm.max_retries`, 默认 3): 针对网络/API 调用异常 (超时、连接失败等),指数退避 2s → 4s → 8s
- **内容校验重试** (`llm.content_retries`, 默认 3): 针对 LLM 返回内容不完整 (chunk_summaries 覆盖不全或 tasks 为空),会清除上一次残留的 `task_summary` 后重新请求;达到最大重试次数后使用最后一次的不完整结果 (标记为 `incomplete`),不会阻断整体流程

### 生成参数

- `llm.temperature` (默认 0.2~0.3): 控制生成的随机性,数值越低输出越稳定/确定,适合结构化 JSON 提取任务

## 真实数据测试结果

在 16 个真实 OpenCode session (83 个 chunks) 上运行:


| 指标                   | 值                          |
| ------------------------ | ----------------------------- |
| 处理 session 数        | 16 (全部成功)               |
| 生成 task 总数         | 31                          |
| 平均每 session task 数 | 1.9                         |
| chunk 总结覆盖率       | 83/83 (100%)                |
| LLM 模型               | hy3-free (OpenCode Zen)     |
| 并发数                 | 4                           |

### Task 分布


| 每 session task 数 | session 数量 |
| -------------------- | -------------- |
| 1                  | 6            |
| 2                  | 6            |
| 3                  | 3            |
| 4                  | 1            |

## 支持的 LLM 提供商

通过修改 `config/code_p2_config.yaml` 切换:


| 提供商               | model                   | base_url                                          | api_key_env          |
| ---------------------- | ------------------------- | --------------------------------------------------- | ---------------------- |
| OpenCode Go (订阅)   | hy3 (默认)              | https://opencode.ai/zen/go/v1                     | OPENCODE_ZEN_API_KEY |
| OpenCode Zen (免费)  | deepseek-v4-flash-free  | https://opencode.ai/zen/v1                        | OPENCODE_ZEN_API_KEY |
| OpenCode Zen (免费)  | mimo-v2.5-free          | https://opencode.ai/zen/v1                        | OPENCODE_ZEN_API_KEY |
| OpenCode Zen (免费)  | nemotron-3-ultra-free   | https://opencode.ai/zen/v1                        | OPENCODE_ZEN_API_KEY |
| DashScope (通义千问) | qwen-plus               | https://dashscope.aliyuncs.com/compatible-mode/v1 | DASHSCOPE_API_KEY    |
| SiliconFlow          | deepseek-ai/DeepSeek-V3 | https://api.siliconflow.cn/v1                     | SILICONFLOW_API_KEY  |
| OpenAI               | gpt-4o-mini             | https://api.openai.com/v1                         | OPENAI_API_KEY       |
| Ollama (本地)        | qwen2.5:7b              | http://localhost:11434/v1                         | (不需要)             |

> **注意**: `hy3-free` 已从 OpenCode Zen 免费层下线。`hy3` 现仅通过 OpenCode Go 订阅端点 (`https://opencode.ai/zen/go/v1`, 模型 ID `hy3`) 提供, API Key 仍取自 `auth.json` 的 `opencode-go` provider。多模型对比测试的端点/认证配置见 `tests/test_p2/config.yaml`。

## 依赖

```
openai>=1.30.0
loguru>=0.7.0
pyyaml>=6.0
tiktoken>=0.5.0
```

## 下一步

Phase 2 的输出作为后续 Phase 的输入:

- **Phase 3**: 将 `tasks.jsonl` 中的 Task 向量化存入 Qdrant `tasks` 集合 (dense embedding of `task_summary`), 支持跨 session 语义检索
- **Phase 3**: 将 `chunks_summary_p2.jsonl` 中的 chunk 总结存入 Qdrant `chunks_summary` 集合, 提供轮次级语义检索
- **Phase 4**: 基于 `chunks.jsonl` 的 `cleaned_text()` 构建 BM25 索引, 与 Phase 3 的 dense 检索融合 (RRF), 实现混合搜索
- Task summary 提供高层语义, chunk summary 提供轮次级语义, cleaned_text 提供全文细节, 形成三层检索结构
