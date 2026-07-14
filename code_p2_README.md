# Phase 2: Task 清单生成

## 概述

Phase 2 是知识图谱系统的核心创新环节:利用 LLM 对 Phase 1 输出的 session chunks 进行整 session 视角的分析,提炼出 1-5 个 session 级 Task。每个 Task 包含简短标签、详细总结和关联的 chunk 列表,为后续 Phase 3 的向量化嵌入和跨 session 检索提供基础。

## 文件结构

```
code_p2_models.py           # Task 数据模型 (dataclass)
code_p2_task_extractor.py   # 核心模块: LLM 调用 + JSON 解析 + 校验
code_p2_config.yaml         # Phase 2 配置 (LLM 参数)
code_p2_main.py             # 入口脚本
code_p2_README.md           # 本文档
```

## 核心流程

```
Phase 1 chunks (JSONL)
        │
   [按 session 分组]
        │
   [构建 prompt] ── 所有 chunk 格式化后注入 SESSION_TASK_PROMPT
        │
   [调用 LLM] ── OpenAI 兼容 API (DeepSeek V4 Flash Free / Qwen / ...)
        │
   [解析 JSON] ── 提取 ```json``` 块或裸 JSON
        │
   [校验] ── 任务数量 [1,5] / chunk_id 合法性 / 覆盖率 / 无重复归属
        │
   Task 列表 (JSONL)
```

## 快速开始

### 1. 设置 API Key

```bash
# 使用 OpenCode Zen (免费 deepseek-v4-flash-free)
export OPENCODE_ZEN_API_KEY='your-api-key'

# 或使用 DashScope (通义千问)
export DASHSCOPE_API_KEY='your-api-key'
```

### 2. 运行

```bash
# 全量运行 (处理 Phase 1 所有 session)
python code_p2_main.py

# 快速验证 (只处理前 3 个 session)
python code_p2_main.py --limit 3

# 指定输入/输出
python code_p2_main.py --chunks ./output/chunks.jsonl --output ./output/tasks.jsonl

# 自定义配置
python code_p2_main.py --config code_p2_config.yaml
```

### 3. 输出

输出文件: `./output/tasks.jsonl`,每行一个 Task JSON 对象:

```json
{
  "task_id": "ses_0a53b6807ffe_T1",
  "session_id": "ses_0a53b6807ffex4xr4a0wNdi5EQ",
  "task_label": "查询 opencode session 概览",
  "task_summary": "通过查询 opencode.db,获取本机所有 22 个 session 的详细信息...",
  "chunk_ids": ["ses_0a53b6807ffex4xr4a0wNdi5EQ_c1"],
  "created_at": "2026-07-14T01:47:29.123456"
}
```

## 真实数据测试结果

在 15 个真实 OpenCode session (86 个 chunks) 上运行:

| 指标 | 值 |
|------|------|
| 处理 session 数 | 15 (全部成功) |
| 生成 task 总数 | 30 |
| 平均每 session task 数 | 2.0 |
| Task 分布 | 1 task: 4 sessions, 2 tasks: 7 sessions, 3 tasks: 4 sessions |
| LLM 模型 | deepseek-v4-flash-free (OpenCode Zen) |
| 总耗时 | ~8.5 分钟 (含 LLM reasoning) |
| 单 session 耗时 | 15-80 秒 (取决于 chunk 数量) |

### Task 关联 chunk 数分布

| 关联 chunk 数 | task 数量 |
|-------------|----------|
| 1 | 10 |
| 2 | 11 |
| 3 | 3 |
| 4 | 1 |
| 5 | 1 |
| 6 | 2 |
| 11 | 2 |

## 技术细节

### Prompt 设计

采用整 session 视角 (ADR-001),将 session 所有 chunk 按顺序格式化后注入 prompt,让 LLM 从全局视角识别 1-5 个独立任务。

关键约束:
- 同一件事可能跨多个对话轮次
- 大多数 session 包含 1-2 个任务
- 任务之间应相对独立,不是子步骤

### Token 预算管理

- 单个 chunk 在 prompt 中最多 2000 tokens (超限自动截断,首尾保留策略)
- 整个 prompt 最多 30000 tokens (超限时按比例缩减每个 chunk)
- LLM 输出 max_tokens=8000 (reasoning 模型需要额外 token 给 thinking)

### Reasoning 模型适配

DeepSeek V4 Flash 是 reasoning 模型,会在 `reasoning_content` 中产生 thinking tokens。处理策略:
- 提高 max_tokens 至 8000,避免 thinking 耗尽 token 导致 content 为空
- Fallback: 当 content 为空时,尝试从 reasoning_content 提取 JSON

### 校验规则

1. JSON 格式合法性
2. 任务数量 [1, 5] (超范围仅警告,不阻断)
3. chunk_id 对应实际 chunk (支持短 ID "c1" 和完整 ID 自动转换)
4. chunk 覆盖率检查 (未覆盖的 chunk 记录警告)
5. 重复归属检查 (一个 chunk 不应属于多个 task)

### 失败重试

指数退避重试: 2s → 4s → 8s,最多 3 次。

## 支持的 LLM 提供商

通过修改 `code_p2_config.yaml` 切换:

| 提供商 | model | base_url | api_key_env |
|--------|-------|----------|-------------|
| OpenCode Zen (免费) | deepseek-v4-flash-free | https://opencode.ai/zen/v1 | OPENCODE_ZEN_API_KEY |
| DashScope (通义千问) | qwen-plus | https://dashscope.aliyuncs.com/compatible-mode/v1 | DASHSCOPE_API_KEY |
| SiliconFlow | deepseek-ai/DeepSeek-V3 | https://api.siliconflow.cn/v1 | SILICONFLOW_API_KEY |
| OpenAI | gpt-4o-mini | https://api.openai.com/v1 | OPENAI_API_KEY |
| Ollama (本地) | qwen2.5:7b | http://localhost:11434/v1 | (不需要) |

## 依赖

```
openai>=1.30.0
loguru>=0.7.0
pyyaml>=6.0
tiktoken>=0.5.0
```

## 下一步

Phase 2 的 Task 输出将作为 Phase 3 的输入:
- **Phase 3**: Qdrant 双 Collection (Tasks + Chunks) 向量化存储
- Task 的 `task_label` + `task_summary` 用于 embedding 和跨 session 检索
- Chunk 的 `cleaned_text()` 用于细粒度检索
