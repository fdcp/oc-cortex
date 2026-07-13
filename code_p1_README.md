# Phase 1 代码骨架

OpenCode Session 知识图谱系统 - Phase 1:数据预处理

## 文件结构

```
code_p1_*.{py,yaml,md,txt}
├── code_p1_config.yaml         # 配置文件
├── code_p1_models.py           # 数据模型(Session, Turn, Chunk, ...)
├── code_p1_utils.py            # 工具(Config 加载, Logger, Token 计数 + 截断)
├── code_p1_session_loader.py   # Session 加载器(JSONL/JSON + Mock + 过滤)
├── code_p1_sqlite_loader.py    # Session 加载器(直接读 OpenCode SQLite DB)
├── code_p1_chunker.py          # 轮次切分(按 user 消息切)
├── code_p1_content_cleaner.py  # 内容整理(降噪 + 减体积)
├── code_p1_main.py             # 主入口
├── code_p1_requirements.txt    # 依赖
└── code_p1_README.md           # 本文件
```

## 流程

```
OpenCode Session (SQLite DB / JSONL / JSON)
    ↓
[Session Loader] 加载 + 过滤(root + user_count >= 2)
    ↓
[Chunker] 按 user 消息切分轮次
    ↓
[Content Cleaner] 整理(工具调用摘要化,失败信息截断)
    ↓
Chunks (JSONL)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r code_p1_requirements.txt
```

### 2. 用 Mock 数据跑一遍(无需真实 session)

```bash
python3 code_p1_main.py --mock
```

**预期输出**:
```
11:42:15 | INFO | Phase 1: 数据预处理启动
11:42:15 | INFO | 使用 mock 数据
11:42:15 | INFO | 原始加载 3 个 session
11:42:15 | INFO | 过滤后保留 2 个 session  (mock_skip 因为只有 1 个 user 被过滤)
11:42:15 | INFO | Session ses_mock_001: 切分出 2 个 chunk
11:42:15 | INFO | Session ses_mock_002: 切分出 2 个 chunk
11:42:15 | INFO | 已写入 4 个 chunk 到 ./output/chunks.jsonl
11:42:15 | INFO | Phase 1 完成
...
```

### 3. 用 OpenCode SQLite 数据库(推荐)

直接读 OpenCode 的 `opencode.db`,无需手动导出:

```bash
# 全量
python3 code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db

# 快速验证(只跑前 3 个 session)
python3 code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3

# 指定输出路径
python3 code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --output ./output/chunks_real.jsonl
```

**实测结果**(2026-07-13,20 个 session):
```
查询到 20 个 session
加载完成: 15 个 session (跳过 5 个 user 消息 < 2 的)
生成 chunk: 86 个
总 raw ≈ 107K tokens → cleaned ≈ 77K tokens
```

各 session 压缩效果:
| Session | Chunks | Raw Tokens | Cleaned Tokens | 压缩率 |
|---------|--------|------------|----------------|--------|
| 通过db文件查询opencode session | 3 | 6,321 | 2,386 | 62% |
| mavis-branch (长对话) | 13 | 17,297 | 17,375 | -1%* |
| mavis-branch (工具密集) | 13 | 17,696 | 12,577 | 29% |
| 安装tmux | 13 | 14,180 | 3,855 | 73% |
| 了解你的能力 | 5 | 18,222 | 8,808 | 52% |

> *纯对话型 session 工具调用少,整理后加的 `[User]`/`[Assistant]` 标签反而增加了少量 token。

### 4. 用 JSONL 数据

把 OpenCode session 导出为 JSONL(每行一个 session),然后:

```bash
python3 code_p1_main.py --source /path/to/opencode_sessions.jsonl --output ./output/chunks.jsonl
```

## 关键设计

### 1. 数据源优先级

`--sqlite` > `--source` > `--mock` > config `mock_data` > config `session_source`

### 2. SQLite 数据模型

OpenCode 使用 SQLite 存储,三层结构:

```
session (id, parent_id, title, time_created, ...)
  └─ message (id, session_id, data=JSON)
       └─ part (id, message_id, data=JSON)
```

- **message.data**: `{ "role": "user"|"assistant", "agent": "...", "model": {...}, "tokens": {...}, "cost": ... }`
- **part.data** 按 type 分类:
  - `text`: `{ "type": "text", "text": "实际文本内容" }`
  - `tool`: `{ "type": "tool", "tool": "bash"|"write"|..., "state": { "status": "completed", "input": {...}, "output": "...", "metadata": { "exit": 0 } } }`
  - `reasoning`: 模型推理过程(Phase 1 跳过)
  - `file`: 图片等附件(Phase 1 跳过)
  - `step-start` / `step-finish`: 步骤标记(Phase 1 跳过)

**过滤规则**:
- `parent_id IS NULL` → root session
- `parent_id IS NOT NULL` → branch session
- user 消息中 `<system-reminder>` 块会被自动清除

### 3. Session 过滤

- **只处理 root session**: 排除分支 session,避免重复处理
- **至少 2 个 user 输入**: 单次问答的 session 没什么可挖掘的

### 4. Chunk 切分

- **轮次边界 = user 消息出现位置**
- 每个 chunk = 1 个 user + 后续所有非 user 直到下一个 user
- 例外: 如果 session 以非 user 开头,跳过首条并 warn

### 5. 内容整理(降噪)

| 角色 | 处理方式 |
|------|---------|
| user / assistant 文本 | 完整保留 |
| bash | `{"type":"bash", "summary":"原命令", "ok":true/false, "error":...}` |
| tool_call | `{"type":"tool_call", "summary":"action", "ok":..., "error":...}` |
| mcp_call | `{"type":"mcp_call", "summary":"target", "ok":..., "error":...}` |
| 失败调用 | error 字符串截断到 200 字符 |

### 6. Token 计数 + 截断

- 用 `tiktoken` 的 `cl100k_base` 编码近似(Qwen3 同源)
- 离线时回退到字符数 / 4 估算
- `safe_truncate()` 保留首 70% + 尾 30%,中间省略号

## 输出格式

`output/chunks.jsonl` 每行一个 chunk:

```json
{
  "chunk_id": "ses_mock_001_c1",
  "session_id": "ses_mock_001",
  "turn_index": 1,
  "user_message": "帮我修一下登录 bug,登录一直失败",
  "assistant_messages": ["好的,我先排查 token 验证逻辑", "找到问题了,是 token 过期判断错了"],
  "tool_calls": [
    {"type": "bash", "summary": "grep -n 'token' src/auth.py", "ok": true, "error": null}
  ],
  "mcp_calls": [],
  "raw_size_tokens": 120,
  "cleaned_size_tokens": 75,
  "created_at": "2026-07-10T10:00:00Z"
}
```

## Mock 数据说明

`get_mock_sessions()` 返回 3 个 session:
- `ses_mock_001`: 2 个 user → 切 2 个 chunk(修 bug + 加测试)
- `ses_mock_002`: 2 个 user → 切 2 个 chunk(性能监控讨论 + 部署)
- `ses_mock_skip`: 1 个 user → **被过滤掉**

跑 `--mock` 能验证:
- ✅ 过滤逻辑(过滤掉 `ses_mock_skip`)
- ✅ 切分逻辑(2 个 session 各切 2 个 chunk)
- ✅ 整理逻辑(bash 命令被摘要化,失败信息被截断)
- ✅ Token 压缩比

## 下一步

Phase 1 跑通后,接 Phase 2:`task_extractor.py` — 用 LLM 整 session 视角抽取 task 清单。

需要的 TODO:
- [x] ~~确认 OpenCode session 实际数据格式~~ — 已适配 SQLite DB(opencode.db)
- [ ] 接真实 LLM(可先 mock 一个 `_mock_llm_call` 函数)
- [ ] 评估切分粒度:目前 1 user = 1 chunk,是否需要再细分?
