# Phase 1 数据预处理

OpenCode Session 知识图谱系统 - Phase 1:数据预处理

## 文件结构

```
oc_sess_graph/
├── src/
│   ├── code_p1_main.py             # 主入口
│   ├── code_p1_models.py           # 数据模型(Session, Turn, Chunk, ...)
│   ├── code_p1_utils.py            # 工具(Config 加载, Logger, Token 计数 + 截断)
│   ├── code_p1_session_loader.py   # Session 加载器(JSONL/JSON + Mock + 过滤)
│   ├── code_p1_sqlite_loader.py    # Session 加载器(直接读 OpenCode SQLite DB)
│   ├── code_p1_chunker.py          # 轮次切分(按 user 消息切 + 后处理合并)
│   └── code_p1_content_cleaner.py  # 内容整理(降噪 + 减体积)
├── config/
│   ├── code_p1_config.yaml         # 配置文件
│   └── code_p1_requirements.txt    # 依赖
└── doc/
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
[Chunker 后处理] 合并空 chunk + 去重连续相同 user_message
    ↓
[Content Cleaner] 整理(工具调用摘要化,失败信息截断)
    ↓
Chunks (JSONL)
```

## 快速开始

> 以下命令均在项目根目录 `oc_sess_graph/` 下执行。

### 1. 安装依赖

```bash
pip install -r config/code_p1_requirements.txt
```

### 2. 用 Mock 数据跑一遍(无需真实 session)

```bash
python3 src/code_p1_main.py --mock
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
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db

# 快速验证(只跑前 3 个 session)
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3

# 指定输出路径
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --output ./output/chunks_real.jsonl
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

### 4. 用 JSONL / JSON 数据

把 OpenCode session 导出为文件,然后:

- **JSONL**(每行一个 session JSON):
  ```bash
  python3 src/code_p1_main.py --source /path/to/opencode_sessions.jsonl --output ./output/chunks.jsonl
  ```
- **JSON**(整个文件是一个 session 对象,或一个 session list):
  ```bash
  python3 src/code_p1_main.py --source /path/to/opencode_sessions.json --output ./output/chunks.jsonl
  ```

`--source` 按文件后缀分发:`.jsonl` 走流式逐行解析,`.json` 整文件加载后兼容 list 或单对象。其他后缀会被拒绝并退出。

## 关键设计

### 1. 数据源优先级

`--sqlite` > `--mock` > `--source` > config `project.mock_data` > config `opencode.session_source`

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

**后处理**(`_postprocess_chunks`):
- 无 assistant 回复的空 chunk(无文本、无工具调用),其 user_message 合并到下一个非空 chunk
- 连续相同 user_message(中间无 assistant 交互)只保留最后一个(去重)
- 合并/去重后重新编号 chunk_id 和 turn_index,保证连续

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
- 离线时回退到 `len(text) * 0.75` 估算(中英文混合经验值)
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
  "created_at": "2026-07-10T10:00:00Z",
  "task_summary": null
}
```

> `task_summary` 由 Phase 2 CoT 填充,Phase 1 输出时为 null。
> 内部字段 `_raw_tool_calls` 仅供 chunker→cleaner 传递,`to_dict()` 时自动剔除。

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

## 配置文件

`config/code_p1_config.yaml`:

```yaml
project:
  data_dir: "./data"
  output_dir: "./output"
  mock_data: true              # 无真实数据时用 mock 演示

opencode:
  session_source: "./data/opencode_sessions.jsonl"
  session_filter:
    type: "root"               # 只处理 root session
    min_user_messages: 2       # 至少 2 个 user 输入

chunking:
  boundary: "user_message"     # 轮次边界
  max_tokens_per_chunk: 30000

content_cleaning:
  preserve_full: ["user", "assistant"]
  summarize: ["bash", "tool_call", "mcp_call"]
  max_error_length: 200        # 错误日志最大长度

logging:
  level: "INFO"
  file: "./logs/phase1.log"
```

> 注意: config 中 `mock_data: true` 是默认值。用真实数据时必须加 `--sqlite` 参数覆盖。

#### 配置项与代码的对应关系

| 配置项 | 状态 | 读取位置 |
|--------|------|----------|
| `project.mock_data` | **wired** | `code_p1_main.py`(无 `--sqlite`/`--source`/`--mock` 时使用) |
| `opencode.session_source` | **wired** | `code_p1_main.py`(同上条件,且后缀必须是 `.jsonl` / `.json`) |
| `opencode.session_filter.type` | **wired** | `code_p1_main.py`(`--sqlite` 模式对应 SQL 中 `parent_id IS NULL`/`branch`)与 `filter_sessions`(JSONL/mock 路径) |
| `opencode.session_filter.min_user_messages` | **wired** | `code_p1_main.py`、`load_sessions_from_sqlite`、`filter_sessions` |
| `logging.level` / `logging.file` | **wired** | `code_p1_main.py` → `setup_logger` |
| `project.data_dir` / `project.output_dir` | **advisory** | 当前 main 未读取;输出路径由 `--output` 决定,数据路径由 `--sqlite`/`--source` 决定 |
| `chunking.boundary` | **advisory** | `code_p1_chunker.py` 已 hardcode 按 user 消息边界切分,不读此字段 |
| `chunking.max_tokens_per_chunk` | **advisory** | chunker 不强制 token 上限;token 截断发生在 Phase 2(`llm.max_tokens_per_chunk`) |
| `content_cleaning.preserve_full` / `summarize` | **advisory** | `code_p1_content_cleaner.py` 已 hardcode user/assistant 全保留、bash/tool_call/mcp_call 摘要化,不读此字段 |
| `content_cleaning.max_error_length` | **advisory** | `clean_chunks()` 默认参数 `max_error_length=200`;`main.py` 未显式传参,故恒为 200,改 YAML 无效 |

> 上述 advisory 项保留在 YAML 中以记录设计意图;若要让它们生效,需在 `code_p1_chunker.py` / `code_p1_content_cleaner.py` / `code_p1_main.py` 中改读 `Config.get(...)`。这与 `src/AGENTS.md` 关于 advisory YAML 的约定一致(参见 P6/P6b 同类情况)。

## 在 Pipeline 中的位置

```
Phase 1 (本阶段) → Phase 2 (Task 抽取) → Phase 3 (向量化) → Phase 4 (检索) → Phase 5 (知识图谱) → Phase 6 (MCP Server)
```

Phase 1 产出的 `chunks.jsonl` 是 Phase 2 的输入。Phase 2 用 LLM 对每个 chunk 生成 `task_summary`,再整 session 视角抽取 task 清单。
