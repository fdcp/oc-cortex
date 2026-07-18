# Phase 1 / Phase 2 运行手册

> 适用版本：2026-07-14 现状

> 项目根目录：`/Users/zhaoxiuwei/Desktop/oc_sess_graph/`

> 流程：Phase 1 (chunks) → Phase 2 (tasks + chunk_summaries) → Phase 3+ (检索/图谱)

---

## 0. TL;DR — 端到端快速跑通

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph

# (1) 安装依赖（一次性）
pip install -r code_p1_requirements.txt

# (2) Phase 1: 从 OpenCode SQLite 抽 chunks
python3 code_p1_main.py \
    --sqlite ~/.local/share/opencode/opencode.db \
    --output ./output/chunks.jsonl

# (3) Phase 2: 配好 API Key 后, 用 LLM 抽 task + chunk 总结
export OPENCODE_ZEN_API_KEY='your-key-here'
python3 code_p2_main.py \
    --chunks ./output/chunks.jsonl \
    --output ./output/tasks.jsonl \
    --concurrency 8
```

跑完会得到：

- `output/chunks.jsonl` — 86 个 chunk（取决于实际 session 数）
- `output/tasks.jsonl` — 27 个 task
- `output/chunks_summary_p2.jsonl` — chunk_id → summary
- `logs/phase1.log` / `logs/phase2.log`

---

## 1. Phase 1：数据预处理

**输入**：OpenCode session 数据
**输出**：`output/chunks.jsonl`（每行一个 Chunk JSON）
**作用**：把 session 按 user 消息切分成"轮次"，并把工具调用降噪整理

### 1.1 入口

```bash
python3 code_p1_main.py [选项]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `code_p1_config.yaml` | 配置文件路径 |
| `--source` | — | JSONL/JSON 数据源（覆盖 config） |
| `--sqlite` | — | OpenCode SQLite db 路径（推荐） |
| `--limit` | — | 最多加载 N 个 session（快速验证用） |
| `--mock` | — | 用内置 mock 数据（不读任何外部文件） |
| `--output` | `./output/chunks.jsonl` | chunks 输出路径 |

### 1.2 数据源优先级

```
--sqlite  >  --source  >  --mock  >  config.mock_data  >  config.session_source
```

任选一种即可；前三者优先级高于配置文件。

### 1.3 方式 A：用 mock 数据（无需任何真实数据）

最快验证方式，3 个内置 session 覆盖过滤/切分/整理全流程：

```bash
python3 code_p1_main.py --mock
```

预期输出：
```
原始加载 3 个 session
过滤后保留 2 个 session（mock_skip 因为只有 1 个 user 被过滤）
Session ses_mock_001: 切分出 2 个 chunk
Session ses_mock_002: 切分出 2 个 chunk
已写入 4 个 chunk 到 ./output/chunks.jsonl
```

### 1.4 方式 B：OpenCode SQLite 数据库（**推荐，真实数据**）

OpenCode 在 `~/.local/share/opencode/opencode.db` 存了所有 session。直接读它：

```bash
# 全量
python3 code_p1_main.py \
    --sqlite ~/.local/share/opencode/opencode.db \
    --output ./output/chunks.jsonl

# 快速验证（只跑前 3 个 session）
python3 code_p1_main.py \
    --sqlite ~/.local/share/opencode/opencode.db \
    --limit 3
```

SQLite loader 会自动：

- 选 `parent_id IS NULL` 的 root session
- 过滤掉 `user 消息 < 2` 的 session
- 清理 user 消息里嵌入的 `<system-reminder>` 块
- 解析 `part.data` 的 `text` / `tool` / 推理等

**macOS 找 db 路径**（用户级）：

```bash
ls -la ~/.local/share/opencode/opencode.db
# 如果不存在，说明 OpenCode 还没在本机写过数据
```

### 1.5 方式 C：JSONL / JSON 文件

把 session 导出为 JSONL（每行一个 session JSON）：

```bash
python3 code_p1_main.py \
    --source /path/to/opencode_sessions.jsonl \
    --output ./output/chunks.jsonl
```

JSONL 单行 schema 参考 `code_p1_models.py`：

```json
{
  "id": "ses_xxx",
  "type": "root",
  "created_at": "2026-07-10T10:00:00Z",
  "turns": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ]
}
```

### 1.6 配置文件（`code_p1_config.yaml`）

```yaml
project:
  data_dir: "./data"
  output_dir: "./output"
  mock_data: true                       # 默认走 mock, 想用真数据改成 false

opencode:
  session_source: "./data/opencode_sessions.jsonl"   # JSONL 路径
  session_filter:
    type: "root"                        # 只处理 root session
    min_user_messages: 2                # 至少 2 个 user 输入

chunking:
  boundary: "user_message"              # 切分边界
  max_tokens_per_chunk: 30000

content_cleaning:
  preserve_full: ["user", "assistant"]  # 完整保留
  summarize: ["bash", "tool_call", "mcp_call"]
  max_error_length: 200                 # 失败日志最大字符数

logging:
  level: "INFO"                         # DEBUG / INFO / WARNING / ERROR
  file: "./logs/phase1.log"
```

> 注意：`project.mock_data: true` 是默认值。如果跑 Phase 1 时不带任何参数，就会走 mock；要走真实 SQLite/JSONL，要么用 `--sqlite` / `--source` 覆盖，要么把 `mock_data` 改成 `false`。

### 1.7 依赖

```bash
pip install -r code_p1_requirements.txt
```

内容：
```
loguru>=0.7.0
pyyaml>=6.0
tiktoken>=0.5.0
openai>=1.30.0     # p2 用的, 提前装
```

> SQLite loader 用的是 Python 内置 `sqlite3`，不需要额外装。

### 1.8 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 跑完说"加载 0 个 session" | db 路径错 / OpenCode 还没在本机产生过数据 | 跑 `ls ~/.local/share/opencode/` 确认；先 `--mock` 验证流程 |
| 报 `ModuleNotFoundError: No module named 'tiktoken'` | 没装依赖 | `pip install -r code_p1_requirements.txt` |
| SQLite 里所有 session 都被跳过 | `min_user_messages: 2` 过滤掉了短 session | 改 `code_p1_config.yaml` 里 `min_user_messages: 1`，或 `--limit N` 拿前 N 个 |
| 想看更详细日志 | 调日志级别 | config 里 `logging.level: "DEBUG"` |
---

## 2. Phase 2：Task 清单生成

**输入**：`output/chunks.jsonl`（Phase 1 产出）
**输出**：
- `output/tasks.jsonl` — Task 列表
- `output/chunks_summary_p2.jsonl` — chunk_id → summary
**作用**：用 LLM 整 session 视角提炼 1-5 个 task，并给每个 chunk 生成一句话总结

### 2.1 入口

```bash
python3 code_p2_main.py [选项]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `code_p2_config.yaml` | 配置文件路径 |
| `--chunks` | config 里 `phase1.chunks_file` | Phase 1 输出的 chunks JSONL |
| `--output` | `./output/tasks.jsonl` | task 输出路径 |
| `--limit` | — | 只处理前 N 个 session（快速验证） |
| `--concurrency` | config 里 `llm.concurrency` | 并发线程数（1=串行） |

### 2.2 准备 API Key（必做）

Phase 2 调 LLM，**必须**有 API Key。通过环境变量传入：

```bash
# OpenCode Zen (免费 deepseek-v4-flash-free, 默认配置)
export OPENCODE_ZEN_API_KEY='your-key-here'

# 或者其他厂商（要同步改 config 的 base_url / model）
export DASHSCOPE_API_KEY='your-key-here'      # 阿里云通义千问
export SILICONFLOW_API_KEY='your-key-here'    # SiliconFlow
export OPENAI_API_KEY='your-key-here'         # OpenAI
```

不设 Key 启动会立刻报错：

```
ValueError: 未找到 API Key: 环境变量 OPENCODE_ZEN_API_KEY 未设置,且未通过参数传入。
```

### 2.3 最小运行

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph

export OPENCODE_ZEN_API_KEY='your-key-here'

python3 code_p2_main.py
```

读完 `code_p2_config.yaml` 的默认配置，从 `./output/chunks.jsonl` 读入，输出到 `./output/tasks.jsonl`。

### 2.4 推荐运行（OpenCode Zen 实测配置）

```bash
export OPENCODE_ZEN_API_KEY='your-key-here'

python3 code_p2_main.py \
    --config code_p2_config.yaml \
    --chunks ./output/chunks.jsonl \
    --output ./output/tasks.jsonl \
    --concurrency 8
```

实测（15 session / 86 chunks）约 93s 完成（并发 8 时）。

### 2.5 快速验证

```bash
# 只处理前 3 个 session, 不开并发
python3 code_p2_main.py --limit 3 --concurrency 1
```

### 2.6 配置文件（`code_p2_config.yaml`）

```yaml
project:
  data_dir: "./data"
  output_dir: "./output"

phase1:
  chunks_file: "./output/chunks.jsonl"     # Phase 1 输出

output:
  chunk_summaries: "./output/chunks_summary_p2.jsonl"

llm:
  provider: "opencode_zen"
  model: "deepseek-v4-flash-free"         # OpenCode Zen 免费模型
  api_key_env: "OPENCODE_ZEN_API_KEY"     # 从哪个环境变量读 key
  base_url: "https://opencode.ai/zen/v1"  # OpenAI 兼容端点
  max_retries: 3
  timeout: 120
  temperature: 0.2
  max_tokens_per_chunk: 2000              # 单 chunk prompt 上限
  max_total_prompt_tokens: 30000          # 整个 prompt 上限
  concurrency: 4                          # 默认 4 线程

logging:
  level: "INFO"
  file: "./logs/phase2.log"
```

### 2.7 切换 LLM 厂商

只要是 OpenAI 兼容 API 都能用。改 config 的 `llm.model` / `llm.base_url` / `llm.api_key_env`：

| 厂商 | model | base_url | api_key_env |
|---|---|---|---|
| **OpenCode Zen**（默认，免费） | `deepseek-v4-flash-free` | `https://opencode.ai/zen/v1` | `OPENCODE_ZEN_API_KEY` |
| DashScope（阿里云） | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| SiliconFlow | `deepseek-ai/DeepSeek-V3` | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Ollama（本地） | `qwen2.5:7b` | `http://localhost:11434/v1` | (不需要) |

切换示例（用 DashScope 跑）：

```bash
export DASHSCOPE_API_KEY='your-key'
# 临时改用 DashScope
python3 code_p2_main.py --chunks ./output/chunks.jsonl --output ./output/tasks_ds.jsonl \
  --concurrency 4

# 注: 想长期用就改 code_p2_config.yaml 里的 llm.* 几行
```

### 2.8 并发 / 串行

- `--concurrency 1` → 串行，调试用，日志最清晰
- `--concurrency N` (N≥2) → 线程池并行；OpenAI SDK 线程安全
- 8 线程是 OpenCode Zen 的甜点档（实测 86 chunk 93s 跑完）

### 2.9 核心设计摘要

- **CoT 三步 Prompt**：步骤 1 逐轮提炼 → 步骤 2 识别任务 → 步骤 3 归类轮次
- **JSON 截断修复**：LLM 输出若被 `max_tokens=16000` 截断，自动修复缺失的 `}` `]` 后重解析
- **未覆盖 chunk 兜底**：LLM 漏归类的 chunk，按 `turn_index` 距离自动塞到最近的 task
- **指数退避重试**：2s → 4s → 8s，最多 3 次
- **Reasoning 模型适配**：DeepSeek V4 的 thinking tokens 不会让 content 为空（`max_tokens=16000` 余量大），且 `content` 为空时回退到 `reasoning_content` 抽 JSON

### 2.10 输出格式

`output/tasks.jsonl`（每行）：

```json
{
  "task_id": "ses_xxx_T1",
  "session_id": "ses_xxx",
  "task_label": "修复登录 bug",
  "task_summary": "用户反馈登录失败，排查发现是 token 过期逻辑错误…",
  "chunk_ids": ["ses_xxx_c1", "ses_xxx_c2"],
  "created_at": "2026-07-14T01:47:29.123456"
}
```

`output/chunks_summary_p2.jsonl`（每行）：
```json
{"chunk_id": "ses_xxx_c1", "summary": "用户请求查看 session 列表…"}
```

> 注意：原 `chunks.jsonl` **不会**被覆盖（Phase 2 只新增 `output/chunks_summary_p2.jsonl`）。如果想让 chunk 本身带 `task_summary` 字段，Phase 3 的 loader 会再把两份文件合并。

### 2.11 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `ValueError: 未找到 API Key: 环境变量 OPENCODE_ZEN_API_KEY 未设置` | 没 export | `export OPENCODE_ZEN_API_KEY='...'` 后再跑 |
| 一直卡住不结束 | 网络问题 / LLM 端点不可达 | `curl https://opencode.ai/zen/v1/models` 试一下；调低 `--concurrency` 到 2 |
| 大批 session 报 "JSON 解析失败且无法修复" | 模型返回非 JSON / 输出太长被截断 | 换更稳的模型（如 gpt-4o-mini）；调高 `llm.max_total_prompt_tokens` 让 chunk 给得多 |
| 想重跑覆盖输出 | 直接覆盖即可 | `--output` 指向同一路径就行，没副作用 |
| 想跑两个不同 LLM 对比 | 输出路径分开 | `--output ./output/tasks_gpt4.jsonl` 之类 |

---

## 3. 端到端复现

完整跑一遍的真实命令序列（基于 2026-07-13 实测）：

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph

# 1. 装依赖
pip install -r code_p1_requirements.txt

# 2. Phase 1: SQLite → chunks.jsonl
python3 code_p1_main.py \
    --sqlite ~/.local/share/opencode/opencode.db \
    --output ./output/chunks.jsonl
# 预期: 15 个 session (跳过 5 个 < 2 user) → 86 个 chunk

# 3. Phase 2: chunks.jsonl → tasks.jsonl + chunk summaries
export OPENCODE_ZEN_API_KEY='your-key-here'
python3 code_p2_main.py \
    --chunks ./output/chunks.jsonl \
    --output ./output/tasks.jsonl \
    --concurrency 8
# 预期: 30 个 task, 86/86 chunk 有总结, ~93s



# 4. 验证产出
ls -la output/
wc -l output/*.jsonl
head -1 output/tasks.jsonl | python3 -m json.tool
```

---

## 4. 文件清单（速查）

| 文件 | 角色 |
|---|---|
| `code_p1_config.yaml` | Phase 1 配置 |
| `code_p1_main.py` | Phase 1 入口 |
| `code_p1_models.py` | `Session` / `Turn` / `Chunk` / `ToolCall` dataclass |
| `code_p1_session_loader.py` | JSONL/JSON 加载 + mock 数据 |
| `code_p1_sqlite_loader.py` | OpenCode SQLite 加载（推荐入口） |
| `code_p1_chunker.py` | 按 user 切轮次 |
| `code_p1_content_cleaner.py` | 工具调用降噪 |
| `code_p1_utils.py` | `Config` / logger / token 计数 / 截断 |
| `code_p1_requirements.txt` | 依赖 |
| `code_p1_README.md` | Phase 1 详细设计文档 |
| `code_p2_config.yaml` | Phase 2 配置 |
| `code_p2_main.py` | Phase 2 入口 |
| `code_p2_task_extractor.py` | CoT prompt + LLM 调用 + 解析校验 |
| `code_p2_models.py` | `Task` dataclass |
| `code_p2_README.md` | Phase 2 详细设计文档 |

---

## 5. 下一步

Phase 1 + 2 跑通后，接 Phase 3：
- `code_p3_main.py` 把 tasks + chunks 向量化写入 Qdrant 三集合 (tasks / chunks_summary / chunks_cleaned_text)
- 详见 `code_p3_README.md`（以及 `图谱方案.md` 第 7 章 Phase 划分）
