## Phase 1 & 2 运行指南

### 前置条件

- Python 3.10+
- OpenCode 的 session 数据（SQLite DB 或 JSONL 文件）

### 安装依赖

```bash
pip install -r config/code_p1_requirements.txt
# 或者手动安装
pip install loguru pyyaml tiktoken openai
```

---

### Phase 1: 数据预处理

从 OpenCode session 数据中提取、过滤、分块、清洗，输出结构化 chunks。

#### 配置文件 code_p1_config.yaml

```yaml
project:
  data_dir: "./data"
  output_dir: "./output"
  mock_data: true               # 无真实数据时用内置 mock

opencode:
  session_source: "./data/opencode_sessions.jsonl"
  session_filter:
    type: "root"                # root / branch / all
    min_user_messages: 2

chunking:
  boundary: "user_message"
  max_tokens_per_chunk: 30000

content_cleaning:
  preserve_full: [user, assistant]
  summarize: [bash, tool_call, mcp_call]
  max_error_length: 200

logging:
  level: "INFO"
  file: "./logs/phase1.log"
```

#### 运行命令

```bash
cd /path/to/oc_sess_graph

# 使用内置 mock 数据 (无需真实 session)
python3 src/code_p1_main.py --mock

# 从 SQLite DB 读取
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db

# 快速验证 (只处理 3 个 session)
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --limit 3

# 从 JSONL 文件读取
python3 src/code_p1_main.py --source ./data/my_sessions.jsonl

# 自定义输出路径（test）
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --output ./output/chunks_real.jsonl
```

**命令行参数：**


| 参数       | 默认值                  | 说明                        |
| ------------ | ------------------------- | ----------------------------- |
| `--config` | `config/code_p1_config.yaml`   | 配置文件路径                |
| `--source` | 无                      | Session JSONL/JSON 文件路径 |
| `--sqlite` | 无                      | OpenCode SQLite DB 路径     |
| `--limit`  | 无                      | 最多处理 N 个 session       |
| `--mock`   | false                   | 使用内置 mock 数据          |
| `--output` | `./output/chunks.jsonl` | 输出文件路径                |

#### 输出

`./output/chunks.jsonl` — 每行一个 chunk：`chunk_id`, `session_id`, `turn_index`, `user_message`, `assistant_messages`, `tool_calls`, `mcp_calls`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`

---

### Phase 2: Task 列表生成 (LLM)

加载 Phase 1 chunks，通过 LLM (CoT prompt) 提取 session 级 task 和 per-chunk summary。

#### 环境变量

```bash
# 必须设置 API Key (根据选择的 provider 设置对应的)
export OPENCODE_ZEN_API_KEY='your-api-key'       # OpenCode Zen (免费 deepseek-v4-flash)
# export DASHSCOPE_API_KEY='your-key'             # DashScope (qwen-plus)
# export SILICONFLOW_API_KEY='your-key'           # SiliconFlow (DeepSeek-V3)
# export OPENAI_API_KEY='your-key'                # OpenAI (gpt-4o-mini)
```

#### 配置文件 code_p2_config.yaml

```yaml
phase1:
  chunks_file: "./output/chunks.jsonl"

output:
  chunk_summaries: "./output/chunks_summary_p2.jsonl"

llm:
  provider: "opencode_zen"
  model: "deepseek-v4-flash-free"
  api_key_env: "OPENCODE_ZEN_API_KEY"
  base_url: "https://opencode.ai/zen/v1"
  max_retries: 3
  timeout: 120
  temperature: 0.2
  max_tokens_per_chunk: 2000
  max_total_prompt_tokens: 30000
  concurrency: 4

logging:
  level: "INFO"
  file: "./logs/phase2.log"
```

**支持的 LLM Provider：**


| Provider            | 模型                      | Base URL                                            | 环境变量               |
| --------------------- | --------------------------- | ----------------------------------------------------- | ------------------------ |
| OpenCode Zen (免费) | `deepseek-v4-flash-free`  | `https://opencode.ai/zen/v1`                        | `OPENCODE_ZEN_API_KEY` |
| DashScope           | `qwen-plus`               | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY`    |
| SiliconFlow         | `deepseek-ai/DeepSeek-V3` | `https://api.siliconflow.cn/v1`                     | `SILICONFLOW_API_KEY`  |
| OpenAI              | `gpt-4o-mini`             | `https://api.openai.com/v1`                         | `OPENAI_API_KEY`       |
| Ollama (本地)       | `qwen2.5:7b`              | `http://localhost:11434/v1`                         | 无需                   |

#### 运行命令

```bash
cd /path/to/oc_sess_graph

# 设置 API Key
export OPENCODE_ZEN_API_KEY='your-api-key'

# 快速验证 (3 个 session)
python3 src/code_p2_main.py --limit 3

# 全量运行
python3 src/code_p2_main.py

# 高并发
python3 src/code_p2_main.py --concurrency 8

# 自定义输入输出
python3 src/code_p2_main.py --chunks ./output/chunks.jsonl --output ./output/tasks.jsonl

# test
export OPENCODE_ZEN_API_KEY=$(python3 -c "import json; d=json.load(open('/Users/zhaoxiuwei/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])") && python3 src/code_p2_main.py --chunks ./output/chunks_real.jsonl --output ./output/tasks_real.jsonl --concurrency 8  2>&1 | tail -30
```

**命令行参数：**


| 参数            | 默认值                 | 说明                                 |
| ----------------- | ------------------------ | -------------------------------------- |
| `--config`      | `config/code_p2_config.yaml`  | 配置文件路径                         |
| `--chunks`      | 无                     | Phase 1 chunks JSONL 路径 (覆盖配置) |
| `--output`      | `./output/tasks.jsonl` | Task 输出路径                        |
| `--limit`       | 无                     | 只处理前 N 个 session                |
| `--concurrency` | 无 (从配置读)          | LLM 并发线程数                       |

#### 输出

1. `./output/tasks.jsonl` — 每行一个 task：`task_id`, `session_id`, `task_label`, `task_summary`, `chunk_ids`, `created_at`
2. `./output/chunks_summary_p2.jsonl` — 每行一个 chunk summary：`chunk_id`, `summary`

---

### 端到端快速开始

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph

# 1. 安装依赖
pip install loguru pyyaml tiktoken openai

# 2. Phase 1: mock 数据验证流程
python3 src/code_p1_main.py --mock

# 3. Phase 1: 真实数据
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db

# 4. Phase 2: 设置 API Key 并运行
export OPENCODE_ZEN_API_KEY='your-api-key'
python3 src/code_p2_main.py --limit 3    # 先快速验证
python3 src/code_p2_main.py               # 全量运行
```

**实际性能参考** (15 sessions, 86 chunks)：Phase 2 产出 30 tasks (平均 2.0/session)，chunk summary 100% 覆盖，concurrency=8 时约 93 秒 (deepseek-v4-flash-free)。
