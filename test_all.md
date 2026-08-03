# 端到端测试指南 — OpenCode Session Knowledge Graph

> 本文档列出从零开始跑通 P1 → P2 → P3 → P4 → P5(triple) → P5(entity) → P5e → MS → P6 → P6b → MCP 全部功能的**详细步骤**,**每一步均列出可能的变体**与预期产物。
>
> **前置约定**
>
> - 项目根:`/Users/zhaoxiuwei/Desktop/oc_sess_graph`
> - 所有命令在项目根执行,且已激活 venv
> - 真实数据源:`~/.local/share/opencode/opencode.db`(OpenCode SQLite)
> - API Key:`~/.local/share/opencode/auth.json` 中 `opencode-go` 字段
> - LLM 端点:
>   - P2 配置默认走 OpenCode Go 订阅 `https://opencode.ai/zen/go/v1`(模型 `hy3`)
>   - P5/P6 配置默认走 OpenCode Zen 免费层 `https://opencode.ai/zen/v1`(模型 `nemotron-3-ultra-free`)
> - 为了「跑所有功能」需要**同时**产出 P5 的 `triple/` 与 `entity/` 两套图谱:
>   - P6b 的决策溯源默认读 `output/triple/knowledge_graph.db`
>   - MCP Server 默认读 `output/entity/knowledge_graph.db`
>   - 故 P5 需分别用两种 `extraction_mode` 各跑一次

---

## Step 0 — 环境准备

### 0.1 安装依赖

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph
bash install.sh            # 完整安装(含 torch/transformers + 模型预下载,~3GB)
# 或
bash install.sh --light    # 轻量安装:跳过 Reranker/Embedding 模型下载
                          # (P4 搜索、P6 graph_rag、MCP graph_rag_search 将不可用)
```

**变体**


| 命令                      | 适用场景                                     | 体积    |
| --------------------------- | ---------------------------------------------- | --------- |
| `bash install.sh`         | 跑全部功能(含 P4 精排、MCP graph_rag_search) | ~3GB    |
| `bash install.sh --light` | 仅跑 P1/P2/P3/P5/MCP 基础(不含 Reranker)     | ~几百MB |

**人工补充**(install.sh 不覆盖,需手动)

```bash
pip install fastapi uvicorn pydantic   # MCP HTTP 模式 (code_mcp_server_http.py)
pip install requests                    # MCP Client (code_mcp_client.py)
```

### 0.2 每次新开终端都要执行的环境变量

```bash
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph
source .venv/bin/activate

# API Key(从 auth.json 注入)
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# HuggingFace 离线模式(避免模型联网检查超时)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 建立输出/日志目录
mkdir -p output logs
```

**变体**


| 场景                      | 调整                                                               |
| --------------------------- | -------------------------------------------------------------------- |
| 无 auth.json / 用别的 key | `export OPENCODE_ZEN_API_KEY=sk-xxxx`                              |
| 想强制重新下载模型        | 不设`TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE`(首次 P3 会下载 ~100MB) |
| 仅跑 P1(不调 LLM)         | 可跳过`OPENCODE_ZEN_API_KEY`,但仍需 `TRANSFORMERS_OFFLINE`         |

---

## Step 1 — Phase 1: 数据预处理(`code_p1_main.py`)

**做什么**:从 SQLite/JSONL/JSON/mock 加载 session → 按 user message 边界切 chunk → 清洗 tool_call 输出 → 为每个 chunk 计算 SHA-256 内容指纹。

P1 的输出文件 `chunks.jsonl` 增加 `content_hash` 字段;若 session/P1 内部数据发生任何变化,P2 自动重跑该 session。

### 1.1 推荐命令(读真实 SQLite,全量模式)

**第一次跑 / 数据变化频繁 / 不想学增量** —— 直接这样跑,每次都对所有 session 重新切分,简单、可靠、不需要理解什么是 checkpoint:

```bash
python3 src/code_p1_main.py \
  --sqlite ~/.local/share/opencode/opencode.db \
  --config config/code_p1_config.yaml \
  --output ./output/chunks.jsonl \
  2>&1 | tee logs/phase1.log
```

跑完后:
- `output/chunks.jsonl` 写完(每行带 `content_hash` 字段)
- **不**会生成 `output/.p1_checkpoint.json`(全量模式保留你之前的习惯)

### 1.2 增量模式——跳过没变的 session

如果你要反复跑 P1(比如在调试 P2 prompt、想多次重跑 P1 重新落 chunks.jsonl),而你的 OpenCode SQLite 里的对话**没变**(没新会话、也没在老会话上续聊),每次都全量切一遍浪费几秒到几十秒。加 `--incremental` 跳过即可:

```bash
python3 src/code_p1_main.py \
  --sqlite ~/.local/share/opencode/opencode.db \
  --config config/code_p1_config.yaml \
  --output ./output/chunks.jsonl \
  --incremental \
  2>&1 | tee logs/phase1.log
```

#### 增量模式的三种情况(看日志的 "复用 / 重切 / 新增" 列就知道)

| 情况 | 日志里看到 | 发生什么 |
|------|------------|---------|
| **第一次跑**(`.p1_checkpoint.json`不存在) | `增量模式: checkpoint 0 session` → 全部 `新增 N` | 全量切,跑完写 checkpoint,记录每个 session 的 turns hash + chunk 内容 hash |
| **再次跑且数据没变** | `复用 N / 重切 0 / 新增 0` | checkpoint 里的 `raw_turns_hash` 跟刚读出的 turns 算的 hash 一致 → 跳过 chunker → 把 `chunks.jsonl` 里该 session 的旧行直接拷过来。**几秒就跑完** |
| **再次跑但有新/变 session** | `复用 X / 重切 m / 新增 n` | 没变的 session 复用;新增 session 或续聊过的 session hash 对不上 → 进 chunker 切一次 → 用新 chunk 写进 `chunks.jsonl` → checkpoint 里这条记录被新 hash 覆盖 |

#### 增量模式日志示例

```text
增量模式: checkpoint 100 session, 旧 chunks.jsonl 含 100 session
Session ses_050: 切分出 3 个 chunk       ← 这条是重切的 session 才打印
Session ses_101: 切分出 2 个 chunk       ← 这是新增的
checkpoint 已更新: ./output/.p1_checkpoint.json (101 session)

Phase 1 完成
  输入 session: 102
  过滤后 session: 102
  输出 chunk: ...
  增量: 复用 100 / 重切 1 / 新增 1
```

跑得顺利的时候你会看到 `复用 N / 重切 0 / 新增 0`,**一条 "切分出" 都不打印**——这意味着 chunker 全部跳过,几秒钟就退出。

### 1.3 强制全量重写(在增量模式下彻底重跑)

两种情况需要它:
- 你改了 chunker 算法本身(`code_p1_chunker._postprocess_chunks` 之类),旧 checkpoint 全部失效需要重切
- 你想清掉旧 checkpoint 状态、完整重来一遍(但还想保留增量模式以备下一次)

```bash
python3 src/code_p1_main.py \
  --sqlite ~/.local/share/opencode/opencode.db \
  --incremental --force
```

`--force` 让 `--incremental` 在这一次退化为全量,**跑完仍会写新 checkpoint**。等价于"删掉 checkpoint 再跑 `--incremental`",但更稳当(checkpoint 文件不需要手动管)。

### 1.4 不需要 `--incremental` 时怎样

不加 `--incremental`,`output/.p1_checkpoint.json` 这个文件**永远不会出现**;P1 行为完全跟之前一样,每次都全量切。

**完全不学增量模式,你也照样能跑通 P2。** P2 自己有 `.p2_checkpoint.json`,靠 `chunks.jsonl` 里每行的 `content_hash` 字段(这个字段无论 P1 全量还是增量都存在)就能判断"P2 上次跑这个 session 时内容是啥",从而决定跳过或重跑。**P1 增量跟 P2 增量相互独立,P1 全量跑不影响 P2 任何能力。**

### 1.5 自定义 checkpoint 路径(可选)

默认 `output/.p1_checkpoint.json`,想换名字/换路径,用 `--checkpoint`:

```bash
# 多实验隔离: 不同数据源用独立 checkpoint, 互不污染
python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db \
  --incremental --checkpoint ./output/.p1_real.json

python3 src/code_p1_main.py --source ./data/imported.jsonl \
  --incremental --checkpoint ./output/.p1_imported.json
```

也可以在 `config/code_p1_config.yaml` 里写 `incremental.checkpoint: ./output/.p1_custom.json`,CLI 不带 `--checkpoint` 时回退到这个值。

### 1.6 数据源/输出/配置变体矩阵


| 变体              | 命令片段                                       | 适用场景                                                                                       |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **SQLite(推荐)**  | `--sqlite ~/.local/share/opencode/opencode.db` | 真实 OpenCode 会话                                                                             |
| **快速验证 N 个** | `--sqlite ... --limit 3`                       | 只跑前 3 个 session,验证流程;**在增量模式下也起作用**(只作用于"待切"的 session)              |
| **JSONL**         | `--source /path/sessions.jsonl`                | 已导出的逐行 JSONL                                                                             |
| **JSON(整文件)**  | `--source /path/sessions.json`                 | 单 session 对象或 list                                                                         |
| **Mock**          | `--mock`                                       | 内置 mock 数据,无需真实 session                                                                |
| **完全默认**      | 不带数据源参数                                 | 按`--sqlite > --mock > --source > config.project.mock_data > config.session_source` 优先级回退 |
| **自定义输出**    | `--output ./output/chunks_dev.jsonl`           | 多实验并跑不覆盖;⚠️ 这会跟增量复用机制冲突(增量从 `--output` 指的那个文件复用旧行,所以切走路径要带 checkpoint 走 |
| **换配置**        | `--config config/my_p1.yaml`                   | 调`session_filter.min_user_messages` 等                                                        |

### 1.7 关键配置(`config/code_p1_config.yaml`)


| 段/字段                                       | 状态        | 说明                                            |
| ----------------------------------------------- | ------------- | ------------------------------------------------- |
| `opencode.session_source`                      | [wired]       | 无 CLI 数据源时默认 JSONL/JSON 路径                  |
| `opencode.session_filter.type`                 | [wired]       | `root` / `branch` 过滤        |
| `opencode.session_filter.min_user_messages`    | [wired]       | 最少 user 消息数                                    |
| `project.mock_data`                            | [wired]       | 无 CLI 数据源时是否用 mock                          |
| `logging.level` / `logging.file`               | [wired]       | 日志                                                |
| `incremental.checkpoint`                       | [wired]       | 增量 checkpoint 路径,默认 `./output/.p1_checkpoint.json`;仅 `--incremental` 模式才读这个值 |
| `chunking.*` / `content_cleaning.*`            | [advisory]    | chunker/cleaner 已 hardcode,改 YAML 不生效         |

### 1.8 预期产物

```
output/chunks.jsonl                  # 每行一个 Chunk JSON (全量/增量都生成)
output/.p1_checkpoint.json           # 仅 --incremental 模式才生成; 全量不加 --incremental 永远不出现
```

`chunks.jsonl` 字段:`chunk_id`, `session_id`, `turn_index`, `user_message`, `assistant_messages`, `tool_calls`, `mcp_calls`, `raw_size_tokens`, `cleaned_size_tokens`, `created_at`, `task_summary`(P1 阶段恒 `null`), `content_hash`(SHA-256 内容指纹,由 P1 chunker 在切分后填入,**不包含 `task_summary`** → 不会因 P2 后续修改导致 hash 变)

`.p1_checkpoint.json` 结构(每个已完成 session 一条记录):
```json
{
  "ses_abc": {
    "turns_hash":       "437cfc...",         // 对该 session 所有 turns 的 SHA-256, 用来短路"对话没变"
    "chunk_count":      3,
    "chunk_ids":        ["ses_abc_c1","_c2","_c3"],
    "content_hash":     "d16671...",         // chunker 切完后对所有 chunk 的链式 SHA-256, 给 P2 用
    "chunker_version":  1,                   // 改 chunker 算法需 bump 此常量, 旧 checkpoint 自动失效
    "processed_at":      "2026-08-03T..."
  }
}
```

短路流程:加载 session 时算 `raw_turns_hash(session.turns)` → 跟 checkpoint 的 `turns_hash` 比对,**匹配**就跳过 chunker,直接复用 `chunks.jsonl` 里该 session 的旧行 → **不匹配**就 chunker 重切,并把新 hash 写回 checkpoint。`chunker_version` 跟代码常量对不上时,**该 session 一律重切**(用于 chunker 算法升级时强制全量重跑所有 session)。

### 1.9 验证命令

```bash
wc -l output/chunks.jsonl
# 看第一行 chunk 的 hash
python3 -c "import json; r=json.loads(open('output/chunks.jsonl').readline()); print(r['chunk_id'], r['content_hash'][:16])"

# 仅在 --incremental 跑过才会有下面这个文件
test -f output/.p1_checkpoint.json && python3 -c "
import json; d=json.load(open('output/.p1_checkpoint.json'))
print(len(d), 'sessions in checkpoint')
print('sample:', list(d.items())[0])"
```

### 1.10 关于 P1 增量值不值得用的简短决策

- chunker 本来就快(几百个 session 几秒),**P1 全量跑通常不亏**
- 你**反复跑 P1 多次**(迭代 P2 时反复清 `chunks.jsonl` 重切)→ 加 `--incremental` 省时间
- 你只跑**一两次 P1 就完事** → 别加,保留全量习惯最简单
- P2 的增量跟 P1 的增量**完全独立**:P1 全量跑 vs 增量跑,P2 都一样能用自己的 checkpoint 增量

> 注:`config/code_p1_config.yaml` 中 `chunking.*` 与 `content_cleaning.*` 为 advisory(chunker/cleaner 已 hardcode),改 YAML 不生效。详见 `doc/code_p1_README.md` wired/advisory 表。
>
> 注:SHA-256 算法在代码中硬编码(无配置项);改 chunker 算法或 `Chunk` 字段集需同时 bump `code_p1_utils.CHUNKER_VERSION`,所有旧 checkpoint 自动失效触发全量重切。

---

## Step 2 — Phase 2: 任务提取(`code_p2_main.py`)

**做什么**:LLM(CoT 三步)从每个 session 的 chunk 中抽 task 清单,并给每个 chunk 生成摘要。

P2 默认启用增量(无需命令行开关):若 `output/.p2_checkpoint.json` 存在且其中某 session 的 `content_hash` 与 P1 在该 session 上的 `session_fingerprint` 一致 → 该 session 跳过 LLM 调用,复用 `tasks.jsonl` 中的旧 task 记录;不一致(P1 改了内容/重切了 chunk)则该 session 进 pending,本次单独调 LLM。`--force` 关闭增量。

### 2.1 推荐命令

```bash
python3 src/code_p2_main.py \
  --config config/code_p2_config.yaml \
  --concurrency 8 \
  --output ./output/tasks.jsonl \
  2>&1 | tee logs/phase2.log
```

### 2.2 变体矩阵


| 变体                | 命令片段                             | 说明                                 |
| --------------------- | -------------------------------------- | -------------------------------------- |
| 默认`chunks_file`   | (来自 config`phase1.chunks_file`)    | 默认读`output/chunks.jsonl`          |
| 换输入文件          | `--chunks ./output/chunks_dev.jsonl` | 对应 Step 1 自定义输出               |
| 自定义输出          | `--output ./output/tasks_v2.jsonl`   | 多实验并跑                           |
| 只跑前 N 个 pending | `--limit 10`                         | 作用于待处理 session(已 done 的不动) |
| 改并发              | `--concurrency 8`                    | 覆盖 config`llm.concurrency`(默认 4) |
| 串行(便于复现/排错) | `--concurrency 1`                    | 顺序执行                             |
| **强制全量重跑**    | `--force`                            | 忽略 checkpoint,所有 session 调 LLM,重写 checkpoint |
| **自定义 checkpoint** | `--checkpoint ./output/.p2_custom.json` | 替换默认路径                          |

### 2.3 LLM 配置变体(改 `config/code_p2_config.yaml`)


| 字段                          | 默认                               | 备注                               |
| ------------------------------- | ------------------------------------ | ------------------------------------ |
| `llm.model`                   | `hy3`                              | OpenCode Go 订阅模型               |
| `llm.base_url`                | `https://opencode.ai/zen/go/v1`    | Go 订阅端点                        |
| `llm.api_key_env`             | `OPENCODE_ZEN_API_KEY`             | 环境变量名                         |
| `llm.concurrency`             | 4                                  | 被`--concurrency` 覆盖             |
| `llm.timeout`                 | 600                                | hy3 推理模型在大 session 上需较长时间,原默认 120 易超时 |
| `llm.max_tokens_per_chunk`    | 2000                               | 单 chunk 在 prompt 中的 token 上限 |
| `llm.max_total_prompt_tokens` | 30000                              | 整 prompt token 预算               |
| `llm.max_retries`             | 3                                  | 网络层重试(2s/4s/8s 退避)          |
| `llm.content_retries`         | 3                                  | 内容校验重试                       |
| `output.chunk_summaries`      | `./output/chunks_summary_p2.jsonl` | 摘要输出                           |
| `output.checkpoint`            | `./output/.p2_checkpoint.json`      | [wired] 增量 checkpoint 路径      |

### 2.4 预期产物

```
output/tasks.jsonl               # 每行一个 Task(task_id, task_label, task_summary, chunk_ids, ...)
output/chunks_summary_p2.jsonl   # 每行 {chunk_id, summary}
output/.p2_checkpoint.json       # 仅在调过 LLM 且有成功 session 后写 (全 done 不动则不写)
```

Checkpoint 结构(P2):每个 session 记录 `content_hash`(即 P1 的 `session_fingerprint`)+ chunk_ids + chunk_count + completed_at。`status` ∈ {`incomplete`,`failed`} 的 session **不写**进 checkpoint,下次自动重跑。

**验证**

```bash
wc -l output/tasks.jsonl output/chunks_summary_p2.jsonl
python3 -c "import json; print(json.loads(open('output/tasks.jsonl').readline()).keys())"
test -f output/.p2_checkpoint.json && python3 -c "import json; d=json.load(open('output/.p2_checkpoint.json')); print(len(d), 'sessions in p2 checkpoint')"
```

> 注:sessions 的 `session_fingerprint` 由 P1 在每个 chunk 上算的 SHA-256 `content_hash` 链式拼接得到,定义在 `code_p1_utils.session_fingerprint`;P2 直接复用,不再本地重算。
>
> 注:输出文件 `tasks.jsonl` 与 `chunks_summary_p2.jsonl` 每次运行按当前 checkpoint 整体重写(temp + rename 原子):done session 用磁盘旧记录,本次 pending session 用新结果。即使增量没跑任何 LLM,文件也会按规范内容重写(等价 no-op)。
>
> 注:**每跑完一个 session 立即同步落盘**(`tasks.jsonl` + `chunks_summary_p2.jsonl` + `.p2_checkpoint.json`,各自 temp+rename 原子写)。中途断电 → 已成功的 session 不会丢;未完成的 session 不写 checkpoint,下次自动续跑。`--concurrency > 1` 时,落盘仍在主线程串行(由 `concurrent.futures.as_completed` 在主线程同步触发),无 IO 竞争。

---

## Step 3 — Phase 3: 向量存储(`code_p3_main.py`)

**做什么**:把 P1 chunks + P2 tasks/summaries 写入 Qdrant 三个集合(dense BGE + BM25 sparse)。

### 3.1 推荐命令

```bash
python3 src/code_p3_main.py \
  --config config/code_p3_config.yaml \
  2>&1 | tee logs/phase3.log
```

### 3.2 变体矩阵


| 变体                | 命令片段                                          | 说明                                     |
| --------------------- | --------------------------------------------------- | ------------------------------------------ |
| 指定 tasks 输入     | `--tasks ./output/tasks_v2.jsonl`                 | 覆盖 config`phase2.tasks_file`           |
| 指定 chunks 输入    | `--chunks ./output/chunks_dev.jsonl`              | 覆盖 config`phase1.chunks_file`          |
| 指定 summaries 输入 | `--summaries ./output/chunks_summary_p2_v2.jsonl` | 覆盖 config`phase2.chunk_summaries_file` |
| 换 Qdrant 路径      | 改 config`qdrant.path`(默认 `./qdrant_data`)      | 多实验隔离                               |

### 3.3 关键配置(`config/code_p3_config.yaml`)


| 段/字段                  | 默认                                               | 影响                                |
| -------------------------- | ---------------------------------------------------- | ------------------------------------- |
| `embedding.model`        | `BAAI/bge-small-zh-v1.5`                           | dense 模型(dim 512)                 |
| `embedding.device`       | `cpu`                                              | `auto/cpu/mps/cuda`(MPS 有挂起风险) |
| `embedding.offline_mode` | `true`                                             | 不联网检查                          |
| `sparse.method`          | `bm25`                                             | `bm25` 或 `bge_m3`                  |
| `sparse.bm25_params`     | k1=1.5, b=0.75                                     | BM25 参数                           |
| `sparse.fuse_k`          | 60                                                 | RRF 融合常数                        |
| `qdrant.path`            | `./qdrant_data`                                    | 嵌入式数据库目录                    |
| 集合名                   | `tasks` / `chunks_summary` / `chunks_cleaned_text` | 见`qdrant.collections`              |

### 3.4 预期产物

```
qdrant_data/   # 三个集合:tasks, chunks_summary, chunks_cleaned_text
```

**验证(P3 自带的搜索 demo)**

```bash
python3 src/code_p3_search_demo.py --query "FlashAttention 实现原理" --top-k 5
# 变体:--mode all(默认,全量 bge+bm25)/ --mode sample(1/5 数据,Qwen3+BGE-M3)
```

> ⚠️ `code_p3_main.py` 在模块顶层 pre-parse `--config`,以在 import `sentence_transformers` 前设 HF 离线环境。改 CLI 时需保留此模式。

---

## Step 4 — Phase 4: 跨 Session 搜索(`code_p4_search_cli.py`)

**做什么**:Dense 搜 `tasks` + BM25/BGE-M3 搜 `chunks_cleaned_text` → RRF 融合 → Qwen3-Reranker 精排 → 展开 chunk 详情。

### 4.1 推荐命令

```bash
python3 src/code_p4_search_cli.py \
  --config config/code_p3_config.yaml \
  --query "FlashAttention 实现原理" \
  --top-k 5
```

### 4.2 变体矩阵


| 变体     | 命令片段        | 说明                                         |
| ---------- | ----------------- | ---------------------------------------------- |
| 单次查询 | `--query "..."` | 直接出结果                                   |
| 跳过精排 | `--no-rerank`   | 仅粗排 RRF,快但精度低                        |
| 交互模式 | `--interactive` | 循环输入,输入`demo` 跑内置 5 条示例,`q` 退出 |
| 改 top-k | `--top-k 10`    | 返回结果数                                   |

### 4.3 关键配置(`config/code_p3_config.yaml` 的 `reranker` 段)


| 字段                  | 默认                       | 备注                |
| ----------------------- | ---------------------------- | --------------------- |
| `reranker.model`      | `Qwen/Qwen3-Reranker-0.6B` | 精排模型            |
| `reranker.device`     | `mps`                      | `auto/cpu/mps/cuda` |
| `reranker.max_length` | 8192                       |                     |
| `reranker.batch_size` | 4                          |                     |

**预期输出**:终端打印每条结果的 `rerank_score`、`hybrid_score`、`task_label`、`task_summary` 及关联 chunk 摘要/预览。

> ⚠️ `code_p4_search_cli.py` 同样在模块顶层 pre-parse `--config`。

---

## Step 5 — Phase 5: 知识图谱构建(`code_p5_main.py`,**跑两次**)

**做什么**:LLM 从 task summary 抽三元组 → Qdrant 相似度 + LLM 实体对齐 → NetworkX 图 → 可选 SQLite 导出。

**为了跑全部功能,需要用 triple 与 entity 两种 `extraction_mode` 各跑一次**(P6b 默认消费 triple;MCP 默认消费 entity)。

### 5.1 Phase 5 (Triple 模式)— 推荐命令

`config/code_p5_config.yaml` 中 `knowledge_graph.extraction_mode: "triple"` 是默认值,直接跑:

```bash
python3 src/code_p5_main.py \
  --config config/code_p5_config.yaml \
  2>&1 | tee logs/phase5_triple.log
```

### 5.2 Phase 5 (Entity 模式)— 推荐命令

复制一份配置并改 `extraction_mode` 为 `entity`,另存为 `config/code_p5_config_entity.yaml`:

```yaml
# config/code_p5_config_entity.yaml 内修改这两行
knowledge_graph:
  extraction_mode: "entity"
  entity_alignment_threshold: 0.92
  # 输出目录与 triple 模式隔离,互不覆盖
  triple_output_dir: "./output/triple"
  entity_output_dir: "./output/entity"
```

然后:

```bash
python3 src/code_p5_main.py \
  --config config/code_p5_config_entity.yaml \
  2>&1 | tee logs/phase5_entity.log
```

### 5.3 变体矩阵(两种模式通用)


| 变体                                   | 命令片段            | 说明                                 |
| ---------------------------------------- | --------------------- | -------------------------------------- |
| 改并发                                 | `--concurrency 8`   | 覆盖 config`llm.concurrency`(默认 4) |
| 只跑前 N 个 task                       | `--limit 50`        | 快速调 prompt                        |
| 跳过对齐(每实体保留原名)               | `--skip-alignment`  | 跳过 Qdrant+LLM 对齐,快但实体分散    |
| 构建后自动可视化                       | `--visualize`       | 额外产`knowledge_graph.html`         |
| 跳过抽取(用现成 triples/entities 文件) | `--skip-extraction` | 复用上次结果只跑对齐/建图            |

### 5.4 关键配置(`config/code_p5_config.yaml`)


| 字段                                         | 默认                         | 影响                                          |
| ---------------------------------------------- | ------------------------------ | ----------------------------------------------- |
| `llm.model`                                  | `nemotron-3-ultra-free`      | Zen 免费层;注释提到 entity 模式可用`hy3-free` |
| `llm.base_url`                               | `https://opencode.ai/zen/v1` | Zen 端点                                      |
| `llm.merge_batch_enable`                     | false                        | true=批量对齐(20 对/批,更快不稳)              |
| `knowledge_graph.extraction_mode`            | `triple`                     | ∈ {`triple`, `entity`}                       |
| `knowledge_graph.entity_alignment_threshold` | 0.92                         | 对齐相似度阈值                                |
| `knowledge_graph.triple_output_dir`          | `./output/triple`            | triple 产物目录                               |
| `knowledge_graph.entity_output_dir`          | `./output/entity`            | entity 产物目录                               |
| `sqlite.enabled`                             | true                         | 自动导出`knowledge_graph.db`                  |
| `graph_rag.*`                                | enabled, bfs_depth=1         | Phase 5e Graph-RAG 参数                       |

### 5.5 预期产物

Triple 模式:

```
output/triple/triples.jsonl
output/triple/entities.jsonl
output/triple/knowledge_graph.gpickle       # NetworkX MultiDiGraph
output/triple/knowledge_graph.html          # 仅 --visualize 时生成
output/triple/knowledge_graph.db            # SQLite (Phase 5e)
```

Entity 模式:

```
output/entity/entities.jsonl
output/entity/knowledge_graph.gpickle
output/entity/knowledge_graph.html           # 仅 --visualize 时生成
output/entity/knowledge_graph.db             # SQLite (MCP 默认读这个)
```

**验证**

```bash
wc -l output/triple/triples.jsonl output/triple/entities.jsonl output/entity/entities.jsonl
ls -la output/triple/knowledge_graph.db output/entity/knowledge_graph.db
```

---

## Step 6 — Phase 5e: SQLite + Graph-RAG(无独立 CLI)

**做什么**:Phase 5 已自动导出 SQLite;`code_p5e_db.KGDatabase` 提供 BFS 扩散查询,`code_p5e_graph_rag.GraphRAGSearcher` 融合向量检索 + 图谱扩散 + Reranker。

### 6.1 无 CLI,通过 Python API 验证

```python
python3 - <<'PY'
import sys; sys.path.insert(0, "src")
from code_p5e_db import KGDatabase
for name, p in [("triple", "output/triple/knowledge_graph.db"),
                ("entity", "output/entity/knowledge_graph.db")]:
    try:
        db = KGDatabase(p)
        s = db.get_stats()               # KGDatabase 实际方法名
        print(f"{name}: {s}")
    except Exception as e:
        print(f"{name}: {e}")
PY
```

### 6.2 Graph-RAG 演示(需 P3 + P5 已就绪)

```python
python3 - <<'PY'
import sys; sys.path.insert(0, "src")
from code_p5e_db import KGDatabase
from code_p5e_graph_rag import GraphRAGSearcher
from code_p4_searcher import SessionSearcher

db = KGDatabase("output/triple/knowledge_graph.db")
searcher = SessionSearcher("config/code_p3_config.yaml")
rag = GraphRAGSearcher(searcher, db)
results, debug = rag.search("FlashAttention", top_k=5)
for r in results:
    print(f"[{r.task_label}] score={r.rerank_score:.3f}")
PY
```

### 6.3 变体


| 变体             | 改什么                                                | 效果                       |
| ------------------ | ------------------------------------------------------- | ---------------------------- |
| bfs_depth        | `config/code_p5_config.yaml` 的 `graph_rag.bfs_depth` | 1=一跳, 2=二跳(更广但更慢) |
| max_expand_nodes | `graph_rag.max_expand_nodes`(30)                      | BFS 扩散上限               |
| graph_weight     | `graph_rag.graph_weight`(0.3)                         | 图谱结果权重加成           |
| 换数据库         | `KGDatabase("output/entity/knowledge_graph.db")`      | 用 entity 模式图           |

> 预期:终端打印 Graph-RAG 排序后的 task 列表,每条附 `rerank_score`。

---

## Step 7 — MS: Milestone 检视(`code_MS_inspect.py`)

**做什么**:检视 Qdrant 集合统计、抽样浏览、检索质量评估。

### 7.1 推荐命令

```bash
python3 src/code_MS_inspect.py \
  --config config/code_p3_config.yaml \
  -n 5
```

### 7.2 变体矩阵


| 变体         | 命令片段                     | 说明                 |
| -------------- | ------------------------------ | ---------------------- |
| 默认 samples | `-n 5`                       | 每集合抽 5 个点      |
| 详细抽样     | `-n 20`                      | 看更多样本           |
| 换配置       | `--config config/my_p3.yaml` | 切到不同 Qdrant 实例 |

> `code_MS_inspect.py` 从 P3 YAML 读集合名/路径/模型,相对路径回退到仓库根,可从任意 CWD 运行。

---

## Step 8 — Phase 6: 跨 Session 总结(`code_p6_cli.py`)

**做什么**:检索 Top-K Task → 收集 chunk 内容 → Token 预算裁剪 → LLM 生成主题总结。

### 8.1 推荐命令

```bash
python3 src/code_p6_cli.py \
  "我最近做过的 FlashAttention 相关工作" \
  --config config/code_p3_config.yaml \
  --top-k 8 \
  --detail summary
```

### 8.2 变体矩阵


| 变体                | 命令片段                                                  | 说明                                                                     |
| --------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| chunk 详情级别      | `--detail summary` / `--detail preview` / `--detail full` | summary(默认,仅摘要)、preview(摘要+前500字)、full(完整原文,token 消耗大) |
| 启用 Graph-RAG 增强 | `--graph-rag`                                             | 复用 P5e Graph-RAG,需 P5 已构建                                          |
| 时间过滤            | `--time-from 2026-07-01 --time-to 2026-07-15`             | 必须 from/to 同时给出                                                    |
| 改 top-k            | `--top-k 5`                                               | 检索 Task 数                                                             |
| Token 预算          | `--max-tokens 12000`                                      | LLM 输入上限                                                             |
| 不带 chunk 原文     | `--no-chunks`                                             | 仅用 task 摘要,最快                                                      |
| JSON 输出           | `--json`                                                  | 结构化输出                                                               |

### 8.3 关键配置(`config/code_p6_config.yaml`,advisory — 实际参数 hardcode 在 `code_p6_summarizer.py`)


| 字段                            | YAML 默认               | 实际                      |
| --------------------------------- | ------------------------- | --------------------------- |
| `summarizer.model`              | `nemotron-3-ultra-free` | hardcode                  |
| `summarizer.max_context_tokens` | 12000                   | 被`--max-tokens` CLI 覆盖 |
| `summarizer.max_tasks`          | 10                      |                           |
| `search.top_k`                  | 8                       | 被`--top-k` CLI 覆盖      |
| `search.use_graph_rag`          | false                   | 被`--graph-rag` CLI 覆盖  |

### 8.4 预期产物

终端打印 LLM 总结(300–800 字,含 task_id 引用);`--json` 时输出 JSON。

**验证**:确认总结中出现的 task_id 可在 `output/tasks.jsonl` 找到。

---

## Step 9 — Phase 6b: 决策溯源 + 骨架总结(`code_p6b_cli.py`)

**做什么**:

1. **trace**:从知识图谱多跳追踪决策链(选用/选型/排除/替换/依赖)
2. **search**:模糊搜索图谱实体
3. **summary**:骨架式主题总结(可注入决策链)

> **数据库选择 (`_init_db`) — 实际行为**
> P6b CLI **完全不读** `config/code_p6b_config.yaml`。`--db` 缺失时按下列顺序自动探测:
>
> 1. `output/triple/knowledge_graph.db`(`triple` 优先,故默认跑 P5 triple 即可被 P6b 用)
> 2. `output/entity/knowledge_graph.db`
> 3. 都没有则报错 "未找到知识图谱数据库 … 请先运行 Phase 5 构建图谱" 并 `sys.exit(1)`
>    想强制用 entity 库:传 `--db output/entity/knowledge_graph.db`。
>
> **`config/code_p6b_config.yaml` 是 advisory**:`decision_categories`、`max_hops` 等都 hardcode 在 `code_p6b_skeleton.py`(全局 `DECISION_CATEGORIES`),改 YAML 不会影响 CLI 行为。
>
> **API Key 只 `summary` 子命令需要**(调 LLM);`trace`、`search` 是纯图/SQLite 查询,无需 `OPENCODE_ZEN_API_KEY`。

### 9.1 决策溯源 `trace`

```bash
python3 src/code_p6b_cli.py trace "FlashAttention" --hops 3 --max-steps 20

# 变体
python3 src/code_p6b_cli.py trace "FlashAttention" --hops 2 --json
python3 src/code_p6b_cli.py trace "OpenCode" --max-steps 10
python3 src/code_p6b_cli.py --db output/entity/knowledge_graph.db trace "FlashAttention"
```


| 变体             | 命令片段                                                                |
| ------------------ | ------------------------------------------------------------------------- |
| 改跳数           | `--hops 2`                                                              |
| 限制显示步骤     | `--max-steps 10`                                                        |
| JSON 输出        | `--json`                                                                |
| 强制用 entity 库 | `--db output/entity/knowledge_graph.db`(否则按 triple→entity 自动探测) |

> `trace` 无网络/无 LLM 调用,可在离线环境运行。

### 9.2 实体搜索 `search`

```bash
python3 src/code_p6b_cli.py search "Flash" --limit 10

# 变体
python3 src/code_p6b_cli.py search "序列" --limit 20
python3 src/code_p6b_cli.py --db output/entity/knowledge_graph.db search "编码"
```

> `search` 同 `trace`,纯 SQLite 查询,无需 API Key。`--db` 同样适用 triple→entity 自动探测。

### 9.3 骨架总结 `summary`

```bash
python3 src/code_p6b_cli.py summary "FlashAttention" \
  --config config/code_p3_config.yaml \
  --max-tasks 10 --hops 3

# 变体
python3 src/code_p6b_cli.py summary "FlashAttention" --no-trace                    # 不注入决策链
python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 5 --hops 2 --json
```


| 变体         | 命令片段        | 说明             |
| -------------- | ----------------- | ------------------ |
| 不注入决策链 | `--no-trace`    | 仅骨架,不带因果  |
| 改搜索 top-k | `--max-tasks 5` |                  |
| 改追溯跳数   | `--hops 2`      |                  |
| JSON 输出    | `--json`        |                  |
| 换搜索配置   | `--config ...`  | 指向不同 P3 YAML |

### 9.4 决策关系分类(`config/code_p6b_config.yaml`,advisory)

实际 hardcode 在 `code_p6b_skeleton.py`:

- 选用: 使用, 集成了, 实现, 实现了, 支持, 通过兼容层加载, 保留fallback模型, 保留
- 选型: 对比, 对比了, 区别于, 适用于, 适合, 支持精度
- 排除: 禁用, 缺少, 阻塞于, 阉割了
- 替换: 替代了, 新增了, 补充了
- 依赖: 属于, 包含, 对应

### 9.5 预期产物

- `trace` — 决策链(按上述 5 类分组)
- `summary` — 结构化总结(400–1000 字:决策主线+选型对比+技术细节+建议)
- `search` — 实体名 + 关联边列表

---

## Step 10 — MCP Server: 知识图谱工具服务

**做什么**:把 KG 封装为 MCP Server,5 个工具:`query_kg`、`search_entities`、`get_entity_info`、`graph_rag_search`、`get_kg_stats`。

**默认 DB**:`output/entity/knowledge_graph.db`(见 `config/code_mcp_config.yaml` 的 `server.db_path`)。

### 10.1 启动模式 — 两种**完全不同**的 HTTP,务必分清


| 启动命令                                                                       | 类型                              | 协议                 | 客户端                                                   |
| -------------------------------------------------------------------------------- | ----------------------------------- | ---------------------- | ---------------------------------------------------------- |
| `python3 src/code_mcp_server.py`                                               | **stdio**(MCP SDK 传输)           | JSON-RPC over stdio  | Claude Code / Codex / OpenCode / QoderWork 等 MCP 客户端 |
| `python3 src/code_mcp_server.py --http`                                        | **streamable-http**(MCP SDK 传输) | MCP`streamable-http` | MCP SDK 客户端,**不是** REST                             |
| `python3 src/code_mcp_server.py --sse`                                         | **SSE**(MCP SDK 传输)             | MCP SSE              | MCP SDK 客户端                                           |
| `uvicorn code_mcp_server_http:app --host 0.0.0.0 --port 8000` (`workdir=src/`) | **REST HTTP API**(FastAPI)        | 普通 HTTP/JSON       | `code_mcp_client.py`、`curl`、任意 HTTP 客户端           |

> `--http` 与 REST **不兼容**:`--http` 是 MCP SDK 的 streamable-http 协议,需要 MCP 客户端按 SDK 协议访问。普通 `curl` 或 `code_mcp_client.py` 想要 REST 路由必须用 `uvicorn` 起 `code_mcp_server_http:app`。

### 10.2 stdio 模式(供 MCP 客户端加载)

```bash
python3 src/code_mcp_server.py
```

启动后等待 JSON-RPC 输入。后台线程预热 `GraphRAGSearcher`(30–50s 加载 embedding+Qdrant+BM25,预热失败不影响其它 4 个工具)。

### 10.3 REST HTTP 模式(供 `code_mcp_client.py` / 调试)

```bash
# 需额外 pip install fastapi uvicorn pydantic(install.sh 不覆盖)
# code_mcp_server_http.py 以 __file__ 解析项目根并加载 code_mcp_config.yaml,
# 从项目根启动需把 src/ 加进 sys.path:
PYTHONPATH=src uvicorn code_mcp_server_http:app --host 0.0.0.0 --port 8000
# 等价:cd src && uvicorn code_mcp_server_http:app --host 0.0.0.0 --port 8000
```

启动后可访问(详见 `/docs` swagger):

- `GET /health` — 健康检查
- `GET /get_stats` — 图谱统计(`db.get_stats()`)
- `GET /search_entities?query=...&limit=...` — 实体搜索
- `GET /get_entity_info?name=...` — 实体详情(节点 + 边)
- `GET /query_kg?entity=...&depth=...&max_nodes=...` — BFS 扩散查询
- `GET /graph_rag_search?query=...&top_k=...&use_graph=...` — Graph-RAG(需预热 SessionSearcher)
- `GET /mcp/tools/list` + `POST /mcp/tools/call` — MCP 兼容别名

### 10.4 配置(`config/code_mcp_config.yaml`)


| 字段                   | 默认                               | 状态         | 谁读取                                                                       |
| ------------------------ | ------------------------------------ | -------------- | ------------------------------------------------------------------------------ |
| `server.db_path`       | `output/entity/knowledge_graph.db` | **wired**    | `code_mcp_server.py` + `code_mcp_server_http.py` 都读                        |
| `server.tasks_file`    | `output/tasks.jsonl`               | **wired**    | 同上                                                                         |
| `context.depth`        | 1                                  | **advisory** | `code_mcp_client.py` 的 `ContextInjector(depth=1,...)` hardcode;改 YAML 无效 |
| `context.max_tasks`    | 10                                 | **advisory** | 同上                                                                         |
| `context.max_entities` | 3                                  | **advisory** | 同上                                                                         |

### 10.5 客户端连接示例(stdio)

**Claude Code** — `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python3",
      "args": ["/Users/zhaoxiuwei/Desktop/oc_sess_graph/src/code_mcp_server.py"],
      "env": { "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1" }
    }
  }
}
```

**Codex** — `~/.codex/config.json`(同上结构)

**OpenCode** — `.opencode/mcp.json`:

```json
{ "mcp": { "knowledge-graph": { "type": "stdio", "command": "python3", "args": [".../code_mcp_server.py"] } } }
```

**QoderWork** — Connectors 设置:名称=knowledge-graph;命令=`python3`;参数=绝对路径;env=`OPENCODE_ZEN_API_KEY=<key>`(可选:MCP server 本身不调 LLM,此 env 仅给客户端自己的链路用)

**验证**:MCP 客户端启动后调用 `get_kg_stats` 应返回节点/边计数;调用 `search_entities "Flash"` 应返回实体列表。

### 10.6 MCP Client / 上下文注入测试(`code_mcp_client.py`,**需先起 10.3 的 REST**)

```bash
pip install requests   # 首次使用

# 必须先起 REST HTTP server(10.3):
#   uvicorn code_mcp_server_http:app --host 0.0.0.0 --port 8000 &
export KG_MCP_URL=${KG_MCP_URL:-http://localhost:8000}   # 可省略,默认即此值

python3 src/code_mcp_client.py health                 # 健康检查
python3 src/code_mcp_client.py tools                  # 列出工具
python3 src/code_mcp_client.py stats                  # 图谱统计
python3 src/code_mcp_client.py query "FlashAttention" 2   # BFS 查询
python3 src/code_mcp_client.py search "Flash"         # 实体搜索
python3 src/code_mcp_client.py context "继续搞 RoPE 优化"   # 新 session 上下文注入
```

> ⚠️ `code_mcp_client.py` 是 **REST** 客户端,直接走普通 HTTP 调 `code_mcp_server_http:app`;它**不**与 stdio / `--http` mode 的 MCP server 互通。`ContextInjector` 的 `depth=1`、`max_tasks=10` 等参数 hardcode 在代码中,**不**读 `code_mcp_config.yaml` 的 `context.*` 段(故而 `context.*` 是 advisory)。

---

## Step 11 — 全流程一键脚本(端到端)

把上面所有步骤串成一条命令(可保存为 `run_all.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /Users/zhaoxiuwei/Desktop/oc_sess_graph
source .venv/bin/activate

export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p output logs

# 前置:仓库内默认不存在 config/code_p5_config_entity.yaml,需先从 triple 配置派生
if [ ! -f config/code_p5_config_entity.yaml ]; then
  python3 - <<'PY'
import yaml, copy
src = yaml.safe_load(open("config/code_p5_config.yaml"))
src["knowledge_graph"]["extraction_mode"] = "entity"
# 输出目录与 triple 隔离,与原 P5 config 默认一致
src["knowledge_graph"]["triple_output_dir"] = "./output/triple"
src["knowledge_graph"]["entity_output_dir"] = "./output/entity"
yaml.safe_dump(src, open("config/code_p5_config_entity.yaml", "w"),
               allow_unicode=True, sort_keys=False)
print("已生成 config/code_p5_config_entity.yaml")
PY
fi

echo "=== P1 ==="; python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db | tee logs/phase1.log
# 增量场景:加 --incremental 跳过 raw_turns_hash 未变的 session
# echo "=== P1 (增量) ==="; python3 src/code_p1_main.py --sqlite ~/.local/share/opencode/opencode.db --incremental | tee logs/phase1.log
echo "=== P2 ==="; python3 src/code_p2_main.py | tee logs/phase2.log
# 增量场景:P2 默认就启用(有 checkpoint 自动比对 content_hash);强制全量加 --force
# echo "=== P2 (全量) ==="; python3 src/code_p2_main.py --force | tee logs/phase2.log
echo "=== P3 ==="; python3 src/code_p3_main.py | tee logs/phase3.log
echo "=== P4 验证 ==="; python3 src/code_p4_search_cli.py --query "FlashAttention 实现原理" --top-k 5

echo "=== P5 (triple) ==="; python3 src/code_p5_main.py | tee logs/phase5_triple.log
echo "=== P5 (entity) ==="; python3 src/code_p5_main.py --config config/code_p5_config_entity.yaml | tee logs/phase5_entity.log

echo "=== MS 检视 ==="; python3 src/code_MS_inspect.py -n 5

echo "=== P6 总结(含 Graph-RAG)==="; python3 src/code_p6_cli.py "FlashAttention 相关工作" --graph-rag --top-k 8 --detail preview
echo "=== P6b 决策溯源(无需 API Key,双 db 测试)==="
python3 src/code_p6b_cli.py trace "FlashAttention" --hops 3                       # 默认 fallback → triple db
python3 src/code_p6b_cli.py --db output/entity/knowledge_graph.db trace "FlashAttention"
echo "=== P6b 骨架总结 ==="; python3 src/code_p6b_cli.py summary "FlashAttention" --max-tasks 10
echo "=== P6b 实体搜索 ==="; python3 src/code_p6b_cli.py search "Flash" --limit 10

# MCP:REST HTTP server + client 健康检查(比 stdio 后台预热 60s 更易验证)
pip install --quiet fastapi uvicorn pydantic requests || true
echo "=== MCP REST HTTP server(后台)==="
PYTHONPATH=src uvicorn code_mcp_server_http:app --host 0.0.0.0 --port 8000 & MCPPID=$!
sleep 5
python3 src/code_mcp_client.py health
python3 src/code_mcp_client.py stats
python3 src/code_mcp_client.py search "Flash"
python3 src/code_mcp_client.py query "FlashAttention" 2
kill $MCPPID
echo "=== DONE ==="
```

---

## 全流程依赖与产物速查


| Phase                         | CLI                                 | 主输入                                                                    | 主产物                                                                        | 必需 env                                                   |
| ------------------------------- | ------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| P1                            | `code_p1_main.py [--incremental] [--force] [--checkpoint PATH]` | SQLite/JSONL/JSON/mock | `output/chunks.jsonl`,`output/.p1_checkpoint.json`(仅 `--incremental`) | —                                                         |
| P2                            | `code_p2_main.py [--force] [--limit N] [--checkpoint PATH]`   | `output/chunks.jsonl`                                                     | `output/tasks.jsonl`, `output/chunks_summary_p2.jsonl`, `output/.p2_checkpoint.json` | `OPENCODE_ZEN_API_KEY`                                     |
| P3                            | `code_p3_main.py`                   | P1+P2 产物                                                                | `qdrant_data/` 3 集合                                                         | `TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE`                    |
| P3 demo                       | `code_p3_search_demo.py`            | Qdrant                                                                    | 终端输出                                                                      | 同上                                                       |
| P4                            | `code_p4_search_cli.py`             | Qdrant                                                                    | 终端结果                                                                      | 同上 + Reranker 模型                                       |
| P5 (triple)                   | `code_p5_main.py`                   | `output/tasks.jsonl` + Qdrant                                             | `output/triple/{triples,entities}.jsonl`, `knowledge_graph.{gpickle,html,db}` | `OPENCODE_ZEN_API_KEY` + HF offline                        |
| P5 (entity)                   | 同上换 config                       | 同上                                                                      | `output/entity/...`                                                           | 同上                                                       |
| P5e                           | (Python API)                        | `knowledge_graph.db`                                                      | Graph-RAG 结果                                                                | —                                                         |
| MS                            | `code_MS_inspect.py`                | Qdrant                                                                    | 终端统计/抽样                                                                 | HF offline                                                 |
| P6                            | `code_p6_cli.py`                    | P3/P4                                                                     | 终端总结                                                                      | `OPENCODE_ZEN_API_KEY` + HF offline                        |
| P6b`trace`/`search`           | `code_p6b_cli.py trace|search`      | KG db                                                                     | 决策链 / 实体列表                                                             | —(纯 SQLite 查询)                                         |
| P6b`summary`                  | `code_p6b_cli.py summary`           | P3/P4 + KG db                                                             | 骨架总结                                                                      | `OPENCODE_ZEN_API_KEY` + HF offline                        |
| MCP stdio /`--http` / `--sse` | `code_mcp_server.py [--http|--sse]` | `output/entity/knowledge_graph.db`(`code_mcp_config.yaml.server.db_path`) | JSON-RPC / MCP 传输                                                           | HF offline(`graph_rag_search` 预热需要,**不**需要 API key) |
| MCP REST HTTP                 | `uvicorn code_mcp_server_http:app`  | 同上                                                                      | REST JSON                                                                     | HF offline(同上);需`pip install fastapi uvicorn pydantic`  |
| MCP Client 测试               | `code_mcp_client.py <cmd>`          | REST server`KG_MCP_URL`(默认 `http://localhost:8000`)                     | 终端结果                                                                      | 需`pip install requests`;先起 REST server                  |

## 常见变体决策树

```
要不要跑 P4 精排?
├── 要  → install.sh(非 --light),配 reranker.device=mps
└── 不要 → install.sh --light,P4 加 --no-rerank

要不要 P6b 决策溯源?
├── 要 → 必须先跑 P5 triple(P6b 默认读 output/triple/knowledge_graph.db)
└── 不要 → 只跑 P5 entity 即可喂 MCP

要不要 MCP graph_rag_search?
├── 要 → install.sh(非 --light) + P5 entity + P3 已就绪(RAG 预热在 MCP 启动时后台跑)
└── 不要 → --light 安装即可

要不要 P6 Graph-RAG 增强?
├── 要 → --graph-rag,需 P5 triple/entity 至少一个就绪
└── 不要 → 默认 P6 不传 --graph-rag
```

## 排错速查


| 现象                                          | 原因 / 对策                                                               |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| P2 报`OPENCODE_ZEN_API_KEY` 缺失              | Step 0.2 未执行,或 auth.json 不存在                                       |
| P3/P4`sentence_transformers` 超时             | 未设`TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`                             |
| P4 MPS 挂起                                   | 改`reranker.device: "cpu"`                                                |
| P5 报 OpenAI choices 空                       | 当前模型返回为空;P2 配置`hy3` 需 OpenCode Go 订阅;无订阅读改用 Zen 免费层 |
| P5 hy3-free 已下线                            | P2 config 注释已说明;改`llm.model` 为可用的 Go 订阅模型或 Zen 免费模型    |
| MCP`client has been closed`                   | 已在历史 commit 修复;确保`code_mcp_server.py` 为最新版本                  |
| P6b trace 无结果                              | `output/triple/knowledge_graph.db` 未生成;先跑 P5 triple                  |
| MCP 默认 DB 不存在                            | P5 entity 未跑;或改`code_mcp_config.yaml` 的 `server.db_path` 指向 triple |
| Entity 模式找不到`code_p5_config_entity.yaml` | 需自行复制 P5 config 改`extraction_mode`                                  |

## 配置文件清单


| 文件                                | Phase        | 状态                                | 关键覆盖字段                                                                                                          |
| ------------------------------------- | -------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `config/code_p1_config.yaml`        | P1           | wired/advisory 混合                 | `session_filter.*`, `session_source`, `mock_data`, `logging.*`, `incremental.checkpoint`(wired); `chunking.*`, `content_cleaning.*`(advisory)   |
| `config/code_p2_config.yaml`        | P2           | wired                               | `llm.model`, `base_url`, `concurrency`, `max_*_tokens`, `llm.timeout=600`, `output.chunk_summaries`, `output.checkpoint`              |
| `config/code_p3_config.yaml`        | P3/P4/P6/P6b | wired                               | `embedding.*`, `sparse.*`, `qdrant.*`, `reranker.*`, `logging.*`                                                      |
| `config/code_p5_config.yaml`        | P5/P5e       | wired                               | `llm.*`, `knowledge_graph.*`, `sqlite.*`, `graph_rag.*`                                                               |
| `config/code_p5_config_entity.yaml` | P5 entity    | wired(需自建,见 Step 5.2 / Step 11) | 同上但`extraction_mode: entity`                                                                                       |
| `config/code_p6_config.yaml`        | P6           | **advisory**(不被 P6 代码读取)      | `code_p6_summarizer.py` 用 hardcode `DEFAULT_MODEL="nemotron-3-ultra-free"`                                           |
| `config/code_p6b_config.yaml`       | P6b          | **advisory**(不被 P6b CLI 读取)     | `code_p6b_skeleton.py` 用全局 `DECISION_CATEGORIES`;`db_path` 不生效,CLI 按 `output/triple → output/entity` fallback |
| `config/code_mcp_config.yaml`       | MCP          | 部分 wired                          | `server.db_path` + `server.tasks_file` wired;`context.*` advisory(被 `code_mcp_client.py` hardcode)                   |
| `config/code_p1_requirements.txt`   | P1 deps      | 文档                                | 仅 P1-only 依赖(loguru/pyyaml/tiktoken)                                                                               |

完。
