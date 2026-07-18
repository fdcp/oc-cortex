---
name: phase2-model-benchmark
description: Run Phase 2 task extraction benchmark comparing multiple LLM models on opencode session data. Evaluates speed, accuracy, summary quality, and chunk coverage. Use when the user asks to test, benchmark, or compare LLM models for Phase 2 task extraction, or mentions "模型对比", "模型测试", "P2测试".
version: 1.0.0
---

# Phase 2 Model Benchmark

Run multi-model comparison tests for Phase 2 task extraction (CoT 3-step: per-turn summarize → task identify → turn-to-task assign). Reuses `tests/test_p2/test_p2.py` from the project.

## Prerequisites

- Project at `~/Desktop/oc_sess_graph` with `output/chunks.jsonl` (Phase 1 output)
- OpenCode Zen API key available at `~/.local/share/opencode/auth.json`

## Workflow

### Step 1: Set up API key

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
```

### Step 2: Determine models

- If the user specified models → use those
- Otherwise → use defaults: `nemotron-3-ultra-free`, `deepseek-v4-flash-free`, `mimo-v2.5-free`, `hy3-free`
- All models must be OpenCode Zen compatible (endpoint: `https://opencode.ai/zen/v1`)

### Step 3: Run the test

```bash
cd ~/Desktop/oc_sess_graph

# With user-specified models
python3 tests/test_p2/test_p2.py \
  --models <model1> <model2> ... \
  --sessions 100 \
  --concurrency 8

# With defaults (omit --models)
python3 tests/test_p2/test_p2.py --sessions 100 --concurrency 8
```

**Parameter guidance:**
- `--sessions 100` → use all sessions (the script picks all if >= total count)
- `--sessions 3` → quick smoke test with 3 representative sessions
- `--concurrency 8` → 8 parallel threads per model (free tier may rate-limit; reduce to 1~4 if errors occur)

### Step 4: Analyze results

The script outputs a comparison table to stdout covering:

| Metric | What it measures |
|--------|-----------------|
| 成功率 | % of sessions that returned valid JSON with tasks |
| 平均耗时 | Mean wall-clock time per session |
| 总耗时 | Total time for all sessions |
| 平均Tasks | Mean tasks extracted per session |
| Label长度 | Mean task_label length (chars) |
| Summary长度 | Mean task_summary length (chars) — longer ≈ more detailed |
| Chunk覆盖 | % of chunks assigned to at least one task |
| Summary覆盖 | % of chunks that got a chunk_summary |

### Step 5: Write report

Save a markdown report to `tests/test_p2/test_p2_models.md` including:
1. Test config (models, sessions, concurrency, date)
2. The comparison table from stdout
3. Per-session latency breakdown (from `tests/test_p2/results.json`)
4. Quality sample: pick one 5-chunk session, show each model's task output side-by-side
5. Analysis: speed ranking, quality ranking, stability, chunk_id accuracy
6. Recommendation table by use case

## Key source files

| File | Role |
|------|------|
| `tests/test_p2/test_p2.py` | Main test script (accepts `--models`, `--sessions`, `--concurrency`, `--output`) |
| `src/code_p2_task_extractor.py` | Phase 2 extractor with `SESSION_TASK_PROMPT` and `TaskExtractor` class |
| `src/code_p2_models.py` | `Task` dataclass |
| `output/chunks.jsonl` | Phase 1 output — test input data |
| `tests/test_p2/results.json` | Detailed per-session results (JSON) |
| `tests/test_p2/test_p2_models.md` | Test report (overwrite with fresh results) |

## Pitfalls

- Free-tier models may rate-limit at high concurrency. If `429 Too Many Requests` errors appear, reduce `--concurrency` to 1~4.
- DeepSeek reasoning models may produce truncated JSON (reasoning tokens consume output budget). The test script has automatic JSON repair and content-retry built in.
- Always use `--sessions 100` (or a number >= total sessions) for a full benchmark. Small samples (3~5) are only for smoke testing.
- The test is slow: 16 sessions × 4 models serially takes ~30-40 minutes. With `--concurrency 8` it drops to ~15 minutes per model.
