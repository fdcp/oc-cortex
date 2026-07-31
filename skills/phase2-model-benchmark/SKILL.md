---
name: phase2-model-benchmark
description: Run Phase 2 task extraction benchmark comparing multiple LLM models on opencode session data. Evaluates speed, accuracy, summary quality, and chunk coverage. Use when the user asks to test, benchmark, or compare LLM models for Phase 2 task extraction, or mentions "模型对比", "模型测试", "P2测试".
version: 1.0.0
---

# Phase 2 Model Benchmark

Run multi-model comparison tests for Phase 2 task extraction (CoT 3-step: per-turn summarize → task identify → turn-to-task assign). Reuses `tests/test_p2/test_p2.py` from the project.

Model / endpoint / auth settings are centralized in `tests/test_p2/config.yaml` and loaded **in order**. Results are written to a dated subfolder under `tests/test_p2/`.

## Prerequisites

- Project at `~/Desktop/oc_sess_graph` with `output/chunks.jsonl` (Phase 1 output)
- OpenCode API key available at `~/.local/share/opencode/auth.json` (managed by the opencode client)

## Endpoints & models (2026-07 status)

| Endpoint key | base_url | auth provider | Example models |
|--------------|----------|---------------|----------------|
| `zen` | `https://opencode.ai/zen/v1` | `opencode-go` | `deepseek-v4-flash-free`, `mimo-v2.5-free`, `nemotron-3-ultra-free` |
| `go` | `https://opencode.ai/zen/go/v1` | `opencode-go` | `hy3`, `glm-5.2`, `kimi-k3` |

> **Note**: `hy3-free` was removed from the Zen free tier. `hy3` is now only available via the **OpenCode Go** subscription endpoint (`.../zen/go/v1`, model ID `hy3`). The test config already routes it correctly.

## Workflow

### Step 1: Review / edit the config

`tests/test_p2/config.yaml` holds the auth file path, endpoint definitions, the ordered `models` list (each with a `name` + `endpoint`), and extractor params. No env-var export is needed — the script reads keys directly from `auth.json`.

To add/reorder/swap models, edit the `models:` list. Each entry:

```yaml
models:
  - name: "hy3"
    endpoint: "go"
  - name: "nemotron-3-ultra-free"
    endpoint: "zen"
```

### Step 2: Run the test

```bash
cd ~/Desktop/oc_sess_graph

# All models from config.yaml (in order), all sessions, 8 threads
python3 tests/test_p2/test_p2.py --sessions 100 --concurrency 8

# Override the model list (endpoint/auth still resolved from config.yaml)
python3 tests/test_p2/test_p2.py --models hy3 nemotron-3-ultra-free --sessions 100

# Quick smoke test (3 representative sessions)
python3 tests/test_p2/test_p2.py --sessions 3
```

**Parameter guidance:**
- `--sessions 100` → use all sessions (the script picks all if >= total count)
- `--sessions 3` → quick smoke test with 3 representative sessions
- `--concurrency 8` → 8 parallel threads per model (free tier may rate-limit; reduce to 1~4 if errors occur)
- `--config <path>` → use an alternate config file
- `--models ...` → override the config model list (order preserved; unknown models fall back to the first endpoint)

### Step 3: Analyze results

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

### Step 4: Results output

The script **automatically** writes to a dated subfolder `tests/test_p2/run_<YYYY-MM-DD_HH-MM>/`:

- `test_p2_results.json` — full per-session detailed results (config, model specs, stats, per-task output)
- `test_p2_models.md` — auto-generated Markdown report (same format as the reference `test_p2_models.md`): test config, comparison table, and analysis conclusions

You may further enrich the generated `test_p2_models.md` by hand (e.g. per-session latency breakdown, side-by-side quality samples for one 5-chunk session, chunk_id accuracy notes) using data from `test_p2_results.json`.

## Key source files

| File | Role |
|------|------|
| `tests/test_p2/config.yaml` | Model / endpoint / auth config, loaded in order |
| `tests/test_p2/test_p2.py` | Main test script (accepts `--config`, `--models`, `--sessions`, `--concurrency`, `--output-dir`) |
| `src/code_p2_task_extractor.py` | Phase 2 extractor with `SESSION_TASK_PROMPT` and `TaskExtractor` class |
| `src/code_p2_models.py` | `Task` dataclass |
| `output/chunks.jsonl` | Phase 1 output — test input data |
| `tests/test_p2/run_<date>/test_p2_results.json` | Detailed per-session results (JSON) |
| `tests/test_p2/run_<date>/test_p2_models.md` | Auto-generated test report |

## Pitfalls

- Free-tier models may rate-limit at high concurrency. If `429 Too Many Requests` errors appear, reduce `--concurrency` to 1~4.
- `hy3` (Go endpoint) is a reasoning model — thinking tokens consume output budget and may truncate JSON. The extractor has automatic JSON repair, `reasoning_content` fallback, and content-retry built in. It is also slower per session (~45-50s) than the Zen free models.
- Always use `--sessions 100` (or a number >= total sessions) for a full benchmark. Small samples (3~5) are only for smoke testing.
- If a key can't be resolved, check that the `auth_provider` in `config.yaml` matches an entry in `auth.json` (currently `opencode-go`).
- The test is slow: 16 sessions × 4 models serially can take ~30-40 minutes. With `--concurrency 8` it drops to ~15 minutes per model.
