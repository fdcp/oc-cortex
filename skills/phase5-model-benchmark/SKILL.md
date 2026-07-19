---
name: phase5-model-benchmark
description: Run Phase 5 knowledge graph benchmark comparing multiple LLM models across triple and entity extraction modes. Evaluates extraction quality, entity alignment, graph structure, and per-stage timing. Use when the user asks to test, benchmark, or compare LLM models for Phase 5 KG construction, or mentions "P5测试", "KG benchmark", "知识图谱模型对比", "triple/entity对比".
version: 1.0.0
---

# Phase 5 KG Model Benchmark

Run multi-model × multi-mode comparison tests for Phase 5 knowledge graph construction. Full pipeline: LLM extraction → entity collection → entity alignment (Qdrant embedding + LLM confirm) → graph construction (NetworkX). Reuses `tests/test_p5/test_p5.py` from the project.

## Prerequisites

- Project at `~/Desktop/oc_sess_graph` with `output/tasks.jsonl` (Phase 2 output)
- OpenCode Zen API key available at `~/.local/share/opencode/auth.json`
- Embedding model cached locally (BAAI/bge-small-zh-v1.5, offline mode)

## Workflow

### Step 1: Set up API key

```bash
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")
```

### Step 2: Determine models and modes

- If the user specified models → use those
- Otherwise → use defaults: `nemotron-3-ultra-free`, `deepseek-v4-flash-free`, `mimo-v2.5-free`, `hy3-free`
- Modes: `triple` (三元组关系图谱), `entity` (实体共现图谱), or both (default)
- All models must be OpenCode Zen compatible (endpoint: `https://opencode.ai/zen/v1`)

### Step 3: Run the test

```bash
cd ~/Desktop/oc_sess_graph

# Full benchmark: 4 models × 2 modes, all tasks
python3 tests/test_p5/test_p5.py

# Specific models
python3 tests/test_p5/test_p5.py --models nemotron-3-ultra-free hy3-free

# Single mode only
python3 tests/test_p5/test_p5.py --modes triple

# Quick smoke test (first 5 tasks)
python3 tests/test_p5/test_p5.py --limit 5

# High concurrency (free tier may rate-limit)
python3 tests/test_p5/test_p5.py --concurrency 8

# Custom output directory
python3 tests/test_p5/test_p5.py --output tests/test_p5/my_run
```

**Parameter guidance:**
- `--limit 0` (default) → all tasks (full benchmark)
- `--limit 5` → quick smoke test
- `--concurrency 4` (default) → safe for free tier; `8` for speed but may hit rate limits
- Full run (31 tasks × 4 models × 2 modes) takes ~30-50 minutes depending on models

**Important:** Triple mode is significantly slower than entity mode. Consider running them separately to avoid timeout:

```bash
# Run triple mode first (slower, ~10-20 min)
python3 tests/test_p5/test_p5.py --modes triple 2>&1 | tee tests/test_p5/test_p5_triple_run.log

# Then entity mode (faster, ~5-10 min)
python3 tests/test_p5/test_p5.py --modes entity 2>&1 | tee tests/test_p5/test_p5_entity_run.log
```

### Step 4: Analyze results

The script outputs comparison tables to stdout grouped by mode:

| Metric | What it measures |
|--------|-----------------|
| 三元组/实体数 | Extraction yield (triples count or unique entities) |
| 平均/t | Mean extraction yield per task |
| 原始实体 | Raw entity count before alignment |
| 唯一实体 | Entities after alignment (lower = more merges) |
| 合并对 | Number of entity pairs merged by LLM confirmation |
| 节点/边 | Final graph structure size |
| 抽取/对齐/图谱(s) | Per-stage wall-clock time |
| 总计(s) | Full pipeline time |

### Step 5: Write report

Save a comprehensive markdown report to `tests/test_p5/test_p5_models.md` including:

1. Test config (models, modes, tasks count, concurrency, date)
2. Reproduction commands (exact bash with env exports)
3. Triple mode comparison table (from stdout / test_p5_results.json)
4. Entity mode comparison table
5. Cross-mode analysis: triple vs entity trade-offs (graph density, speed, semantic richness)
6. Per-model analysis: strengths, weaknesses, failure patterns
7. Graph topology: top-5 high-degree nodes per model×mode, avg/max degree
8. Error analysis: JSON parse failures, empty content retries, alignment instability
9. Recommendation table by use case (best quality, fastest, most stable)
10. Known issues and file listing

## Key source files

| File | Role |
|------|------|
| `tests/test_p5/test_p5.py` | Main benchmark script (accepts `--models`, `--modes`, `--limit`, `--concurrency`, `--output`, `--config`) |
| `src/code_p5_kg_builder.py` | KGBuilder class: extraction prompts, entity alignment, graph construction |
| `src/code_p5_models.py` | `Triple`, `Entity`, `KGStats` dataclasses |
| `src/code_p5_main.py` | Phase 5 main entry (production flow, not benchmark) |
| `config/code_p5_config.yaml` | Phase 5 config (LLM, embedding, Qdrant, KG settings) |
| `output/tasks.jsonl` | Phase 2 output — benchmark input data |
| `tests/test_p5/test_p5_results.json` | Machine-readable per-run results (auto-generated) |
| `tests/test_p5/test_p5_report.md` | Auto-generated summary report |
| `tests/test_p5/test_p5_models.md` | Comprehensive test report (manually written, overwrite with fresh results) |

## Output structure

Each model×mode run produces a subdirectory `tests/test_p5/{model}__{mode}/`:

- Triple mode: `triples.jsonl`, `knowledge_graph.gpickle`, `knowledge_graph.json`
- Entity mode: `entity_extract.jsonl`, `inverted_index.json`, `knowledge_graph.gpickle`, `knowledge_graph.json`

These per-model output dirs are gitignored (`tests/test_p5/*__triple/`, `tests/test_p5/*__entity/`). Only the script, reports, and JSON results are committed.

## Pitfalls

- **DeepSeek triple mode catastrophic failure**: reasoning model puts JSON in `reasoning_content` instead of `content`; `_extract_json_from_response` often fails to extract valid triple JSON. Entity mode works because simpler `{"entities": [...]}` format is parseable. Expect ~1.4 triples/task vs ~9/task for other models.
- **Mimo alignment instability**: LLM merge confirmation calls frequently return empty content → 0 merge pairs in entity mode. Also slowest overall model.
- **Nemotron list format**: occasionally returns JSON array instead of `{"triples": [...]}` object → `'list' object has no attribute 'get'` errors (non-fatal, those tasks are skipped).
- **Hy3 entity leakage**: may extract session IDs (e.g., `ses_12c6bd...`) as entities despite prompt exclusion rules.
- **Isolated Qdrant**: each benchmark run uses `tempfile.mkdtemp()` for its entities collection to prevent cross-run contamination. Temp dirs are cleaned up automatically.
- **Timeout risk**: full triple mode across 4 models can exceed 10 minutes. Run triple and entity modes as separate commands, or use `is_background` for long runs.
- **Free-tier rate limiting**: at concurrency > 4, `429 Too Many Requests` may appear. Reduce `--concurrency` to 1-2 if errors occur.
- **HF offline mode**: the script sets `HF_HUB_OFFLINE=1` before importing `sentence_transformers`. Embedding model must be pre-downloaded.

## Verification

After the benchmark completes:
1. Check `test_p5_results.json` — all runs should have `"success": true` (except known deepseek triple issue)
2. Verify per-model output dirs contain expected files (triples.jsonl or entity_extract.jsonl + graph files)
3. Cross-check graph node/edge counts in JSON against the stdout comparison table
4. Confirm the report `test_p5_models.md` covers all models × modes with analysis
