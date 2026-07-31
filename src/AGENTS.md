# SOURCE MODULES

## OVERVIEW
`src/` is a flat, phase-prefixed Python module set. Each phase exposes one or more direct CLI scripts and reusable classes; there is no package initializer or installed module namespace.

## STRUCTURE
| Prefix | Responsibility | Main boundary |
|--------|----------------|---------------|
| `code_p1_*` | Load sessions, model records, chunk, clean | Writes `output/chunks.jsonl` |
| `code_p2_*` | LLM task and chunk-summary extraction | Reads P1 output; writes task JSONL |
| `code_p3_*` | Embeddings, BM25, Qdrant storage | Reads P1/P2 outputs; writes embedded DB |
| `code_p4_*` | Cross-session hybrid search and reranking | Reuses `Phase3Store` |
| `code_p5_*` | Triple/entity extraction, alignment, graph output | Uses LLM and Qdrant |
| `code_p5e_*` | SQLite graph persistence and Graph-RAG | Reused by P6b and MCP |
| `code_p6*` | Summary and decision-trace workflows | Consumes search and graph services |
| `code_mcp_*` | MCP stdio/HTTP server and client | External integration boundary |

## WHERE TO LOOK
- `code_p1_utils.py`: shared `Config`, logging, token counting, and truncation helpers.
- `code_p2_task_extractor.py`: threaded LLM extraction, retries, and JSON recovery.
- `code_p3_hf_config.py`: must run before importing Hugging Face model libraries.
- `code_p3_qdrant_store.py`: collection setup, dense/sparse retrieval, and RRF support.
- `code_p3_search_demo.py`: standalone interactive search demo (argparse CLI).
- `code_p4_searcher.py`: central search orchestration and result aggregation.
- `code_p4_reranker.py`: `Qwen3Reranker-0.6B` CausalLM re-scoring; reuses `_detect_device` from P3.
- `code_p5_kg_builder.py`: largest module; extraction, entity alignment, and graph construction.
- `code_p5_visualize.py`: pyvis HTML export from a `.gpickle`; runnable as `--visualize` from P5 main.
- `code_p5_benchmark.py`: model comparison harness for triple extraction (separate from `tests/`).
- `code_p5e_db.py`: SQLite schema, graph import, entity lookup, and BFS expansion.
- `code_p5e_graph_rag.py`: combines vector retrieval, graph expansion, and reranking.
- `code_mcp_server.py`: stdio MCP server (FastMCP, 5 KG tools).
- `code_mcp_server_http.py`: FastAPI HTTP MCP server; thin wrappers over `KGDatabase`.
- `code_mcp_client.py`: HTTP client + `ContextInjector` for cross-session memory injection.
- `code_MS_inspect.py`: Qdrant quality/stats inspection (direct `QdrantClient`, `argparse --config/--samples`; reads collections/paths/model from the Phase 3 YAML, resolving relative paths against the repo root so it runs from any CWD).

## CONVENTIONS
- Keep imports compatible with direct execution from the repository root; sibling imports use names such as `from code_p1_models import Chunk`.
- Add CLI behavior to an existing `*_main.py`, `*_cli.py`, or dedicated runnable utility rather than introducing a package entry point.
- Use `Config.load()` and the phase YAML files for runtime settings instead of hard-coding new paths or model parameters.
- Keep generated records JSON-serializable and preserve the existing JSONL schemas when changing pipeline stages.
- Use `loguru` for runtime logging and `argparse` for command-line flags.

## COMPLEXITY HOTSPOTS
- `code_p5_kg_builder.py` (1592 lines): `KGBuilder` conflates 6 concerns — LLM calling, triple extraction, entity extraction, Qdrant+LLM entity alignment, graph construction, and persistence. Primary refactor candidate; split alignment (~350 lines) and persistence first.
- `code_p3_qdrant_store.py` (988 lines): `Phase3Store` (~850-line class) is single-responsibility but large; the BGE-M3 sparse path adds conditional complexity.
- `code_p2_task_extractor.py` (835 lines): `_parse_and_validate` (~170 lines) is oversized and schema-driven.
- `code_p6b_skeleton.py` (621 lines): well-factored, but imports private `_get_client`/`DEFAULT_MODEL` from `code_p6_summarizer` — refactor to a shared client helper.

## RUNTIME CONSTRAINTS
- Hugging Face environment variables must be set before `sentence_transformers` or `transformers` imports. Existing modules use `code_p3_hf_config.setup_hf_env()` or set the variables directly.
- Qdrant is file-backed. `code_mcp_server.py` uses a single-worker executor so heavy initialization/search remains off the async loop and SQLite thread affinity is respected.
- P3 configuration is shared by P4, P6, and P6b. Changes to embedding, reranker, Qdrant, or file paths have cross-phase effects.
- LLM calls use `OPENCODE_ZEN_API_KEY`; do not embed credentials in code or YAML.
- `code_p3_main.py` and `code_p4_search_cli.py` pre-parse `--config` at module level (before `main()`) to set `TRANSFORMERS_OFFLINE` before HuggingFace imports — preserve this pattern when editing those entry points.
- P1 phase-specific YAML (`code_p1_config.yaml`) mixes wired and advisory keys (tagged inline as `[wired]`/`[advisory]`): `project.mock_data`, `opencode.session_source`, `opencode.session_filter.*`, and `logging.*` are read via `Config`; `chunking.*` and `content_cleaning.*` (except the unused `max_error_length` default) are hardcoded in `code_p1_chunker.py`/`code_p1_content_cleaner.py` and not read from YAML.
- P6/P6b phase-specific YAML (`code_p6_config.yaml`, `code_p6b_config.yaml`) is advisory only; the settings are hardcoded in `code_p6_summarizer.py` and `code_p6b_skeleton.py`. Wiring these to `Config.load()` is an open improvement, not the current behavior.

## ANTI-PATTERNS
- Do not create nested Python packages solely to group phases; the flat layout is part of the current import contract.
- Do not silently swallow malformed records in new code without matching the surrounding loader's logging/error behavior.
- Do not initialize heavyweight models at import time unless the existing CLI's pre-import environment setup is preserved.
- Do not write pipeline products into tracked source or configuration paths; use the configured `output/`, `logs/`, or `qdrant_data/` locations.
- Do not import private API (`_get_client`, `DEFAULT_MODEL`) across phase boundaries; use a shared client helper instead.
- Do not move the module-level `--config` pre-parse in HuggingFace entry points behind `main()`; it must run before `sentence_transformers`/`transformers` imports.
