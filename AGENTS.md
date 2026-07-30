# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-30
**Commit:** a7ca072
**Branch:** main

## OVERVIEW
OpenCode Session Knowledge Graph is a Python 3.10+ pipeline that turns OpenCode session history into searchable tasks, vector indexes, a knowledge graph, Graph-RAG results, summaries, and decision traces. The repository is script-driven rather than packaged.

## STRUCTURE
```text
oc_sess_graph/
├── src/          # Flat phase-prefixed Python implementation
├── config/       # YAML runtime configuration (7 yaml + code_p1_requirements.txt)
├── prompts/      # LLM prompt templates extracted from code (4 .md)
├── tests/        # Standalone P2/P5 model benchmarks, NOT pytest
├── doc/          # Phase READMEs, SVG diagrams, patent disclosures, how-to-run
├── skills/       # OpenCode benchmark skill definitions (phase2, phase5)
├── tools/        # DOCX generation scripts for patent disclosures (md2docx + 3 patchers)
├── data/         # Empty placeholder for JSONL session imports (SQLite is primary)
├── lib/          # Vendored browser assets for pyvis graph visualization
├── output/       # Generated pipeline products; ignored
├── qdrant_data/  # Embedded Qdrant database; ignored
├── logs/         # Runtime logs; ignored
└── (root)        # README.md, install.sh, requirements.txt, 图谱方案.md, patent .md drafts
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Session loading, chunking, cleaning | `src/code_p1_*.py` | Produces `output/chunks.jsonl` |
| LLM task extraction | `src/code_p2_*.py`, `prompts/p2_session_task_extraction.md` | Produces tasks and chunk summaries |
| Vector indexing and search | `src/code_p3_*.py`, `src/code_p4_*.py` | Qdrant is embedded at `qdrant_data/` |
| Knowledge graph construction | `src/code_p5_*.py`, `src/code_p5e_*.py` | Triple/entity modes and SQLite Graph-RAG |
| Summaries and decision traces | `src/code_p6*.py`, `prompts/p6_p6b_summarization.md` | Consumes search and graph outputs |
| MCP integration | `src/code_mcp_*.py`, `config/code_mcp_config.yaml` | stdio server plus HTTP debug variant |
| Runtime settings | `config/` | `code_p3_config.yaml` is shared by P3/P4/P6/P6b |
| Design docs and diagrams | `doc/` | See `doc/AGENTS.md`; phase READMEs + SVG figures + patent drafts |
| Patent DOCX generation | `tools/` | md2docx + patch_formula/code_block/footer; macOS `qlmanage` only |
| Benchmark runs | `tests/` | See `tests/AGENTS.md` |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Config` | class | `src/code_p1_utils.py` | Singleton YAML config with dotted-key access |
| `TaskExtractor` | class | `src/code_p2_task_extractor.py` | Threaded LLM task extraction, JSON repair, validation |
| `Phase3Store` | class | `src/code_p3_qdrant_store.py` | Dense/sparse vector storage, 3-collection Qdrant lifecycle |
| `SessionSearcher` | class | `src/code_p4_searcher.py` | Hybrid retrieval, RRF, reranking, chunk→task aggregation |
| `Qwen3Reranker` | class | `src/code_p4_reranker.py` | Qwen3-Reranker-0.6B CausalLM relevance re-scoring |
| `KGBuilder` | class | `src/code_p5_kg_builder.py` | LLM extraction, entity alignment, graph construction (1592 lines) |
| `UnionFind` | class | `src/code_p5_kg_builder.py` | Disjoint-set for entity deduplication |
| `KGDatabase` | class | `src/code_p5e_db.py` | SQLite graph persistence and BFS queries |
| `GraphRAGSearcher` | class | `src/code_p5e_graph_rag.py` | Vector plus graph retrieval fusion |
| `SessionSummarizer` | class | `src/code_p6_summarizer.py` | LLM summary generation from retrieved tasks |
| `DecisionTracer` | class | `src/code_p6b_skeleton.py` | Multi-hop decision-chain tracing |
| `SummarySkeleton` | class | `src/code_p6b_skeleton.py` | Topic→skeleton→LLM structured summary pipeline |
| `ContextInjector` | class | `src/code_mcp_client.py` | Auto-inject cross-session memory into new sessions |

## CONVENTIONS
- Run scripts from the repository root with `python3 src/<script>.py`; this is not an installed Python package.
- Source files use the flat `code_p{phase}_*.py` naming scheme and import sibling modules by bare module name.
- Python CLIs use stdlib `argparse`; logging uses `loguru`; JSONL is the normal interchange format.
- Python 3.10 or newer is required. Dependencies are listed in `requirements.txt`; there is no `pyproject.toml` or dev-tool configuration.
- Set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` before Hugging Face model imports. Set `OPENCODE_ZEN_API_KEY` for LLM phases.
- Preserve `ensure_ascii=False` for Chinese JSON output and UTF-8 file handling.

## ANTI-PATTERNS (THIS PROJECT)
- Do not treat benchmark scripts as pytest tests; they are direct CLI programs and require generated pipeline inputs.
- Do not commit `output/`, `qdrant_data/`, `logs/`, databases, caches, or model artifacts; `.gitignore` defines the repository boundary.
- Do not assume a service-based Qdrant deployment; the supported path is local embedded Qdrant storage.
- Do not import `sentence_transformers` or `transformers` before the offline environment setup where the module requires it.
- Do not change shared `code_p3_config.yaml` semantics without checking P3, P4, P6, and P6b consumers.
- Do not treat `code_p6_config.yaml` / `code_p6b_config.yaml` as authoritative; their settings are hardcoded in `code_p6_summarizer.py` and `code_p6b_skeleton.py` and the YAML is advisory only.

## COMMANDS
```bash
bash install.sh                 # Full environment and model setup
bash install.sh --light         # Skip heavy model dependencies/downloads
python3 src/code_p1_main.py     # P1: sessions -> chunks
python3 src/code_p2_main.py     # P2: chunks -> tasks/summaries
python3 src/code_p3_main.py     # P3: build Qdrant indexes
python3 src/code_p5_main.py     # P5: build triple graph
python3 src/code_p4_search_cli.py --query "..."
python3 src/code_p6b_cli.py summary "..."
python3 tests/test_p2/test_p2.py
python3 tests/test_p5/test_p5.py
```

## NOTES
- The normal data flow is P1 -> P2 -> P3, with P4 as search, P5/P5e as graph construction and retrieval, and P6/P6b/MCP as consumers.
- `install.sh` is the de facto build system. There is no CI, Docker configuration, Makefile, or standard Python packaging metadata.
- `code_p5_kg_builder.py` (`KGBuilder`, 1592 lines) is the largest module and the primary refactor candidate — it conflates LLM calling, triple/entity extraction, Qdrant+LLM entity alignment, graph construction, and persistence.
- `code_p3_main.py` and `code_p4_search_cli.py` pre-parse `--config` at module level before `main()` to set `TRANSFORMERS_OFFLINE` before HuggingFace imports; preserve this pattern when editing those entry points.
- `data/` is an empty placeholder; the primary input path is SQLite at `--sqlite ~/.local/share/opencode/opencode.db`.
