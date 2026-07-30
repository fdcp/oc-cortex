# DOCUMENTATION

## OVERVIEW
`doc/` holds per-phase design docs, cross-phase tutorials, SVG/HTML figures, a how-to-run history, and patent disclosure drafts (the reason `tools/` exists).

## STRUCTURE
| File | Phase | Type |
|------|-------|------|
| `code_p1_README.md` … `code_p5_README.md` | P1–P5 | Per-phase design docs (概述→文件结构→核心流程→关键设计→输出格式→配置) |
| `code_p5e.md` | P5e | SQLite schema + Graph-RAG extension design |
| `code_p6.md` | P6 | Summarizer architecture, token budget, time filter |
| `code_p6b.md` | P6b | DecisionTracer + SummarySkeleton + relation categories |
| `code_mcp.md` | MCP | MCP protocol wrapper, 4 client config examples |
| `code_retrieval_principles.md` | cross | Dense/sparse/hybrid/coarse/fine search tutorial (all search phases) |
| `code_search_tools.md` | P3/P4/MS | When to use each of the 3 search scripts |
| `code_MS_tuning_notes.md` | P1/P2 | Quality audit + prompt/chunk tuning recommendations |
| `fig1_system_architecture.svg` | all | System architecture |
| `fig2_task_extraction.svg` | P2 | Task extraction flow |
| `fig3_entity_alignment.svg` | P5 | Entity alignment flow |
| `fig4_graph_rag_fusion.svg` | P5e | Graph-RAG fusion design |
| `fig5_decision_tracing.svg` | P6b | Decision tracing flow |
| `fig6_json_repair.svg` | P2/P5 | LLM JSON repair logic |
| `diagram_phase3_vector_store.html` | P3 | Interactive Qdrant collection architecture |
| `how_to_run.md`, `how_to_run_p12.md`, `how_to_run_p12_v2.md` | P1/P2/P3 | Superseded run guides; `README.md` is now authoritative |
| `专利交底书_*.md` / `*.docx` (Ver1–Ver5) | — | Patent disclosure drafts + generated DOCX |
| `交底书_跨会话知识图谱.md` | — | Alternate-structure patent draft |

## WHERE TO LOOK
- Per-phase design rationale: `code_pN_README.md` / `code_pNe.md` / `code_p6*.md`.
- Search algorithm theory and tool selection: `code_retrieval_principles.md`, `code_search_tools.md`.
- Quality tuning feedback into P1/P2: `code_MS_tuning_notes.md`.
- Figures are embedded in the matching README; keep SVG numbering aligned with phase order (`figN_*`).
- Patent DOCX regeneration pipeline: `专利交底书_*.md` → `../tools/md2docx.py` → `../tools/patch_formula.py`, `patch_code_block.py`, `patch_footer.py`.

## CONVENTIONS
- Phase READMEs follow a fixed section order (概述 → 文件结构 → 核心流程 → 关键设计 → 输出格式 → 配置); keep new phase docs in this shape.
- SVG figures are named `fig{N}_{subject}.svg` with `N` increasing along the pipeline; embed them in the relevant README.
- Patent docs are versioned `..._Ver{N}.docx`; the `.md` source under `doc/` is the Ver2 base, root-level `专利交底书_*.md` is Ver1.
- Mixed Chinese prose with English technical terms is the doc-standard language style.

## ANTI-PATTERNS
- Do not treat `how_to_run*.md` as current; they predate `README.md` and are kept for history only.
- Do not regenerate the patent DOCX outside macOS — `tools/md2docx.py` relies on `qlmanage` for SVG→PNG conversion.
- Do not commit large binary patent DOCX casually without confirming `.gitignore` policy; Ver3–Ver5 are already tracked.