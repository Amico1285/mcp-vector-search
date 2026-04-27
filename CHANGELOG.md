# Changelog

## 0.2.0 — 2026-04-27

### New
- `PREVIEW_CHARS_OUTPUT` controls how much per-file content is included in `search_files` results:
  - `0` → paths + scores only (agentic flow: the model reads files itself via `Read`/`Grep`/`Glob`)
  - `N` → first `N` characters of each file as a snippet
  - `-1` → full file content
- Every `search_files` response now includes a one-line **Pipeline budget** header showing how many candidates each stage produced (vector search → reranker → AI filter → final cut by `MAX_RESULTS`). Makes it obvious which knob to turn when results feel off.
- Every response also includes a one-line **Score interpretation** header (whether scores come from the reranker — higher is better — or from raw semantic distance — lower is better).
- File scores are now rendered with 5 decimal places (e.g. `0.71234`) for finer-grained comparison.

### Breaking
- `PREVIEW_LINES_OUTPUT` is **removed**. Replace with `PREVIEW_CHARS_OUTPUT` in your `.mcp.json`. Mapping: previous default of `80` lines roughly corresponds to ~3000 chars of typical markdown/code; the new default is `0` (no preview), which is friendlier for agentic workflows.

## 0.1.0 — 2026-04-27

Initial public release.
- MCP server for semantic code/docs search via vector embeddings.
- Embedding providers: VoyageAI, OpenAI, Ollama.
- Hybrid retrieval: dense + BM25 + RRF (off by default).
- VoyageAI reranker (rerank-2.5).
- Optional AI filter via Claude CLI.
- ChromaDB persistent storage with chunking and incremental updates.
