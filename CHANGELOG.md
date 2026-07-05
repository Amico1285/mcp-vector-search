# Changelog

## 0.5.0 — 2026-07-05

### New
- **voyage-context-4 support.** The contextual embedding provider now accepts the whole `voyage-context-*` family (`voyage-context-3`, `voyage-context-4`, …) instead of being hardcoded to `voyage-context-3`. Set `VOYAGE_EMBEDDING_MODEL=voyage-context-4` to use it.
- **`VOYAGE_MAX_DOC_TOKENS`** (default `25600`) — per-document token budget for contextual models. A document whose chunks sum to more than this is split into parts, each embedded as its own contextualized example so it fits the model's 32000-token context window. `.mcp.json.example` documents it alongside the other `VOYAGE_*` chunking knobs.

### Fixed
- **Large corpora no longer abort indexing with "too many SQL variables".** ChromaDB's local (rusqlite) backend binds one SQL variable per returned row, so a single unbounded `collection.get()` overflowed SQLite's 32766-variable limit once a collection held more than ~32766 records. This aborted `update_db()` on large corpora *after* every embedding had already been written (e.g. during the final stats pass). All unbounded reads now page through the collection well below the limit.
- **Documents larger than the context window are split instead of failing.** Contextual embeddings do not support truncation, so any document whose chunks summed to more than 32000 tokens previously failed with "the example … does not fit into the model's context window of 32000 tokens". Oversized documents are now split into parts (bounded by `VOYAGE_MAX_DOC_TOKENS`); the per-document threshold was previously mis-compared against the whole-request budget (120000) instead of the per-document context window.
- **Empty documents no longer fail a whole batch.** Contextual requests now pack multiple documents per API call and skip empty/whitespace-only inputs, which the Voyage API rejects (one bad input used to fail the entire batch).
- **`exclude_dirs` matches exact path segments** relative to the codebase root instead of doing a substring match against the absolute path. Previously an exclude entry that was a substring of the codebase's own parent folder name silently excluded everything (0 files indexed).
- **No more "Changing the distance function" crash on config/reindex.** The cosine-space metadata is set only when the collection is created and stripped from every `collection.modify()` call — ChromaDB rejects `hnsw:space` in `modify()`.
- **`update_db()` reports its own run, not the previous one.** The `running` status is written synchronously before the background thread starts, so an immediate `get_server_info()` (or `wait=True`) no longer surfaces the prior run's stale `completed`/`error` status.

### Internal
- New `chroma_utils.py`: `get_all_records()` (paginated `get`) and `delete_by_ids()` (paginated `delete`), used across the updater and searcher.
- Contextual provider rewritten to batch documents per request and preserve the strict ascending-`doc_index` ordering the database updater relies on. Removed dead `_process_small_document` / `_process_medium_document`.

## 0.4.2 — 2026-04-27

### Documentation
- Added a "Recommended Configurations" section to `README.md` with four copy-paste-ready presets (documentation / code / local / cheap-cloud) and the reasoning behind each. Without these, models setting up the MCP for a new project tend to copy the first Voyage Quick Start block whether it fits the corpus or not.
- Added a corresponding decision table to `CLAUDE.md` so Claude Code agents working in this repo are nudged to pick a config that matches the corpus type instead of inventing one from scratch.

No code changes.

## 0.4.1 — 2026-04-27

### Documentation
- Removed phantom environment variables from `README.md` and `OLLAMA_SETUP.md` that were never read by the server: `ENABLE_CHUNKING`, `MAX_CHUNK_TOKENS`, `MIN_CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `OPENAI_EMBEDDING_DIMENSIONS`. The OpenAI/Ollama Quick Start examples no longer set settings the code silently ignores. Voyage chunking is documented under the `VOYAGE_*` prefixed variables that actually exist.
- Replaced the last leftover `PREVIEW_LINES_OUTPUT` (`OLLAMA_SETUP.md`) with `PREVIEW_CHARS_OUTPUT`.
- Documented previously hidden variables: `DB_NAME`, `RERANKER_USE_CHUNKS`, the full Hybrid Search suite (`HYBRID_SEARCH_ENABLED`, `BM25_*`, `RRF_*`).
- Made the three Quick Start config blocks (Voyage / OpenAI / Ollama) symmetric in the variables they set.
- Verify section now leads with the installed binary and shows the module-form invocation as an explicit fallback when the binary isn't on `PATH`.

No code changes.

## 0.4.0 — 2026-04-27

### New
- `search_files(query, path_prefix=...)` — optional second argument restricts results to files inside a given relative subtree of the indexed codebase (e.g. `path_prefix="frontend"` or `path_prefix="src/auth"`). Useful when one database covers multiple sub-projects and the question clearly belongs to one of them. Default is `None` — search everywhere, which is what you almost always want; the docstring explicitly tells the model not to set it speculatively. Path traversal (`..`) is rejected.
- The `Pipeline:` budget header now includes a `path_prefix='X' kept N` segment when the filter is applied, so the agent can see how aggressive the scope was.

### Internal
- `searcher.search()` accepts a `path_prefix` keyword argument and applies the filter between vector search and reranker (skips reranker work on excluded files). `last_search_stats` now records `path_prefix` and `after_path_filter`.

## 0.3.0 — 2026-04-27

### New
- `update_db(wait: bool = False)` — pass `wait=True` to block until indexing finishes (with a 600s timeout) and get the full report back in one call. Removes the need to poll `get_server_info()` from agentic flows.

### Fixed
- `reset_db()` now preserves the saved configuration (file extensions, excludes) by reading it from the existing collection metadata and recreating an empty collection with the same config. Previously the docstring claimed `update_db()` could be called right after, but it would fail with "Collection does not exist" — you had to call `set_config()` again. Now the documented one-step recovery actually works:
  ```
  reset_db()
  update_db(wait=True)
  ```
- `reset_db()` docstring rewritten to describe the new behaviour and to make explicit that `set_config()` is only required if no prior configuration exists.

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
