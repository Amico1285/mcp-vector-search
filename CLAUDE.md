# CLAUDE.md

Notes for Claude Code agents and human contributors working in this repository.
This file is auto-loaded by Claude Code when used inside the repo.

User-facing documentation lives in [README.md](README.md). This file describes
the **codebase itself** — where things live, how to extend them, what to keep
in sync.

## What this is

A Model Context Protocol (MCP) server that exposes a small set of tools backed
by a ChromaDB vector index built from a target codebase. The flagship tool is
`search_files`. Three embedding providers are supported (Voyage AI, OpenAI,
Ollama); a Voyage reranker and a Claude-CLI-based AI filter sit between the
initial vector search and the final result list.

## Repository layout

```
code_search_mcp/
  __main__.py              # Entry point; FastMCP startup (`main()`).
  server.py                # MCP tools + result formatting (Pipeline header,
                           # PREVIEW_CHARS_OUTPUT branching).
  searcher.py              # Search pipeline (vector → path_prefix → reranker
                           # → AI filter). Writes self.last_search_stats.
  database_updater.py      # Index/re-index files into ChromaDB.
  env_config.py            # SOURCE OF TRUTH for every supported env var.
  project_analyzer.py      # set_config(analyze=True) heuristics.
  file_processor.py        # File reading, truncation helpers.
  embedding_providers/     # voyage / voyage_context / openai / ollama
                           # subclassing base.py:EmbeddingProvider.
  aifilter/                # Claude CLI integration for the filter step.
  hybrid/                  # Optional BM25 + RRF fusion (off by default).
```

There is no `tests/` or `scripts/` directory in this repo on purpose — those
live in a separate research monorepo where the project is iterated on. Don't
add references to research artefacts in the public-facing docs.

## Local development

```bash
git clone https://github.com/Amico1285/mcp-vector-search.git
cd mcp-vector-search
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

`mcp-vector-search` becomes available on PATH as the entry point. Run it
stand-alone to confirm the FastMCP banner:

```bash
mcp-vector-search
```

## Picking a configuration for a new corpus

When wiring this MCP into a new project, **don't reach for the first Voyage Quick
Start block by default.** Different corpora benefit from very different settings.
Pick the closest match below and read the full preset (including reasoning) in
[README.md → Recommended Configurations](README.md#recommended-configurations).

| Corpus type | Embedding model | Chunking | Reranker | `PREVIEW_CHARS_OUTPUT` |
|---|---|---|---|---|
| Markdown docs / knowledge base | `voyage-context-3` | yes, 64 tokens | on | 200 |
| Source code / mixed monorepo | `voyage-3-large` (dim 1024) | off | on | 0 |
| Local / privacy-sensitive | Ollama `snowflake-arctic-embed2` | (default) | off | 100 |
| Cheap/fast cloud | OpenAI `text-embedding-3-large` | off | off, but `AI_FILTER_ENABLED=true` | 0 |

Two rules of thumb that the table encodes:

- **Long-form prose retrieves better with small contextualised chunks** — that's why docs use `voyage-context-3` + 64-token chunks instead of whole-file embeddings.
- **`PREVIEW_CHARS_OUTPUT` is a UX choice, not a quality choice.** If the agent will mostly inspect files via `Read`/`Grep`/`Glob` (typical for code), keep it 0. If the agent will skim a list and pick one (typical for docs), 100–200 chars saves an extra `Read` per query.

If unsure, copy the matching preset from `README.md` and tune from there.

## Smoke-testing changes

Point the server at any directory of code or docs you have nearby:

```bash
export CODEBASE_PATH="$HOME/some-project"
export DB_NAME=smoke
export EMBEDDING_PROVIDER=voyage
export VOYAGE_API_KEY=...
mcp-vector-search
```

Then hook it into Claude Code via `.mcp.json` (see [`.mcp.json.example`](.mcp.json.example)),
run `update_db(wait=True)`, and exercise `search_files(...)` with realistic
queries. Watch the `Pipeline:` header in each result — it is the single best
diagnostic for whether each stage of the pipeline did what you expected.

For pure unit-level work that doesn't need an LLM, `searcher.CodebaseSearcher`
can be imported and called directly without going through the MCP layer.

## Source of truth: environment variables

Every env var the server reads is declared as an `ENV_*` constant and listed in
the `DEFAULTS` dict in `env_config.py`. **The env-var tables in `README.md`
must stay in sync with that dict.** When you add a new variable:

1. Add an `ENV_<NAME>` string constant.
2. Add an entry in `DEFAULTS`.
3. Add a typed `get_<name>()` accessor.
4. Document it in `README.md` under the appropriate section table.

Anything not present in `env_config.py` is not actually read — putting it in
the README would be documentation rot. (We had a bunch of that before 0.4.1;
don't reintroduce it.)

## Adding a new embedding provider

1. Create `code_search_mcp/embedding_providers/<name>.py`, subclassing
   `EmbeddingProvider` from `base.py`. Implement `embed_query`,
   `embed_documents`, and `embed_documents_with_metadata` (the last enables
   chunked indexing — without it, files are vectorised whole).
2. Register the provider in
   `embedding_providers/__init__.py:create_embedding_provider()`.
3. Add provider-specific env vars to `env_config.py`.
4. Add a Quick Start config block in `README.md` and a row in the
   embedding-provider section.
5. Add the import dependency under `[project.optional-dependencies].<name>`
   in `pyproject.toml` and include it in the `all` extra.

## Search pipeline mental model

`searcher.search(query, path_prefix=None)`:

1. Embed the query, run vector search against ChromaDB →
   `SEMANTIC_SEARCH_N_RESULTS` candidates (deduped by file in chunked mode).
2. If `path_prefix` is set, drop candidates whose absolute path doesn't start
   with `<CODEBASE_PATH>/<path_prefix>/`. `..` segments are rejected.
3. If `RERANKER_ENABLED`, run Voyage rerank-2.5 and drop everything below
   `RERANKER_THRESHOLD`.
4. If `AI_FILTER_ENABLED` and the Claude CLI is on PATH, ask Claude to mark
   relevance.
5. Truncate to `MAX_RESULTS`.

After every call, `searcher.last_search_stats` carries per-stage counts.
`server.py:_pipeline_budget_line` reads that dict to render the `Pipeline:`
header. If you change the pipeline, update both `last_search_stats` and the
formatting line at the same time, otherwise the header will lie.

## Releasing

Single-package repo, no automated release pipeline. To cut a release:

1. Bump `version` in `pyproject.toml`.
2. Add a `CHANGELOG.md` entry under the new version. Use `### Breaking` when
   removing or renaming an env var or tool parameter.
3. Commit, tag (`git tag v<version>`), push (`git push && git push --tags`).

Users install via
`pip install git+https://github.com/Amico1285/mcp-vector-search.git`. PyPI is
not wired up yet — adding a publish-on-tag GitHub Action is on the wish list.
