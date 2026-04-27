# MCP Vector Search

A high-performance MCP (Model Context Protocol) server designed specifically for Claude Code that enables precision semantic search through codebases and documentation. Built for agentic search. 

Unlike traditional search tools that flood agents with irrelevant data, MCP Vector Search uses a sophisticated 3-stage pipeline to deliver only the most relevant results.

## Search Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Claude Code     │    │  Vector Search  │    │   Reranker      │    │   AI Filter     │
│ Query           │───▶│                 │───▶│                 │───▶│                 │───▶ Results
│ "find auth code"│    │ ChromaDB        │    │ VoyageAI        │    │ Claude CLI      │
│                 │    │ ~20 candidates  │    │ Relevance       │    │ Precision       │
└─────────────────┘    └─────────────────┘    │ Scoring         │    │ Filtering       │
                                              └─────────────────┘    └─────────────────┘

Stage Impact:           Recall: ~100%          Precision: ~19%       Precision: ~55%
                       Precision: ~5%          Speed: +1s            Speed: +14s
                       Speed: ~0.2s            (Optional)            (Optional)
```

**Pipeline Stages:**
- **Vector Search**: Semantic similarity matching using embeddings
- **Reranker**: Advanced relevance scoring (VoyageAI rerank models)  
- **AI Filter**: Intelligent result filtering using Claude for precision

## Performance Analysis

All performance tests conducted on 506 Netwrix Auditor knowledge base articles. Each article paired with a specific question, automated testing scripts attempt to find correct articles. Results show real-world search accuracy across different configurations.

### Stage 1: Semantic Search Results

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR | Mean Position | Avg Search Time |
|--------------|----------|----------|----------|-----------|-----------|-----------|-----|---------------|-----------------|
| **voyage-3-large-2048dimensions** | 86.03% | 93.41% | 97.60% | 100.00% | 100.00% | 100.00% | 0.907 | 1.37 | 0.232s |
| **voyage-3-large-1024dimensions** | 84.43% | 93.81% | 98.20% | 100.00% | 100.00% | 100.00% | 0.900 | 1.37 | 0.227s |
| **voyage-3.5-lite-2048dimensions** | 83.23% | 93.01% | 96.41% | 99.20% | 99.60% | 99.80% | 0.888 | 1.79 | 0.249s |
| **openai-3-large-chunk-1024** | 77.05% | 92.61% | 97.41% | 99.00% | 99.40% | 99.80% | 0.856 | 1.75 | 0.371s |
| **ollama-snowflake-chunk-128** | 72.26% | 87.62% | 91.42% | 95.61% | 97.41% | 98.40% | 0.807 | 4.01 | 0.139s |

Voyage AI leads with 2048-dimension models. Chunking disabled by default, using full context window. Large files auto-chunk when exceeding limits.

**Optimal chunking:** OpenAI (1024 tokens), Ollama (128 tokens).

**Enterprise usage:** Voyage AI (open docs only), OpenAI (company-supported), Ollama (local/private).

### Stage 2: Reranker Analysis

#### Relevance Score Thresholds Distribution

| Configuration | Min | 1% | 3% | 5% | 10% | 25% | Median | Mean | Max |
|--------------|-----|----|----|----|----|-----|--------|------|-----|
| **voyage-3-large-2048-rerank-2.5** | 0.746 | 0.797 | 0.844 | 0.859 | 0.891 | 0.922 | 0.941 | 0.932 | 0.973 |

Reranker requires Voyage AI API key, open documentation only.

`RERANKER_THRESHOLD` controls precision/recall trade-off. Netwrix Auditor dataset shows minimum relevance 0.746 for correct files. Safe threshold 0.7+ includes all relevant files.

**Recommended thresholds:** Documentation (0.6-0.7), Code (0.5).

#### Threshold Impact on Document Count (501 questions dataset)

| Threshold | Description | Mean Docs | Median | Min | Max | Precision* | 
|-----------|-------------|-----------|--------|-----|-----|-----------|
| **0.700** | Current (baseline) | 6.52 | 5 | 1 | 20 | 15.3% |
| **0.797** | Safe (99% pass) | 3.83 | 2 | 0 | 20 | 26.1% |
| **0.844** | Balanced (97% pass) | 2.65 | 1 | 0 | 16 | 37.7% |
| **0.859** | Aggressive (95% pass) | 2.45 | 1 | 0 | 15 | 40.8% |

*Precision = 1 / Mean Docs × 100%

Higher thresholds reduce document count and improve precision from 15.3% to 40.8%.

### Stage 3: AI Filter Analysis

#### Model Performance Testing (v4 Precision Prompt)

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **claude-sonnet-4-20250514** | 0.900 | 0.900 | 0.900 | 0.900 |
| **claude-opus-4-1-20250805** | 0.818 | 0.900 | 0.857 | 0.850 |
| **claude-3-5-haiku-20241022** | 0.750 | 0.300 | 0.429 | 0.600 |

Claude Sonnet 4: 90% precision/recall - optimal for filtering.
Claude Opus 4: 81.8% precision - acceptable with more false positives.
Claude Haiku: 30% recall - unsuitable for filtering.

### Precision Impact Analysis

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Position | Precision* | Avg Search Time |
|--------------|----------|----------|----------|-----------|-----|---------------|------------|----------------|
| **voyage-3-large-rerank-threshold-0.888** | 81.4% | 88.8% | 89.4% | 90.2% | 0.852 | 1.17 | **54.9%** | ~1.0s |
| **voyage-3-large-rerank-aifilter** | 77.45% | 82.04% | 83.23% | 84.63% | 0.800 | 1.22 | **55.25%** | 14.78s |
| **voyage-3-large-2048-rerank** | 88.02% | 96.61% | 98.40% | 100.00% | 0.927 | 1.26 | **18.98%** | 0.985s |
| **voyage-3-large-2048dimensions** | 86.03% | 93.41% | 97.60% | 100.00% | 0.907 | 1.37 | **5.00%** | 0.232s |

*Precision = 1 / Mean Documents Returned

**Key Findings:**
- **Optimal approach**: Reranker with threshold=0.888 achieves same precision (54.9%) as AI Filter but with better recall
- **Performance comparison** at ~55% precision:
  - Reranker (0.888): Recall@1=81.4%, Recall@5=89.4%, Speed=1s
  - AI Filter: Recall@1=77.45%, Recall@5=83.23%, Speed=15s

**Recommendations by Use Case:**
- **Open documentation**: Use Voyage AI with high reranker threshold (0.5-0.8) for optimal speed and precision
- **Company codebases**: Use OpenAI models with AI Filter to achieve similar precision results when Voyage AI unavailable

**Why Precision is Critical for Claude Code:**

Precision is the most important metric for agent search systems due to Claude Code's limited context window. Low precision has severe consequences:

**Context Window Waste**: With 5% precision (semantic search only), returning 20 documents means only 1 is relevant - wasting 95% of valuable context on irrelevant data. Claude Code agents often perform multiple searches, and context depletion ends conversations prematurely.

**Model Confusion**: Irrelevant documents can mislead the model, causing incorrect conclusions or actions.

**Semantic Search Limitations**: 
- Always returns K results regardless of relevance
- No concept of "no relevant results" - will return 4 documents even when none are relevant  
- Cannot handle variable result counts - returns 4 when 20 relevant files exist

**Reranker Solution**: Unlike semantic search, reranker evaluates query-document relevance with threshold filtering:
- **No relevant docs**: Returns empty results (threshold not met)
- **Multiple relevant docs**: Returns all above threshold
- **Adaptive results**: Result count varies based on actual relevance

This threshold-based filtering is the key improvement that makes search practical for agent use.

## Prerequisites

- Python 3.10+
- One of the following embedding providers:
  - [VoyageAI API key](https://www.voyageai.com/) for cloud embeddings
  - [OpenAI API key](https://platform.openai.com/api-keys) for cloud embeddings
  - [Ollama](https://ollama.ai/) for local embeddings
- (Optional) [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) for AI filtering

## Installation

### Option A: Install from GitHub (recommended)

Pick the extra that matches your embedding provider:

```bash
# VoyageAI only
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[voyage]"

# OpenAI only
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[openai]"

# Ollama only
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[ollama]"

# Everything
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[all]"
```

This installs the `mcp-vector-search` command into your active environment.

### Option B: From source (for development)

```bash
git clone https://github.com/Amico1285/mcp-vector-search.git
cd mcp-vector-search
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -e ".[all]"
```

### Verify

```bash
mcp-vector-search --help 2>/dev/null || python -m code_search_mcp
```

You should see the FastMCP startup banner. Press `Ctrl+C` to stop.

## Quick Start with Claude Code

This MCP server is designed to work seamlessly with Claude Code, providing agents with powerful search and database management capabilities.

### 1. Add MCP server to Claude Code

**Option A: Using Claude CLI command**
```bash
# Add as project-scoped server (creates .mcp.json in project root)
claude mcp add code-search --scope project mcp-vector-search
```

**Option B: Manual configuration**
1. Copy `.mcp.json.example` to `.mcp.json` in your project root and fill in the values
2. Exit Claude Code
3. Open Claude Code again in the same project
4. When prompted "Use MCP servers found in this project?", answer "Yes"
5. The MCP server is now connected

### 2. Configure the server

Edit the `.mcp.json` file to add your configuration. Choose one of the embedding providers:

#### Option A: VoyageAI (Cloud) - Best Performance with Reranker
```json
{
  "mcpServers": {
    "code-search": {
      "command": "mcp-vector-search",
      "env": {
        "CODEBASE_PATH": "/path/to/your/codebase",
        
        "EMBEDDING_PROVIDER": "voyage",
        "VOYAGE_API_KEY": "your-voyage-api-key-here",
        "VOYAGE_EMBEDDING_MODEL": "voyage-3-large",
        "VOYAGE_OUTPUT_DIMENSION": "2048",
        
        "SEMANTIC_SEARCH_N_RESULTS": "20",
        
        "RERANKER_ENABLED": "true",
        "RERANKER_THRESHOLD": "0.7",
        "RERANKER_MODEL": "rerank-2.5",
        "RERANKER_INSTRUCTIONS": "",
        
        "AI_FILTER_ENABLED": "false",
        "AI_FILTER_MODEL": "claude-sonnet-4-20250514",
        "AI_FILTER_TIMEOUT_SECONDS": "120",
        
        "MAX_RESULTS": "10",
        
        "LOGGING_VERBOSE": "false",
        "LOGGING_FILE_ENABLED": "false",
        "LOGGING_FILE_PATH": "Logs/search_operations.log",
        
        "PREVIEW_LINES_VECTORIZATION": "-1",
        "PREVIEW_LINES_STORAGE": "-1",
        "PREVIEW_LINES_RERANKER": "-1",
        "PREVIEW_LINES_AI_FILTER": "-1",
        "PREVIEW_LINES_OUTPUT": "80"
      }
    }
  }
}
```

#### Option B: OpenAI (Cloud) - Good Performance, Flexible
```json
{
  "mcpServers": {
    "code-search": {
      "command": "mcp-vector-search",
      "env": {
        "CODEBASE_PATH": "/path/to/your/codebase",
        
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        "ENABLE_CHUNKING": "true",
        "MAX_CHUNK_TOKENS": "1024",

        "SEMANTIC_SEARCH_N_RESULTS": "20",
        
        "AI_FILTER_ENABLED": "true",
        "AI_FILTER_MODEL": "claude-sonnet-4-20250514",
        "AI_FILTER_TIMEOUT_SECONDS": "120",
        
        "MAX_RESULTS": "10",
        
        "LOGGING_VERBOSE": "false",
        "LOGGING_FILE_ENABLED": "false",
        "LOGGING_FILE_PATH": "Logs/search_operations.log",
        
        "PREVIEW_LINES_VECTORIZATION": "-1",
        "PREVIEW_LINES_STORAGE": "-1",
        "PREVIEW_LINES_AI_FILTER": "-1",
        "PREVIEW_LINES_OUTPUT": "80"
      }
    }
  }
}
```

#### Option C: Ollama (Local) - Complete Privacy
```json
{
  "mcpServers": {
    "code-search": {
      "command": "mcp-vector-search",
      "env": {
        "CODEBASE_PATH": "/path/to/your/codebase",
        
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "snowflake-arctic-embed2",
        "ENABLE_CHUNKING": "true",
        "MAX_CHUNK_TOKENS": "128",
        
        "SEMANTIC_SEARCH_N_RESULTS": "20",
        
        "AI_FILTER_ENABLED": "true",
        "AI_FILTER_MODEL": "claude-sonnet-4-20250514",
        "AI_FILTER_TIMEOUT_SECONDS": "120",
        
        "MAX_RESULTS": "10",
        
        "LOGGING_VERBOSE": "false",
        "LOGGING_FILE_ENABLED": "false",
        "LOGGING_FILE_PATH": "Logs/search_operations.log",
        
        "PREVIEW_LINES_VECTORIZATION": "-1",
        "PREVIEW_LINES_STORAGE": "-1",
        "PREVIEW_LINES_AI_FILTER": "-1",
        "PREVIEW_LINES_OUTPUT": "80"
      }
    }
  }
}
```


**Note**: If `mcp-vector-search` is installed inside a virtualenv that is not on Claude Code's `PATH`, set `command` to the absolute path of that binary, e.g. `/path/to/venv/bin/mcp-vector-search`.

### 3. Reconnect to the server

In Claude Code, use the `/mcp` command to reconnect and verify the server is running.

### 4. Agent Capabilities

Once connected, Claude Code agents can autonomously:

#### Initialize and manage the database:

**Step 1: Create configuration**
```
"Please analyze my codebase and create search configuration"
```

The agent will:
- Analyze your codebase structure and detect frameworks
- Generate configuration showing which files will be indexed
- Display total file count and exclusion patterns
- Allow you to review and adjust the configuration

**Step 2: Start vectorization**
```
"Start vectorizing the database with current configuration"
```

After reviewing the configuration, the agent uses a separate tool to begin vectorization. For large codebases, enable logging in `.mcp.json` to track progress:
```json
"LOGGING_VERBOSE": "true",
"LOGGING_FILE_ENABLED": "true"
```

#### Search with natural language:
```
"Find the authentication implementation"
"Show me where error handling happens"
"Locate the API endpoint definitions"
"Find documentation about deployment"
```

#### Update and reconfigure on demand:
```
"Update the search index with the latest changes"
"Exclude .txt files from search"
"Exclude /folder files from search"
"Add .md files to the search scope"
```

## Configuration

The server behavior is customized via environment variables.

**Important**: After changing configuration in `.mcp.json`, you need to restart Claude Code for changes to take effect (this also restarts the MCP server). To preserve your conversation progress, use:
```bash
claude --continue
```

### Required Variables
| Variable | Description | Default |
|----------|-------------|----------|
| `CODEBASE_PATH` | Path to the codebase to index | (required) |
| `EMBEDDING_PROVIDER` | Embedding provider to use | voyage |

### Embedding Provider Settings

#### VoyageAI (Cloud)
| Variable | Description | Default |
|----------|-------------|----------|
| `VOYAGE_API_KEY` | Your VoyageAI API key | (required) |
| `VOYAGE_EMBEDDING_MODEL` | VoyageAI model for embeddings | voyage-code-3 |
| `VOYAGE_OUTPUT_DIMENSION` | Output vector dimensions (256/512/1024/2048) | (model default) |
| `VOYAGE_ENABLE_CHUNKING` | Enable chunking for Voyage models | true |
| `VOYAGE_MAX_CHUNK_TOKENS` | Maximum tokens per chunk for Voyage | (model-specific) |
| `VOYAGE_CHUNK_OVERLAP_TOKENS` | Overlap tokens between chunks for Voyage | (model-specific) |
| `VOYAGE_MIN_CHUNK_TOKENS` | Minimum tokens per chunk for Voyage | (model-specific) |

#### OpenAI (Cloud)
| Variable | Description | Default |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | (required) |
| `OPENAI_EMBEDDING_MODEL` | OpenAI model for embeddings | text-embedding-3-large |
| `OPENAI_EMBEDDING_DIMENSIONS` | Optional dimension reduction | (model default) |
| `OPENAI_BATCH_SIZE` | Batch size for OpenAI API requests | 2048 |

#### Ollama (Local)
| Variable | Description | Default |
|----------|-------------|----------|
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_EMBEDDING_MODEL` | Ollama model name | snowflake-arctic-embed2 |

### Chunking Settings (Universal)
| Variable | Description | Default |
|----------|-------------|----------|
| `ENABLE_CHUNKING` | Enable automatic file chunking for large files | true |
| `MAX_CHUNK_TOKENS` | Maximum tokens per chunk | Model-specific |
| `MIN_CHUNK_TOKENS` | Minimum tokens per chunk | Model-specific |
| `CHUNK_OVERLAP_TOKENS` | Number of overlapping tokens between chunks | Model-specific |

### Search Pipeline Settings
| Variable | Description | Default |
|----------|-------------|----------|
| `SEMANTIC_SEARCH_N_RESULTS` | Initial candidates from vector search | 20 |
| `RERANKER_ENABLED` | Enable VoyageAI reranker | true |
| `RERANKER_THRESHOLD` | Minimum relevance score | 0.5 |
| `RERANKER_MODEL` | Reranker model name | rerank-2.5 |
| `RERANKER_INSTRUCTIONS` | Custom instructions for reranker to improve relevance | (empty) |
| `AI_FILTER_ENABLED` | Enable Claude CLI filtering | true |
| `AI_FILTER_MODEL` | Claude model for filtering | claude-sonnet-4-20250514 |
| `AI_FILTER_TIMEOUT_SECONDS` | Timeout for Claude CLI | 120 |
| `MAX_RESULTS` | Maximum results returned | 10 |

### Logging Settings
| Variable | Description | Default |
|----------|-------------|----------|
| `LOGGING_VERBOSE` | Enable verbose console logging | false |
| `LOGGING_FILE_ENABLED` | Enable file logging | false |
| `LOGGING_FILE_PATH` | Path to log file | Logs/search_operations.log |

### Preview Lines Settings

Controls how many lines of code are used at each stage of the search pipeline:

| Variable | Description | Default | Details |
|----------|-------------|---------|---------|
| `PREVIEW_LINES_VECTORIZATION` | Lines used to create search embeddings | 30 | First N lines of each file are vectorized for semantic search. Higher = better context but slower. -1 = entire file |
| `PREVIEW_LINES_STORAGE` | Lines stored in database | -1 | How much code to save per file. -1 = entire file (recommended) |
| `PREVIEW_LINES_RERANKER` | Lines sent to VoyageAI reranker | 100 | Code context for relevance scoring. Balance between accuracy and speed. -1 = entire file |
| `PREVIEW_LINES_AI_FILTER` | Lines sent to Claude for filtering | 40 | Code context for AI relevance evaluation. More lines = better judgment. -1 = entire file |
| `PREVIEW_LINES_OUTPUT` | Lines shown in search results | 80 | How much code you see in the final output. -1 = entire file |

All settings are configured via environment variables in your MCP server configuration.

## Advanced Usage

### Use Cases

This tool excels in various scenarios:

- **Large Codebases**: Navigate complex projects with thousands of files effortlessly
- **Documentation Search**: Find relevant documentation sections instantly
- **Code Reviews**: Quickly locate related code sections during reviews
- **Onboarding**: Help new team members explore and understand the codebase
- **Refactoring**: Find all instances of patterns that need updating
- **Debugging**: Locate error handling and logging implementations



## Troubleshooting

### `command not found: mcp-vector-search`

The package was installed but the binary isn't on Claude Code's `PATH`. This is typical when the install lives inside a virtualenv.

In `.mcp.json`, point at the absolute path to the binary:
```json
"command": "/absolute/path/to/venv/bin/mcp-vector-search"
```

Or invoke the module directly:
```json
"command": "/absolute/path/to/venv/bin/python",
"args": ["-m", "code_search_mcp"]
```

### `ModuleNotFoundError: No module named 'fastmcp'` (or any other dep)

Dependencies were not installed for the active Python environment. Reinstall with the right extra:
```bash
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[all]"
```

### `externally-managed-environment`

Modern macOS/Linux protect system Python. Always install into a virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate
pip install "git+https://github.com/Amico1285/mcp-vector-search.git#egg=mcp-vector-search[all]"
```

### `CODEBASE_PATH environment variable not set`

Add it to the `env` block of your `.mcp.json` — see [`.mcp.json.example`](.mcp.json.example).

### `Claude CLI not found`

The AI filter is optional. Set `AI_FILTER_ENABLED=false`, or install the [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli).

### Search returns no results

1. Check `get_server_info()` — has the codebase been vectorized yet?
2. Run `update_db()` to (re)build the index.
3. Verify the embedding provider's API key is correct.
4. Enable verbose logs:
   ```json
   "LOGGING_VERBOSE": "true",
   "LOGGING_FILE_ENABLED": "true"
   ```
   and tail `Logs/search_operations.log`.

### Using one installation across multiple projects

Install once, then point each project's `.mcp.json` at the same binary with a different `DB_NAME`:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "mcp-vector-search",
      "env": {
        "CODEBASE_PATH": "/absolute/path/to/project1",
        "DB_NAME": "project1_db",
        "EMBEDDING_PROVIDER": "voyage",
        "VOYAGE_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Windows

If `mcp-vector-search` doesn't resolve through Claude Code on Windows, point at the venv binary directly:
```json
"command": "C:\\path\\to\\venv\\Scripts\\mcp-vector-search.exe"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments

- [VoyageAI](https://www.voyageai.com/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Anthropic](https://www.anthropic.com/) for Claude CLI
- [FastMCP](https://github.com/anthropics/fastmcp) for the MCP framework