# Ollama Setup Guide for Local Embeddings

This guide will help you set up Ollama for generating embeddings locally, without sending your code to external APIs.

## Table of Contents
- [Installation](#installation)
- [Recommended Models](#recommended-models)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

## Installation

### Step 1: Install Ollama

#### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from website
# Visit https://ollama.ai and download the installer
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download the installer from [ollama.ai](https://ollama.ai)

### Step 2: Start Ollama Server

```bash
# Start the Ollama server (runs on port 11434 by default)
ollama serve
```

Keep this running in a separate terminal window.

### Step 3: Download Embedding Models

```bash
# Recommended: Qwen3 Embedding Model (best quality)
ollama pull qwen3-embedding-8b:q5_k_m

# Alternative: Smaller but still good
ollama pull qwen3-embedding-4b:q5_k_m

# Popular alternatives
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

### Step 4: Verify Installation

```bash
# List installed models
ollama list

# Test embedding generation
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "test"
}'
```

## Recommended Models

### Qwen3 Family (Top Performance)
| Model | Size | Quality | Speed | Memory | Dimensions |
|-------|------|---------|-------|--------|------------|
| qwen3-embedding-8b:q5_k_m | 5.5GB | Excellent | Medium | 8GB RAM | 2560 |
| qwen3-embedding-4b:q5_k_m | 2.8GB | Very Good | Fast | 4GB RAM | 2560 |
| qwen3-embedding-0.6b | 430MB | Good | Very Fast | 2GB RAM | 2560 |

### Alternative Models
| Model | Size | Use Case | Notes | Dimensions |
|-------|------|----------|-------|------------|
| nomic-embed-text | 274MB | General text | Fast, reliable | 768 |
| mxbai-embed-large | 670MB | Multilingual | Good for mixed codebases | 1024 |
| snowflake-arctic-embed | 669MB | Technical content | Good for code | 1024 |

## Configuration

### Basic Configuration

Add to your `.mcp.json` file:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "code_search_mcp"],
      "cwd": "/path/to/mcp-vector-search",
      "env": {
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "qwen3-embedding-8b:q5_k_m",
        
        "CODEBASE_PATH": "/path/to/your/codebase",
        
        "SEMANTIC_SEARCH_N_RESULTS": "20",
        "MAX_RESULTS": "10",
        
        "RERANKER_ENABLED": "false",
        "AI_FILTER_ENABLED": "false"
      }
    }
  }
}
```

### Advanced Configuration

For optimal performance with local models:

```json
{
  "env": {
    "EMBEDDING_PROVIDER": "ollama",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_EMBEDDING_MODEL": "qwen3-embedding-8b:q5_k_m",
    
    "SEMANTIC_SEARCH_N_RESULTS": "30",
    "MAX_RESULTS": "15",
    
    "RERANKER_ENABLED": "false",
    "AI_FILTER_ENABLED": "true",
    "AI_FILTER_MODEL": "claude-sonnet-4-20250514",
    
    "PREVIEW_LINES_VECTORIZATION": "50",
    "PREVIEW_LINES_OUTPUT": "100"
  }
}
```

## Usage

### First Time Setup

1. **Start Ollama server**:
   ```bash
   ollama serve
   ```

2. **Pull your chosen model**:
   ```bash
   ollama pull qwen3-embedding-8b:q5_k_m
   ```

3. **Update your `.mcp.json`** with Ollama configuration

4. **Create initial database**:
   ```python
   # In Claude Code
   update_db()  # Will use Ollama for embeddings
   ```

5. **Search your codebase**:
   ```python
   search_files("authentication logic")
   ```

### Switching Between Providers

```json
# To use VoyageAI (cloud)
"EMBEDDING_PROVIDER": "voyage"

# To use Ollama (local)
"EMBEDDING_PROVIDER": "ollama"
```

**Important**: When switching providers, reset and rebuild your database:
```python
reset_db()  # Clear existing embeddings
update_db()  # Re-index with new provider
```

## Troubleshooting

### Common Issues

#### 1. "Ollama server not responding"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

#### 2. "Model not found"
```bash
# List available models
ollama list

# Pull the required model
ollama pull qwen3-embedding-8b:q5_k_m
```

#### 3. "Dimension mismatch error"
- Different models generate different embedding dimensions (e.g., Qwen3: 2560, nomic-embed-text: 768)
- Switching between models requires resetting the database: `reset_db()`
- The system now uses native dimensions from each model automatically

#### 4. "Slow embedding generation"
- Try a smaller model (qwen3-embedding-4b or nomic-embed-text)
- Ensure Ollama has enough memory allocated
- Consider reducing batch size in heavy workloads

#### 5. "Fallback to VoyageAI"
The system automatically falls back to VoyageAI if Ollama is unavailable. Check:
- Ollama server is running
- Model is downloaded
- Network connectivity to localhost:11434

### Checking Ollama Status

```python
# Test script to verify Ollama setup
python3 test_ollama_integration.py
```

## Performance Tips

### Model Selection
- **Large codebases (>10k files)**: Use qwen3-embedding-4b for faster indexing
- **Small codebases (<1k files)**: Use qwen3-embedding-8b for best quality
- **Quick prototyping**: Use nomic-embed-text (fastest)

### Memory Management
```bash
# Set Ollama memory limit (example: 8GB)
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2
```

### Batch Processing
The integration automatically batches embeddings for optimal performance:
- Default batch size: 128 documents
- Adjust in `embedding_providers.py` if needed

### CPU vs GPU
Ollama automatically uses GPU if available:
```bash
# Check GPU usage
ollama ps  # Shows running models and resource usage
```

## Comparison with VoyageAI

| Aspect | Ollama (Local) | VoyageAI (Cloud) |
|--------|----------------|------------------|
| Privacy | ✅ Fully local | ❌ Sends code to API |
| Cost | ✅ Free | ❌ Per-token pricing |
| Speed | Depends on hardware | Generally faster |
| Quality | Very good (Qwen3) | Excellent (voyage-code-3) |
| Setup | Requires installation | Just API key |
| Offline | ✅ Works offline | ❌ Requires internet |

## Advanced: Using Custom Models

You can use any Ollama model that supports embeddings:

1. **Create custom modelfile**:
   ```dockerfile
   FROM llama3
   PARAMETER embedding_only true
   ```

2. **Build custom model**:
   ```bash
   ollama create my-embed-model -f Modelfile
   ```

3. **Use in configuration**:
   ```json
   "OLLAMA_EMBEDDING_MODEL": "my-embed-model"
   ```

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.ai/library)
- [Qwen3 Embedding Models](https://huggingface.co/collections/Qwen/qwen3-66e81a66c2dd3c92a6c01026)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (embedding model rankings)