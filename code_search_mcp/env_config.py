"""Environment configuration helper for code search MCP server."""
import os
from typing import Any, Optional, List


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value."""
    return os.getenv(key, default)


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on', 't', 'y')


def get_int_env(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_float_env(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


# Environment variable names
ENV_EMBEDDING_PROVIDER = 'EMBEDDING_PROVIDER'
ENV_VOYAGE_EMBEDDING_MODEL = 'VOYAGE_EMBEDDING_MODEL'
ENV_VOYAGE_MAX_CHUNK_TOKENS = 'VOYAGE_MAX_CHUNK_TOKENS'
ENV_VOYAGE_CHUNK_OVERLAP_TOKENS = 'VOYAGE_CHUNK_OVERLAP_TOKENS'
ENV_VOYAGE_MIN_CHUNK_TOKENS = 'VOYAGE_MIN_CHUNK_TOKENS'
ENV_VOYAGE_ENABLE_CHUNKING = 'VOYAGE_ENABLE_CHUNKING'
ENV_VOYAGE_OUTPUT_DIMENSION = 'VOYAGE_OUTPUT_DIMENSION'
ENV_OLLAMA_BASE_URL = 'OLLAMA_BASE_URL'
ENV_OLLAMA_EMBEDDING_MODEL = 'OLLAMA_EMBEDDING_MODEL'
ENV_OPENAI_API_KEY = 'OPENAI_API_KEY'
ENV_OPENAI_EMBEDDING_MODEL = 'OPENAI_EMBEDDING_MODEL'
ENV_OPENAI_BATCH_SIZE = 'OPENAI_BATCH_SIZE'
ENV_SEMANTIC_SEARCH_N_RESULTS = 'SEMANTIC_SEARCH_N_RESULTS'
ENV_RERANKER_ENABLED = 'RERANKER_ENABLED'
ENV_RERANKER_THRESHOLD = 'RERANKER_THRESHOLD'
ENV_RERANKER_MODEL = 'RERANKER_MODEL'
ENV_RERANKER_INSTRUCTIONS = 'RERANKER_INSTRUCTIONS'
ENV_RERANKER_USE_CHUNKS = 'RERANKER_USE_CHUNKS'
ENV_AI_FILTER_ENABLED = 'AI_FILTER_ENABLED'
ENV_AI_FILTER_MODEL = 'AI_FILTER_MODEL'
ENV_AI_FILTER_TIMEOUT_SECONDS = 'AI_FILTER_TIMEOUT_SECONDS'
ENV_MAX_RESULTS = 'MAX_RESULTS'
ENV_LOGGING_VERBOSE = 'LOGGING_VERBOSE'
ENV_LOGGING_FILE_ENABLED = 'LOGGING_FILE_ENABLED'
ENV_LOGGING_FILE_PATH = 'LOGGING_FILE_PATH'
ENV_PREVIEW_LINES_VECTORIZATION = 'PREVIEW_LINES_VECTORIZATION'
ENV_PREVIEW_LINES_STORAGE = 'PREVIEW_LINES_STORAGE'
ENV_PREVIEW_LINES_RERANKER = 'PREVIEW_LINES_RERANKER'
ENV_PREVIEW_LINES_AI_FILTER = 'PREVIEW_LINES_AI_FILTER'
ENV_PREVIEW_CHARS_OUTPUT = 'PREVIEW_CHARS_OUTPUT'
ENV_DB_NAME = 'DB_NAME'

# Hybrid Search Environment Variables
ENV_HYBRID_SEARCH_ENABLED = 'HYBRID_SEARCH_ENABLED'
ENV_BM25_ONLY_MODE = 'BM25_ONLY_MODE'
ENV_RRF_K_PARAMETER = 'RRF_K_PARAMETER'
ENV_RRF_WEIGHTS_ENABLED = 'RRF_WEIGHTS_ENABLED'
ENV_RRF_VECTOR_WEIGHT = 'RRF_VECTOR_WEIGHT'
ENV_RRF_BM25_WEIGHT = 'RRF_BM25_WEIGHT'
ENV_BM25_K1_PARAMETER = 'BM25_K1_PARAMETER'
ENV_BM25_B_PARAMETER = 'BM25_B_PARAMETER'
ENV_BM25_N_RESULTS = 'BM25_N_RESULTS'
ENV_BM25_MIN_TOKEN_LENGTH = 'BM25_MIN_TOKEN_LENGTH'
ENV_BM25_REMOVE_STOPWORDS = 'BM25_REMOVE_STOPWORDS'
ENV_BM25_LANGUAGE = 'BM25_LANGUAGE'
ENV_BM25_USE_STEMMING = 'BM25_USE_STEMMING'
ENV_BM25_USE_CHUNKING = 'BM25_USE_CHUNKING'

# Default values (same as original search_config.json)
DEFAULTS = {
    ENV_EMBEDDING_PROVIDER: 'voyage',
    ENV_VOYAGE_EMBEDDING_MODEL: 'voyage-code-3',
    ENV_VOYAGE_OUTPUT_DIMENSION: None,  # None means use model default
    ENV_OLLAMA_BASE_URL: 'http://localhost:11434',
    ENV_OLLAMA_EMBEDDING_MODEL: 'snowflake-arctic-embed2',
    ENV_OPENAI_API_KEY: '',
    ENV_OPENAI_EMBEDDING_MODEL: 'text-embedding-3-large',
    ENV_OPENAI_BATCH_SIZE: 2048,  # Maximum batch size for OpenAI API
    ENV_SEMANTIC_SEARCH_N_RESULTS: 20,
    ENV_RERANKER_ENABLED: True,
    ENV_RERANKER_THRESHOLD: 0.5,
    ENV_RERANKER_MODEL: 'rerank-2.5',
    ENV_RERANKER_INSTRUCTIONS: '',  # Empty by default, user can set custom instructions
    ENV_RERANKER_USE_CHUNKS: False,  # If True, send chunk text to reranker instead of full file
    ENV_AI_FILTER_ENABLED: True,
    ENV_AI_FILTER_MODEL: 'claude-sonnet-4-20250514',
    ENV_AI_FILTER_TIMEOUT_SECONDS: 120,
    ENV_MAX_RESULTS: 10,
    ENV_LOGGING_VERBOSE: False,
    ENV_LOGGING_FILE_ENABLED: False,
    ENV_LOGGING_FILE_PATH: 'Logs/search_operations.log',
    ENV_PREVIEW_LINES_VECTORIZATION: 30,
    ENV_PREVIEW_LINES_STORAGE: -1,
    ENV_PREVIEW_LINES_RERANKER: 100,  # Increased to utilize 32K context of rerank-2.5
    ENV_PREVIEW_LINES_AI_FILTER: 40,
    ENV_PREVIEW_CHARS_OUTPUT: 0,  # 0 = paths + scores only; N = first N chars; -1 = whole file
    ENV_DB_NAME: 'codebase_files',
    # Hybrid Search defaults
    ENV_HYBRID_SEARCH_ENABLED: False,
    ENV_BM25_ONLY_MODE: False,
    ENV_RRF_K_PARAMETER: 60,
    ENV_RRF_WEIGHTS_ENABLED: False,
    ENV_RRF_VECTOR_WEIGHT: 0.6,
    ENV_RRF_BM25_WEIGHT: 0.4,
    ENV_BM25_K1_PARAMETER: 1.2,
    ENV_BM25_B_PARAMETER: 0.75,
    ENV_BM25_N_RESULTS: 20,
    ENV_BM25_MIN_TOKEN_LENGTH: 2,
    ENV_BM25_REMOVE_STOPWORDS: True,
    ENV_BM25_LANGUAGE: 'english',
    ENV_BM25_USE_STEMMING: False,
    ENV_BM25_USE_CHUNKING: False
}


def get_embedding_provider() -> str:
    """Get embedding provider type ('voyage', 'ollama', or 'openai')."""
    return get_env(ENV_EMBEDDING_PROVIDER, DEFAULTS[ENV_EMBEDDING_PROVIDER])


def get_voyage_embedding_model() -> str:
    """Get VoyageAI model for embeddings."""
    return get_env(ENV_VOYAGE_EMBEDDING_MODEL, DEFAULTS[ENV_VOYAGE_EMBEDDING_MODEL])


def get_voyage_max_chunk_tokens() -> Optional[int]:
    """Get max chunk size in tokens for Voyage."""
    val = get_env(ENV_VOYAGE_MAX_CHUNK_TOKENS)
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def get_voyage_chunk_overlap_tokens() -> Optional[int]:
    """Get chunk overlap in tokens for Voyage."""
    val = get_env(ENV_VOYAGE_CHUNK_OVERLAP_TOKENS)
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def get_voyage_min_chunk_tokens() -> Optional[int]:
    """Get min chunk size in tokens for Voyage."""
    val = get_env(ENV_VOYAGE_MIN_CHUNK_TOKENS)
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def get_voyage_enable_chunking() -> bool:
    """Get whether chunking is enabled for Voyage."""
    val = get_env(ENV_VOYAGE_ENABLE_CHUNKING, 'true')
    return val.lower() in ('true', '1', 'yes', 'on')


def get_voyage_output_dimension() -> Optional[int]:
    """Get Voyage output dimension (optional)."""
    val = get_env(ENV_VOYAGE_OUTPUT_DIMENSION)
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def get_ollama_base_url() -> str:
    """Get Ollama server base URL."""
    return get_env(ENV_OLLAMA_BASE_URL, DEFAULTS[ENV_OLLAMA_BASE_URL])


def get_ollama_embedding_model() -> str:
    """Get Ollama model for embeddings."""
    return get_env(ENV_OLLAMA_EMBEDDING_MODEL, DEFAULTS[ENV_OLLAMA_EMBEDDING_MODEL])


def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return get_env(ENV_OPENAI_API_KEY, DEFAULTS[ENV_OPENAI_API_KEY])


def get_openai_embedding_model() -> str:
    """Get OpenAI model for embeddings."""
    return get_env(ENV_OPENAI_EMBEDDING_MODEL, DEFAULTS[ENV_OPENAI_EMBEDDING_MODEL])


def get_openai_batch_size() -> int:
    """Get OpenAI batch size for embedding requests."""
    return get_int_env(ENV_OPENAI_BATCH_SIZE, DEFAULTS[ENV_OPENAI_BATCH_SIZE])


def get_semantic_search_n_results() -> int:
    """Get number of initial semantic search results."""
    return get_int_env(ENV_SEMANTIC_SEARCH_N_RESULTS, DEFAULTS[ENV_SEMANTIC_SEARCH_N_RESULTS])


def get_reranker_enabled() -> bool:
    """Get whether reranker is enabled."""
    return get_bool_env(ENV_RERANKER_ENABLED, DEFAULTS[ENV_RERANKER_ENABLED])


def get_reranker_threshold() -> float:
    """Get reranker threshold."""
    return get_float_env(ENV_RERANKER_THRESHOLD, DEFAULTS[ENV_RERANKER_THRESHOLD])


def get_reranker_model() -> str:
    """Get reranker model name."""
    return get_env(ENV_RERANKER_MODEL, DEFAULTS[ENV_RERANKER_MODEL])


def get_reranker_instructions() -> str:
    """Get reranker instructions for instruction-following capability."""
    return get_env(ENV_RERANKER_INSTRUCTIONS, DEFAULTS[ENV_RERANKER_INSTRUCTIONS])


def get_reranker_use_chunks() -> bool:
    """Get whether reranker should use chunk text instead of full file content."""
    return get_bool_env(ENV_RERANKER_USE_CHUNKS, DEFAULTS[ENV_RERANKER_USE_CHUNKS])


def get_ai_filter_enabled() -> bool:
    """Get whether AI filter is enabled."""
    # Also check legacy USE_AI_FILTER for backward compatibility
    if os.getenv('USE_AI_FILTER') is not None:
        return get_bool_env('USE_AI_FILTER', False)
    return get_bool_env(ENV_AI_FILTER_ENABLED, DEFAULTS[ENV_AI_FILTER_ENABLED])


def get_ai_filter_model() -> str:
    """Get AI filter model name."""
    return get_env(ENV_AI_FILTER_MODEL, DEFAULTS[ENV_AI_FILTER_MODEL])


def get_ai_filter_timeout_seconds() -> int:
    """Get AI filter timeout in seconds."""
    return get_int_env(ENV_AI_FILTER_TIMEOUT_SECONDS, DEFAULTS[ENV_AI_FILTER_TIMEOUT_SECONDS])


def get_max_results() -> int:
    """Get maximum number of results to return."""
    return get_int_env(ENV_MAX_RESULTS, DEFAULTS[ENV_MAX_RESULTS])


def get_logging_verbose() -> bool:
    """Get whether verbose logging is enabled."""
    return get_bool_env(ENV_LOGGING_VERBOSE, DEFAULTS[ENV_LOGGING_VERBOSE])


def get_logging_file_enabled() -> bool:
    """Get whether file logging is enabled."""
    return get_bool_env(ENV_LOGGING_FILE_ENABLED, DEFAULTS[ENV_LOGGING_FILE_ENABLED])


def get_logging_file_path() -> str:
    """Get logging file path."""
    return get_env(ENV_LOGGING_FILE_PATH, DEFAULTS[ENV_LOGGING_FILE_PATH])


def get_preview_lines_vectorization() -> int:
    """Get number of lines for vectorization."""
    return get_int_env(ENV_PREVIEW_LINES_VECTORIZATION, DEFAULTS[ENV_PREVIEW_LINES_VECTORIZATION])


def get_preview_lines_storage() -> int:
    """Get number of lines for storage (-1 for unlimited)."""
    return get_int_env(ENV_PREVIEW_LINES_STORAGE, DEFAULTS[ENV_PREVIEW_LINES_STORAGE])


def get_preview_lines_reranker() -> int:
    """Get number of lines for reranker."""
    return get_int_env(ENV_PREVIEW_LINES_RERANKER, DEFAULTS[ENV_PREVIEW_LINES_RERANKER])


def get_preview_lines_ai_filter() -> int:
    """Get number of lines for AI filter."""
    return get_int_env(ENV_PREVIEW_LINES_AI_FILTER, DEFAULTS[ENV_PREVIEW_LINES_AI_FILTER])


def get_preview_chars_output() -> int:
    """Get how many characters of file content to include per result.

    Values:
      0  -> paths and scores only (agentic flow: agent reads files via Read/Grep/Glob)
      N  -> include the first N characters of each file as a snippet
      -1 -> include the entire file content
    """
    return get_int_env(ENV_PREVIEW_CHARS_OUTPUT, DEFAULTS[ENV_PREVIEW_CHARS_OUTPUT])


def get_db_name() -> str:
    """Get database name for ChromaDB collection."""
    return get_env(ENV_DB_NAME, DEFAULTS[ENV_DB_NAME])


# ========== HYBRID SEARCH CONFIGURATION ==========

def get_hybrid_search_enabled() -> bool:
    """Get whether hybrid search (vector + BM25 + RRF) is enabled."""
    return get_bool_env(ENV_HYBRID_SEARCH_ENABLED, DEFAULTS[ENV_HYBRID_SEARCH_ENABLED])


def get_bm25_only_mode() -> bool:
    """Get whether to use only BM25 search without vector search."""
    return get_bool_env(ENV_BM25_ONLY_MODE, DEFAULTS[ENV_BM25_ONLY_MODE])


# ========== RRF CONFIGURATION ==========

def get_rrf_k_parameter() -> int:
    """Get RRF k parameter for reciprocal rank fusion."""
    return get_int_env(ENV_RRF_K_PARAMETER, DEFAULTS[ENV_RRF_K_PARAMETER])


def get_rrf_weights_enabled() -> bool:
    """Get whether weighted RRF is enabled instead of standard RRF."""
    return get_bool_env(ENV_RRF_WEIGHTS_ENABLED, DEFAULTS[ENV_RRF_WEIGHTS_ENABLED])


def get_rrf_vector_weight() -> float:
    """Get weight for vector search results in weighted RRF."""
    return get_float_env(ENV_RRF_VECTOR_WEIGHT, DEFAULTS[ENV_RRF_VECTOR_WEIGHT])


def get_rrf_bm25_weight() -> float:
    """Get weight for BM25 search results in weighted RRF."""
    return get_float_env(ENV_RRF_BM25_WEIGHT, DEFAULTS[ENV_RRF_BM25_WEIGHT])


# ========== BM25 CONFIGURATION ==========

def get_bm25_k1_parameter() -> float:
    """Get BM25 k1 parameter controlling term frequency saturation."""
    return get_float_env(ENV_BM25_K1_PARAMETER, DEFAULTS[ENV_BM25_K1_PARAMETER])


def get_bm25_b_parameter() -> float:
    """Get BM25 b parameter controlling document length normalization."""
    return get_float_env(ENV_BM25_B_PARAMETER, DEFAULTS[ENV_BM25_B_PARAMETER])


def get_bm25_n_results() -> int:
    """Get number of top results from BM25 search to pass to RRF."""
    return get_int_env(ENV_BM25_N_RESULTS, DEFAULTS[ENV_BM25_N_RESULTS])


def get_bm25_min_token_length() -> int:
    """Get minimum token length for BM25 indexing."""
    return get_int_env(ENV_BM25_MIN_TOKEN_LENGTH, DEFAULTS[ENV_BM25_MIN_TOKEN_LENGTH])


def get_bm25_remove_stopwords() -> bool:
    """Get whether to remove stopwords during BM25 indexing and search."""
    return get_bool_env(ENV_BM25_REMOVE_STOPWORDS, DEFAULTS[ENV_BM25_REMOVE_STOPWORDS])


def get_bm25_language() -> str:
    """Get language for BM25 stemming and stopwords."""
    return get_env(ENV_BM25_LANGUAGE, DEFAULTS[ENV_BM25_LANGUAGE])


def get_bm25_use_stemming() -> bool:
    """Get whether to use stemming in BM25 text processing."""
    return get_bool_env(ENV_BM25_USE_STEMMING, DEFAULTS[ENV_BM25_USE_STEMMING])


def get_bm25_use_chunking() -> bool:
    """Get whether to index chunks separately for BM25 (instead of full files)."""
    return get_bool_env(ENV_BM25_USE_CHUNKING, DEFAULTS[ENV_BM25_USE_CHUNKING])


# ========== VALIDATION FUNCTIONS ==========

def validate_hybrid_config() -> List[str]:
    """Validate hybrid search configuration and return any errors."""
    errors = []
    
    if not get_hybrid_search_enabled():
        return errors  # No validation needed if hybrid search is disabled
    
    # RRF validation
    k_param = get_rrf_k_parameter()
    if k_param < 1 or k_param > 1000:
        errors.append(f"RRF_K_PARAMETER must be between 1-1000, got {k_param}")
    
    # Weights validation
    if get_rrf_weights_enabled():
        v_weight = get_rrf_vector_weight()  
        b_weight = get_rrf_bm25_weight()
        if v_weight < 0 or v_weight > 1:
            errors.append(f"RRF_VECTOR_WEIGHT must be between 0-1, got {v_weight}")
        if b_weight < 0 or b_weight > 1:
            errors.append(f"RRF_BM25_WEIGHT must be between 0-1, got {b_weight}")
        if abs(v_weight + b_weight - 1.0) > 0.01:
            errors.append(f"RRF weights should sum to ~1.0, got {v_weight + b_weight}")
    
    # BM25 validation  
    k1 = get_bm25_k1_parameter()
    if k1 < 0.1 or k1 > 3.0:
        errors.append(f"BM25_K1_PARAMETER should be between 0.1-3.0, got {k1}")
        
    b = get_bm25_b_parameter()
    if b < 0.0 or b > 1.0:
        errors.append(f"BM25_B_PARAMETER must be between 0.0-1.0, got {b}")
    
    n_results = get_bm25_n_results()
    if n_results < 5 or n_results > 1000:
        errors.append(f"BM25_N_RESULTS should be between 5-1000, got {n_results}")
    
    return errors