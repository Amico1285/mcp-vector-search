"""Embedding providers for code search MCP server."""
import logging
from typing import Optional

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .voyage import VoyageProvider
from .voyage_context import VoyageContextProvider
from .ollama import OllamaProvider
from .openai_provider import OpenAIProvider
from .utils import count_tokens, split_by_tokens, split_code_by_structure

logger = logging.getLogger(__name__)


def create_embedding_provider(provider_type: str = None) -> EmbeddingProvider:
    """Factory function to create embedding provider based on configuration.
    
    Args:
        provider_type: Type of provider ('voyage', 'ollama', or 'openai'). 
                      If None, reads from environment.
    
    Returns:
        EmbeddingProvider instance
    """
    import os
    
    try:
        # Import from parent module
        from ..env_config import (
            get_embedding_provider,
            get_voyage_embedding_model,
            get_voyage_output_dimension,
            get_openai_embedding_model,
            get_ollama_embedding_model,
            get_ollama_base_url
        )
    except ImportError:
        # Fallback if env_config not available
        logger.warning("env_config not available, using defaults")
        get_embedding_provider = lambda: os.getenv("EMBEDDING_PROVIDER", "voyage")
        get_voyage_embedding_model = lambda: os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3-large")
        get_voyage_output_dimension = lambda: int(os.getenv("VOYAGE_OUTPUT_DIMENSION")) if os.getenv("VOYAGE_OUTPUT_DIMENSION") else None
        get_openai_embedding_model = lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        get_ollama_embedding_model = lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "snowflake-arctic-embed2")
        get_ollama_base_url = lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if provider_type is None:
        provider_type = get_embedding_provider()
    
    logger.info(f"[EMBEDDING_PROVIDER] Creating provider of type: {provider_type}")
    
    # Get universal chunking configuration (applies to all providers)
    enable_chunking = os.getenv("ENABLE_CHUNKING", os.getenv("VOYAGE_ENABLE_CHUNKING", "true")).lower() == "true"
    max_chunk_tokens = os.getenv("MAX_CHUNK_TOKENS", os.getenv("VOYAGE_MAX_CHUNK_TOKENS"))
    min_chunk_tokens = os.getenv("MIN_CHUNK_TOKENS", os.getenv("VOYAGE_MIN_CHUNK_TOKENS"))
    chunk_overlap_tokens = os.getenv("CHUNK_OVERLAP_TOKENS", os.getenv("VOYAGE_CHUNK_OVERLAP_TOKENS"))
    
    # Convert to integers if present
    max_chunk_tokens = int(max_chunk_tokens) if max_chunk_tokens else None
    min_chunk_tokens = int(min_chunk_tokens) if min_chunk_tokens else None  
    chunk_overlap_tokens = int(chunk_overlap_tokens) if chunk_overlap_tokens else None
    
    logger.info(f"[EMBEDDING_PROVIDER] Universal chunking config:")
    logger.info(f"  Enable chunking: {enable_chunking}")
    logger.info(f"  Max chunk tokens: {max_chunk_tokens}")
    logger.info(f"  Min chunk tokens: {min_chunk_tokens}")
    logger.info(f"  Chunk overlap: {chunk_overlap_tokens}")
    
    if provider_type == 'voyage':
        model = get_voyage_embedding_model()
        logger.info(f"[EMBEDDING_PROVIDER] Voyage model: {model}")
        
        # Check if it's voyage-context-3
        if model == "voyage-context-3":
            # Use specialized context provider
            logger.info("[EMBEDDING_PROVIDER] Using VoyageContextProvider for voyage-context-3")
            # Context model has specific defaults if not provided
            if max_chunk_tokens is None:
                max_chunk_tokens = 64  # Default for context model
            if min_chunk_tokens is None:
                min_chunk_tokens = 1  # Default for context model
            return VoyageContextProvider(
                max_chunk_tokens=max_chunk_tokens,
                min_chunk_tokens=min_chunk_tokens
            )
        else:
            # Use standard voyage provider
            output_dimension = get_voyage_output_dimension()
            logger.info(f"[EMBEDDING_PROVIDER] Using VoyageProvider for {model}, output_dimension: {output_dimension}")
            return VoyageProvider(
                model=model,
                enable_chunking=enable_chunking,
                max_chunk_tokens=max_chunk_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                output_dimension=output_dimension
            )
    
    elif provider_type == 'openai':
        # Use new OpenAI provider
        model = get_openai_embedding_model()
        
        logger.info(f"[EMBEDDING_PROVIDER] Using OpenAIProvider for {model}")
        return OpenAIProvider(
            model=model,
            enable_chunking=enable_chunking,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens
        )
    
    elif provider_type == 'ollama':
        # Use new Ollama provider
        model = get_ollama_embedding_model()
        base_url = get_ollama_base_url()
        
        logger.info(f"[EMBEDDING_PROVIDER] Using OllamaProvider for {model}")
        return OllamaProvider(
            model=model, 
            base_url=base_url,
            enable_chunking=enable_chunking,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens
        )
    
    else:
        # Unknown provider type
        raise ValueError(f"Unknown embedding provider type: {provider_type}")


# Export all classes
__all__ = [
    'EmbeddingProvider',
    'ChunkedEmbeddingResult',
    'VoyageProvider',
    'VoyageContextProvider',
    'OllamaProvider',
    'OpenAIProvider',
    'create_embedding_provider',
    'count_tokens',
    'split_by_tokens',
    'split_code_by_structure'
]