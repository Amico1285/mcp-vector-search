"""Ollama local embedding provider."""
import os
import logging
from typing import List, Dict, Optional, Tuple
import requests

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .utils import count_tokens, split_by_tokens

logger = logging.getLogger(__name__)


class OllamaProvider(EmbeddingProvider):
    """Ollama local embedding provider.
    
    Supports local embedding models running via Ollama server.
    Handles large documents by truncating to model limits.
    """
    
    # Token limits for different models
    TOKEN_LIMITS = {
        "snowflake-arctic-embed2": 8192,
        "snowflake-arctic-embed": 8192,
        "nomic-embed-text": 8192,
        "mxbai-embed-large": 512,
        "all-minilm": 256,
        # Add more models as needed
    }
    
    def __init__(
        self,
        model: str = "snowflake-arctic-embed2",
        base_url: str = "http://localhost:11434",
        enable_chunking: bool = True,
        max_chunk_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None
    ):
        """Initialize Ollama provider.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            enable_chunking: Whether to enable automatic chunking for large files
            max_chunk_tokens: Maximum tokens per chunk (default: 95% of model limit)
            chunk_overlap_tokens: Number of overlapping tokens between chunks
        """
        self.model = model
        self.base_url = base_url
        self.dimension = None  # Will be set after first embedding
        self.batch_size = 128  # Batch size for processing
        self.enable_chunking = enable_chunking
        
        # Get token limit for this model
        self.token_limit = self.TOKEN_LIMITS.get(model, 8192)
        
        # Configure chunking parameters
        if max_chunk_tokens is None:
            # Default to 95% of token limit for safety
            max_chunk_tokens = int(self.token_limit * 0.95)
        
        if chunk_overlap_tokens is None:
            chunk_overlap_tokens = 200
        
        # Validate chunk size
        if max_chunk_tokens > self.token_limit:
            logger.warning(
                f"[Ollama] max_chunk_tokens ({max_chunk_tokens}) exceeds model limit "
                f"({self.token_limit}), using {int(self.token_limit * 0.95)}"
            )
            max_chunk_tokens = int(self.token_limit * 0.95)
        
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        
        # Check if ollama library is available
        try:
            import ollama
            self.ollama = ollama
            self.use_library = True
        except ImportError:
            logger.warning("Ollama library not installed, falling back to REST API")
            self.use_library = False
        
        # Check server availability
        self._check_server()
        
        logger.info(f"[Ollama] Initialized provider")
        logger.info(f"[Ollama] Model: {model}")
        logger.info(f"[Ollama] Base URL: {base_url}")
        logger.info(f"[Ollama] Token limit: {self.token_limit}")
        logger.info(f"[Ollama] Chunking: {'enabled' if enable_chunking else 'disabled'}")
        if enable_chunking:
            logger.info(f"[Ollama] Max chunk tokens: {self.max_chunk_tokens}")
            logger.info(f"[Ollama] Chunk overlap: {self.chunk_overlap_tokens}")
        logger.info(f"[Ollama] Using library: {self.use_library}")
    
    def _check_server(self):
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama server not responding at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to Ollama server at {self.base_url}: {e}")
    
    def get_token_limit(self) -> int:
        """Get the maximum token limit for this provider."""
        return self.token_limit
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_query("test")
            self.dimension = len(test_embedding)
        return self.dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        result = self.embed_documents_with_metadata(texts)
        return result.embeddings
    
    def embed_documents_with_metadata(self, texts: List[str]) -> ChunkedEmbeddingResult:
        """Generate embeddings with metadata.
        
        This method handles large documents by chunking them if needed.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            ChunkedEmbeddingResult with embeddings and metadata
        """
        if not texts:
            return ChunkedEmbeddingResult(embeddings=[], metadata=[], chunks=[])
        
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        
        for doc_idx, text in enumerate(texts):
            tokens = count_tokens(text)
            logger.info(f"[Ollama] Processing document {doc_idx}: {tokens:,} tokens")
            
            if not self.enable_chunking or tokens <= self.max_chunk_tokens:
                # Small document or chunking disabled - embed whole
                embeddings, metadata, chunks = self._process_single_document(text, doc_idx)
            else:
                # Large document - chunk it
                embeddings, metadata, chunks = self._process_chunked_document(text, doc_idx)
            
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
            all_chunks.extend(chunks)
        
        # Set dimension from first embedding if not set
        if all_embeddings and self.dimension is None:
            self.dimension = len(all_embeddings[0])
            logger.info(f"[Ollama] Detected embedding dimension: {self.dimension}")
        
        logger.info(f"[Ollama] Generated {len(all_embeddings)} embeddings total")
        
        return ChunkedEmbeddingResult(
            embeddings=all_embeddings,
            metadata=all_metadata,
            chunks=all_chunks
        )
    
    def _process_single_document(self, text: str, doc_idx: int) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """Process a document without chunking.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        tokens = count_tokens(text)
        
        # Check if text exceeds token limit
        if tokens > self.token_limit:
            logger.warning(
                f"[Ollama] Document {doc_idx} has {tokens} tokens, exceeds limit {self.token_limit}. "
                f"Text will be truncated."
            )
        
        try:
            if self.use_library:
                response = self.ollama.embed(
                    model=self.model,
                    input=[text],
                    truncate=True  # Allow truncation for oversized documents
                )
                embedding = response['embeddings'][0]
            else:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": [text],
                        "truncate": True
                    },
                    timeout=60
                )
                response.raise_for_status()
                embedding = response.json()['embeddings'][0]
            
            return (
                [embedding],
                [{
                    'type': 'full_file',
                    'doc_index': doc_idx,
                    'tokens': tokens,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'truncated': tokens > self.token_limit
                }],
                [text]
            )
        except Exception as e:
            logger.error(f"[Ollama] Failed to embed document {doc_idx}: {e}")
            raise
    
    def _process_chunked_document(self, text: str, doc_idx: int) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """Process a large document by chunking it.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        total_tokens = count_tokens(text)
        logger.info(f"[Ollama] Chunking document {doc_idx} ({total_tokens:,} tokens)")
        
        # Split into chunks
        chunks = split_by_tokens(
            text,
            max_tokens=self.max_chunk_tokens,
            overlap_tokens=self.chunk_overlap_tokens
        )
        
        # Filter empty chunks
        chunks = [c for c in chunks if c and c.strip()]
        
        if not chunks:
            logger.warning(f"[Ollama] No valid chunks after splitting document {doc_idx}")
            return [], [], []
        
        logger.info(f"[Ollama] Document {doc_idx} split into {len(chunks)} chunks")
        
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        
        # Process chunks in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            try:
                if self.use_library:
                    response = self.ollama.embed(
                        model=self.model,
                        input=batch_chunks,
                        truncate=False  # Chunks should already fit
                    )
                    embeddings = response['embeddings']
                else:
                    response = requests.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model,
                            "input": batch_chunks,
                            "truncate": False
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    embeddings = response.json()['embeddings']
                
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_idx = batch_start + j
                    all_embeddings.append(embedding)
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'type': 'chunk',
                        'doc_index': doc_idx,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'tokens': count_tokens(chunk),
                        'truncated': False
                    })
                
                if len(batch_chunks) > 1:
                    logger.info(
                        f"[Ollama] Processed batch of {len(batch_chunks)} chunks "
                        f"({batch_start + 1}-{batch_end}/{len(chunks)})"
                    )
                    
            except Exception as e:
                logger.error(f"[Ollama] Failed to embed chunk batch starting at {batch_start}: {e}")
                raise
        
        return all_embeddings, all_metadata, all_chunks
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        # Use embed_documents for consistency
        result = self.embed_documents_with_metadata([text])
        return result.embeddings[0] if result.embeddings else []
    
    def validate_api_key(self) -> bool:
        """Validate that the server is accessible.
        
        For Ollama, this just checks server availability.
        
        Returns:
            True if server is accessible
        """
        try:
            self._check_server()
            return True
        except Exception as e:
            logger.error(f"[Ollama] Server validation failed: {e}")
            return False