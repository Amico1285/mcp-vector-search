"""OpenAI embedding provider using text-embedding-3 models."""
import os
import logging
from typing import List, Dict, Optional, Tuple

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .utils import count_tokens, split_by_tokens

logger = logging.getLogger(__name__)


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3 models.
    
    Supports the latest OpenAI embedding models with optional dimensionality reduction.
    Handles large documents by truncating to model limits.
    """
    
    # Token limits for different models
    TOKEN_LIMITS = {
        "text-embedding-3-large": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-ada-002": 8191,
    }
    
    # Default dimensions for each model
    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        enable_chunking: bool = True,
        max_chunk_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None
    ):
        """Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name (text-embedding-3-large or text-embedding-3-small)
            enable_chunking: Whether to enable automatic chunking for large files
            max_chunk_tokens: Maximum tokens per chunk (default: model limit)
            chunk_overlap_tokens: Number of overlapping tokens between chunks
        """
        self.model = model
        self.enable_chunking = enable_chunking
        
        # Get batch size from environment or use default
        import os
        self.batch_size = int(os.getenv('OPENAI_BATCH_SIZE', '2048'))
        
        # Get token limit for this model
        self.token_limit = self.TOKEN_LIMITS.get(model, 8191)
        
        # Configure chunking parameters
        if max_chunk_tokens is None:
            # Default to full token limit
            max_chunk_tokens = self.token_limit
        
        if chunk_overlap_tokens is None:
            chunk_overlap_tokens = 0
        
        # Validate chunk size
        if max_chunk_tokens > self.token_limit:
            logger.warning(
                f"[OpenAI] max_chunk_tokens ({max_chunk_tokens}) exceeds model limit "
                f"({self.token_limit}), using {self.token_limit}"
            )
            max_chunk_tokens = self.token_limit
        
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise RuntimeError("OpenAI library not installed. Run: pip install openai")
        
        # Use native dimensions for the model
        self.dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        
        logger.info(f"[OpenAI] Initialized provider")
        logger.info(f"[OpenAI] Model: {model}")
        logger.info(f"[OpenAI] Native dimension: {self.dimension}")
        logger.info(f"[OpenAI] Token limit: {self.token_limit}")
        logger.info(f"[OpenAI] Batch size: {self.batch_size}")
        logger.info(f"[OpenAI] Chunking: {'enabled' if enable_chunking else 'disabled'}")
        if enable_chunking:
            logger.info(f"[OpenAI] Max chunk tokens: {self.max_chunk_tokens}")
            logger.info(f"[OpenAI] Chunk overlap: {self.chunk_overlap_tokens}")
    
    def get_token_limit(self) -> int:
        """Get the maximum token limit for this provider."""
        return self.token_limit
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
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
            logger.info(f"[OpenAI] Processing document {doc_idx}: {tokens:,} tokens")
            
            if not self.enable_chunking or tokens <= self.max_chunk_tokens:
                # Small document or chunking disabled - embed whole
                embeddings, metadata, chunks = self._process_single_document(text, doc_idx)
            else:
                # Large document - chunk it
                embeddings, metadata, chunks = self._process_chunked_document(text, doc_idx)
            
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"[OpenAI] Generated {len(all_embeddings)} embeddings total")
        
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
        
        # Truncate text if it exceeds token limit and chunking is disabled
        if not self.enable_chunking and tokens > self.token_limit:
            logger.warning(
                f"[OpenAI] Document {doc_idx} has {tokens} tokens, exceeds limit {self.token_limit}. "
                f"Truncating to {self.token_limit} tokens."
            )
            # Truncate text to token limit
            from .utils import split_by_tokens
            truncated = split_by_tokens(text, self.token_limit, overlap_tokens=0)[0]
            text = truncated
            tokens = self.token_limit
        
        try:
            # Prepare request parameters
            kwargs = {
                "input": [text],
                "model": self.model
            }
            
            
            # Make API call
            response = self.client.embeddings.create(**kwargs)
            embedding = response.data[0].embedding
            
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
            logger.error(f"[OpenAI] Failed to embed document {doc_idx}: {e}")
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
        logger.info(f"[OpenAI] Chunking document {doc_idx} ({total_tokens:,} tokens)")
        
        # Split into chunks
        chunks = split_by_tokens(
            text,
            max_tokens=self.max_chunk_tokens,
            overlap_tokens=self.chunk_overlap_tokens
        )
        
        # Filter empty chunks
        chunks = [c for c in chunks if c and c.strip()]
        
        if not chunks:
            logger.warning(f"[OpenAI] No valid chunks after splitting document {doc_idx}")
            return [], [], []
        
        logger.info(f"[OpenAI] Document {doc_idx} split into {len(chunks)} chunks")
        
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        
        # OpenAI allows batch processing - use configured batch size
        batch_size = self.batch_size
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            try:
                # Prepare request parameters
                kwargs = {
                    "input": batch_chunks,
                    "model": self.model
                }
                
                # Make API call
                response = self.client.embeddings.create(**kwargs)
                embeddings = [item.embedding for item in response.data]
                
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
                        f"[OpenAI] Processed batch of {len(batch_chunks)} chunks "
                        f"({batch_start + 1}-{batch_end}/{len(chunks)})"
                    )
                    
            except Exception as e:
                logger.error(f"[OpenAI] Failed to embed chunk batch starting at {batch_start}: {e}")
                raise
        
        return all_embeddings, all_metadata, all_chunks
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Prepare request parameters
            kwargs = {
                "input": [text],
                "model": self.model
            }
            
            
            # Make API call
            response = self.client.embeddings.create(**kwargs)
            
            # Return the first (and only) embedding
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"[OpenAI] Error generating query embedding: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    def validate_api_key(self) -> bool:
        """Validate that the API key works.
        
        Returns:
            True if API key is valid
        """
        try:
            # Try to generate a test embedding
            self.embed_query("test")
            return True
        except Exception as e:
            logger.error(f"[OpenAI] API key validation failed: {e}")
            return False