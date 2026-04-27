"""VoyageAI embedding provider for standard (non-contextual) models."""
import os
import logging
from typing import List, Dict, Optional
import voyageai

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .utils import count_tokens, split_by_tokens

logger = logging.getLogger(__name__)


class VoyageProvider(EmbeddingProvider):
    """Provider for standard Voyage embedding models.
    
    Supports all Voyage models except voyage-context-3.
    Handles large documents by chunking them into manageable pieces.
    """
    
    # Token limits for different models
    TOKEN_LIMITS = {
        "voyage-3-large": 32000,
        "voyage-3.5": 32000,
        "voyage-3.5-lite": 32000,
        "voyage-code-3": 32000,
        "voyage-finance-2": 32000,
        "voyage-law-2": 16000,
        "voyage-code-2": 16000,
    }
    
    # Batch limits (tokens) for different models
    BATCH_LIMITS = {
        "voyage-3.5-lite": 1000000,  # 1M tokens
        "voyage-3.5": 320000,         # 320K tokens
        "voyage-3-large": 120000,     # 120K tokens
        "voyage-code-3": 120000,      # 120K tokens
        "voyage-finance-2": 120000,   # 120K tokens
        "voyage-law-2": 120000,       # 120K tokens
        "voyage-code-2": 120000,      # 120K tokens
    }
    
    # Default dimensions for models
    MODEL_DIMENSIONS = {
        "voyage-3-large": 1024,
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-code-2": 1536,
    }
    
    def __init__(
        self,
        model: str = "voyage-3-large",
        enable_chunking: bool = True,
        max_chunk_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
        output_dimension: Optional[int] = None
    ):
        """Initialize VoyageAI provider.
        
        Args:
            model: VoyageAI model name
            enable_chunking: Whether to enable automatic chunking for large files
            max_chunk_tokens: Maximum tokens per chunk (default: 95% of model limit)
            chunk_overlap_tokens: Number of overlapping tokens between chunks
            min_chunk_tokens: Minimum tokens per chunk
            output_dimension: Optional output dimension (256, 512, 1024, 2048)
        """
        self.model = model
        self.enable_chunking = enable_chunking
        
        # Get limits for this model
        self.token_limit = self.TOKEN_LIMITS.get(model, 32000)
        self.batch_limit = self.BATCH_LIMITS.get(model, 120000)
        
        # Configure chunking
        if max_chunk_tokens is None:
            # Default to 95% of token limit for safety
            max_chunk_tokens = int(self.token_limit * 0.95)
        
        if chunk_overlap_tokens is None:
            chunk_overlap_tokens = 200
        
        if min_chunk_tokens is None:
            min_chunk_tokens = 500
        
        # Validate chunk size
        if max_chunk_tokens > self.token_limit:
            logger.warning(
                f"[Voyage] max_chunk_tokens ({max_chunk_tokens}) exceeds model limit "
                f"({self.token_limit}), using {int(self.token_limit * 0.95)}"
            )
            max_chunk_tokens = int(self.token_limit * 0.95)
        
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.output_dimension = output_dimension
        
        # Set dimension
        if output_dimension:
            self.dimension = output_dimension
        else:
            self.dimension = self.MODEL_DIMENSIONS.get(model, 1024)
        
        # Initialize Voyage client
        self.client = voyageai.Client()
        
        logger.info(
            f"[Voyage] Initialized {model} provider: "
            f"token_limit={self.token_limit}, batch_limit={self.batch_limit}, "
            f"max_chunk={self.max_chunk_tokens}, dimension={self.dimension}"
        )
    
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
        """Generate embeddings with chunking metadata.
        
        This method handles large documents by chunking them if needed.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            ChunkedEmbeddingResult with embeddings and metadata
        """
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        
        for doc_idx, text in enumerate(texts):
            tokens = count_tokens(text)
            logger.info(f"[Voyage] Processing document {doc_idx}: {tokens:,} tokens")
            
            if not self.enable_chunking or tokens <= self.max_chunk_tokens:
                # Small document or chunking disabled - embed whole
                embeddings, metadata, chunks = self._process_single_document(text, doc_idx)
            else:
                # Large document - chunk it
                embeddings, metadata, chunks = self._process_chunked_document(text, doc_idx)
            
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
            all_chunks.extend(chunks)
        
        return ChunkedEmbeddingResult(
            embeddings=all_embeddings,
            metadata=all_metadata,
            chunks=all_chunks
        )
    
    def _process_single_document(self, text: str, doc_idx: int) -> tuple:
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
                f"[Voyage] Document {doc_idx} has {tokens} tokens, exceeds limit {self.token_limit}. "
                f"Text will be truncated."
            )
        
        try:
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="document",
                truncation=True,  # Allow truncation for oversized documents
                output_dimension=self.output_dimension
            )
            
            embedding = result.embeddings[0]
            
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
            logger.error(f"[Voyage] Failed to embed document {doc_idx}: {e}")
            raise
    
    def _process_chunked_document(self, text: str, doc_idx: int) -> tuple:
        """Process a large document by chunking it.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        total_tokens = count_tokens(text)
        logger.info(f"[Voyage] Chunking document {doc_idx} ({total_tokens:,} tokens)")
        
        # Split into chunks
        chunks = split_by_tokens(
            text,
            max_tokens=self.max_chunk_tokens,
            overlap_tokens=self.chunk_overlap_tokens
        )
        
        # Filter empty chunks
        chunks = [c for c in chunks if c and c.strip()]
        
        if not chunks:
            logger.warning(f"[Voyage] No valid chunks after splitting document {doc_idx}")
            return [], [], []
        
        logger.info(f"[Voyage] Document {doc_idx} split into {len(chunks)} chunks")
        
        # Process chunks in batches if needed
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        
        # Calculate how many chunks can fit in a batch
        avg_chunk_tokens = total_tokens // len(chunks)
        chunks_per_batch = max(1, self.batch_limit // avg_chunk_tokens)
        
        for i in range(0, len(chunks), chunks_per_batch):
            batch_chunks = chunks[i:i + chunks_per_batch]
            
            try:
                result = self.client.embed(
                    texts=batch_chunks,
                    model=self.model,
                    input_type="document",
                    truncation=False,  # Chunks should already fit
                    output_dimension=self.output_dimension
                )
                
                embeddings = result.embeddings
                
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_idx = i + j
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
                        f"[Voyage] Processed batch of {len(batch_chunks)} chunks "
                        f"({i+1}-{min(i+len(batch_chunks), len(chunks))}/{len(chunks)})"
                    )
                    
            except Exception as e:
                logger.error(f"[Voyage] Failed to embed chunk batch starting at {i}: {e}")
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
            result = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="query",
                output_dimension=self.output_dimension
            )
            
            return result.embeddings[0]
            
        except Exception as e:
            logger.error(f"[Voyage] Query embedding failed: {e}")
            raise
    
    def validate_api_key(self) -> bool:
        """Validate that the API key works.
        
        Returns:
            True if API key is valid
        """
        try:
            self.embed_query("test")
            return True
        except Exception as e:
            logger.error(f"[Voyage] API key validation failed: {e}")
            return False