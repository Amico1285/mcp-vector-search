"""Base class for all embedding providers."""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        pass
    
    @abstractmethod
    def get_token_limit(self) -> int:
        """Get the maximum token limit for this provider.
        
        Returns:
            Maximum number of tokens that can be processed
        """
        pass
    
    def get_safe_token_limit(self) -> int:
        """Get a safe token limit with some buffer.
        
        Returns:
            Safe token limit (usually 95% of actual limit)
        """
        return int(self.get_token_limit() * 0.95)
    
    def should_chunk(self, text: str, token_count: Optional[int] = None) -> bool:
        """Check if text should be chunked based on token limit.
        
        Args:
            text: Text to check
            token_count: Pre-calculated token count (optional)
            
        Returns:
            True if text exceeds safe token limit
        """
        if token_count is None:
            # Import here to avoid circular dependency
            from .utils import count_tokens
            token_count = count_tokens(text)
        
        return token_count > self.get_safe_token_limit()


class ChunkedEmbeddingResult:
    """Result of embedding with chunking information."""
    
    def __init__(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict],
        chunks: Optional[List[str]] = None
    ):
        """Initialize chunked embedding result.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata for each embedding
            chunks: Optional list of chunk texts
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.chunks = chunks
    
    def __repr__(self) -> str:
        return (
            f"ChunkedEmbeddingResult("
            f"embeddings={len(self.embeddings)}, "
            f"metadata={len(self.metadata)})"
        )