"""VoyageAI voyage-context-3 embedding provider for contextual embeddings."""
import os
import logging
from typing import List, Dict, Optional, Tuple
import voyageai

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .utils import count_tokens, split_by_tokens

logger = logging.getLogger(__name__)


class VoyageContextProvider(EmbeddingProvider):
    """Provider for voyage-context-3 contextual embeddings.
    
    This provider handles the special contextualized embedding API
    that processes chunks in context with each other.
    """
    
    # API limits for voyage-context-3
    MAX_TOKENS_PER_REQUEST = 120000  # Total tokens across all inputs
    MAX_CHUNKS_PER_REQUEST = 5461    # Total chunks across all inputs (actual API limit)
    MAX_INPUTS_PER_REQUEST = 1000    # Number of documents
    TOKEN_LIMIT = 32000              # Max tokens per document part
    
    def __init__(
        self,
        max_chunk_tokens: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
        output_dimension: Optional[int] = None
    ):
        """Initialize voyage-context-3 provider.
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk (default: 2048)
            min_chunk_tokens: Minimum tokens per chunk (default: 64)
            output_dimension: Optional output dimension (256, 512, 1024, 2048)
        """
        self.model = "voyage-context-3"
        
        # Chunk configuration - use reasonable defaults for contextual embeddings
        if max_chunk_tokens is None:
            max_chunk_tokens = 64
        if min_chunk_tokens is None:
            min_chunk_tokens = 1
        
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.output_dimension = output_dimension
        self.dimension = output_dimension if output_dimension else 1024
        
        # Initialize Voyage client
        self.client = voyageai.Client()
        
        logger.info(f"[VoyageContext] Initialized with chunk_size={max_chunk_tokens}, dimension={self.dimension}")
    
    def get_token_limit(self) -> int:
        """Get the maximum token limit for this provider."""
        return self.TOKEN_LIMIT
    
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
        """Generate contextual embeddings with chunking metadata.
        
        This method handles large documents by:
        1. Splitting very large documents into manageable parts
        2. Chunking each part into small pieces
        3. Using contextualized_embed API to maintain context
        
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
            logger.info(f"[VoyageContext] Processing document {doc_idx}: {tokens:,} tokens")
            logger.info(f"[VoyageContext] max_chunk_tokens={self.max_chunk_tokens}, MAX_TOKENS_PER_REQUEST={self.MAX_TOKENS_PER_REQUEST}")
            
            # Process based on document size
            if tokens <= self.max_chunk_tokens:
                # Small document - single chunk
                logger.info(f"[VoyageContext] Calling _process_small_document (tokens {tokens} <= {self.max_chunk_tokens})")
                embeddings, metadata, chunks = self._process_small_document(text, doc_idx)
            elif tokens <= self.MAX_TOKENS_PER_REQUEST:
                # Medium document - fits in one API call
                logger.info(f"[VoyageContext] Calling _process_medium_document (tokens {tokens} <= {self.MAX_TOKENS_PER_REQUEST})")
                embeddings, metadata, chunks = self._process_medium_document(text, doc_idx)
            else:
                # Large document - needs multiple API calls
                logger.info(f"[VoyageContext] Calling _process_large_document (tokens {tokens} > {self.MAX_TOKENS_PER_REQUEST})")
                embeddings, metadata, chunks = self._process_large_document(text, doc_idx)
            
            all_embeddings.extend(embeddings)
            all_metadata.extend(metadata)
            all_chunks.extend(chunks)
        
        return ChunkedEmbeddingResult(
            embeddings=all_embeddings,
            metadata=all_metadata,
            chunks=all_chunks
        )
    
    def _process_small_document(self, text: str, doc_idx: int) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """Process a document that fits in a single chunk.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        try:
            # Single chunk, single input
            result = self.client.contextualized_embed(
                inputs=[[text]],
                model=self.model,
                input_type="document",
                output_dimension=self.output_dimension
            )
            
            embedding = result.results[0].embeddings[0]
            tokens = count_tokens(text)
            
            return (
                [embedding],
                [{
                    'type': 'full_file',
                    'doc_index': doc_idx,
                    'tokens': tokens,
                    'chunk_index': 0,
                    'total_chunks': 1
                }],
                [text]
            )
        except Exception as e:
            logger.error(f"[VoyageContext] Failed to embed small document: {e}")
            raise
    
    def _process_medium_document(self, text: str, doc_idx: int) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """Process a document that needs chunking but fits in one API call.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        # Split into chunks
        chunks = split_by_tokens(
            text,
            max_tokens=self.max_chunk_tokens,
            overlap_tokens=0  # No overlap for contextual embeddings
        )
        
        # Filter empty chunks
        chunks = [c for c in chunks if c and c.strip()]
        
        if not chunks:
            logger.warning(f"[VoyageContext] No valid chunks after splitting")
            return [], [], []
        
        logger.info(f"[VoyageContext] Document split into {len(chunks)} chunks")
        
        # Check both chunk count AND total tokens limit for contextualized API
        total_tokens = sum(count_tokens(chunk) for chunk in chunks)
        if len(chunks) > self.MAX_CHUNKS_PER_REQUEST or total_tokens > self.TOKEN_LIMIT:
            logger.warning(
                f"[VoyageContext] Too many chunks ({len(chunks)}) or tokens ({total_tokens}), "
                f"processing as large document"
            )
            return self._process_large_document(text, doc_idx)
        
        try:
            # Send all chunks as one document (inner list)
            result = self.client.contextualized_embed(
                inputs=[chunks],  # Single document with multiple chunks
                model=self.model,
                input_type="document",
                output_dimension=self.output_dimension
            )
            
            embeddings = result.results[0].embeddings
            
            # Create metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    'type': 'chunk',
                    'doc_index': doc_idx,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'tokens': count_tokens(chunk)
                })
            
            return embeddings, metadata, chunks
            
        except Exception as e:
            logger.error(f"[VoyageContext] Failed to embed medium document: {e}")
            raise
    
    def _process_large_document(self, text: str, doc_idx: int) -> Tuple[List[List[float]], List[Dict], List[str]]:
        """Process a very large document that exceeds API limits.
        
        Strategy:
        1. Split document into parts of ~100K tokens
        2. Process each part separately
        3. Maintain chunk indices across parts
        
        Args:
            text: Large document text
            doc_idx: Document index
            
        Returns:
            Tuple of (embeddings, metadata, chunks)
        """
        total_tokens = count_tokens(text)
        logger.info(f"[VoyageContext] Large document ({total_tokens:,} tokens), splitting into parts")
        
        # Split into parts that fit within token limits
        # For voyage-context-3 with contextual embeddings:
        # - Total tokens in chunks must not exceed 32k (TOKEN_LIMIT)
        # - With small chunks (e.g. 64 tokens), we can have max ~500 chunks
        # To be safe, let's limit to 400 chunks per part
        max_chunks_per_part = min(400, self.TOKEN_LIMIT // self.max_chunk_tokens)
        part_size = max_chunks_per_part * self.max_chunk_tokens  # e.g., 400 * 64 = 25600 tokens
        logger.info(f"[VoyageContext] Using part_size={part_size} (max {max_chunks_per_part} chunks of {self.max_chunk_tokens} tokens)")
        parts = split_by_tokens(text, max_tokens=part_size, overlap_tokens=0)
        
        logger.info(f"[VoyageContext] Split into {len(parts)} parts")
        
        all_embeddings = []
        all_metadata = []
        all_chunks = []
        global_chunk_idx = 0
        
        for part_idx, part_text in enumerate(parts):
            part_tokens = count_tokens(part_text)
            logger.info(f"[VoyageContext] Processing part {part_idx + 1}/{len(parts)} ({part_tokens:,} tokens)")
            
            # Chunk this part
            chunks = split_by_tokens(
                part_text,
                max_tokens=self.max_chunk_tokens,
                overlap_tokens=0
            )
            
            # Filter empty chunks
            chunks = [c for c in chunks if c and c.strip()]
            
            if not chunks:
                continue
            
            try:
                # Process this part's chunks
                result = self.client.contextualized_embed(
                    inputs=[chunks],
                    model=self.model,
                    input_type="document",
                    output_dimension=self.output_dimension
                )
                
                embeddings = result.results[0].embeddings
                
                # Create metadata with global chunk indices
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    all_embeddings.append(embedding)
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'type': 'chunk',
                        'doc_index': doc_idx,
                        'chunk_index': global_chunk_idx,
                        'total_chunks': None,  # Will be updated after processing all parts
                        'part_index': part_idx,
                        'total_parts': len(parts),
                        'tokens': count_tokens(chunk)
                    })
                    global_chunk_idx += 1
                
                logger.info(f"[VoyageContext] Part {part_idx + 1} created {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"[VoyageContext] Failed to embed part {part_idx + 1}: {e}")
                raise
        
        # Update total chunks count
        total_chunks = len(all_chunks)
        for meta in all_metadata:
            meta['total_chunks'] = total_chunks
        
        logger.info(f"[VoyageContext] Large document processed: {len(parts)} parts, {total_chunks} total chunks")
        
        return all_embeddings, all_metadata, all_chunks
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # For queries, use single-element inner list
            result = self.client.contextualized_embed(
                inputs=[[text]],
                model=self.model,
                input_type="query",
                output_dimension=self.output_dimension
            )
            
            if result.results and result.results[0].embeddings:
                return result.results[0].embeddings[0]
            else:
                return []
                
        except Exception as e:
            logger.error(f"[VoyageContext] Query embedding failed: {e}")
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
            logger.error(f"[VoyageContext] API key validation failed: {e}")
            return False