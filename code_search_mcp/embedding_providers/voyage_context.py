"""VoyageAI voyage-context-* embedding provider for contextual embeddings."""
import os
import logging
from typing import List, Dict, Optional, Tuple
import voyageai

from .base import EmbeddingProvider, ChunkedEmbeddingResult
from .utils import count_tokens, split_by_tokens

logger = logging.getLogger(__name__)


class VoyageContextProvider(EmbeddingProvider):
    """Provider for voyage-context-* contextual embeddings (context-3, context-4, ...).

    This provider handles the special contextualized embedding API
    that processes chunks in context with each other.
    """

    # API limits for the voyage-context-* family
    MAX_TOKENS_PER_REQUEST = 120000  # Total tokens across all inputs in one request
    MAX_CHUNKS_PER_REQUEST = 16000   # Total chunks across all inputs (Voyage limit: 16K)
    MAX_INPUTS_PER_REQUEST = 1000    # Max documents (inner lists) per request
    TOKEN_LIMIT = 32000              # Model context window per example (one document)
    # A single contextualized_embed "example" (one document = one inner list of its
    # chunks) is embedded together and must fit the model's 32000-token context
    # window — contextual embeddings do NOT support truncation. count_tokens() uses
    # tiktoken cl100k_base, which can differ from Voyage's own tokenizer, so we cap
    # each document well below TOKEN_LIMIT and split anything larger into parts.
    PER_DOC_TOKEN_LIMIT = 25600      # Safe per-document token budget (< 32000)
    
    def __init__(
        self,
        max_chunk_tokens: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
        output_dimension: Optional[int] = None,
        model: str = "voyage-context-3",
        max_doc_tokens: Optional[int] = None
    ):
        """Initialize a voyage-context-* contextual provider.

        Args:
            max_chunk_tokens: Maximum tokens per chunk (default: 64)
            min_chunk_tokens: Minimum tokens per chunk (default: 1)
            output_dimension: Optional output dimension (256, 512, 1024, 2048)
            model: Voyage contextual model name (e.g. "voyage-context-3",
                "voyage-context-4"). All are called via contextualized_embed.
            max_doc_tokens: Max tokens for one document (one contextualized_embed
                example). Documents above this are split into parts so each part
                fits the model's context window. Defaults to PER_DOC_TOKEN_LIMIT.
        """
        self.model = model

        # Chunk configuration - use reasonable defaults for contextual embeddings
        if max_chunk_tokens is None:
            max_chunk_tokens = 64
        if min_chunk_tokens is None:
            min_chunk_tokens = 1

        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.output_dimension = output_dimension
        self.dimension = output_dimension if output_dimension else 1024

        # Per-document (per-example) token budget. A document whose chunks sum to
        # more than this is split into parts, each embedded as its own example so
        # it stays within the model's context window. Kept below TOKEN_LIMIT for
        # margin against the tiktoken-vs-Voyage tokenizer difference.
        if max_doc_tokens is None:
            max_doc_tokens = self.PER_DOC_TOKEN_LIMIT
        if max_doc_tokens >= self.TOKEN_LIMIT:
            logger.warning(
                f"[VoyageContext] max_doc_tokens ({max_doc_tokens}) >= context window "
                f"({self.TOKEN_LIMIT}); capping at {self.PER_DOC_TOKEN_LIMIT} for safety margin"
            )
            max_doc_tokens = self.PER_DOC_TOKEN_LIMIT
        if max_doc_tokens < self.max_chunk_tokens:
            max_doc_tokens = self.max_chunk_tokens
        self.max_doc_tokens = max_doc_tokens

        # Initialize Voyage client
        self.client = voyageai.Client()

        logger.info(
            f"[VoyageContext] Initialized with chunk_size={max_chunk_tokens}, "
            f"max_doc_tokens={self.max_doc_tokens}, dimension={self.dimension}"
        )
    
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

        Documents are pre-chunked client-side, then packed into as few
        contextualized_embed requests as the API limits allow (up to
        MAX_INPUTS_PER_REQUEST documents / MAX_CHUNKS_PER_REQUEST chunks /
        MAX_TOKENS_PER_REQUEST tokens per request). Each document is still sent
        as its own inner list, so its chunks stay contextualized within the
        document. A document that alone exceeds per-request limits is split into
        parts individually.

        The returned arrays are grouped by document and ordered by input index
        (embeddings[i], chunks[i] and metadata[i] describe the same chunk, and
        all chunks of document N precede those of document N+1). The database
        updater relies on this strict ordering.

        Args:
            texts: List of document strings to embed

        Returns:
            ChunkedEmbeddingResult with embeddings, metadata and chunk texts
        """
        # 1. Pre-chunk every document; classify batchable vs oversized
        results_by_doc: Dict[int, Tuple[List, List, List]] = {}
        batchable = []   # [{'doc_idx', 'chunks', 'tokens'}]
        oversized = []   # [(doc_idx, text)]

        for doc_idx, text in enumerate(texts):
            doc_tokens = count_tokens(text)
            if doc_tokens <= self.max_chunk_tokens:
                # Single chunk — but skip empty/whitespace-only documents, since
                # the API rejects empty strings and one bad input fails the whole batch.
                chunks = [text] if (text and text.strip()) else []
            else:
                chunks = [c for c in split_by_tokens(
                    text, max_tokens=self.max_chunk_tokens, overlap_tokens=0
                ) if c and c.strip()]

            if not chunks:
                logger.info(f"[VoyageContext] Document {doc_idx}: no non-empty chunks, skipping")
                results_by_doc[doc_idx] = ([], [], [])
                continue

            total_tokens = sum(count_tokens(c) for c in chunks)
            # A document is sent as one contextualized_embed example, so its chunks
            # are embedded together and must fit the model's context window. Route
            # anything above max_doc_tokens to per-document part-splitting;
            # MAX_TOKENS_PER_REQUEST (120000) only bounds a whole multi-doc request.
            if len(chunks) > self.MAX_CHUNKS_PER_REQUEST or total_tokens > self.max_doc_tokens:
                oversized.append((doc_idx, text))
            else:
                batchable.append({'doc_idx': doc_idx, 'chunks': chunks, 'tokens': total_tokens})

        # 2. Greedily pack batchable documents into request-sized groups
        batches = []
        cur, cur_chunks, cur_tokens = [], 0, 0
        for item in batchable:
            n_chunks, n_tokens = len(item['chunks']), item['tokens']
            if cur and (
                len(cur) + 1 > self.MAX_INPUTS_PER_REQUEST
                or cur_chunks + n_chunks > self.MAX_CHUNKS_PER_REQUEST
                or cur_tokens + n_tokens > self.MAX_TOKENS_PER_REQUEST
            ):
                batches.append(cur)
                cur, cur_chunks, cur_tokens = [], 0, 0
            cur.append(item)
            cur_chunks += n_chunks
            cur_tokens += n_tokens
        if cur:
            batches.append(cur)

        logger.info(
            f"[VoyageContext] Embedding {len(texts)} document(s) via {self.model} "
            f"(dim={self.dimension}): {len(batchable)} batchable in {len(batches)} "
            f"API request(s), {len(oversized)} oversized (per-doc splitting)"
        )

        # 3. One contextualized_embed call per batch
        for b_idx, batch in enumerate(batches):
            inputs = [item['chunks'] for item in batch]
            n_chunks = sum(len(x) for x in inputs)
            n_tokens = sum(item['tokens'] for item in batch)
            logger.info(
                f"[VoyageContext] Request {b_idx + 1}/{len(batches)}: "
                f"{len(inputs)} doc(s), {n_chunks} chunk(s), {n_tokens} token(s)"
            )
            try:
                result = self.client.contextualized_embed(
                    inputs=inputs,
                    model=self.model,
                    input_type="document",
                    output_dimension=self.output_dimension
                )
            except Exception as e:
                logger.error(f"[VoyageContext] Request {b_idx + 1} failed: {e}")
                raise

            for i, item in enumerate(batch):
                doc_idx = item['doc_idx']
                chunks = item['chunks']
                embeddings = result.results[i].embeddings
                metadata = [{
                    'type': 'full_file' if len(chunks) == 1 else 'chunk',
                    'doc_index': doc_idx,
                    'chunk_index': c_i,
                    'total_chunks': len(chunks),
                    'tokens': count_tokens(chunk),
                } for c_i, chunk in enumerate(chunks)]
                results_by_doc[doc_idx] = (list(embeddings), metadata, list(chunks))

        # 4. Oversized documents handled individually (part-splitting)
        for doc_idx, text in oversized:
            logger.info(f"[VoyageContext] Oversized document {doc_idx}: splitting into parts")
            embeddings, metadata, chunks = self._process_large_document(text, doc_idx)
            results_by_doc[doc_idx] = (embeddings, metadata, chunks)

        # 5. Assemble strictly in input order (grouped by doc, ascending doc_index)
        all_embeddings, all_metadata, all_chunks = [], [], []
        for doc_idx in range(len(texts)):
            emb, meta, chks = results_by_doc.get(doc_idx, ([], [], []))
            all_embeddings.extend(emb)
            all_metadata.extend(meta)
            all_chunks.extend(chks)

        logger.info(
            f"[VoyageContext] Done: {len(all_embeddings)} embedding(s) across "
            f"{len(texts)} document(s) in {len(batches)} request(s) + {len(oversized)} oversized"
        )
        return ChunkedEmbeddingResult(
            embeddings=all_embeddings,
            metadata=all_metadata,
            chunks=all_chunks
        )
    
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
        
        # Split into parts that each fit a single contextualized_embed example.
        # For the voyage-context-* contextual models the whole example (a part and
        # all its chunks) is embedded together, so it must stay under the model's
        # context window. We cap each part at max_doc_tokens (< TOKEN_LIMIT) for
        # margin against the tiktoken-vs-Voyage tokenizer difference.
        part_size = self.max_doc_tokens
        max_chunks_per_part = max(1, part_size // self.max_chunk_tokens)
        logger.info(f"[VoyageContext] Using part_size={part_size} (up to {max_chunks_per_part} chunks of {self.max_chunk_tokens} tokens)")
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