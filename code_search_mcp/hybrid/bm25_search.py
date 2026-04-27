"""BM25 search engine implementation."""
import math
import pickle
import json
from typing import Dict, List, NamedTuple, Optional, Any
from pathlib import Path
from collections import Counter
import logging

from .text_processor import TextProcessor

logger = logging.getLogger('code_searcher.hybrid.bm25')


class BM25Result(NamedTuple):
    """BM25 search result."""
    document_id: str
    score: float
    matched_terms: List[str]


class BM25Index:
    """BM25 index containing all necessary data structures."""
    
    def __init__(self):
        # Core index structures
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> {tokens, length, content_preview}
        self.term_doc_freq: Dict[str, int] = {}  # term -> document frequency
        self.term_freq_in_doc: Dict[str, Dict[str, int]] = {}  # doc -> term -> frequency
        
        # Statistics
        self.total_documents = 0
        self.total_tokens = 0
        self.average_doc_length = 0.0
        
        # IDF cache for performance
        self._idf_cache: Dict[str, float] = {}
    
    def clear(self):
        """Clear all index data."""
        self.documents.clear()
        self.term_doc_freq.clear()
        self.term_freq_in_doc.clear()
        self._idf_cache.clear()
        self.total_documents = 0
        self.total_tokens = 0
        self.average_doc_length = 0.0


class BM25SearchEngine:
    """BM25 search engine with configurable parameters."""
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        language: str = 'english',
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        use_stemming: bool = False
    ):
        """
        Initialize BM25 search engine.
        
        Args:
            k1: Controls term frequency saturation (1.2 is typical)
            b: Controls document length normalization (0.75 is typical)
            language: Language for text processing
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum token length
            use_stemming: Whether to use stemming
        """
        self.k1 = k1
        self.b = b
        
        # Initialize text processor
        self.text_processor = TextProcessor(
            language=language,
            remove_stopwords=remove_stopwords,
            min_token_length=min_token_length,
            use_stemming=use_stemming
        )
        
        # Initialize index
        self.index = BM25Index()
        
        logger.info(f"[BM25] Initialized with k1={k1}, b={b}, language={language}")
    
    def add_document(self, document_id: str, content: str, content_preview: str = None):
        """
        Add a document to the BM25 index.
        
        Args:
            document_id: Unique document identifier
            content: Full document content for indexing
            content_preview: Optional preview content for display (defaults to content)
        """
        # Process text to get tokens
        tokens = self.text_processor.process_code_text(content)
        
        if not tokens:
            logger.warning(f"[BM25] No tokens found for document {document_id}")
            return
        
        # Count term frequencies in this document
        term_counts = Counter(tokens)
        doc_length = len(tokens)
        
        # Store document information
        self.index.documents[document_id] = {
            'tokens': tokens,
            'length': doc_length,
            'content_preview': content_preview or content[:500] + ('...' if len(content) > 500 else ''),
            'term_counts': dict(term_counts)
        }
        
        # Update term-document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.index.term_doc_freq[term] = self.index.term_doc_freq.get(term, 0) + 1
        
        # Store term frequencies for this document
        self.index.term_freq_in_doc[document_id] = dict(term_counts)
        
        # Update statistics
        self.index.total_documents += 1
        self.index.total_tokens += doc_length
        self.index.average_doc_length = self.index.total_tokens / self.index.total_documents
        
        # Clear IDF cache since document frequencies changed
        self.index._idf_cache.clear()
        
        if self.index.total_documents % 100 == 0:
            logger.info(f"[BM25] Indexed {self.index.total_documents} documents")
    
    def remove_document(self, document_id: str):
        """
        Remove a document from the BM25 index.
        
        Args:
            document_id: Document identifier to remove
        """
        if document_id not in self.index.documents:
            logger.warning(f"[BM25] Document {document_id} not found for removal")
            return
        
        # Get document info before removal
        doc_info = self.index.documents[document_id]
        doc_length = doc_info['length']
        unique_terms = set(doc_info['tokens'])
        
        # Update term-document frequencies
        for term in unique_terms:
            if term in self.index.term_doc_freq:
                self.index.term_doc_freq[term] -= 1
                if self.index.term_doc_freq[term] <= 0:
                    del self.index.term_doc_freq[term]
        
        # Remove document
        del self.index.documents[document_id]
        del self.index.term_freq_in_doc[document_id]
        
        # Update statistics
        self.index.total_documents -= 1
        self.index.total_tokens -= doc_length
        if self.index.total_documents > 0:
            self.index.average_doc_length = self.index.total_tokens / self.index.total_documents
        else:
            self.index.average_doc_length = 0.0
        
        # Clear IDF cache
        self.index._idf_cache.clear()
    
    def _calculate_idf(self, term: str) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.
        
        Args:
            term: Term to calculate IDF for
            
        Returns:
            IDF score for the term
        """
        if term in self.index._idf_cache:
            return self.index._idf_cache[term]
        
        df = self.index.term_doc_freq.get(term, 0)
        if df == 0:
            idf = 0.0
        else:
            # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
            N = self.index.total_documents
            idf = math.log((N - df + 0.5) / (df + 0.5))
        
        self.index._idf_cache[term] = idf
        return idf
    
    def _calculate_bm25_score(self, query_terms: List[str], document_id: str) -> tuple:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            document_id: Document to score
            
        Returns:
            Tuple of (total_score, matched_terms)
        """
        if document_id not in self.index.documents:
            return 0.0, []
        
        doc_info = self.index.documents[document_id]
        doc_length = doc_info['length']
        term_freqs = self.index.term_freq_in_doc.get(document_id, {})
        
        total_score = 0.0
        matched_terms = []
        
        for term in query_terms:
            if term in term_freqs:
                # Term exists in document
                tf = term_freqs[term]  # Term frequency in document
                idf = self._calculate_idf(term)
                
                # BM25 score formula:
                # IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.index.average_doc_length)
                
                score = idf * (numerator / denominator)
                total_score += score
                matched_terms.append(term)
        
        return total_score, matched_terms
    
    def search(self, query: str, n_results: int = 10) -> List[BM25Result]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            
        Returns:
            List of BM25Result objects ordered by relevance score
        """
        if not query.strip():
            return []
        
        if self.index.total_documents == 0:
            logger.warning("[BM25] No documents in index")
            return []
        
        # Process query to get terms
        query_terms = self.text_processor.process_code_text(query)
        
        if not query_terms:
            logger.warning(f"[BM25] No terms found in query: '{query}'")
            return []
        
        logger.debug(f"[BM25] Query terms: {query_terms}")
        
        # Score all documents
        scored_docs = []
        for doc_id in self.index.documents:
            score, matched_terms = self._calculate_bm25_score(query_terms, doc_id)
            if len(matched_terms) > 0:  # Only include documents that match query terms
                scored_docs.append(BM25Result(
                    document_id=doc_id,
                    score=score,
                    matched_terms=matched_terms
                ))
        
        # Sort by score (descending) and limit results
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        results = scored_docs[:n_results]
        
        logger.debug(f"[BM25] Found {len(results)} results for query '{query}'")
        
        return results
    
    def get_document_content(self, document_id: str) -> Optional[str]:
        """
        Get document content preview by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document content preview or None if not found
        """
        doc_info = self.index.documents.get(document_id)
        return doc_info['content_preview'] if doc_info else None
    
    def save_index(self, index_path: Path):
        """
        Save BM25 index to disk.
        
        Args:
            index_path: Path to save index to
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save binary index data
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata as JSON for inspection
        metadata_path = index_path.with_suffix('.meta.json')
        metadata = {
            'total_documents': self.index.total_documents,
            'total_tokens': self.index.total_tokens,
            'average_doc_length': self.index.average_doc_length,
            'vocabulary_size': len(self.index.term_doc_freq),
            'parameters': {
                'k1': self.k1,
                'b': self.b,
            },
            'text_processor_stats': self.text_processor.get_stats()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[BM25] Saved index to {index_path} ({self.index.total_documents} documents)")
    
    def load_index(self, index_path: Path) -> bool:
        """
        Load BM25 index from disk.
        
        Args:
            index_path: Path to load index from
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not index_path.exists():
            logger.error(f"[BM25] Index file not found: {index_path}")
            return False
        
        try:
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
            
            logger.info(f"[BM25] Loaded index from {index_path} ({self.index.total_documents} documents)")
            return True
            
        except Exception as e:
            logger.error(f"[BM25] Failed to load index: {e}")
            return False

    def check_text_processor_match(self, index_path: Path) -> bool:
        """
        Check if saved index's text processor settings match current settings.

        Args:
            index_path: Path to the index file

        Returns:
            True if settings match, False if mismatch or metadata not found
        """
        metadata_path = index_path.with_suffix('.meta.json')

        if not metadata_path.exists():
            logger.warning(f"[BM25] No metadata file found at {metadata_path}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            saved_stats = metadata.get('text_processor_stats', {})
            current_stats = self.text_processor.get_stats()

            # Check critical text processing parameters that affect indexing
            critical_params = ['remove_stopwords', 'use_stemming', 'language', 'min_token_length']

            for param in critical_params:
                saved_val = saved_stats.get(param)
                current_val = current_stats.get(param)

                if saved_val != current_val:
                    logger.info(f"[BM25] Text processor mismatch: {param} = {saved_val} (saved) vs {current_val} (current)")
                    return False

            logger.debug(f"[BM25] Text processor settings match saved index")
            return True

        except Exception as e:
            logger.error(f"[BM25] Failed to check metadata: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index.total_documents == 0:
            return {
                'total_documents': 0,
                'status': 'empty'
            }
        
        # Calculate vocabulary stats
        vocab_stats = {}
        if self.index.term_doc_freq:
            doc_freqs = list(self.index.term_doc_freq.values())
            vocab_stats = {
                'size': len(doc_freqs),
                'max_doc_freq': max(doc_freqs),
                'min_doc_freq': min(doc_freqs),
                'avg_doc_freq': sum(doc_freqs) / len(doc_freqs)
            }
        
        return {
            'total_documents': self.index.total_documents,
            'total_tokens': self.index.total_tokens,
            'average_doc_length': self.index.average_doc_length,
            'vocabulary': vocab_stats,
            'parameters': {
                'k1': self.k1,
                'b': self.b
            },
            'text_processor': self.text_processor.get_stats(),
            'status': 'ready'
        }