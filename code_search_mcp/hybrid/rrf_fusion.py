"""Reciprocal Rank Fusion implementation for combining search results."""
from typing import List, Dict, Any, NamedTuple, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger('code_searcher.hybrid.rrf')


@dataclass
class SearchResult:
    """Generic search result from any search method."""
    document_id: str
    score: float
    source: str  # 'vector', 'bm25', or 'fusion'
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FusedResult(NamedTuple):
    """Result after RRF fusion."""
    document_id: str
    rrf_score: float
    vector_rank: Optional[int]  # Rank in vector results (1-based, None if not found)
    bm25_rank: Optional[int]    # Rank in BM25 results (1-based, None if not found)
    vector_score: Optional[float]
    bm25_score: Optional[float]
    content: Optional[str]
    metadata: Optional[Dict[str, Any]]


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining vector search and BM25 results.
    
    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    where rank_i(d) is the rank of document d in ranking i
    """
    
    def __init__(
        self,
        k_parameter: int = 60,
        use_weights: bool = False,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        Initialize RRF fusion.
        
        Args:
            k_parameter: RRF k parameter (typically 60)
            use_weights: Whether to use weighted RRF
            vector_weight: Weight for vector search results (if use_weights=True)
            bm25_weight: Weight for BM25 results (if use_weights=True)
        """
        self.k = k_parameter
        self.use_weights = use_weights
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Validate weights
        if use_weights:
            if abs(vector_weight + bm25_weight - 1.0) > 0.01:
                logger.warning(f"[RRF] Weights don't sum to 1.0: vector={vector_weight}, bm25={bm25_weight}")
        
        logger.info(f"[RRF] Initialized with k={k_parameter}, weights={use_weights}")
    
    def fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[FusedResult]:
        """
        Fuse vector search and BM25 results using RRF.
        
        Args:
            vector_results: Results from vector search (format from CodebaseSearcher)
            bm25_results: Results from BM25 search (format from BM25SearchEngine)
            
        Returns:
            List of fused results ordered by RRF score (descending)
        """
        logger.debug(f"[RRF] Fusing {len(vector_results)} vector + {len(bm25_results)} BM25 results")
        
        # Convert results to normalized format
        vector_search_results = self._normalize_vector_results(vector_results)
        bm25_search_results = self._normalize_bm25_results(bm25_results)
        
        # Create rankings (document_id -> rank)
        vector_ranking = {result.document_id: idx + 1 for idx, result in enumerate(vector_search_results)}
        bm25_ranking = {result.document_id: idx + 1 for idx, result in enumerate(bm25_search_results)}
        
        # Get all unique documents
        all_documents = set(vector_ranking.keys()) | set(bm25_ranking.keys())
        
        logger.debug(f"[RRF] Total unique documents: {len(all_documents)}")
        
        # Calculate RRF scores
        fused_results = []
        
        for doc_id in all_documents:
            # Get ranks (None if document not in ranking)
            vector_rank = vector_ranking.get(doc_id)
            bm25_rank = bm25_ranking.get(doc_id)
            
            # Calculate RRF score
            rrf_score = self._calculate_rrf_score(vector_rank, bm25_rank)
            
            # Get original scores and content
            vector_score = None
            bm25_score = None
            content = None
            metadata = {}
            
            # Find vector result data
            for result in vector_search_results:
                if result.document_id == doc_id:
                    vector_score = result.score
                    content = result.content
                    if result.metadata:
                        metadata.update(result.metadata)
                    break
            
            # Find BM25 result data  
            for result in bm25_search_results:
                if result.document_id == doc_id:
                    bm25_score = result.score
                    if not content:  # Use BM25 content if vector didn't provide it
                        content = result.content
                    if result.metadata:
                        metadata.update(result.metadata)
                    break
            
            fused_results.append(FusedResult(
                document_id=doc_id,
                rrf_score=rrf_score,
                vector_rank=vector_rank,
                bm25_rank=bm25_rank,
                vector_score=vector_score,
                bm25_score=bm25_score,
                content=content,
                metadata=metadata if metadata else None
            ))
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        if fused_results:
            logger.debug(f"[RRF] Fusion complete, top score: {fused_results[0].rrf_score:.4f}")
        else:
            logger.debug("[RRF] Fusion complete, no results")
        
        return fused_results
    
    def _calculate_rrf_score(self, vector_rank: Optional[int], bm25_rank: Optional[int]) -> float:
        """
        Calculate RRF score for a document given its ranks.
        
        Args:
            vector_rank: Rank in vector results (1-based, None if not found)
            bm25_rank: Rank in BM25 results (1-based, None if not found)
            
        Returns:
            RRF score
        """
        score = 0.0
        
        if self.use_weights:
            # Weighted RRF
            if vector_rank is not None:
                score += self.vector_weight * (1 / (self.k + vector_rank))
            if bm25_rank is not None:
                score += self.bm25_weight * (1 / (self.k + bm25_rank))
        else:
            # Standard RRF
            if vector_rank is not None:
                score += 1 / (self.k + vector_rank)
            if bm25_rank is not None:
                score += 1 / (self.k + bm25_rank)
        
        return score
    
    def _normalize_vector_results(self, results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        Normalize vector search results to SearchResult format.
        
        Vector results format from CodebaseSearcher:
        [{'path': str, 'content': str, 'score': float, ...}]
        """
        normalized = []
        
        for result in results:
            # Use file path as document ID 
            doc_id = result.get('path', '')
            if not doc_id:
                logger.warning(f"[RRF] Vector result missing 'path': {result}")
                continue
            
            normalized.append(SearchResult(
                document_id=doc_id,
                score=result.get('score', 0.0),
                source='vector',
                content=result.get('content', ''),
                metadata=result
            ))
        
        return normalized
    
    def _normalize_bm25_results(self, results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        Normalize BM25 results to SearchResult format.
        
        BM25 results format from BM25SearchEngine:
        [{'document_id': str, 'score': float, 'content': str, ...}]
        """
        normalized = []
        
        for result in results:
            # BM25 results already have document_id
            doc_id = result.get('document_id', '')
            if not doc_id:
                logger.warning(f"[RRF] BM25 result missing 'document_id': {result}")
                continue
            
            normalized.append(SearchResult(
                document_id=doc_id,
                score=result.get('score', 0.0),
                source='bm25',
                content=result.get('content', ''),
                metadata=result
            ))
        
        return normalized
    
    def convert_to_searcher_format(self, fused_results: List[FusedResult]) -> List[Dict[str, Any]]:
        """
        Convert fused results back to CodebaseSearcher format for compatibility.
        
        Args:
            fused_results: Results from RRF fusion
            
        Returns:
            List in CodebaseSearcher format: [{'path': str, 'content': str, 'score': float, ...}]
        """
        converted = []
        
        for result in fused_results:
            converted_result = {
                'path': result.document_id,
                'content': result.content or '',
                'score': result.rrf_score,
                # Add RRF-specific metadata
                'rrf_metadata': {
                    'vector_rank': result.vector_rank,
                    'bm25_rank': result.bm25_rank,
                    'vector_score': result.vector_score,
                    'bm25_score': result.bm25_score,
                    'fusion_method': 'weighted_rrf' if self.use_weights else 'standard_rrf'
                }
            }
            
            # Include original metadata if available, but don't override core fields
            if result.metadata:
                for key, value in result.metadata.items():
                    if key not in ['path', 'content', 'score', 'rrf_metadata']:
                        converted_result[key] = value
            
            converted.append(converted_result)
        
        return converted
    
    def analyze_fusion_quality(self, fused_results: List[FusedResult]) -> Dict[str, Any]:
        """
        Analyze fusion quality and provide insights.
        
        Args:
            fused_results: Results from fusion
            
        Returns:
            Analysis dictionary with fusion statistics
        """
        if not fused_results:
            return {'status': 'no_results'}
        
        total_docs = len(fused_results)
        vector_only = sum(1 for r in fused_results if r.vector_rank and not r.bm25_rank)
        bm25_only = sum(1 for r in fused_results if r.bm25_rank and not r.vector_rank)
        both_methods = sum(1 for r in fused_results if r.vector_rank and r.bm25_rank)
        
        # Calculate rank correlation for documents found by both methods
        rank_correlations = []
        for result in fused_results:
            if result.vector_rank and result.bm25_rank:
                rank_correlations.append((result.vector_rank, result.bm25_rank))
        
        analysis = {
            'total_documents': total_docs,
            'found_by_vector_only': vector_only,
            'found_by_bm25_only': bm25_only,
            'found_by_both': both_methods,
            'fusion_diversity': (vector_only + bm25_only) / total_docs if total_docs > 0 else 0,
            'method_agreement': both_methods / total_docs if total_docs > 0 else 0,
            'rank_correlations': rank_correlations[:10],  # Sample of rank correlations
            'top_result_source': self._analyze_top_results(fused_results[:5])
        }
        
        return analysis
    
    def _analyze_top_results(self, top_results: List[FusedResult]) -> Dict[str, int]:
        """Analyze which methods contributed to top results."""
        sources = {'vector_only': 0, 'bm25_only': 0, 'both': 0}
        
        for result in top_results:
            if result.vector_rank and result.bm25_rank:
                sources['both'] += 1
            elif result.vector_rank:
                sources['vector_only'] += 1
            elif result.bm25_rank:
                sources['bm25_only'] += 1
        
        return sources
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion configuration and statistics."""
        return {
            'k_parameter': self.k,
            'use_weights': self.use_weights,
            'vector_weight': self.vector_weight if self.use_weights else None,
            'bm25_weight': self.bm25_weight if self.use_weights else None,
            'fusion_method': 'weighted_rrf' if self.use_weights else 'standard_rrf'
        }