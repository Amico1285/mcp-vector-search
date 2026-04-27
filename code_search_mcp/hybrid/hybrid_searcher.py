"""Hybrid searcher combining vector search and BM25 with RRF fusion."""
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from ..searcher import CodebaseSearcher
from .. import env_config
from .bm25_search import BM25SearchEngine, BM25Result
from .rrf_fusion import RRFFusion

logger = logging.getLogger('code_searcher.hybrid.searcher')


class HybridSearcher(CodebaseSearcher):
    """
    Enhanced searcher that combines vector search and BM25 using RRF fusion.
    
    Falls back to standard vector search if hybrid search is disabled.
    """
    
    def __init__(self, codebase_path: str, use_ai_filter: bool = False, db_name: str = 'codebase_files', db_base_path: str = None):
        """
        Initialize hybrid searcher.
        
        Args:
            codebase_path: Path to the codebase
            use_ai_filter: Whether to use AI filtering
            db_name: Name of the database/collection
            db_base_path: Optional base path for database storage
        """
        # Initialize parent CodebaseSearcher
        super().__init__(codebase_path, use_ai_filter, db_name, db_base_path)
        
        # Check if hybrid search is enabled
        self.hybrid_enabled = env_config.get_hybrid_search_enabled()
        
        if self.hybrid_enabled:
            # Initialize BM25 search engine
            self.bm25_engine = self._initialize_bm25_engine()
            
            # Initialize RRF fusion
            self.rrf_fusion = self._initialize_rrf_fusion()
            
            logger.info("[HYBRID] Hybrid search initialized")
        else:
            self.bm25_engine = None
            self.rrf_fusion = None
            logger.info("[HYBRID] Hybrid search disabled, using vector search only")
    
    def _initialize_bm25_engine(self) -> Optional[BM25SearchEngine]:
        """Initialize BM25 search engine with configuration."""
        try:
            bm25_engine = BM25SearchEngine(
                k1=env_config.get_bm25_k1_parameter(),
                b=env_config.get_bm25_b_parameter(),
                language=env_config.get_bm25_language(),
                remove_stopwords=env_config.get_bm25_remove_stopwords(),
                min_token_length=env_config.get_bm25_min_token_length(),
                use_stemming=env_config.get_bm25_use_stemming()
            )
            
            # Load BM25 index if it exists
            bm25_index_path = self._get_bm25_index_path()
            if bm25_index_path.exists():
                if bm25_engine.load_index(bm25_index_path):
                    logger.info(f"[HYBRID] Loaded BM25 index from {bm25_index_path}")
                else:
                    logger.warning(f"[HYBRID] Failed to load BM25 index, will create new one")
            else:
                logger.warning(f"[HYBRID] BM25 index not found at {bm25_index_path}")
            
            return bm25_engine
            
        except Exception as e:
            logger.error(f"[HYBRID] Failed to initialize BM25 engine: {e}")
            return None
    
    def _initialize_rrf_fusion(self) -> RRFFusion:
        """Initialize RRF fusion with configuration."""
        return RRFFusion(
            k_parameter=env_config.get_rrf_k_parameter(),
            use_weights=env_config.get_rrf_weights_enabled(),
            vector_weight=env_config.get_rrf_vector_weight(),
            bm25_weight=env_config.get_rrf_bm25_weight()
        )
    
    def _get_bm25_index_path(self) -> Path:
        """Get path to BM25 index file."""
        # Use same logic as database_updater.py
        db_base = os.path.join(os.path.dirname(__file__), '..', '..', 'DBs')
        bm25_db_name = f"{self.db_name}_bm25"
        bm25_path = Path(db_base) / bm25_db_name
        bm25_path.mkdir(parents=True, exist_ok=True)
        
        return bm25_path / 'index.pkl'
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach (vector + BM25 + RRF) or fallback to vector only.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        # Validate configuration
        config_errors = env_config.validate_hybrid_config()
        if config_errors:
            logger.warning(f"[HYBRID] Configuration errors, falling back to vector search: {config_errors}")
            return super().search(query)
        
        # Check for BM25-only mode
        if env_config.get_bm25_only_mode():
            if not self.bm25_engine:
                logger.error("[BM25] BM25-only mode requested but BM25 engine not available")
                return []
            if env_config.get_logging_verbose():
                logger.info("[BM25] Using BM25 search only")
            return self._bm25_only_search(query)
        
        # Check if hybrid search is enabled and components are available
        if not self.hybrid_enabled or not self.bm25_engine or not self.rrf_fusion:
            if env_config.get_logging_verbose():
                logger.info("[HYBRID] Using vector search only")
            return super().search(query)
        
        # Perform hybrid search
        return self._hybrid_search(query)
    
    def _hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 results with RRF.
        
        Args:
            query: Search query
            
        Returns:
            List of fused search results
        """
        if env_config.get_logging_verbose():
            logger.info(f"[HYBRID] Starting hybrid search for: '{query}'")
        
        try:
            # Step 1: Get vector search results (without reranker/AI filter)
            vector_results = self._get_vector_results(query)
            
            # Step 2: Get BM25 search results
            bm25_results = self._get_bm25_results(query)
            
            if env_config.get_logging_verbose():
                logger.info(f"[HYBRID] Vector: {len(vector_results)} results, BM25: {len(bm25_results)} results")
            
            # If no results from either method, return empty
            if not vector_results and not bm25_results:
                logger.info(f"[HYBRID] No results found for query: '{query}'")
                return []
            
            # Step 3: Fuse results using RRF
            fused_results = self.rrf_fusion.fuse_results(vector_results, bm25_results)
            
            # Step 4: Convert back to searcher format
            hybrid_results = self.rrf_fusion.convert_to_searcher_format(fused_results)
            
            if env_config.get_logging_verbose():
                logger.info(f"[HYBRID] RRF fusion produced {len(hybrid_results)} results")
                # Log fusion analysis
                analysis = self.rrf_fusion.analyze_fusion_quality(fused_results)
                logger.info(f"[HYBRID] Fusion analysis: {analysis}")
            
            # Step 5: Apply existing pipeline (reranker + AI filter)
            final_results = self._apply_post_fusion_pipeline(query, hybrid_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"[HYBRID] Error in hybrid search: {e}")
            logger.info("[HYBRID] Falling back to vector search")
            return super().search(query)
    
    def _get_vector_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Get vector search results without reranker/AI filter.
        
        Args:
            query: Search query
            
        Returns:
            Raw vector search results
        """
        # Temporarily disable reranker and AI filter to get raw vector results
        original_reranker = env_config.get_reranker_enabled()
        original_ai_filter = self.ai_filter
        
        try:
            # Disable post-processing temporarily
            os.environ['RERANKER_ENABLED'] = 'false'
            self.ai_filter = None
            
            # Get raw vector results
            vector_results = super().search(query)
            
            return vector_results
            
        finally:
            # Restore original settings
            os.environ['RERANKER_ENABLED'] = str(original_reranker).lower()
            self.ai_filter = original_ai_filter
    
    def _get_bm25_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Get BM25 search results.
        
        Args:
            query: Search query
            
        Returns:
            BM25 search results in compatible format
        """
        if not self.bm25_engine:
            return []
        
        # Get number of BM25 results to retrieve
        n_results = env_config.get_bm25_n_results()
        
        try:
            # Perform BM25 search
            bm25_raw_results = self.bm25_engine.search(query, n_results)
            
            # Convert to compatible format
            bm25_results = []
            for result in bm25_raw_results:
                # Get document content from BM25 engine
                content = self.bm25_engine.get_document_content(result.document_id) or ''
                
                bm25_results.append({
                    'document_id': result.document_id,
                    'score': result.score,
                    'content': content,
                    'matched_terms': result.matched_terms,
                    'search_method': 'bm25'
                })
            
            return bm25_results

        except Exception as e:
            logger.error(f"[HYBRID] Error in BM25 search: {e}")
            return []

    def _convert_bm25_to_standard_format(self, bm25_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert BM25 results to standard CodebaseSearcher format.

        Handles both full file IDs and chunk IDs (format: file_path::chunk_N).
        When chunking is used, deduplicates results by file path, keeping
        the highest scoring chunk for each file.

        Args:
            bm25_results: Results from _get_bm25_results with keys:
                - document_id: file path or chunk ID (file_path::chunk_N)
                - score: BM25 score
                - content: file/chunk content
                - matched_terms: terms that matched
                - search_method: 'bm25'

        Returns:
            Results in standard format with keys:
                - path: file path (deduplicated)
                - content: file content
                - score: BM25 score (max across chunks)
                - bm25_metadata: BM25-specific data
        """
        # First pass: group by file path and keep best scoring chunk
        file_results = {}  # file_path -> best result

        for result in bm25_results:
            document_id = result.get('document_id', '')
            score = result.get('score', 0.0)

            # Extract file path from chunk ID if present
            # Format: file_path::chunk_N
            if '::chunk_' in document_id:
                file_path = document_id.rsplit('::chunk_', 1)[0]
            else:
                file_path = document_id

            # Keep the result with highest score for each file
            if file_path not in file_results or score > file_results[file_path]['score']:
                file_results[file_path] = {
                    'path': file_path,
                    'content': result.get('content', ''),
                    'score': score,
                    'bm25_metadata': {
                        'matched_terms': result.get('matched_terms', []),
                        'search_method': result.get('search_method', 'bm25'),
                        'chunk_id': document_id if '::chunk_' in document_id else None
                    }
                }

        # Convert to list and sort by score (descending)
        converted = list(file_results.values())
        converted.sort(key=lambda x: x['score'], reverse=True)

        if env_config.get_logging_verbose():
            original_count = len(bm25_results)
            deduped_count = len(converted)
            if original_count != deduped_count:
                logger.info(f"[BM25] Deduplicated {original_count} chunks to {deduped_count} files")

        return converted

    def _apply_post_fusion_pipeline(self, query: str, hybrid_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply reranker and AI filter to hybrid results.
        
        Args:
            query: Original query
            hybrid_results: Results from RRF fusion
            
        Returns:
            Final processed results
        """
        processed_results = hybrid_results
        
        # Apply reranker if enabled
        if env_config.get_reranker_enabled() and processed_results:
            if env_config.get_logging_verbose():
                logger.info(f"[HYBRID] Applying reranker to {len(processed_results)} results")
            
            try:
                processed_results = self._apply_reranker(query, processed_results)
                if env_config.get_logging_verbose():
                    logger.info(f"[HYBRID] Reranker returned {len(processed_results)} results")
            except Exception as e:
                logger.error(f"[HYBRID] Reranker failed: {e}")
        
        # Apply AI filter if enabled
        if self.ai_filter and processed_results:
            if env_config.get_logging_verbose():
                logger.info(f"[HYBRID] Applying AI filter to {len(processed_results)} results")
            
            try:
                processed_results = self.ai_filter.filter_search_results(
                    query,
                    processed_results,
                    return_all_with_scores=False
                )
                if env_config.get_logging_verbose():
                    logger.info(f"[HYBRID] AI filter returned {len(processed_results)} results")
            except Exception as e:
                logger.error(f"[HYBRID] AI filter failed: {e}")
        
        # Limit to max results
        max_results = env_config.get_max_results()
        if len(processed_results) > max_results:
            processed_results = processed_results[:max_results]
            if env_config.get_logging_verbose():
                logger.info(f"[HYBRID] Limited to top {max_results} results")
        
        return processed_results
    
    def _bm25_only_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform BM25-only search without vector component.
        
        Args:
            query: Search query
            
        Returns:
            List of BM25 search results
        """
        if env_config.get_logging_verbose():
            logger.info(f"[BM25] Starting BM25-only search for: '{query}'")
        
        try:
            # Get BM25 search results
            bm25_results = self._get_bm25_results(query)
            
            if env_config.get_logging_verbose():
                logger.info(f"[BM25] Found {len(bm25_results)} BM25 results")
            
            # Convert BM25 results to standard format
            processed_results = self._convert_bm25_to_standard_format(bm25_results)
            
            # Apply reranker if enabled
            if env_config.get_reranker_enabled() and processed_results:
                if env_config.get_logging_verbose():
                    logger.info(f"[BM25] Applying reranker to {len(processed_results)} results")
                
                try:
                    processed_results = self._apply_reranker(query, processed_results)
                    if env_config.get_logging_verbose():
                        logger.info(f"[BM25] Reranker returned {len(processed_results)} results")
                except Exception as e:
                    logger.error(f"[BM25] Reranker failed: {e}")
            
            # Apply AI filter if enabled
            if self.ai_filter and processed_results:
                if env_config.get_logging_verbose():
                    logger.info(f"[BM25] Applying AI filter to {len(processed_results)} results")
                
                try:
                    processed_results = self.ai_filter.filter_search_results(
                        query,
                        processed_results,
                        return_all_with_scores=False
                    )
                    if env_config.get_logging_verbose():
                        logger.info(f"[BM25] AI filter returned {len(processed_results)} results")
                except Exception as e:
                    logger.error(f"[BM25] AI filter failed: {e}")
            
            # Limit to max results
            max_results = env_config.get_max_results()
            if len(processed_results) > max_results:
                processed_results = processed_results[:max_results]
                if env_config.get_logging_verbose():
                    logger.info(f"[BM25] Limited to top {max_results} results")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"[BM25] BM25-only search failed: {e}")
            return []
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hybrid search components.
        
        Returns:
            Dictionary with hybrid search statistics
        """
        stats = {
            'hybrid_enabled': self.hybrid_enabled,
            'config_valid': len(env_config.validate_hybrid_config()) == 0
        }
        
        if self.hybrid_enabled:
            # BM25 statistics
            if self.bm25_engine:
                stats['bm25'] = self.bm25_engine.get_stats()
            else:
                stats['bm25'] = {'status': 'not_initialized'}
            
            # RRF statistics
            if self.rrf_fusion:
                stats['rrf'] = self.rrf_fusion.get_fusion_stats()
            else:
                stats['rrf'] = {'status': 'not_initialized'}
            
            # Index paths
            stats['bm25_index_path'] = str(self._get_bm25_index_path())
            stats['bm25_index_exists'] = self._get_bm25_index_path().exists()
        
        return stats
    
    def rebuild_bm25_index(self, documents: List[Dict[str, str]]) -> bool:
        """
        Rebuild BM25 index with new documents.
        
        Args:
            documents: List of documents with 'id', 'content', and optional 'preview' keys
            
        Returns:
            True if successful, False otherwise
        """
        if not self.hybrid_enabled or not self.bm25_engine:
            logger.warning("[HYBRID] Cannot rebuild BM25 index: hybrid search not enabled or BM25 engine not initialized")
            return False
        
        try:
            logger.info(f"[HYBRID] Rebuilding BM25 index with {len(documents)} documents")
            
            # Clear existing index
            self.bm25_engine.index.clear()
            
            # Add documents to BM25 index
            for doc in documents:
                doc_id = doc['id']
                content = doc['content']
                preview = doc.get('preview', content[:500] + ('...' if len(content) > 500 else ''))
                
                self.bm25_engine.add_document(doc_id, content, preview)
            
            # Save index to disk
            bm25_index_path = self._get_bm25_index_path()
            self.bm25_engine.save_index(bm25_index_path)
            
            logger.info(f"[HYBRID] BM25 index rebuilt and saved to {bm25_index_path}")
            return True
            
        except Exception as e:
            logger.error(f"[HYBRID] Failed to rebuild BM25 index: {e}")
            return False