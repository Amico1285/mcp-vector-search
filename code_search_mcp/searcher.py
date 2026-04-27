"""Search logic for codebase files."""
import os
import json
import sys
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import chromadb
from .aifilter import AIFilter
from . import env_config
from .embedding_providers import create_embedding_provider

# Setup logging
logger = logging.getLogger('code_searcher')
logger.setLevel(logging.INFO)

# Console handler - always active for MCP stderr output
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# File handler - only if enabled in environment
file_handler = None
if env_config.get_logging_file_enabled():
    # Create log file path
    log_file_path = env_config.get_logging_file_path()
    # Handle relative paths
    if not os.path.isabs(log_file_path):
        log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file_path)
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

# Add handlers only if they don't exist (to avoid duplicates)
if not logger.handlers:
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)


class CodebaseSearcher:
    """Handles search operations for codebase files."""
    
    def __init__(self, codebase_path: str, use_ai_filter: bool = False, db_name: str = 'codebase_files', db_base_path: str = None):
        """Initialize the searcher with codebase path.
        
        Args:
            codebase_path: Path to the codebase
            use_ai_filter: Whether to use AI filtering
            db_name: Name of the database/collection (default: 'codebase_files')
            db_base_path: Optional base path for database storage (default: DBs/<db_name>)
        """
        self.codebase_path = Path(codebase_path)
        if not self.codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {codebase_path}")
        
        self.db_name = db_name
        
        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider()
        
        # Initialize ChromaDB
        if db_base_path:
            # Use custom base path
            db_path = os.path.join(db_base_path, db_name)
        else:
            # Use default path
            db_path = os.path.join(os.path.dirname(__file__), '..', 'DBs', db_name)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=db_name)
        except:
            # Collection doesn't exist - database needs to be created first
            raise RuntimeError(
                f"Database '{db_name}' not initialized. Please run 'update_db()' first to create and populate the database."
            )
        
        # Initialize AI filter based on parameter
        self.ai_filter = None
        if use_ai_filter:
            try:
                model = env_config.get_ai_filter_model()
                timeout = env_config.get_ai_filter_timeout_seconds()
                self.ai_filter = AIFilter(model=model, timeout_seconds=timeout)
                logger.info(f"AI filter initialized with model: {model}")
            except RuntimeError as e:
                logger.warning(f"AI filter not available: {e}")
                self.ai_filter = None
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.embedding_provider.embed_documents(texts)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for files matching the query.
        
        Pipeline:
        1. Semantic search for initial results
        2. VoyageAI reranker (if enabled)
        3. AI filtering (if enabled)
        4. Return top K results
        
        Args:
            query: Search query
            
        Returns:
            List of search results with file path and preview
        """
        
        try:
            # Use environment values
            semantic_n = env_config.get_semantic_search_n_results()
            max_results = env_config.get_max_results()
            
            if env_config.get_logging_verbose():
                logger.info(f"[SEARCHER] Starting search for: '{query}'")
                logger.info(f"[SEARCHER] Config: semantic_n={semantic_n}, max_results={max_results}, " +
                           f"reranker={env_config.get_reranker_enabled()}, " +
                           f"ai_filter={env_config.get_ai_filter_enabled()}")
            
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_query(query)
            
            # Check if we're in chunking mode by looking for chunk records
            test_data = self.collection.get(limit=1, where={"is_file_record": False})
            is_chunking_mode = bool(test_data['ids'])
            
            if is_chunking_mode and env_config.get_logging_verbose():
                logger.info("[SEARCHER] Chunking mode detected, using iterative search")
            
            # Format initial results and deduplicate by file
            formatted_results = []
            seen_files = set()
            
            if is_chunking_mode:
                # ITERATIVE SEARCH FOR CHUNKING MODE
                # Query more chunks to ensure we get enough unique files
                batch_size = semantic_n
                max_iterations = 10  # Safety limit
                iteration = 0
                
                while len(formatted_results) < semantic_n and iteration < max_iterations:
                    # Calculate how many more results we need
                    needed = semantic_n - len(formatted_results)
                    # Query with a buffer (assume average 3 chunks per file)
                    query_size = min(batch_size + needed * 3, 500)  # Cap at 500 for performance
                    
                    if env_config.get_logging_verbose():
                        logger.info(f"[SEARCHER] Iteration {iteration + 1}: querying {query_size} chunks, have {len(formatted_results)} files")
                    
                    # Query chunks (include documents for chunk text)
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=query_size,
                        where={"is_file_record": False},  # Only search chunks
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Process results
                    if results['ids'] and results['ids'][0]:
                        for i, chunk_id in enumerate(results['ids'][0]):
                            # Get file path from metadata
                            metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                            file_path = metadata.get('file_path', chunk_id)
                            
                            # Skip if we already have this file
                            if file_path in seen_files:
                                continue
                            seen_files.add(file_path)
                            
                            # Get full file content from file record
                            file_id = f"file:{file_path}"
                            try:
                                file_data = self.collection.get(
                                    ids=[file_id],
                                    include=['documents']
                                )
                                if file_data['documents']:
                                    document_content = file_data['documents'][0]
                                else:
                                    # File record not found, skip this result
                                    logger.warning(f"[SEARCHER] File record not found for {file_path}")
                                    continue
                            except:
                                logger.warning(f"[SEARCHER] Error fetching file record for {file_path}")
                                continue
                            
                            # Get chunk text from the query results
                            chunk_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ""

                            formatted_results.append({
                                'path': file_path,
                                'content': document_content,
                                'chunk_text': chunk_text,  # Store chunk text for reranker
                                'score': results['distances'][0][i] if results['distances'] else None
                            })
                            
                            # Stop if we have enough results
                            if len(formatted_results) >= semantic_n:
                                break
                    
                    # If no more results available, stop
                    if not results['ids'] or not results['ids'][0] or len(results['ids'][0]) < query_size:
                        if env_config.get_logging_verbose():
                            logger.info(f"[SEARCHER] No more results available, stopping at {len(formatted_results)} files")
                        break
                    
                    iteration += 1
                    batch_size = min(batch_size * 2, 200)  # Increase batch size for next iteration
                
            else:
                # NON-CHUNKING MODE - original logic
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=semantic_n,
                    where={"is_file_record": False},  # Only search chunks, not files
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['ids'] and results['ids'][0]:
                    for i, chunk_id in enumerate(results['ids'][0]):
                        # Get file path from metadata
                        metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                        file_path = metadata.get('file_path', chunk_id)
                        
                        # Skip if we already have this file
                        if file_path in seen_files:
                            continue
                        seen_files.add(file_path)
                        
                        # Get full file content from file record
                        file_id = f"file:{file_path}"
                        try:
                            file_data = self.collection.get(
                                ids=[file_id],
                                include=['documents']
                            )
                            if file_data['documents']:
                                document_content = file_data['documents'][0]
                            else:
                                # Fallback to chunk content if file record not found
                                document_content = results['documents'][0][i] if results['documents'] and results['documents'][0] else 'No content available'
                        except:
                            # Fallback to chunk content
                            document_content = results['documents'][0][i] if results['documents'] and results['documents'][0] else 'No content available'

                        # Get chunk text from query results
                        chunk_text = results['documents'][0][i] if results['documents'] and results['documents'][0] else ""

                        formatted_results.append({
                            'path': file_path,
                            'content': document_content,
                            'chunk_text': chunk_text,  # Store chunk text for reranker
                            'score': results['distances'][0][i] if results['distances'] else None
                        })
            
            if env_config.get_logging_verbose():
                logger.info(f"[SEARCHER] Semantic search found {len(formatted_results)} results")
            
            # Step 2: Apply reranker if enabled
            if env_config.get_reranker_enabled() and formatted_results:
                reranked_results = self._apply_reranker(query, formatted_results)
                if env_config.get_logging_verbose():
                    logger.info(f"[SEARCHER] Reranker: {len(formatted_results)} -> {len(reranked_results)} results")
                formatted_results = reranked_results
            
            # Step 3: Apply AI filtering if available
            if self.ai_filter and formatted_results:
                if env_config.get_logging_verbose():
                    logger.info(f"[SEARCHER] Applying AI filter to {len(formatted_results)} results")
                
                filtered_results = self.ai_filter.filter_search_results(
                    query, 
                    formatted_results,
                    return_all_with_scores=False
                )
                
                if env_config.get_logging_verbose():
                    logger.info(f"[SEARCHER] AI filter returned {len(filtered_results)} results")
                
                formatted_results = filtered_results
            
            # Step 4: Limit to max_results
            if len(formatted_results) > max_results:
                formatted_results = formatted_results[:max_results]
                if env_config.get_logging_verbose():
                    logger.info(f"[SEARCHER] Limited to top {max_results} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _apply_reranker(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply VoyageAI reranker to filter and reorder results."""
        if not results:
            return results
        
        # Get reranker settings
        reranker_lines = env_config.get_preview_lines_reranker()
        use_chunks = env_config.get_reranker_use_chunks()

        # Prepare documents for reranking
        documents = []
        for result in results:
            if use_chunks and result.get('chunk_text'):
                # Use chunk text directly (from vector search)
                doc_text = f"File: {result['path']}\n{result['chunk_text']}"
                if env_config.get_logging_verbose():
                    logger.debug(f"[RERANKER] Using chunk text ({len(result['chunk_text'])} chars) for {result['path']}")
            else:
                # Truncate full content for reranker
                truncated_content = self._truncate_preview(result['content'], reranker_lines)
                doc_text = f"File: {result['path']}\n{truncated_content}"
            documents.append(doc_text)
        
        try:
            # Check if we're using VoyageAI provider (reranker currently only supported with VoyageAI)
            if env_config.get_embedding_provider() != 'voyage':
                logger.info("[RERANKER] Reranker only supported with VoyageAI provider, skipping")
                return results
            
            # Get instructions for instruction-following capability
            instructions = env_config.get_reranker_instructions()
            
            # Form query with instructions if provided
            if instructions:
                enhanced_query = f"{instructions}\n\n{query}"
                if env_config.get_logging_verbose():
                    logger.info(f"[RERANKER] Using instructions")
            else:
                enhanced_query = query
            
            # Use VoyageAI reranker
            import voyageai
            voyage_client = voyageai.Client()
            reranking = voyage_client.rerank(
                query=enhanced_query, 
                documents=documents, 
                model=env_config.get_reranker_model()
            )
            
            # Filter by threshold and reorder
            threshold = env_config.get_reranker_threshold()
            reranked_results = []
            
            for rank_result in reranking.results:
                if rank_result.relevance_score >= threshold:
                    # Get original result and add rerank score
                    original_result = results[rank_result.index].copy()
                    original_result['rerank_score'] = rank_result.relevance_score
                    reranked_results.append(original_result)
            
            # Sort by rerank score (highest first)
            reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Simplified logging
            if env_config.get_logging_verbose() and reranking.results:
                scores = [r.relevance_score for r in reranking.results]
                logger.info(f"[RERANKER] Scores range: {min(scores):.3f} - {max(scores):.3f}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"[RERANKER] Error: {e}")
            raise  # Propagate error - user should see configuration problems
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        try:
            # Count only unique files, not chunks
            # First try to get file records
            try:
                collection_data = self.collection.get(
                    where={"is_file_record": True}
                )
                total_files = len(collection_data['ids']) if collection_data['ids'] else 0
            except:
                # Fallback: count unique file paths from all records
                collection_data = self.collection.get()
                unique_files = set()
                if collection_data.get('metadatas'):
                    for metadata in collection_data['metadatas']:
                        if metadata and 'file_path' in metadata:
                            unique_files.add(metadata['file_path'])
                total_files = len(unique_files)
            
            # Get last update time from status file - unique per database
            status_path = os.path.join(os.path.dirname(__file__), '..', 'status', f'last_update_{self.db_name}.json')
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    last_update = status.get('last_update_time', 'Never')
            else:
                last_update = 'Never'
            
            return {
                'total_files': total_files,
                'last_update': last_update,
                'db_size': f"{total_files} vectors"
            }
        except Exception as e:
            return {
                'total_files': 0,
                'last_update': 'Error',
                'db_size': str(e)
            }
    
    def _truncate_preview(self, content: str, max_lines: int) -> str:
        """
        Truncate content to specified number of lines.
        
        Args:
            content: Full content text
            max_lines: Maximum number of lines (-1 for unlimited)
            
        Returns:
            Truncated content text
        """
        if max_lines == -1:
            return content
        
        lines = content.split('\n')
        if len(lines) <= max_lines:
            return content
        
        # Truncate and add ellipsis
        return '\n'.join(lines[:max_lines]) + '\n...'