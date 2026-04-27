"""Database update and vectorization logic for code search."""
import fnmatch
import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import chromadb
from . import env_config
from .embedding_providers import create_embedding_provider
from .embedding_providers.base import ChunkedEmbeddingResult
from .embedding_providers.voyage import VoyageProvider
from .project_analyzer import ProjectAnalyzer

# Hybrid search imports (conditional - only used if hybrid search is enabled)
try:
    from .hybrid.bm25_search import BM25SearchEngine
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Setup logging
logger = logging.getLogger('code_searcher.db_updater')
logger.setLevel(logging.INFO)

# Set log level based on verbose setting
verbose = env_config.get_logging_verbose()

# Console handler - always active for MCP stderr output
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
console_formatter = logging.Formatter('[DB_UPDATE] %(message)s')
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
    file_formatter = logging.Formatter('%(asctime)s - [DB_UPDATE] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

# Add handlers only if they don't exist (to avoid duplicates)
if not logger.handlers:
    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)

# ALSO configure logging for embedding providers
voyage_logger = logging.getLogger('code_search_mcp.embedding_providers.voyage_context')
voyage_logger.setLevel(logging.INFO)
if not voyage_logger.handlers:
    voyage_logger.addHandler(console_handler)
    if file_handler:
        voyage_logger.addHandler(file_handler)

class DatabaseUpdater:
    """Handles database creation and updates with intelligent project analysis."""
    
    # Default file extensions to process
    DEFAULT_EXTENSIONS = {
        '.ts', '.tsx', '.js', '.jsx', '.py', '.md', '.mdx', 
        '.json', '.yml', '.yaml', '.sh', '.rs', '.go', '.java', '.rb'
    }
    
    # Default directories to exclude
    DEFAULT_EXCLUDE_DIRS = {
        'node_modules', '.git', 'dist', 'build', 'coverage', 
        '.next', 'out', 'target', '__pycache__', 'venv', '.venv',
        'env', '.env', 'vendor', '.cache', '.pytest_cache'
    }
    
    def __init__(self, codebase_path: str, db_name: str = 'codebase_files', db_base_path: str = None):
        """Initialize the database updater.
        
        Args:
            codebase_path: Path to the codebase to index
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
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        self.collection = None
        self.config = None
        
        # Status file path - unique per database
        self.status_path = Path(__file__).parent.parent / 'status' / f'update_status_{db_name}.json'
        self.status_path.parent.mkdir(exist_ok=True)
    
    def _update_status(self, message: str, progress: Optional[Dict] = None):
        """Update status file with current progress."""
        status = {
            'status': 'running',
            'message': message
        }
        if progress:
            status['progress'] = progress
        
        try:
            with open(self.status_path, 'w') as f:
                json.dump(status, f)
        except:
            pass  # Don't fail the update if status can't be written
    
    def analyze_project(self) -> Dict:
        """Analyze project structure and detect frameworks."""
        logger.info("Analyzing project structure...")
        self._update_status("Analyzing project structure...")
        
        # Use ProjectAnalyzer for the actual analysis
        analyzer = ProjectAnalyzer(str(self.codebase_path))
        config = analyzer.analyze_project()
        
        return config
    
    def get_or_create_collection(self) -> Tuple[chromadb.Collection, bool]:
        """Get existing collection or create new one."""
        try:
            self.collection = self.chroma_client.get_collection(name=self.db_name)
            metadata = self.collection.metadata
            self.config = metadata.get('config', {})
            if isinstance(self.config, str):
                self.config = json.loads(self.config)
            return self.collection, True
        except:
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.db_name,
                metadata={"hnsw:space": "cosine"}
            )
            return self.collection, False
    
    def calculate_file_hash(self, content: str) -> str:
        """Calculate MD5 hash of file content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def should_process_file(self, file_path: Path, config: Dict) -> bool:
        """Check if file should be processed based on configuration."""
        # Check extension
        if file_path.suffix not in config['extensions']:
            logger.debug(f"[FILTER] Skipping {file_path.name}: extension {file_path.suffix} not in {config['extensions']}")
            return False
        
        # Check exclude directories - only check in path parts, not in filename
        path_parts = file_path.parts[:-1]  # Exclude the filename itself
        for exclude_dir in config['exclude_dirs']:
            if any(exclude_dir in part or part == exclude_dir for part in path_parts):
                logger.info(f"[FILTER] Skipping {file_path.name}: excluded directory '{exclude_dir}' in path")
                return False
        
        # Check exclude patterns with proper glob matching
        for pattern in config.get('exclude_patterns', []):
            # Use fnmatch for glob-style pattern matching
            if fnmatch.fnmatch(file_path.name, pattern):
                return False
        
        return True
    
    def scan_files(self, config: Dict, specific_files: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Scan codebase and return file information.
        
        Args:
            config: Configuration dictionary
            specific_files: Optional list of relative file paths to process (instead of scanning all)
        """
        files_data = {}
        self._update_status("Scanning files...")
        
        if specific_files:
            # Process only specific files
            all_files = []
            for rel_path in specific_files:
                file_path = self.codebase_path / rel_path
                if file_path.exists() and file_path.is_file():
                    all_files.append(file_path)
                else:
                    # Try to find the file by name if full path doesn't work
                    possible_files = list(self.codebase_path.rglob(Path(rel_path).name))
                    if possible_files:
                        all_files.append(possible_files[0])
                    else:
                        logger.warning(f"File not found: {rel_path}")
            total = len(all_files)
            processed = 0
            
            for file_path in all_files:
                # For specific files, skip extension/exclude checks
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        file_hash = self.calculate_file_hash(content)
                        relative_path = str(file_path.relative_to(self.codebase_path))
                        
                        files_data[str(file_path)] = {
                            'path': str(file_path),
                            'relative_path': relative_path,
                            'content': content,
                            'file_hash': file_hash,
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        processed += 1
                        if processed % 10 == 0:
                            self._update_status(f"Scanning files: {processed}/{total}", 
                                              {'current': processed, 'total': total})
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        else:
            # OPTIMIZED: Use os.walk with directory pruning instead of rglob('*')
            import os
            
            # Track progress
            processed = 0
            scanned_dirs = 0
            skipped_dirs = 0
            
            # Get exclude directories for efficient filtering
            exclude_dirs = set(config.get('exclude_dirs', []))
            
            # Walk the filesystem with pruning
            for root, dirs, files in os.walk(self.codebase_path, followlinks=False):
                root_path = Path(root)
                
                # Prune directories before entering them
                # This modifies dirs in-place to skip excluded directories
                original_dir_count = len(dirs)
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
                skipped_dirs += original_dir_count - len(dirs)
                scanned_dirs += 1
                
                # Log progress for large projects
                if scanned_dirs % 100 == 0:
                    logger.debug(f"Scanned {scanned_dirs} directories, skipped {skipped_dirs} excluded dirs")
                
                # Process files in current directory
                for file_name in files:
                    file_path = root_path / file_name
                    
                    # Check if file should be processed
                    if self.should_process_file(file_path, config):
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            file_hash = self.calculate_file_hash(content)
                            relative_path = str(file_path.relative_to(self.codebase_path))
                            
                            files_data[str(file_path)] = {
                                'path': str(file_path),
                                'relative_path': relative_path,
                                'content': content,
                                'file_hash': file_hash,
                                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            }
                            
                            processed += 1
                            if processed % 10 == 0:
                                self._update_status(f"Scanning files: {processed} found", 
                                                  {'current': processed})
                        except Exception as e:
                            logger.warning(f"Error processing {file_path}: {e}")
            
            logger.info(f"Scan complete: {processed} files found, {scanned_dirs} dirs scanned, {skipped_dirs} dirs skipped")
        
        return files_data
    
    def vectorize_files(self, files_data: Dict[str, Dict], batch_size: int = 10) -> int:
        """Vectorize files and store in ChromaDB with chunking support."""
        files_list = list(files_data.values())
        total_processed = 0
        total_chunks_created = 0
        
        self._update_status(f"Vectorizing {len(files_list)} files...")
        logger.info(f"Starting vectorization of {len(files_list)} files in batches of {batch_size}")
        
        # Get vectorization lines setting
        vectorization_lines = env_config.get_preview_lines_vectorization()
        logger.info(f"Using first {vectorization_lines} lines for embeddings")
        
        # Check if provider supports chunking
        supports_chunking = hasattr(self.embedding_provider, 'embed_documents_with_metadata')
        logger.info(f"Provider supports chunking: {supports_chunking}")
        
        for i in range(0, len(files_list), batch_size):
            batch = files_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files_list) + batch_size - 1) // batch_size
            
            self._update_status(f"Vectorizing batch {batch_num}/{total_batches}",
                              {'current': total_processed, 'total': len(files_list)})
            
            # Prepare texts for vectorization
            texts = []
            for file_info in batch:
                content = file_info['content']
                if vectorization_lines != -1:
                    lines = content.split('\n')
                    content = '\n'.join(lines[:vectorization_lines])
                # VoyageAI doesn't accept empty strings, use placeholder for empty files
                if not content or content.strip() == '':
                    content = "[Empty file]"
                texts.append(content)
            
            # Generate embeddings
            try:
                logger.info(f"Generating embeddings for batch {batch_num}/{total_batches}")
                logger.info(f"[DEBUG] Provider type: {type(self.embedding_provider).__name__}")
                logger.info(f"[DEBUG] Provider module: {self.embedding_provider.__module__}")
                logger.info(f"[DEBUG] Number of texts in batch: {len(texts)}")
                
                # Log text sizes
                for i, text in enumerate(texts):
                    text_size = len(text)
                    from code_search_mcp.embedding_providers.utils import count_tokens
                    text_tokens = count_tokens(text)
                    logger.info(f"[DEBUG] Text {i} in batch: {text_size} chars, {text_tokens} tokens")
                
                if supports_chunking:
                    # Use new chunking-aware method
                    logger.info(f"[DEBUG] Calling embed_documents_with_metadata with {len(texts)} texts")
                    
                    # Inspect the method source
                    import inspect
                    try:
                        method_source = inspect.getsource(self.embedding_provider.embed_documents_with_metadata)
                        # Log first 200 chars to see which method is being called
                        logger.info(f"[INSPECT] Method source preview: {method_source[:200]}...")
                        # Check if it contains _process_large_document
                        if "_process_large_document" in method_source:
                            logger.info(f"[INSPECT] Method DOES contain _process_large_document call")
                        else:
                            logger.info(f"[INSPECT] Method DOES NOT contain _process_large_document call")
                    except Exception as e:
                        logger.info(f"[INSPECT] Error getting method source: {e}")
                    
                    result = self.embedding_provider.embed_documents_with_metadata(texts)
                    logger.info(f"[DEBUG] Result: {len(result.embeddings)} embeddings, {len(result.chunks)} chunks")
                    
                    # Process results with chunking support
                    all_ids = []
                    all_embeddings = []
                    all_documents = []
                    all_metadatas = []
                    
                    embedding_idx = 0
                    for file_idx, file_info in enumerate(batch):
                        # First, store the full file once (without embedding)
                        file_id = f"file:{file_info['path']}"
                        all_ids.append(file_id)
                        all_embeddings.append([0.0] * self.embedding_provider.get_dimension())  # Placeholder embedding
                        all_documents.append(file_info['content'])  # Full file content
                        all_metadatas.append({
                            'file_hash': file_info['file_hash'],
                            'relative_path': file_info['relative_path'],
                            'last_modified': file_info['last_modified'],
                            'chunk_type': 'file',
                            'is_file_record': True,  # Mark as file record
                            'file_path': file_info['path']
                        })
                        
                        # Now add chunks with their embeddings
                        while embedding_idx < len(result.metadata):
                            meta = result.metadata[embedding_idx]
                            if meta['doc_index'] != file_idx:
                                break
                            
                            # Generate chunk ID
                            chunk_id = f"chunk:{file_info['path']}#{meta['chunk_index']}"
                            
                            all_ids.append(chunk_id)
                            all_embeddings.append(result.embeddings[embedding_idx])
                            # Store chunk text for reranker use
                            # result.chunks contains text for each embedding in same order
                            chunk_text = result.chunks[embedding_idx] if result.chunks else ""
                            all_documents.append(chunk_text)
                            
                            # Prepare metadata with reference to file
                            chunk_metadata = {
                                'file_hash': file_info['file_hash'],
                                'relative_path': file_info['relative_path'],
                                'last_modified': file_info['last_modified'],
                                'chunk_type': meta['type'],
                                'chunk_index': meta.get('chunk_index', 0),
                                'total_chunks': meta.get('total_chunks', 1),
                                'chunk_tokens': meta.get('tokens', 0),
                                'file_path': file_info['path'],  # Reference to file
                                'is_file_record': False  # Mark as chunk record
                            }
                            all_metadatas.append(chunk_metadata)
                            
                            total_chunks_created += 1
                            embedding_idx += 1
                    
                    # Add all chunks to ChromaDB in batches (ChromaDB has a limit of 5461 per batch)
                    if all_ids:
                        chromadb_batch_size = 5000  # Safe limit below ChromaDB's max of 5461
                        for idx in range(0, len(all_ids), chromadb_batch_size):
                            batch_end = min(idx + chromadb_batch_size, len(all_ids))
                            self.collection.add(
                                ids=all_ids[idx:batch_end],
                                embeddings=all_embeddings[idx:batch_end],
                                documents=all_documents[idx:batch_end],
                                metadatas=all_metadatas[idx:batch_end]
                            )
                            logger.info(f"Added batch {idx//chromadb_batch_size + 1}: {batch_end - idx} items (total: {batch_end}/{len(all_ids)})")
                        logger.info(f"Total added {len(all_ids)} chunks/files to database")
                    
                else:
                    # Fallback to old method (no chunking support)
                    embeddings = self.embedding_provider.embed_documents(texts)
                    
                    # Prepare data for ChromaDB  
                    ids = [f"file:{f['path']}" for f in batch]
                    documents = [f['content'] for f in batch]
                    metadatas = [
                        {
                            'file_hash': f['file_hash'],
                            'relative_path': f['relative_path'],
                            'last_modified': f['last_modified'],
                            'chunk_type': 'full_file',
                            'chunk_index': 0,
                            'total_chunks': 1,
                            'file_path': f['path']
                        }
                        for f in batch
                    ]
                    
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    total_chunks_created += len(ids)
                
                total_processed += len(batch)
                logger.info(f"✓ Batch {batch_num}/{total_batches} complete. {total_processed}/{len(files_list)} files, {total_chunks_created} chunks")
                
            except Exception as e:
                logger.error(f"✗ Error vectorizing batch {batch_num}: {e}")
                raise
        
        logger.info(f"Vectorization complete: {total_processed} files, {total_chunks_created} chunks created")
        return total_processed
    
    def incremental_update(self, config: Dict, specific_files: Optional[List[str]] = None) -> Dict:
        """Perform incremental update of the database with chunk support.
        
        Args:
            config: Configuration dictionary
            specific_files: Optional list of relative file paths to process
        """
        logger.info("Performing incremental update...")
        self._update_status("Checking for file changes...")
        
        # Get current files
        current_files = self.scan_files(config, specific_files)
        logger.info(f"Found {len(current_files)} files in codebase")
        
        # Get existing files from database
        existing_data = self.collection.get()
        existing_files = {}
        existing_ids_by_path = {}  # Map file paths to all their IDs (including chunks)
        
        if existing_data['ids']:
            for i, entry_id in enumerate(existing_data['ids']):
                metadata = existing_data['metadatas'][i] if existing_data['metadatas'] else {}
                
                # Extract file path from ID (handles both "file:path" and "chunk:path#n")
                if entry_id.startswith('file:'):
                    file_path = entry_id[5:]  # Remove "file:" prefix
                elif entry_id.startswith('chunk:'):
                    # Remove "chunk:" prefix and chunk number
                    file_path = entry_id[6:].split('#')[0]
                else:
                    # Legacy format - just the path
                    file_path = entry_id
                
                # Track all IDs for each file path
                if file_path not in existing_ids_by_path:
                    existing_ids_by_path[file_path] = []
                existing_ids_by_path[file_path].append(entry_id)
                
                # Store file info (only once per file)
                if file_path not in existing_files:
                    existing_files[file_path] = {
                        'file_hash': metadata.get('file_hash', ''),
                        'relative_path': metadata.get('relative_path', '')
                    }
        
        logger.info(f"Found {len(existing_files)} unique files in database")
        logger.info(f"Total entries (including chunks): {len(existing_data['ids']) if existing_data['ids'] else 0}")
        
        # Detect changes
        current_paths = set(current_files.keys())
        existing_paths = set(existing_files.keys())
        
        # Check which existing files should now be excluded
        excluded_files = set()
        for file_path in existing_paths:
            # Check if this file should still be processed according to new config
            path_obj = self.codebase_path / file_path
            if path_obj.exists() and not self.should_process_file(path_obj, config):
                excluded_files.add(file_path)
        
        if excluded_files:
            logger.info(f"Found {len(excluded_files)} files that are now excluded by configuration")
        
        new_files = current_paths - existing_paths
        deleted_files = existing_paths - current_paths
        
        # Check for modified files
        modified_files = set()
        for path in current_paths & existing_paths:
            if current_files[path]['file_hash'] != existing_files[path]['file_hash']:
                modified_files.add(path)
        
        stats = {
            'new': len(new_files),
            'modified': len(modified_files),
            'deleted': len(deleted_files),
            'excluded': len(excluded_files)
        }
        
        logger.info(f"Changes detected: {stats['new']} new, {stats['modified']} modified, {stats['deleted']} deleted, {stats['excluded']} excluded")
        
        # Delete removed, modified, and now-excluded files (including all their chunks)
        files_to_delete = deleted_files | modified_files | excluded_files
        if files_to_delete:
            # Collect all IDs to delete (file and chunk IDs)
            ids_to_delete = []
            for file_path in files_to_delete:
                if file_path in existing_ids_by_path:
                    ids_to_delete.extend(existing_ids_by_path[file_path])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(files_to_delete)} files ({len(ids_to_delete)} total entries) from database")
        
        # Process new and modified files
        files_to_process = {}
        for path in new_files | modified_files:
            files_to_process[path] = current_files[path]
        
        if files_to_process:
            processed = self.vectorize_files(files_to_process)
            logger.info(f"Added/updated {processed} files")
        
        return stats
    
    def full_vectorization(self, config: Dict, specific_files: Optional[List[str]] = None) -> int:
        """Perform full vectorization of the codebase.
        
        Args:
            config: Configuration dictionary
            specific_files: Optional list of relative file paths to process
        """
        logger.info("Performing full vectorization...")
        if specific_files:
            logger.info(f"Processing {len(specific_files)} specific files")
        self._update_status("Starting full vectorization...")
        
        # Scan all files
        files_data = self.scan_files(config, specific_files)
        logger.info(f"Found {len(files_data)} files to process")
        self._update_status(f"Found {len(files_data)} files to vectorize")
        
        # Vectorize and store
        total = self.vectorize_files(files_data)
        
        return total
    
    def update_database(self, analyze: bool = False, specific_files: Optional[List[str]] = None) -> str:
        """Main method to update or create the database.
        
        Args:
            analyze: DEPRECATED - configuration should be created with set_config()
            specific_files: Optional list of relative file paths to process (for testing)
        """
        logger.info("=" * 50)
        logger.info("Starting database update")
        logger.info(f"Codebase path: {self.codebase_path}")
        logger.info(f"Options: specific_files={len(specific_files) if specific_files else 'None'}")
        
        # Get or create collection
        collection, exists = self.get_or_create_collection()
        
        if self.config:
            # Use existing configuration
            config = self.config
            logger.info("Using existing configuration from database")
        else:
            # No configuration found - create default configuration for new database
            if analyze:
                # Try to analyze project (requires ProjectAnalyzer)
                try:
                    from .project_analyzer import ProjectAnalyzer
                    analyzer = ProjectAnalyzer(str(self.codebase_path))
                    config = analyzer.analyze_project()
                    logger.info(f"Project analysis complete. Detected: {config.get('detected_frameworks', ['generic'])}")
                except ImportError:
                    logger.warning("ProjectAnalyzer not available, using default configuration")
                    config = None
            
            if not analyze or config is None:
                # Use default configuration
                config = {
                    'extensions': list(self.DEFAULT_EXTENSIONS),
                    'exclude_dirs': list(self.DEFAULT_EXCLUDE_DIRS),
                    'exclude_patterns': ['*.test.*', '*.spec.*'],
                    'project_type': 'default',
                    'analysis_date': datetime.now().isoformat()
                }
                logger.info("Using default configuration")
        
        logger.info(f"Extensions: {config['extensions']}")
        logger.info(f"Exclude dirs: {config['exclude_dirs'][:10]}")  # Log first 10
        
        # Update collection metadata with configuration
        self.collection.modify(metadata={'config': json.dumps(config)})
        
        # Prepare result message
        result = []
        result.append("=== Database Update Report ===\n")
        
        if not exists:
            logger.info("Database does not exist. Starting full vectorization...")
            result.append("Creating new database...\n")
            result.append(f"Project type detected: {', '.join(config.get('detected_frameworks', ['generic']))}")
            result.append(f"Extensions to index: {', '.join(config['extensions'])}")
            result.append(f"Excluding: {', '.join(config['exclude_dirs'][:5])}...")
            result.append("")
            
            # Perform full vectorization
            total = self.full_vectorization(config, specific_files)
            result.append(f"✓ Successfully indexed {total} files")
            logger.info(f"Full vectorization complete. Indexed {total} files")
            
        else:
            logger.info("Database exists. Starting incremental update...")
            # Perform incremental update
            stats = self.incremental_update(config, specific_files)
            
            result.append("Incremental update completed:")
            result.append(f"• New files: {stats['new']}")
            result.append(f"• Modified files: {stats['modified']}")
            result.append(f"• Deleted files: {stats['deleted']}")
            result.append(f"• Excluded files removed: {stats.get('excluded', 0)}")
            
            total_changes = stats['new'] + stats['modified'] + stats['deleted'] + stats.get('excluded', 0)
            if total_changes == 0:
                result.append("\n✓ Database is already up to date")
                logger.info("No changes detected. Database is up to date")
            else:
                result.append(f"\n✓ Updated {total_changes} files")
                logger.info(f"Incremental update complete. Updated {total_changes} files")
        
        # Add current statistics
        collection_stats = self.collection.count()
        # Count actual unique files (not chunks)
        all_data = self.collection.get()
        unique_files = set()
        for i, entry_id in enumerate(all_data['ids']):
            if entry_id.startswith('file:'):
                unique_files.add(entry_id[5:])  # Remove 'file:' prefix
            elif entry_id.startswith('chunk:'):
                # Extract file path from chunk ID
                file_path = entry_id[6:].split('#')[0]  # Remove 'chunk:' and chunk number
                unique_files.add(file_path)
        
        actual_file_count = len(unique_files)
        result.append(f"\nTotal files indexed: {actual_file_count}")
        result.append(f"Total database entries: {collection_stats} (includes chunks)")
        logger.info(f"Total files indexed: {actual_file_count}, database entries: {collection_stats}")
        
        # Mark database as vectorized
        metadata = self.collection.metadata
        metadata['vectorized'] = 'true'
        self.collection.modify(metadata=metadata)
        
        # HYBRID SEARCH: Build BM25 index if enabled
        if env_config.get_hybrid_search_enabled() and HYBRID_AVAILABLE:
            try:
                logger.info("Building BM25 index for hybrid search...")
                self._update_status("Building BM25 index...")
                
                if self._build_bm25_index(specific_files):
                    result.append("\n✓ BM25 index built successfully")
                    logger.info("BM25 index built successfully")
                else:
                    result.append("\n⚠ BM25 index build failed")
                    logger.warning("BM25 index build failed")
            except Exception as e:
                logger.error(f"BM25 indexing error: {e}")
                result.append(f"\n⚠ BM25 indexing error: {e}")
        elif env_config.get_hybrid_search_enabled():
            result.append("\n⚠ Hybrid search enabled but hybrid modules not available")
            logger.warning("Hybrid search enabled but hybrid modules not available")
        
        logger.info("Database update completed successfully")
        logger.info("=" * 50)
        
        return "\n".join(result)
    
    def _build_bm25_index(self, specific_files: Optional[List[str]] = None) -> bool:
        """
        Build BM25 index for hybrid search.
        
        Args:
            specific_files: Optional list of specific files to index
            
        Returns:
            True if successful, False otherwise
        """
        if not HYBRID_AVAILABLE:
            logger.error("Hybrid search modules not available")
            return False
        
        try:
            # Initialize BM25 engine with configuration
            bm25_engine = BM25SearchEngine(
                k1=env_config.get_bm25_k1_parameter(),
                b=env_config.get_bm25_b_parameter(),
                language=env_config.get_bm25_language(),
                remove_stopwords=env_config.get_bm25_remove_stopwords(),
                min_token_length=env_config.get_bm25_min_token_length(),
                use_stemming=env_config.get_bm25_use_stemming()
            )
            
            # Get documents from ChromaDB to index with BM25
            documents = self._extract_documents_for_bm25(specific_files)
            
            if not documents:
                logger.warning("No documents found for BM25 indexing")
                return True  # Not an error, just nothing to index
            
            logger.info(f"Building BM25 index for {len(documents)} documents")
            
            # Add documents to BM25 index
            for doc in documents:
                bm25_engine.add_document(
                    document_id=doc['id'],
                    content=doc['content'],
                    content_preview=doc.get('preview', doc['content'][:500])
                )
            
            # Save BM25 index to disk
            bm25_index_path = self._get_bm25_index_path()
            bm25_engine.save_index(bm25_index_path)
            
            logger.info(f"BM25 index saved to {bm25_index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _extract_documents_for_bm25(self, specific_files: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Extract documents from ChromaDB for BM25 indexing.
        
        Args:
            specific_files: Optional list of specific files to extract
            
        Returns:
            List of document dictionaries with 'id', 'content', and 'preview' keys
        """
        documents = []
        
        try:
            # Get all data from collection
            if specific_files:
                # For specific files, get all documents and filter by file path later
                # This is needed because file paths might be stored as relative or absolute
                collection_data = self.collection.get(include=['documents', 'metadatas'])
                logger.info(f"Filtering from {len(collection_data.get('ids', []))} documents for {len(specific_files)} specific files")
            else:
                collection_data = self.collection.get(include=['documents', 'metadatas'])
            
            if not collection_data['ids']:
                logger.warning("No documents found in ChromaDB collection")
                return documents
            
            logger.info(f"Processing {len(collection_data['ids'])} records from ChromaDB")
            
            # Track files we've already processed (to avoid duplicates from chunks)
            processed_files = set()
            
            # Convert specific_files to normalized paths for comparison
            target_files = set()
            if specific_files:
                for file_path in specific_files:
                    # Normalize path by removing leading separators and converting to posix
                    normalized = Path(file_path).as_posix().lstrip('/')
                    target_files.add(normalized)
                logger.info(f"Target files: {list(target_files)[:5]}...")  # Log first 5 for debugging
            
            for i, doc_id in enumerate(collection_data['ids']):
                metadata = collection_data['metadatas'][i] if collection_data['metadatas'] else {}
                document_content = collection_data['documents'][i] if collection_data['documents'] else ""
                
                # For BM25, we want to index full file content, not chunks
                # So we look for file records or extract file path from chunk records
                file_path = metadata.get('file_path', '')
                
                if not file_path:
                    # Try to extract from document ID
                    if doc_id.startswith('file:'):
                        file_path = doc_id[5:]  # Remove 'file:' prefix
                    elif doc_id.startswith('chunk:'):
                        # Extract file path from chunk ID: "chunk:path#index"
                        file_path = doc_id.split('#')[0][6:]  # Remove 'chunk:' prefix and chunk index
                
                if not file_path:
                    continue
                
                # Normalize file_path for comparison
                normalized_file_path = Path(file_path).as_posix().lstrip('/')
                
                # If specific_files is provided, check if this file should be included
                if specific_files and normalized_file_path not in target_files:
                    # Also check if any target file ends with this path (for relative matching)
                    matches = any(target.endswith(normalized_file_path) or normalized_file_path.endswith(target) 
                                for target in target_files)
                    if not matches:
                        continue
                
                if file_path in processed_files:
                    continue
                
                # For file records, use the document content directly
                # For chunk records, we need to get the full file content
                content = document_content
                
                # If this is a chunk record or the content is empty, try to read the full file
                if (not content or doc_id.startswith('chunk:')) and file_path:
                    try:
                        full_file_path = self.codebase_path / file_path
                        if full_file_path.exists():
                            content = full_file_path.read_text(encoding='utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        continue
                
                if content and file_path:
                    documents.append({
                        'id': file_path,  # Use file path as document ID for BM25
                        'content': content,
                        'preview': content[:500] + ('...' if len(content) > 500 else '')
                    })
                    processed_files.add(file_path)
            
            logger.info(f"Extracted {len(documents)} unique documents for BM25 indexing")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting documents for BM25: {e}")
            return []
    
    def _get_bm25_index_path(self) -> Path:
        """Get path for BM25 index storage."""
        # Use same base path as ChromaDB but with _bm25 suffix
        if hasattr(self, 'db_base_path'):
            db_base = getattr(self, 'db_base_path', None)
        else:
            db_base = None
            
        if db_base:
            base_path = Path(db_base)
        else:
            base_path = Path(os.path.dirname(__file__)) / '..' / 'DBs'
        
        bm25_db_name = f"{self.db_name}_bm25"
        bm25_path = base_path / bm25_db_name
        bm25_path.mkdir(parents=True, exist_ok=True)
        
        return bm25_path / 'index.pkl'