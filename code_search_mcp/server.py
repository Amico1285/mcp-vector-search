"""Code search MCP server implementation."""
import os
import json
import threading
import logging
import subprocess
from typing import Optional
from pathlib import Path
from datetime import datetime
from fastmcp import FastMCP
from . import env_config
from .project_analyzer import ProjectAnalyzer

# Setup logger
logger = logging.getLogger('code_searcher.server')

# Create MCP server instance
app = FastMCP("code-search")

# Lazy initialization of searcher
_searcher = None

def get_searcher():
    """Get or create the searcher instance."""
    global _searcher
    if _searcher is None:
        codebase_path = os.getenv('CODEBASE_PATH')
        if not codebase_path:
            raise ValueError("CODEBASE_PATH environment variable is required")
        
        # Get AI filter setting from environment
        use_ai_filter = env_config.get_ai_filter_enabled()
        
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Use HybridSearcher if hybrid search is enabled, otherwise use CodebaseSearcher
        if env_config.get_hybrid_search_enabled():
            try:
                from .hybrid import HybridSearcher
                _searcher = HybridSearcher(codebase_path, use_ai_filter=use_ai_filter, db_name=db_name)
                logger.info(f"[SERVER] Initialized HybridSearcher (BM25 + Vector search) with database '{db_name}'")
            except ImportError as e:
                logger.warning(f"[SERVER] HybridSearcher unavailable ({e}), falling back to CodebaseSearcher")
                from .searcher import CodebaseSearcher
                _searcher = CodebaseSearcher(codebase_path, use_ai_filter=use_ai_filter, db_name=db_name)
        else:
            from .searcher import CodebaseSearcher
            _searcher = CodebaseSearcher(codebase_path, use_ai_filter=use_ai_filter, db_name=db_name)
            logger.info(f"[SERVER] Initialized CodebaseSearcher (Vector search only) with database '{db_name}'")
    return _searcher


def _result_score_value(result):
    """Pick the most informative score for a result (rerank_score wins over semantic distance)."""
    if result.get('rerank_score') is not None:
        return result['rerank_score'], 'rerank'
    if result.get('score') is not None:
        return result['score'], 'semantic'
    return None, None


def _pipeline_budget_line(stats: dict) -> str:
    """One-line summary of how the search pipeline shrunk the candidate pool."""
    if not stats:
        return ""
    parts = [f"SEMANTIC_SEARCH_N_RESULTS={stats['semantic_n']} -> {stats['after_semantic']} candidates"]
    if stats['reranker_used']:
        parts.append(
            f"reranker (threshold {stats['reranker_threshold']:.2f}) kept {stats['after_reranker']}"
        )
    else:
        parts.append("reranker disabled")
    if stats['ai_filter_used']:
        parts.append(f"AI filter kept {stats['after_ai_filter']}")
    parts.append(f"MAX_RESULTS={stats['max_results']} -> returned {stats['returned']}")
    return "Pipeline: " + " | ".join(parts)


def _score_interpretation_line(stats: dict) -> str:
    """One-line guidance on how to read the score column."""
    use_rerank = stats and stats.get('reranker_used')
    if use_rerank:
        return (
            f"Score: reranker relevance (model: {env_config.get_reranker_model()}). "
            f"HIGHER = more relevant. Files below RERANKER_THRESHOLD are filtered out; "
            f"scores below ~0.50 are typically not relevant."
        )
    return "Score: semantic distance (no reranker). LOWER = more relevant. No fixed threshold."


def _truncate_chars(content: str, max_chars: int) -> str:
    """Cut content to max_chars, append ellipsis if truncated. -1 means no cut."""
    if max_chars == -1 or len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."


@app.tool()
def search_files(query: str) -> str:
    """
    Semantic file search. Search through codebase files using natural language queries.

    Always returns: a header (pipeline budget + score interpretation), then a
    numbered list of `path — score`. Per-file content preview is controlled by
    PREVIEW_CHARS_OUTPUT:
      - 0  -> no preview (agentic flow: read selected files via Read/Grep/Glob)
      - N  -> first N characters of each file
      - -1 -> entire file

    Args:
        query: Search query (e.g., "button component", "theme configuration")

    Examples:
        search_files("button dropdown component")
    """
    try:
        searcher = get_searcher()
        results = searcher.search(query)
        stats = getattr(searcher, 'last_search_stats', None)

        if not results:
            return f"No files found matching query: '{query}'"

        preview_chars = env_config.get_preview_chars_output()

        header = [f"Found {len(results)} candidate files for '{query}'."]
        pipeline_line = _pipeline_budget_line(stats)
        if pipeline_line:
            header.append(pipeline_line)
        header.append(_score_interpretation_line(stats))

        body = [""]
        for i, result in enumerate(results, 1):
            score_val, _ = _result_score_value(result)
            score_str = f"{score_val:.5f}" if score_val is not None else "n/a"
            if preview_chars == 0:
                body.append(f"{i}. {result['path']} — {score_str}")
            else:
                body.append(f"=== [{i}] {result['path']} — {score_str} ===")
                body.append(_truncate_chars(result.get('content', ''), preview_chars))
                body.append("")

        if preview_chars == 0:
            body.append("")
            body.append("Open the most relevant files via Read/Grep/Glob to inspect their content.")

        return "\n".join(header + body)

    except Exception as e:
        return f"Error searching files: {str(e)}"


@app.tool()
def get_server_info() -> str:
    """Get search server configuration and database statistics."""
    try:
        # Check for update status first
        db_name = env_config.get_db_name()
        status_path = Path(__file__).parent.parent / 'status' / f'update_status_{db_name}.json'
        update_status = None
        if status_path.exists():
            try:
                with open(status_path, 'r') as f:
                    update_status = json.load(f)
                    
                # Check if the process is still running
                if update_status.get('status') == 'running':
                    pid = update_status.get('pid')
                    if pid:
                        # Check if process exists
                        process_exists = False
                        try:
                            import psutil
                            process_exists = psutil.pid_exists(pid)
                        except ImportError:
                            # psutil not installed, use os.kill
                            import signal
                            try:
                                os.kill(pid, 0)
                                process_exists = True
                            except (OSError, ProcessLookupError, PermissionError):
                                process_exists = False
                        
                        if not process_exists:
                            # Process doesn't exist, clear stale status
                            logger.warning(f"[SERVER] Process {pid} not found, clearing stale status file")
                            status_path.unlink()
                            update_status = None
            except:
                pass
        
        # If update is running, show that first
        if update_status and update_status.get('status') == 'running':
            result = ["Database Update In Progress", "=" * 30]
            result.append(f"Status: {update_status.get('message', 'Processing...')}")
            
            progress = update_status.get('progress', {})
            if progress:
                current = progress.get('current', 0)
                total = progress.get('total', 0)
                if total > 0:
                    percentage = (current / total) * 100
                    result.append(f"Progress: {current}/{total} ({percentage:.1f}%)")
            
            result.append("\nUse get_server_info() to check for completion.")
            return "\n".join(result)
        
        # If update completed recently, show that
        if update_status and update_status.get('status') == 'completed':
            completion_message = "✓ Last update completed successfully\n\n"
        elif update_status and update_status.get('status') == 'error':
            completion_message = f"✗ Last update failed: {update_status.get('message', 'Unknown error')}\n\n"
        else:
            completion_message = ""
        
        # === Configuration Checks ===
        health_issues = []
        
        # 1. Check CODEBASE_PATH
        codebase_path = os.getenv('CODEBASE_PATH')
        codebase_status = "✓"
        if not codebase_path:
            codebase_status = "✗"
            health_issues.append("CODEBASE_PATH not set - set it in MCP configuration")
        elif codebase_path in ['/path/to/your/codebase', 'your-codebase-path', '/your/codebase/path']:
            codebase_status = "✗"
            health_issues.append("CODEBASE_PATH is still default - update it to your actual codebase path")
        elif not Path(codebase_path).exists():
            codebase_status = "✗"
            health_issues.append(f"CODEBASE_PATH does not exist: {codebase_path}")
        elif not Path(codebase_path).is_dir():
            codebase_status = "✗"
            health_issues.append(f"CODEBASE_PATH is not a directory: {codebase_path}")
        
        # 2. Check VoyageAI API key
        voyage_api_key = os.getenv('VOYAGE_API_KEY')
        voyage_status = "✓"
        voyage_message = "configured"
        
        if not voyage_api_key:
            voyage_status = "✗"
            voyage_message = "not set"
            health_issues.append("VOYAGE_API_KEY not set - add it to MCP configuration")
        elif voyage_api_key in ['your-voyage-api-key-here', 'your-key', 'YOUR_API_KEY']:
            voyage_status = "✗"
            voyage_message = "placeholder value"
            health_issues.append("VOYAGE_API_KEY is still placeholder - set your actual API key")
        else:
            # Test the API key with a minimal request
            try:
                import voyageai
                client = voyageai.Client()
                model = env_config.get_voyage_embedding_model()
                
                # Use appropriate method based on model
                if model == "voyage-context-3":
                    # Use contextualized_embed for voyage-context-3
                    result = client.contextualized_embed(
                        [["test"]],  # List of lists for contextualized API
                        model=model,
                        input_type="document"
                    )
                else:
                    # Use regular embed for other models
                    result = client.embed(
                        ["test"],
                        model=model,
                        input_type="document"
                    )
                voyage_message = "valid and working"
            except Exception as e:
                voyage_status = "✗"
                voyage_message = f"invalid or not working"
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    health_issues.append("VOYAGE_API_KEY is invalid - check your API key")
                elif "rate" in error_msg.lower():
                    voyage_message = "rate limited but valid"
                    voyage_status = "⚠"
                else:
                    health_issues.append(f"VoyageAI API error: {error_msg[:100]}")
        
        # 3. Check AI Filter (Claude CLI) if enabled
        ai_filter_status = ""
        ai_filter_message = ""
        if env_config.get_ai_filter_enabled():
            ai_filter_status = "✓"
            ai_filter_message = "enabled"
            # Check if claude command exists
            try:
                import subprocess
                result = subprocess.run(['claude', '--version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    ai_filter_message = "enabled and available"
                else:
                    ai_filter_status = "✗"
                    ai_filter_message = "enabled but Claude CLI not working"
                    health_issues.append("AI Filter enabled but Claude CLI not working - install Claude CLI or set AI_FILTER_ENABLED=false")
            except FileNotFoundError:
                ai_filter_status = "✗"
                ai_filter_message = "enabled but Claude CLI not found"
                health_issues.append("AI Filter enabled but Claude CLI not installed - install Claude CLI or set AI_FILTER_ENABLED=false")
            except subprocess.TimeoutExpired:
                ai_filter_status = "⚠"
                ai_filter_message = "enabled but Claude CLI slow to respond"
            except Exception as e:
                ai_filter_status = "✗"
                ai_filter_message = f"enabled but error: {str(e)[:50]}"
                health_issues.append(f"AI Filter check failed: {str(e)[:100]}")
        
        # Try to get database status
        try:
            searcher = get_searcher()
            stats = searcher.get_stats()
            db_status = f"Files Indexed: {stats.get('total_files', 0)}"
        except RuntimeError as e:
            db_status = "Database not initialized. Run update_db() first."
        except Exception as e:
            db_status = f"Database error: {str(e)[:50]}"
        
        # Get configuration details from environment
        reranker_enabled = env_config.get_reranker_enabled()
        reranker_threshold = env_config.get_reranker_threshold()
        max_results = env_config.get_max_results()
        
        # Build result
        result = [completion_message]
        
        # If there are health issues, show them prominently
        if health_issues:
            result.append("⚠️  Configuration Issues")
            result.append("=" * 30)
            for issue in health_issues:
                result.append(f"• {issue}")
            result.append("")
        
        result.append("Search Server Status")
        result.append("=" * 30)
        result.append(f"{codebase_status} Codebase Path: {codebase_path if codebase_path else 'Not configured'}")
        result.append(f"{voyage_status} VoyageAI API: {voyage_message}")
        if ai_filter_status:
            result.append(f"{ai_filter_status} AI Filter: {ai_filter_message}")
        result.append("")
        
        preview_chars = env_config.get_preview_chars_output()
        if preview_chars == 0:
            preview_label = "0 (paths + scores only — agentic flow)"
        elif preview_chars == -1:
            preview_label = "-1 (entire file)"
        else:
            preview_label = f"{preview_chars} chars"

        result.append("Configuration:")
        result.append(f"- Reranker: {'Enabled' if reranker_enabled else 'Disabled'}")
        result.append(f"- Reranker Threshold: {reranker_threshold}")
        result.append(f"- Max Results: {max_results}")
        result.append(f"- Preview Chars Output: {preview_label}")
        result.append("")
        
        result.append("Database:")
        result.append(f"- {db_status}")
        
        # Add warning if system not ready
        if health_issues:
            result.append("")
            result.append("[!] System not fully configured. Fix issues above before using.")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Server info unavailable: {str(e)}"


def _run_update_in_background(codebase_path: str, analyze_project: bool):
    """Background function to run database update."""
    # Make status file unique per database
    db_name = env_config.get_db_name()
    status_path = Path(__file__).parent.parent / 'status' / f'update_status_{db_name}.json'
    status_path.parent.mkdir(exist_ok=True)
    
    try:
        # Set initial status with PID and timestamp
        import os
        with open(status_path, 'w') as f:
            json.dump({
                'status': 'running',
                'message': 'Starting database update...',
                'pid': os.getpid(),
                'started_at': datetime.now().isoformat()
            }, f)
        
        # Import here to avoid circular imports
        from .database_updater import DatabaseUpdater
        
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Run the actual update
        updater = DatabaseUpdater(codebase_path, db_name=db_name)
        result = updater.update_database(analyze=analyze_project)
        
        # Set completed status
        with open(status_path, 'w') as f:
            json.dump({
                'status': 'completed',
                'message': result
            }, f)
            
    except Exception as e:
        # Set error status
        with open(status_path, 'w') as f:
            json.dump({
                'status': 'error',
                'message': f"Error: {str(e)}"
            }, f)


@app.tool()
def update_db() -> str:
    """
    Update vector database using the existing configuration.
    
    This will vectorize files according to the configuration created by set_config().
    If no configuration exists, you must first run set_config() to create one.
    
    The update process:
    - For new databases: Performs full vectorization of all matching files
    - For existing databases: Performs incremental updates (only changed files)
    - Automatically removes files that no longer match configuration
    - Preserves chunks and embeddings for unchanged files
    
    Before running this, use set_config() to:
    - Analyze your project structure
    - Configure which files to index
    - Preview what will be vectorized
    
    Examples:
        # First time setup:
        set_config(analyze=True)  # Analyze and create configuration
        update_db()               # Vectorize files
        
        # Incremental updates:
        update_db()  # Only updates changed files
    """
    try:
        codebase_path = os.getenv('CODEBASE_PATH')
        if not codebase_path:
            return "Error: CODEBASE_PATH environment variable is required"
        
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Check if configuration exists
        db_path = Path(__file__).parent.parent / 'DBs' / db_name
        if not db_path.exists():
            return """No configuration found!

Please create a configuration first:
  set_config(analyze=True)  # Auto-detect project structure
  
Or manually configure:
  set_config(extensions=".py,.js,.md")  # Specify file types"""
        
        # Check configuration in database
        import chromadb
        try:
            client = chromadb.PersistentClient(path=str(db_path))
            collection = client.get_collection(name=db_name)
            metadata = collection.metadata
            config = metadata.get('config', {})
            
            if not config:
                return """No configuration found in database!

Please create a configuration first:
  set_config(analyze=True)  # Auto-detect project structure"""
                
        except Exception as e:
            return f"""Database error: {str(e)}

Try creating a new configuration:
  set_config(analyze=True)"""
        
        # Check if update is already running
        db_name = env_config.get_db_name()
        status_path = Path(__file__).parent.parent / 'status' / f'update_status_{db_name}.json'
        if status_path.exists():
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    if status.get('status') == 'running':
                        return "Database update is already running. Use get_server_info() to check progress."
            except:
                pass
        
        # Start update in background thread - no analyze parameter needed
        thread = threading.Thread(
            target=_run_update_in_background,
            args=(codebase_path, False),  # Always use existing config
            daemon=True
        )
        thread.start()
        
        return """✓ Database update started in background

Use get_server_info() to check progress.
The update will continue even if this conversation ends."""
        
    except Exception as e:
        return f"Error starting database update: {str(e)}"


@app.tool()
def reset_db() -> str:
    """
    Reset the vector database by removing all stored embeddings and metadata.
    
    WARNING: This will delete all indexed data. You will need to run update_db() 
    to re-index your codebase after resetting.
    
    Use this when:
    - Changing embedding model (VOYAGE_EMBEDDING_MODEL)
    - Changing vectorization settings (PREVIEW_LINES_VECTORIZATION)
    - Database is corrupted or inconsistent
    - Testing different configurations
    
    Returns:
        Confirmation message or error
    
    Examples:
        reset_db()  # Reset the database
    """
    try:
        # Check if update is currently running
        db_name = env_config.get_db_name()
        status_path = Path(__file__).parent.parent / 'status' / f'update_status_{db_name}.json'
        if status_path.exists():
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    if status.get('status') == 'running':
                        return "❌ Cannot reset database while update is running. Please wait for it to complete."
            except:
                pass
        
        # Log the reset operation if verbose logging is enabled
        if env_config.get_logging_verbose():
            logger.info("[RESET_DB] Starting database reset operation")
        
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Get database path
        db_path = Path(__file__).parent.parent / 'DBs' / db_name
        
        # Check if database exists
        if not db_path.exists():
            return "No database found to reset."
        
        # Complete cleanup of ChromaDB
        import chromadb
        import shutil
        
        # Ensure directory exists
        db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use ChromaDB API to cleanly delete collection
            client = chromadb.PersistentClient(path=str(db_path))
            
            # List all collections
            collections = client.list_collections()
            deleted_count = 0
            
            # Delete each collection
            for collection in collections:
                try:
                    client.delete_collection(name=collection.name)
                    deleted_count += 1
                    if env_config.get_logging_verbose():
                        logger.info(f"[RESET_DB] Deleted collection: {collection.name}")
                except Exception as e:
                    if env_config.get_logging_verbose():
                        logger.warning(f"[RESET_DB] Could not delete collection {collection.name}: {e}")
            
            # Always clean up orphaned UUID directories after deleting collections
            if db_path.exists():
                orphaned_count = 0
                for item in db_path.iterdir():
                    if item.is_dir() and len(item.name) == 36:  # UUID format
                        try:
                            shutil.rmtree(item)
                            orphaned_count += 1
                            if env_config.get_logging_verbose():
                                logger.info(f"[RESET_DB] Removed orphaned directory: {item.name}")
                        except:
                            pass
                
                if orphaned_count > 0 and env_config.get_logging_verbose():
                    logger.info(f"[RESET_DB] Cleaned up {orphaned_count} orphaned directories")
            
            if env_config.get_logging_verbose():
                logger.info(f"[RESET_DB] Database reset completed. Deleted {deleted_count} collections")
                
        except Exception as e:
            if env_config.get_logging_verbose():
                logger.error(f"[RESET_DB] Failed to reset database: {e}")
            raise  # Re-raise to be caught by outer try-catch
        
        # Clear status files
        status_dir = Path(__file__).parent.parent / 'status'
        status_files_cleared = 0
        if status_dir.exists():
            for file in status_dir.glob('*.json'):
                try:
                    file.unlink()
                    status_files_cleared += 1
                except:
                    pass
        
        if status_files_cleared > 0 and env_config.get_logging_verbose():
            logger.info(f"[RESET_DB] Cleared {status_files_cleared} status files")
        
        # Clear the global searcher instance
        global _searcher
        _searcher = None
        
        if env_config.get_logging_verbose():
            logger.info("[RESET_DB] Reset operation completed successfully")
        
        return """✓ Database reset complete

All indexed data has been cleared.
Run update_db() to re-index your codebase."""
        
    except Exception as e:
        error_msg = f"Error resetting database: {str(e)}"
        if env_config.get_logging_verbose():
            logger.error(f"[RESET_DB] {error_msg}")
        return error_msg


@app.tool()
def get_config() -> str:
    """
    Get the current database configuration including file extensions and exclusions.
    
    Shows:
    - File extensions being indexed
    - Excluded directories
    - Excluded file patterns
    - Project type and detected frameworks
    
    Returns:
        Current configuration in readable format
    
    Examples:
        get_config()  # View current configuration
    """
    try:
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Check if database exists
        db_path = Path(__file__).parent.parent / 'DBs' / db_name
        if not db_path.exists():
            return "No database found. Run update_db() first to create a database."
        
        # Get configuration from ChromaDB
        import chromadb
        client = chromadb.PersistentClient(path=str(db_path))
        
        try:
            collection = client.get_collection(name=db_name)
            metadata = collection.metadata
            config = metadata.get('config', {})
            
            if isinstance(config, str):
                config = json.loads(config)
            
            if not config:
                return "No configuration found in database. Run update_db() to initialize."
            
            # Format configuration for display
            result = ["=== Current Database Configuration ===\n"]
            
            # Project info
            if 'detected_frameworks' in config:
                result.append(f"Project Type: {', '.join(config['detected_frameworks'])}")
            if 'analysis_date' in config:
                result.append(f"Last Analyzed: {config['analysis_date']}")
            result.append("")
            
            # Extensions
            result.append("File Extensions:")
            extensions = config.get('extensions', [])
            if extensions:
                # Group extensions nicely
                ext_line = ""
                for ext in sorted(extensions):
                    if len(ext_line) + len(ext) + 2 > 60:  # Start new line if too long
                        result.append(f"  {ext_line}")
                        ext_line = ext
                    else:
                        ext_line = f"{ext_line}, {ext}" if ext_line else ext
                if ext_line:
                    result.append(f"  {ext_line}")
            else:
                result.append("  (none)")
            result.append("")
            
            # Excluded directories
            result.append("Excluded Directories:")
            exclude_dirs = config.get('exclude_dirs', [])
            if exclude_dirs:
                for exc in sorted(exclude_dirs)[:15]:  # Show first 15
                    result.append(f"  • {exc}")
                if len(exclude_dirs) > 15:
                    result.append(f"  ... and {len(exclude_dirs) - 15} more")
            else:
                result.append("  (none)")
            result.append("")
            
            # Excluded patterns
            result.append("Excluded Patterns:")
            exclude_patterns = config.get('exclude_patterns', [])
            if exclude_patterns:
                for pattern in exclude_patterns:
                    result.append(f"  • {pattern}")
            else:
                result.append("  (none)")
            
            # Statistics
            result.append("")
            result.append("Database Statistics:")
            result.append(f"  Total Files: {collection.count()}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error reading configuration: {str(e)}"
            
    except Exception as e:
        return f"Error accessing database: {str(e)}"


@app.tool()
def set_config(
    extensions: Optional[str] = None,
    exclude_dirs: Optional[str] = None,
    exclude_patterns: Optional[str] = None,
    append: bool = False,
    analyze: bool = False
) -> str:
    """
    Create or update the database configuration for file indexing.
    
    If no configuration exists, this will create one. Use analyze=True to automatically
    detect your project structure and suggest optimal settings.
    
    WARNING: Changing configuration will trigger automatic cleanup on next update_db().
    Files that no longer match the configuration will be removed from the database.
    
    Args:
        extensions: Comma-separated file extensions to index (e.g., ".py,.js,.tsx")
        exclude_dirs: Comma-separated directories to exclude (e.g., "node_modules,build,dist")
        exclude_patterns: Comma-separated file patterns to exclude (e.g., "*.test.*,*.spec.*")
        append: If True, append to existing config instead of replacing
        analyze: If True, analyze project structure to auto-detect optimal configuration
    
    Returns:
        Configuration details including what will be indexed
    
    Examples:
        set_config(analyze=True)  # Auto-detect configuration for new project
        set_config(extensions=".py,.md,.txt")  # Index only specific file types
        set_config(exclude_dirs="tests,docs", append=True)  # Add more exclusions
    """
    try:
        # Get codebase path
        codebase_path = os.getenv('CODEBASE_PATH')
        if not codebase_path:
            return "Error: CODEBASE_PATH environment variable is required"
        
        # Initialize ProjectAnalyzer for analysis and file counting
        analyzer = ProjectAnalyzer(codebase_path)
        
        # Get database name from environment
        db_name = env_config.get_db_name()
        
        # Check if database/configuration exists
        db_path = Path(__file__).parent.parent / 'DBs' / db_name
        import chromadb
        
        config = None
        collection = None
        is_new_config = False
        
        if db_path.exists():
            # Try to get existing configuration
            try:
                client = chromadb.PersistentClient(path=str(db_path))
                collection = client.get_collection(name=db_name)
                metadata = collection.metadata
                config = metadata.get('config', {})
                if isinstance(config, str):
                    config = json.loads(config)
            except:
                # Collection doesn't exist yet
                pass
        
        # If no config exists, we need to create one
        if not config:
            is_new_config = True
            
            # If user provided specific config without analyze, use that
            if not analyze and (extensions or exclude_dirs or exclude_patterns):
                config = {
                    'extensions': sorted([ext.strip() for ext in extensions.split(',')] if extensions else list(analyzer.DEFAULT_EXTENSIONS)),
                    'exclude_dirs': sorted([exc.strip() for exc in exclude_dirs.split(',')] if exclude_dirs else list(analyzer.DEFAULT_EXCLUDE_DIRS)),
                    'exclude_patterns': [pat.strip() for pat in exclude_patterns.split(',')] if exclude_patterns else ['*.test.*', '*.spec.*'],
                    'project_type': 'manual',
                    'analysis_date': datetime.now().isoformat()
                }
            # If analyze=True or no parameters provided, analyze the project
            elif analyze or (not extensions and not exclude_dirs and not exclude_patterns):
                config = analyzer.analyze_project()
                logger.info(f"Project analysis complete. Detected: {config.get('detected_frameworks', ['generic'])}")
            else:
                # Use defaults
                config = {
                    'extensions': sorted(list(analyzer.DEFAULT_EXTENSIONS)),
                    'exclude_dirs': sorted(list(analyzer.DEFAULT_EXCLUDE_DIRS)),
                    'exclude_patterns': ['*.test.*', '*.spec.*'],
                    'project_type': 'default',
                    'analysis_date': datetime.now().isoformat()
                }
        else:
            # Update existing configuration
            changes = []
            
            # If analyze=True, re-analyze and replace the config
            if analyze:
                config = analyzer.analyze_project()
                changes.append("Re-analyzed project structure")
            else:
                # Update specific parameters
                if extensions is not None:
                    new_extensions = [ext.strip() for ext in extensions.split(',') if ext.strip()]
                    if append and 'extensions' in config:
                        existing = set(config['extensions'])
                        existing.update(new_extensions)
                        config['extensions'] = sorted(list(existing))
                        changes.append(f"Added extensions: {', '.join(new_extensions)}")
                    else:
                        config['extensions'] = sorted(new_extensions)
                        changes.append(f"Set extensions to: {', '.join(new_extensions)}")
                
                if exclude_dirs is not None:
                    new_excludes = [exc.strip() for exc in exclude_dirs.split(',') if exc.strip()]
                    if append and 'exclude_dirs' in config:
                        existing = set(config['exclude_dirs'])
                        existing.update(new_excludes)
                        config['exclude_dirs'] = sorted(list(existing))
                        changes.append(f"Added exclude dirs: {', '.join(new_excludes)}")
                    else:
                        config['exclude_dirs'] = sorted(new_excludes)
                        changes.append(f"Set exclude dirs to: {', '.join(new_excludes)}")
                
                if exclude_patterns is not None:
                    new_patterns = [pat.strip() for pat in exclude_patterns.split(',') if pat.strip()]
                    if append and 'exclude_patterns' in config:
                        existing = set(config['exclude_patterns'])
                        existing.update(new_patterns)
                        config['exclude_patterns'] = sorted(list(existing))
                        changes.append(f"Added exclude patterns: {', '.join(new_patterns)}")
                    else:
                        config['exclude_patterns'] = sorted(new_patterns)
                        changes.append(f"Set exclude patterns to: {', '.join(new_patterns)}")
                
                if not changes and not is_new_config:
                    return "No configuration changes specified."
        
        # Count files that will be processed
        stats = analyzer.count_files(config)
        
        # Save configuration
        config['last_modified'] = datetime.now().isoformat()
        
        # Create or update collection
        if not db_path.exists():
            os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        if collection is None:
            # Create new collection with configuration
            try:
                collection = client.create_collection(
                    name=db_name,
                    metadata={'config': json.dumps(config), 'vectorized': 'false'}
                )
            except:
                # Collection might exist, get it
                collection = client.get_collection(name=db_name)
                collection.modify(metadata={'config': json.dumps(config), 'vectorized': 'false'})
        else:
            # Update existing collection
            metadata = collection.metadata
            metadata['config'] = json.dumps(config)
            if is_new_config:
                metadata['vectorized'] = 'false'
            collection.modify(metadata=metadata)
        
        # Build detailed result message
        result = []
        
        if is_new_config:
            result.append("✓ Configuration created successfully!\n")
            if config.get('detected_frameworks'):
                result.append(f"Detected project type: {', '.join(config['detected_frameworks'])}")
            result.append("")
        else:
            result.append("✓ Configuration updated successfully!\n")
            if changes:
                result.append("Changes made:")
                for change in changes:
                    result.append(f"  • {change}")
                result.append("")
        
        # Show what will be indexed
        result.append("=== Files to be indexed ===")
        result.append(f"Total files: {stats['total_files']}")
        result.append(f"Total size: {stats['total_size'] / (1024*1024):.1f} MB")
        result.append("")
        
        # Show breakdown by extension
        if stats['by_extension']:
            result.append("By extension:")
            for ext in sorted(stats['by_extension'].keys()):
                info = stats['by_extension'][ext]
                result.append(f"  {ext}: {info['count']} files ({info['size'] / (1024*1024):.1f} MB)")
        result.append("")
        
        # Show excluded directories if any
        if stats['excluded_dirs_count']:
            result.append("Excluded directories (file count):")
            for dir_name in sorted(stats['excluded_dirs_count'].keys())[:10]:
                result.append(f"  • {dir_name}: {stats['excluded_dirs_count'][dir_name]} files")
            if len(stats['excluded_dirs_count']) > 10:
                result.append(f"  ... and {len(stats['excluded_dirs_count']) - 10} more")
            result.append("")
        
        # Show sample files
        if stats['sample_files']:
            result.append("Sample files that will be indexed:")
            for file_path in stats['sample_files']:
                result.append(f"  • {file_path}")
            result.append("")
        
        # Instructions
        result.append("=== Next steps ===")
        if stats['total_files'] == 0:
            result.append("⚠️  No files match the current configuration!")
            result.append("Adjust the configuration with:")
            result.append("  set_config(extensions=\".py,.js\")  # Add extensions")
            result.append("  set_config(exclude_dirs=\"\", append=False)  # Clear exclusions")
        else:
            result.append("To modify configuration:")
            result.append("  set_config(exclude_dirs=\"tests,docs\", append=True)  # Add exclusions")
            result.append("  set_config(extensions=\".py,.md\")  # Change file types")
            result.append("")
            result.append("Ready to vectorize?")
            result.append("  Run: update_db()")
        
        if env_config.get_logging_verbose():
            logger.info(f"[SET_CONFIG] Configuration {'created' if is_new_config else 'updated'}")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error configuring database: {str(e)}"