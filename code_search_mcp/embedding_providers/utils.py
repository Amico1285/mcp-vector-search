"""Utilities for embedding providers."""
import logging
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Name of the encoding to use
        
    Returns:
        Number of tokens
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except ImportError:
        raise RuntimeError(
            "tiktoken is required for token counting. "
            "Please install it with: pip install tiktoken"
        )
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise


def estimate_tokens(text: str) -> int:
    """Estimate token count without tiktoken.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    # Heuristic: ~3 characters per token for code
    # Adjust based on content type
    if "import" in text or "def " in text or "class " in text:
        # Likely code
        return len(text) // 2
    else:
        # Likely text
        return len(text) // 4


def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """Split text into chunks based on token count.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks
        encoding_name: Name of the encoding to use
        
    Returns:
        List of text chunks
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        
        tokens = encoding.encode(text)
        chunks = []
        
        step = max_tokens - overlap_tokens
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Stop if we've processed all tokens
            if i + max_tokens >= len(tokens):
                break
        
        return chunks
        
    except ImportError:
        raise RuntimeError(
            "tiktoken is required for token-based splitting. "
            "Please install it with: pip install tiktoken"
        )


def split_by_characters(text: str, max_chars: int, overlap_chars: int = 0) -> List[str]:
    """Split text into chunks based on character count.
    
    Args:
        text: Text to split
        max_chars: Maximum characters per chunk
        overlap_chars: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    step = max_chars - overlap_chars
    
    for i in range(0, len(text), step):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        
        if i + max_chars >= len(text):
            break
    
    return chunks


def split_code_by_structure(
    content: str,
    max_tokens: int,
    min_chunk_tokens: int = 100
) -> List[Dict]:
    """Split code intelligently based on structure (functions, classes).
    
    Args:
        content: Code content to split
        max_tokens: Maximum tokens per chunk
        min_chunk_tokens: Minimum tokens for a chunk
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    lines = content.split('\n')
    
    # Find logical boundaries (functions, classes)
    boundaries = find_code_boundaries(lines)
    
    current_chunk = []
    current_tokens = 0
    chunk_start_line = 0
    
    for i, line in enumerate(lines):
        # Check current chunk size with this line
        test_chunk = '\n'.join(current_chunk + [line])
        test_tokens = count_tokens(test_chunk)
        
        # Check if adding this line would exceed limit
        if test_tokens > max_tokens:
            # Chunk is full, need to save it
            if current_chunk:  # Make sure we have content
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i - 1,
                    'tokens': current_tokens
                })
                
                # Start new chunk with current line
                current_chunk = [line]
                current_tokens = count_tokens(line + '\n')
                chunk_start_line = i
            else:
                # Single line exceeds max_tokens - split it anyway
                current_chunk = [line]
                current_tokens = test_tokens
        else:
            # Line fits, add it to current chunk
            current_chunk.append(line)
            current_tokens = test_tokens
            
            # Only split at boundaries if we're getting close to limit
            if i in boundaries and current_tokens > max_tokens * 0.8:
                # Good place to split before getting too close to limit
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i,
                    'tokens': current_tokens
                })
                
                # Start new chunk
                current_chunk = []
                current_tokens = 0
                chunk_start_line = i + 1
    
    # Add remaining content
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'start_line': chunk_start_line,
            'end_line': len(lines) - 1,
            'tokens': count_tokens(chunk_text)
        })
    
    return chunks


def find_code_boundaries(lines: List[str]) -> List[int]:
    """Find logical boundaries in code (end of functions, classes).
    
    Args:
        lines: Lines of code
        
    Returns:
        List of line indices that are good split points
    """
    boundaries = []
    
    # Patterns that indicate logical boundaries
    patterns = [
        r'^def\s+\w+',      # Function definition
        r'^class\s+\w+',    # Class definition
        r'^async\s+def\s+', # Async function
        r'^#\s*%%',         # Jupyter cell marker
        r'^if\s+__name__',  # Main block
    ]
    
    for i, line in enumerate(lines):
        # Check if this line starts a new logical unit
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                # Mark the line before as a boundary
                if i > 0:
                    boundaries.append(i - 1)
                break
    
    return boundaries


def add_context_to_chunks(
    chunks: List[Dict],
    original_content: str,
    context_lines: int = 5
) -> List[Dict]:
    """Add context from surrounding code to chunks.
    
    Args:
        chunks: List of chunk dictionaries
        original_content: Original full content
        context_lines: Number of context lines to include
        
    Returns:
        List of chunks with added context
    """
    lines = original_content.split('\n')
    
    # Find imports and global definitions
    imports = []
    for i, line in enumerate(lines[:50]):  # Check first 50 lines
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line)
    
    import_context = '\n'.join(imports) + '\n\n' if imports else ''
    
    # Add context to each chunk
    for chunk in chunks:
        # Prepend imports if not already included
        if chunk['start_line'] > 10 and import_context:
            chunk['text'] = import_context + "# ... (imports above) ...\n\n" + chunk['text']
            chunk['has_context'] = True
        else:
            chunk['has_context'] = False
    
    return chunks


def merge_small_chunks(
    chunks: List[Dict],
    min_tokens: int,
    max_tokens: int
) -> List[Dict]:
    """Merge small chunks together to minimize fragmentation.
    
    Args:
        chunks: List of chunk dictionaries
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged = []
    current = chunks[0].copy()
    
    for next_chunk in chunks[1:]:
        combined_tokens = current['tokens'] + next_chunk['tokens']
        
        # Check if we can merge
        if current['tokens'] < min_tokens and combined_tokens <= max_tokens:
            # Merge chunks
            current['text'] = current['text'] + '\n' + next_chunk['text']
            current['end_line'] = next_chunk['end_line']
            current['tokens'] = combined_tokens
        else:
            # Save current and start new
            merged.append(current)
            current = next_chunk.copy()
    
    # Add last chunk
    merged.append(current)
    
    return merged