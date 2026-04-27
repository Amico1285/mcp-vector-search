"""File processing utilities for extracting JSDoc and content."""
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from . import env_config


class FileProcessor:
    """Process TypeScript/JavaScript files to extract JSDoc and content."""
    
    # File extensions to process
    VALID_EXTENSIONS = {'.ts', '.tsx', '.js', '.jsx', '.md', '.mdx', '.json', '.yml', '.yaml', '.sh'}
    
    
    # JSON files to include (others will be excluded)
    ALLOWED_JSON_FILES = {
        'package.json',
        'tsconfig.json',
        'chromatic.config.json',
        'vite.config.json',
        'vitest.config.json'
    }
    @staticmethod
    def get_first_n_lines(content: str, n: int = 30) -> str:
        """Get first n lines of content."""
        if n == -1:
            return content
        lines = content.split('\n')
        return '\n'.join(lines[:n])
    
    
    
    @staticmethod
    def calculate_file_hash(content: str) -> str:
        """Calculate hash of file content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    @classmethod
    def process_file(cls, file_path: Path) -> Optional[Dict]:
        """
        Process a single file and extract relevant information.
        
        Returns:
            Dictionary with processed file data or None if file should be skipped
        """
        # Check if valid extension
        if file_path.suffix not in cls.VALID_EXTENSIONS:
            return None
        
        # Skip node_modules and other irrelevant paths
        path_str = str(file_path)
        skip_patterns = ['node_modules', '.git', 'dist', 'build', 'coverage', '.next', 'out']
        if any(pattern in path_str for pattern in skip_patterns):
            return None
        
        # Skip Claude command files
        if '/.claude/commands/' in path_str:
            return None
        
        # For JSON files, only process allowed ones
        if file_path.suffix == '.json':
            if file_path.name not in cls.ALLOWED_JSON_FILES:
                # Also check for package-lock.json and similar
                if 'package-lock' in file_path.name or 'yarn.lock' in file_path.name:
                    return None
                # Skip if not in allowed list
                return None
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            file_hash = cls.calculate_file_hash(content)
            
            # Get content based on storage_lines setting from environment
            storage_lines = env_config.get_preview_lines_storage()
            stored_content = cls.get_first_n_lines(content, storage_lines)
            
            return {
                'path': str(file_path),
                'content': stored_content,
                'file_hash': file_hash,
                'relative_path': file_path.name  # Just filename for now
            }
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    @classmethod
    def process_directory(cls, directory: Path) -> List[Dict]:
        """Process all valid files in a directory recursively."""
        processed_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                result = cls.process_file(file_path)
                if result:
                    processed_files.append(result)
        
        return processed_files