"""Project analyzer for intelligent configuration detection."""
import fnmatch
import json
import logging
from pathlib import Path
from typing import Dict, Set
from datetime import datetime

logger = logging.getLogger('code_searcher.project_analyzer')


class ProjectAnalyzer:
    """Analyzes project structure and suggests optimal configuration."""
    
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
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'next.js': {
            'files': ['next.config.js', 'next.config.ts'],
            'dirs': ['.next', 'pages', 'app'],
            'extensions': {'.tsx', '.ts', '.jsx', '.js', '.mdx'},
            'exclude': {'.next', 'out'}
        },
        'react': {
            'files': ['package.json'],  # check for react in dependencies
            'dirs': ['src/components'],
            'extensions': {'.tsx', '.ts', '.jsx', '.js'},
            'exclude': {'build'}
        },
        'vue': {
            'files': ['vue.config.js', 'nuxt.config.js'],
            'dirs': ['components', 'pages'],
            'extensions': {'.vue', '.js', '.ts'},
            'exclude': {'dist', '.nuxt'}
        },
        'python': {
            'files': ['requirements.txt', 'pyproject.toml', 'setup.py'],
            'dirs': ['src', 'tests'],
            'extensions': {'.py', '.pyx', '.ipynb'},
            'exclude': {'__pycache__', 'venv', '.venv', '*.egg-info', '.pytest_cache'}
        },
        'django': {
            'files': ['manage.py'],
            'dirs': ['templates', 'static'],
            'extensions': {'.py', '.html', '.css'},
            'exclude': {'migrations', '__pycache__', 'media', 'staticfiles'}
        },
        'rust': {
            'files': ['Cargo.toml'],
            'dirs': ['src', 'tests'],
            'extensions': {'.rs', '.toml'},
            'exclude': {'target'}
        },
        'go': {
            'files': ['go.mod'],
            'dirs': ['pkg', 'cmd'],
            'extensions': {'.go'},
            'exclude': {'vendor'}
        }
    }
    
    # Binary and system file extensions to always exclude
    BINARY_EXTENSIONS = {
        # Executables and libraries
        '.exe', '.dll', '.so', '.dylib', '.db', '.sqlite',
        '.pyc', '.pyo', '.class', '.o', '.a', '.lib',
        
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp',
        
        # Media
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac', '.mkv',
        
        # Archives
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2',
        
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        
        # Data and temporary files
        '.bin', '.dat', '.log', '.cache', '.tmp', '.temp',
        '.backup', '.bak', '.swp', '.lock', '.pid',
        
        # Version control and diffs
        '.v1', '.v2', '.orig', '.rej',
        
        # Log variations (case-sensitive file systems)
        '.LOG', '.2024', '.2025', '.2023', '.2022', '.2021', '.2020',
        
        # Configuration files that are usually not code
        '.cfg', '.CFG', '.properties', '.ini', '.INI',
        
        # Other binary formats
        '.ttf', '.otf', '.woff', '.woff2', '.eot',  # Fonts
        '.sketch', '.fig', '.xd',  # Design files
        '.sqlite3', '.db3',  # Additional database formats
        '.DS_Store', '.thumbs.db'  # System files
    }
    
    def __init__(self, codebase_path: str):
        """Initialize the analyzer with codebase path."""
        self.codebase_path = Path(codebase_path)
        if not self.codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {codebase_path}")
    
    def analyze_project(self) -> Dict:
        """Analyze project structure and detect frameworks."""
        logger.info("Analyzing project structure...")
        
        detected_frameworks = []
        suggested_extensions = set(self.DEFAULT_EXTENSIONS)
        suggested_excludes = set(self.DEFAULT_EXCLUDE_DIRS)
        
        # Check for each framework
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            # Check for framework-specific files
            for file_pattern in patterns['files']:
                if (self.codebase_path / file_pattern).exists():
                    detected_frameworks.append(framework)
                    suggested_extensions.update(patterns['extensions'])
                    suggested_excludes.update(patterns['exclude'])
                    break
            
            # Check for framework-specific directories
            for dir_pattern in patterns.get('dirs', []):
                if (self.codebase_path / dir_pattern).exists():
                    if framework not in detected_frameworks:
                        detected_frameworks.append(framework)
                        suggested_extensions.update(patterns['extensions'])
                        suggested_excludes.update(patterns['exclude'])
                    break
        
        # Special case: Check package.json for React/Vue
        package_json_path = self.codebase_path / 'package.json'
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                    deps = {**package_data.get('dependencies', {}), 
                           **package_data.get('devDependencies', {})}
                    
                    if 'react' in deps and 'react' not in detected_frameworks:
                        detected_frameworks.append('react')
                        suggested_extensions.update(self.FRAMEWORK_PATTERNS['react']['extensions'])
                        suggested_excludes.update(self.FRAMEWORK_PATTERNS['react']['exclude'])
                    
                    if 'vue' in deps and 'vue' not in detected_frameworks:
                        detected_frameworks.append('vue')
                        suggested_extensions.update(self.FRAMEWORK_PATTERNS['vue']['extensions'])
                        suggested_excludes.update(self.FRAMEWORK_PATTERNS['vue']['exclude'])
            except Exception as e:
                logger.warning(f"Could not parse package.json: {e}")
        
        # Count files by extension to suggest additional ones
        # OPTIMIZED: Use os.walk with directory pruning
        import os
        extension_counts = {}
        
        for root, dirs, files in os.walk(self.codebase_path, followlinks=False):
            # Prune excluded directories before traversing
            dirs[:] = [d for d in dirs if d not in suggested_excludes and not d.startswith('.')]
            
            # Count file extensions in current directory
            for file_name in files:
                ext = Path(file_name).suffix
                if ext:
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # If no framework detected, be more conservative with file extensions
        if not detected_frameworks:
            # Common code extensions to consider for generic projects
            COMMON_CODE_EXTENSIONS = {
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
                '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala', '.r',
                '.md', '.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.scss',
                '.sql', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
                '.dockerfile', '.makefile', '.cmake', '.gradle', '.sbt',
                '.vue', '.svelte', '.astro', '.mdx'
            }
            
            for ext, count in extension_counts.items():
                # Only add known code extensions, not everything
                if count >= 1 and ext in COMMON_CODE_EXTENSIONS and ext not in suggested_extensions:
                    suggested_extensions.add(ext)
                    logger.info(f"Auto-detected code extension {ext} ({count} files)")
        else:
            # For framework projects, be more selective
            for ext, count in extension_counts.items():
                if count > 5 and ext not in suggested_extensions:
                    # Common code/config extensions worth including
                    if ext in {'.scss', '.css', '.sql', '.graphql', '.prisma'}:
                        suggested_extensions.add(ext)
        
        config = {
            'extensions': sorted(list(suggested_extensions)),
            'exclude_dirs': sorted(list(suggested_excludes)),
            'exclude_patterns': ['*.test.*', '*.spec.*', '*.min.js', '*.min.css'],
            'detected_frameworks': detected_frameworks,
            'project_type': detected_frameworks[0] if detected_frameworks else 'generic',
            'analysis_date': datetime.now().isoformat()
        }
        
        return config
    
    def count_files(self, config: Dict) -> Dict:
        """Count files that would be processed with given configuration.
        
        Returns dict with statistics about files that would be indexed.
        """
        import os
        
        stats = {
            'by_extension': {},
            'total_files': 0,
            'total_size': 0,
            'excluded_dirs_count': {},
            'sample_files': []
        }
        
        # Get exclude directories for efficient filtering
        exclude_dirs = set(config.get('exclude_dirs', []))
        
        # OPTIMIZED: Use os.walk with directory pruning
        for root, dirs, files in os.walk(self.codebase_path, followlinks=False):
            root_path = Path(root)
            
            # Track excluded directories being skipped
            for d in dirs[:]:  # Iterate on a copy
                if d in exclude_dirs or d.startswith('.'):
                    dirs.remove(d)  # Remove from dirs to skip traversal
                    stats['excluded_dirs_count'][d] = stats['excluded_dirs_count'].get(d, 0) + 1
            
            # Process files in current directory
            for file_name in files:
                file_path = root_path / file_name
                
                # Check if file should be processed
                if not self._should_process_file(file_path, config):
                    continue
                
                # File would be included
                ext = file_path.suffix
                if ext not in stats['by_extension']:
                    stats['by_extension'][ext] = {'count': 0, 'size': 0}
                
                try:
                    size = file_path.stat().st_size
                    stats['by_extension'][ext]['count'] += 1
                    stats['by_extension'][ext]['size'] += size
                    stats['total_files'] += 1
                    stats['total_size'] += size
                    
                    # Collect sample files (first 5)
                    if len(stats['sample_files']) < 5:
                        stats['sample_files'].append(str(file_path.relative_to(self.codebase_path)))
                except:
                    pass
        
        return stats
    
    def _should_process_file(self, file_path: Path, config: Dict) -> bool:
        """Check if file should be processed based on configuration."""
        # Check extension
        if file_path.suffix not in config['extensions']:
            return False
        
        # Check exclude directories - only check in path parts, not in filename
        path_parts = file_path.parts[:-1]  # Exclude the filename itself
        for exclude_dir in config['exclude_dirs']:
            if any(exclude_dir in part or part == exclude_dir for part in path_parts):
                return False
        
        # Check exclude patterns with proper glob matching
        for pattern in config.get('exclude_patterns', []):
            # Use fnmatch for glob-style pattern matching
            if fnmatch.fnmatch(file_path.name, pattern):
                return False
        
        return True