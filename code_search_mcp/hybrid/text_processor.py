"""Text preprocessing utilities for BM25 search."""
import re
import string
from typing import List, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger('code_searcher.hybrid.text_processor')

class TextProcessor:
    """Text preprocessing for BM25 search with configurable options."""
    
    def __init__(
        self,
        language: str = 'english',
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        use_stemming: bool = False
    ):
        """
        Initialize text processor.
        
        Args:
            language: Language for stopwords and stemming
            remove_stopwords: Whether to remove stopwords
            min_token_length: Minimum token length to keep
            use_stemming: Whether to use stemming (requires nltk)
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.use_stemming = use_stemming
        
        # Initialize stopwords
        self.stopwords = self._load_stopwords() if remove_stopwords else set()
        
        # Initialize stemmer if needed
        self.stemmer = None
        if use_stemming:
            self.stemmer = self._init_stemmer()
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for the specified language."""
        # English stopwords (basic set)
        english_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'or', 'not', 'this', 'these',
            'they', 'we', 'you', 'your', 'have', 'had', 'can', 'could', 'should',
            'would', 'all', 'any', 'been', 'do', 'does', 'did', 'get', 'got'
        }
        
        if self.language == 'english':
            return english_stopwords
        else:
            # For other languages, we'd load appropriate stopwords
            logger.warning(f"Stopwords for language '{self.language}' not implemented, using English")
            return english_stopwords
    
    def _init_stemmer(self):
        """Initialize stemmer if available."""
        try:
            from nltk.stem import PorterStemmer
            import nltk
            # Try to download required data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            return PorterStemmer()
        except ImportError:
            logger.warning("NLTK not available for stemming. Install with: pip install nltk")
            return None
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize text and apply all preprocessing steps.
        
        Args:
            text: Raw text to process
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Basic tokenization - split on whitespace and punctuation
        # Keep underscores for code tokens (function_name, variable_name)
        tokens = re.findall(r'\b\w+(?:_\w+)*\b', text)
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= self.min_token_length]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def process_code_text(self, text: str) -> List[str]:
        """
        Specialized processing for code text with programming-specific features.
        
        Args:
            text: Code text to process
            
        Returns:
            List of processed tokens including code-specific tokens
        """
        # Standard processing
        tokens = self.tokenize_and_process(text)
        
        # Additional code-specific tokenization
        code_tokens = self._extract_code_tokens(text)
        
        # Combine and deduplicate
        all_tokens = tokens + code_tokens
        seen = set()
        unique_tokens = []
        for token in all_tokens:
            if token not in seen and len(token) >= self.min_token_length:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def _extract_code_tokens(self, text: str) -> List[str]:
        """
        Extract programming-specific tokens.
        
        Args:
            text: Code text
            
        Returns:
            List of code-specific tokens
        """
        code_tokens = []
        
        # Extract camelCase and PascalCase words
        camel_case_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b'
        camel_matches = re.findall(camel_case_pattern, text)
        for match in camel_matches:
            # Split camelCase into parts: getUserData -> ["get", "user", "data"] 
            parts = re.sub(r'([A-Z])', r' \1', match).split()
            code_tokens.extend([part.lower() for part in parts if len(part) >= self.min_token_length])
        
        # Extract snake_case words (already handled by main tokenization)
        
        # Extract method calls: methodName() -> methodName
        method_pattern = r'\b(\w+)\s*\('
        method_matches = re.findall(method_pattern, text)
        code_tokens.extend([match.lower() for match in method_matches if len(match) >= self.min_token_length])
        
        # Extract file extensions: .py, .js, .tsx
        ext_pattern = r'\.(\w+)\b'
        ext_matches = re.findall(ext_pattern, text)
        code_tokens.extend([match.lower() for match in ext_matches if len(match) >= self.min_token_length])
        
        return code_tokens
    
    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            'language': self.language,
            'remove_stopwords': self.remove_stopwords,
            'stopwords_count': len(self.stopwords),
            'min_token_length': self.min_token_length,
            'use_stemming': self.use_stemming,
            'stemmer_available': self.stemmer is not None
        }