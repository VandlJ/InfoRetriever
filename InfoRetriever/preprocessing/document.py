from typing import Dict, Optional, List
from .tokenizer import Tokenizer, RegexMatchTokenizer, Token
from .preprocess import PreprocessingPipeline

class Document:
    """
    Represents a document in the information retrieval system.
    Stores document contents and preprocessed data.
    """
    
    def __init__(self, id=None, title: str = "", content: str = "", metadata: Dict = None, 
                 abstract: str = "", text: str = "", doc_id=None):
        """
        Initialize a document with content.
        
        Args:
            id: Unique identifier for the document
            doc_id: Alternative name for id (for compatibility)
            title: Document title
            content: Main document text (alternative to text)
            text: Main document text (alternative to content)
            abstract: Document abstract/summary
            metadata: Additional document metadata
        """
        # Handle doc_id as an alternative to id
        self.id = doc_id if id is None else id
        self.title = title
        self.metadata = metadata or {}
        
        # Handle both content and text as alternative ways to provide the main text
        self.content = content or text
        self.text = self.content  # Alias for compatibility
        
        # For backward compatibility
        self.abstract = abstract or self.metadata.get("abstract", "")
        
        # Combined text for processing
        self.combined_text = f"{title} {self.abstract} {self.content}".strip()
        self.tokens = None
        self.processed_tokens = None
        self.tf_idf_vector = None  # Will be populated during TF-IDF processing
    
    def tokenize(self, tokenizer: Tokenizer = None) -> 'Document':
        """
        Tokenize the document content.
        
        Args:
            tokenizer: Tokenizer to use (defaults to RegexMatchTokenizer)
            
        Returns:
            Self for chaining operations
        """
        tokenizer = tokenizer or RegexMatchTokenizer()
        # Tokenize the combined text
        self.tokens = tokenizer.tokenize(self.combined_text)
        return self
    
    def preprocess(self, preprocessing_pipeline: PreprocessingPipeline) -> 'Document':
        """
        Preprocess the document tokens.
        
        Args:
            preprocessing_pipeline: Pipeline of preprocessors to apply
            
        Returns:
            Self for chaining operations
        """
        if self.tokens is None:
            self.tokenize()
        
        self.processed_tokens = preprocessing_pipeline.preprocess(self.tokens, self.combined_text)
        return self
    
    def get_preprocessed_terms(self) -> List[str]:
        """
        Get the preprocessed terms from the document.
        
        Returns:
            List of preprocessed terms (non-empty)
        """
        if not self.processed_tokens:
            return []
        
        return [token.processed_form for token in self.processed_tokens 
                if token.processed_form]  # Filter out empty tokens
