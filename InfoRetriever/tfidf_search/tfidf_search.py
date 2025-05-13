import json
import math
import os
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

from ..preprocessing.tokenizer import RegexMatchTokenizer, TokenType
from ..preprocessing.preprocess import (
    PreprocessingPipeline,
    LowercasePreprocessor,
    StopWordsPreprocessor,
    NonsenseTokenPreprocessor,
    RemoveDiacriticsPreprocessor
)
from ..preprocessing.stem_preprocessor import StemPreprocessor
from ..preprocessing.document import Document

# For cosine similarity calculation
def compute_cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector as a dictionary {word: tf_idf_score}
        vec2: Second vector as a dictionary {word: tf_idf_score}
        
    Returns:
        Cosine similarity score
    """
    # Find common words
    common_words = set(vec1.keys()) & set(vec2.keys())
    
    # Calculate dot product
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(score**2 for score in vec1.values()))
    magnitude2 = math.sqrt(sum(score**2 for score in vec2.values()))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)


class InvertedIndex:
    """Inverted index mapping words to document occurrences."""
    
    def __init__(self):
        self.index = defaultdict(dict)  # {word: {doc_id: freq}}
        self.document_count = 0
        self.documents = {}  # {doc_id: Document}
    
    def add_document(self, document: Document, word_freq: Dict[str, int]):
        """
        Add a document to the inverted index.
        
        Args:
            document: Document object to add
            word_freq: Dictionary mapping words to their frequencies in the document
        """
        doc_id = document.id
        self.documents[doc_id] = document
        
        # Update document frequency for each word
        for word, freq in word_freq.items():
            if word:  # Skip empty strings
                self.index[word][doc_id] = freq
        
        self.document_count += 1
    
    def get_document_frequency(self, word: str) -> int:
        """
        Get the number of documents containing the given word.
        
        Args:
            word: The word to check
            
        Returns:
            Number of documents containing the word
        """
        return len(self.index.get(word, {}))
    
    def get_inverse_document_frequency(self, word: str) -> float:
        """
        Calculate the inverse document frequency for a word.
        IDF(t) = log10(N/DF(t))
        
        Args:
            word: The word to calculate IDF for
            
        Returns:
            IDF value for the word
        """
        df = self.get_document_frequency(word)
        if df == 0:
            return 0
        return math.log10(self.document_count / df)


def preprocess_text(text: str, pipeline: PreprocessingPipeline) -> Dict[str, int]:
    """
    Preprocess text and return word frequencies.
    
    Args:
        text: Text to preprocess
        pipeline: Preprocessing pipeline to use
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # Apply preprocessing pipeline
    processed_tokens = pipeline.preprocess(tokens, text)
    
    # Build word frequency dictionary from processed tokens
    word_freq = Counter()
    for token in processed_tokens:
        if token.processed_form and token.token_type == TokenType.WORD:  # Skip empty tokens or non-words
            word_freq[token.processed_form] += 1
    
    return word_freq


def compute_tf(word_freq: Dict[str, int]) -> Dict[str, float]:
    """
    Compute term frequency (TF) for each word.
    TF(t,d) = 1 + log10(f(t,d)) if f(t,d) > 0, else 0
    
    Args:
        word_freq: Dictionary mapping words to their frequencies
        
    Returns:
        Dictionary mapping words to their TF scores
    """
    tf_scores = {}
    for word, freq in word_freq.items():
        if freq > 0:
            tf_scores[word] = 1 + math.log10(freq)
        else:
            tf_scores[word] = 0
    return tf_scores


def compute_tf_idf_vectors(inverted_index: InvertedIndex) -> Dict[int, Dict[str, float]]:
    """
    Compute TF-IDF vectors for all documents in the inverted index.
    
    Args:
        inverted_index: The inverted index containing documents
        
    Returns:
        Dictionary mapping document IDs to their TF-IDF vectors
    """
    tf_idf_vectors = {}
    
    # Create a reverse index to quickly find words for each document
    # This avoids checking every word for every document
    doc_to_words = {}
    
    # Build the reverse index
    for word, doc_dict in inverted_index.index.items():
        for doc_id, freq in doc_dict.items():
            if doc_id not in doc_to_words:
                doc_to_words[doc_id] = {}
            doc_to_words[doc_id][word] = freq
    
    # Now process each document using the reverse index
    for doc_id, document in inverted_index.documents.items():
        # Skip if the document already has a vector
        if hasattr(document, 'tf_idf_vector') and document.tf_idf_vector:
            tf_idf_vectors[doc_id] = document.tf_idf_vector
            continue
            
        if doc_id not in doc_to_words:
            # Empty document, create empty vector
            document.tf_idf_vector = {}
            tf_idf_vectors[doc_id] = {}
            continue
        
        # Get word frequencies for this document
        word_freq = doc_to_words[doc_id]
        
        # Compute TF scores
        tf_scores = compute_tf(word_freq)
        
        # Compute TF-IDF vector
        tf_idf_vector = {}
        for word, tf_score in tf_scores.items():
            idf = inverted_index.get_inverse_document_frequency(word)
            tf_idf_vector[word] = tf_score * idf
        
        # Store TF-IDF vector in document and in result dictionary
        document.tf_idf_vector = tf_idf_vector
        tf_idf_vectors[doc_id] = tf_idf_vector
    
    return tf_idf_vectors


class TFIDFSearchEngine:
    """TF-IDF search engine implementation"""
    
    def __init__(self, documents=None, use_stemming=True, use_stop_words=False, config=None):
        """
        Initialize the TF-IDF search engine.
        
        Args:
            documents: Optional list of document dictionaries to initialize with
            use_stemming: Whether to use stemming in preprocessing (if config not provided)
            use_stop_words: Whether to use stop words in preprocessing (if config not provided)
            config: Configuration dictionary (overrides use_stemming and use_stop_words)
        """
        self.inverted_index = InvertedIndex()
        self.doc_vectors = None
        self.config = config
        
        # Load config if not provided
        if not self.config:
            self._load_config()
            
        # Create preprocessing pipeline based on config
        self.pipeline = self._create_preprocessing_pipeline()
        
        # Process documents if provided
        if documents:
            self.add_documents(documents)
    
    def _load_config(self):
        """Load configuration from file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config: {e}, using default settings")
        
        if not self.config:
            # Use default configuration
            self.config = {
                "preprocessing": {
                    "lowercase": True,
                    "remove_diacritics": True,
                    "stop_words": {"use": False},
                    "nonsense_tokens": {"remove": True, "min_word_length": 2}
                },
                "stemming": {
                    "use": True,
                    "language": "cz"
                },
                "pipeline_order": [
                    "tokenize", "lowercase", "remove_diacritics", 
                    "nonsense_tokens", "stemming"
                ]
            }
    
    def _create_preprocessing_pipeline(self):
        """Create preprocessing pipeline based on configuration"""
        preprocessors = []
        
        # Get preprocessing config
        preproc_config = self.config.get("preprocessing", {})
        stemming_config = self.config.get("stemming", {})
        
        # Add preprocessors in the order specified in config
        pipeline_order = self.config.get("pipeline_order", [])
        
        for step in pipeline_order:
            if step == "lowercase" and preproc_config.get("lowercase", True):
                preprocessors.append(LowercasePreprocessor())
            
            elif step == "remove_diacritics" and preproc_config.get("remove_diacritics", True):
                preprocessors.append(RemoveDiacriticsPreprocessor())
            
            elif step == "stop_words" and preproc_config.get("stop_words", {}).get("use", False):
                language = preproc_config.get("stop_words", {}).get("language", "both")
                # Get data directory for stopwords
                data_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "preprocessing", "data"
                )
                preprocessors.append(StopWordsPreprocessor(language=language, stop_words_dir=data_dir))
            
            elif step == "nonsense_tokens" and preproc_config.get("nonsense_tokens", {}).get("remove", True):
                min_length = preproc_config.get("nonsense_tokens", {}).get("min_word_length", 2)
                preprocessors.append(NonsenseTokenPreprocessor(min_word_length=min_length))
            
            elif step == "stemming" and stemming_config.get("use", True):
                language = stemming_config.get("language", "cz")
                stemmer_path = stemming_config.get("stemmer_path", None)
                preprocessors.append(StemPreprocessor(language=language, stemmer_path=stemmer_path))
        
        # If no preprocessors from config, use defaults
        if not preprocessors:
            preprocessors = [
                LowercasePreprocessor(),
                RemoveDiacriticsPreprocessor(),
                NonsenseTokenPreprocessor(min_word_length=2),
                StemPreprocessor()
            ]
        
        return PreprocessingPipeline(preprocessors)
    
    def add_documents(self, documents_list):
        """
        Add documents to the search engine.
        
        Args:
            documents_list: List of document dictionaries with title, abstract, text fields
        """
        # Get the starting index based on existing documents
        start_index = len(self.inverted_index.documents)
        
        for i, doc_data in enumerate(documents_list):
            # Create document object with a unique ID
            doc_id = doc_data.get("id", str(start_index + i))
            # Convert string IDs to integers for internal processing
            int_doc_id = int(doc_id) if doc_id.isdigit() else hash(doc_id) % 10**9
            
            document = Document(
                doc_id=int_doc_id,
                title=doc_data.get("title", ""),
                content=doc_data.get("text", doc_data.get("content", "")),
                metadata={"abstract": doc_data.get("abstract", ""), "original_id": doc_id}
            )
            
            # Get combined text for processing
            combined_text = f"{document.title} {document.metadata.get('abstract', '')} {document.content}".strip()
            
            # Preprocess document text
            word_freq = preprocess_text(combined_text, self.pipeline)
            
            # Add to inverted index
            self.inverted_index.add_document(document, word_freq)
        
        # Compute TF-IDF vectors only for new documents to save memory
        new_vectors = compute_tf_idf_vectors(self.inverted_index)
        
        # Initialize doc_vectors if it doesn't exist
        if self.doc_vectors is None:
            self.doc_vectors = {}
            
        # Update with new vectors
        self.doc_vectors.update(new_vectors)
    
    def search(self, query, top_k=5):
        """
        Search for documents matching the query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Preprocess query
        word_freq = preprocess_text(query, self.pipeline)
        
        # Compute TF for query terms
        tf_scores = compute_tf(word_freq)
        
        # Compute TF-IDF for query
        query_vector = {}
        for word, tf_score in tf_scores.items():
            idf = self.inverted_index.get_inverse_document_frequency(word)
            query_vector[word] = tf_score * idf
        
        # Find most similar documents
        return self._rank_documents(query_vector, top_k)
    
    def _rank_documents(self, query_vector, top_k=5):
        """
        Rank documents by similarity to query vector.
        
        Args:
            query_vector: TF-IDF vector for query
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        similarities = []
        
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = compute_cosine_similarity(query_vector, doc_vector)
            similarities.append((self.inverted_index.documents[doc_id], similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]


# Command-line interface
def main():
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='TF-IDF Search Engine')
    parser.add_argument('--input', help='Input JSON file with documents')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--top', type=int, default=5, help='Number of top results to display')
    parser.add_argument('--no-stemming', action='store_true', help='Disable stemming')
    parser.add_argument('--no-stop-words', action='store_true', help='Disable stop words removal')
    args = parser.parse_args()
    
    # Load documents
    try:
        if args.input:
            print(f"Loading documents from {args.input}...")
            with open(args.input, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            # Create sample test documents
            print("No input file provided. Using test documents...")
            documents = [
                {
                    "title": "Artificial Intelligence Basics",
                    "abstract": "An introduction to AI concepts and applications.",
                    "text": "Artificial Intelligence encompasses machine learning, neural networks, and natural language processing."
                },
                {
                    "title": "Introduction to Machine Learning",
                    "abstract": "Overview of machine learning techniques and algorithms.",
                    "text": "Machine learning algorithms include supervised learning, unsupervised learning, and reinforcement learning."
                },
                {
                    "title": "Natural Language Processing",
                    "abstract": "NLP techniques for processing and understanding human language.",
                    "text": "NLP applications include sentiment analysis, machine translation, and text summarization."
                },
                {
                    "title": "Deep Learning Fundamentals",
                    "abstract": "Introduction to deep neural networks.",
                    "text": "Deep learning uses multi-layered neural networks to learn representations from data."
                },
                {
                    "title": "Reinforcement Learning Applications",
                    "abstract": "Applications of reinforcement learning in various domains.",
                    "text": "Reinforcement learning has been applied to game playing, robotics, and recommendation systems."
                }
            ]
        
        # Create search engine
        engine = TFIDFSearchEngine(
            documents=documents, 
            use_stemming=not args.no_stemming,
            use_stop_words=not args.no_stop_words
        )
        
        # Use query from command line or default
        query = args.query or "machine learning algorithms"
        print(f"\nQuery: '{query}'")
        
        # Search and display results
        results = engine.search(query, top_k=args.top)
        
        print("\n" + "="*50)
        print("SEARCH RESULTS")
        print("="*50)
        
        if not results:
            print("No matching documents found.")
        else:
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Document: \"{doc.title}\"")
                abstract = doc.metadata.get('abstract', '')
                print(f"   Abstract: {abstract[:100]}{'...' if len(abstract) > 100 else ''}")
                print(f"   Similarity: {score:.4f}")
                print()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
