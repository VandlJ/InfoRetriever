import json
import os
import argparse
import time
from collections import defaultdict

from InfoRetriever.preprocessing.tokenizer import RegexMatchTokenizer, TokenType
from InfoRetriever.preprocessing.preprocess import (
    LowercasePreprocessor, 
    RemoveDiacriticsPreprocessor,
    StopWordsPreprocessor,
    NonsenseTokenPreprocessor,
    PreprocessingPipeline
)
from InfoRetriever.preprocessing.stem_preprocessor import StemPreprocessor

class InvertedIndexBuilder:
    def __init__(self, config=None):
        self.index = defaultdict(set)  # term -> set of document IDs
        self.all_docs = set()          # all document IDs
        self.document_count = 0
        
        # Get configuration
        self.config = config
        if not self.config:
            # Default config if none is provided
            self.config = self._load_default_config()
        
        # Create tokenizer and preprocessors based on config
        self.tokenizer = RegexMatchTokenizer()
        self.preprocessors = self._create_preprocessing_pipeline()
    
    def _load_default_config(self):
        """Load default configuration or create it if not exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}, using default settings")
        else:
            # Try to create default config
            try:
                from InfoRetriever.preprocessing.create_default_config import create_default_config
                create_default_config(config_path)
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error creating config: {e}, using default settings")
        
        # Return basic default configuration
        return {
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
                RemoveDiacriticsPreprocessor()
            ]
        
        return PreprocessingPipeline(preprocessors, name="IndexingPipeline")
    
    def preprocess_text(self, text):
        """Tokenize and preprocess text using the preprocessing pipeline"""
        # Handle None or empty text
        if not text:
            return []
            
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Apply preprocessing using the configurable pipeline
        self.preprocessors.preprocess(tokens, text)
        
        # Return only non-empty processed token forms
        return [token.processed_form for token in tokens 
                if token.token_type == TokenType.WORD and token.processed_form]
    
    def build_from_json(self, json_file, vocab_file=None):
        """
        Build inverted index from a JSON file containing documents.
        Optionally filter using a vocabulary file.
        
        Args:
            json_file: Path to JSON file with documents
            vocab_file: Optional path to preprocessed vocabulary file
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Load vocabulary if provided (to filter terms)
        vocab = set()
        if vocab_file and os.path.exists(vocab_file):
            print(f"Loading vocabulary from {vocab_file}...")
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # Handle weighted vocab format (term count)
                    parts = line.strip().split()
                    if parts:
                        vocab.add(parts[0])
            print(f"Loaded {len(vocab)} terms in vocabulary")
        
        # Load documents
        try:
            print(f"Loading documents from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            start_time = time.time()
            missing_titles = 0
            
            # Process documents
            if isinstance(documents, list):
                # Process list format
                for i, doc in enumerate(documents):
                    doc_id = i + 1  # Start IDs from 1
                    
                    # Extract text
                    text = self._extract_document_text(doc)
                    if text:
                        self._index_document(doc_id, text, vocab)
                    else:
                        missing_titles += 1
            else:
                # Process dictionary format
                for doc_id, doc in documents.items():
                    # Convert string doc_id to int if needed
                    doc_id = int(doc_id) if isinstance(doc_id, str) and doc_id.isdigit() else doc_id
                    
                    # Extract text
                    text = self._extract_document_text(doc)
                    if text:
                        self._index_document(doc_id, text, vocab)
                    else:
                        missing_titles += 1
            
            end_time = time.time()
            print(f"Indexed {self.document_count} documents in {end_time - start_time:.2f} seconds")
            if missing_titles > 0:
                print(f"Warning: {missing_titles} documents had no title and were skipped")
            print(f"Total terms in inverted index: {len(self.index)}")
            print(f"Total unique document IDs: {len(self.all_docs)}")
            
            return True
            
        except Exception as e:
            print(f"Error building inverted index: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            return False
    
    def _index_document(self, doc_id, text, vocab=None):
        """
        Index a single document.
        
        Args:
            doc_id: Document ID
            text: Text to index (typically the title)
            vocab: Optional vocabulary set to filter terms
        """
        self.all_docs.add(doc_id)
        self.document_count += 1
        
        # Preprocess the text
        terms = self.preprocess_text(text)
        
        # Add each term to the index (if in vocabulary or no vocabulary filter)
        for term in terms:
            if not vocab or term in vocab:
                self.index[term].add(doc_id)
    
    def save_to_json(self, output_file):
        """
        Save the inverted index to a JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        # Convert sets to lists for JSON serialization
        serializable_index = {term: list(doc_ids) for term, doc_ids in self.index.items()}
        
        # Add metadata
        output_data = {
            "metadata": {
                "document_count": self.document_count,
                "term_count": len(self.index),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "index": serializable_index
        }
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Inverted index saved to {output_file}")
    
    def print_sample(self, sample_size=10):
        """Print a sample of the inverted index"""
        print("\nInverted Index Sample:")
        print("-" * 60)
        
        # Get sample terms (alphabetically sorted)
        sample_terms = sorted(self.index.keys())[:sample_size]
        
        for term in sample_terms:
            doc_ids = self.index[term]
            print(f"'{term}' → {len(doc_ids)} documents: {sorted(doc_ids)[:5]}{'...' if len(doc_ids) > 5 else ''}")
        
        print("-" * 60)

    def _extract_document_text(self, doc):
        """
        Extract text from a document for indexing.
        
        Args:
            doc: Document object (dict, string, etc.)
        
        Returns:
            str: Extracted text or empty string if no text found
        """
        # If doc is a string, return it directly
        if isinstance(doc, str):
            return doc
                
        # If doc is a dict, extract available text fields
        if isinstance(doc, dict):
            # Try title, abstract, and text fields
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            text = doc.get('text', '')
            
            # Combine available fields, prioritizing title
            combined_text = ' '.join(filter(None, [title, abstract, text[:1000] if text else '']))
            
            return combined_text
        
        # No suitable text found
        return ""

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build inverted index from documents')
    parser.add_argument('documents', help='Path to documents JSON file')
    parser.add_argument('--vocab', help='Path to preprocessed vocabulary file')
    parser.add_argument('--output', default='inverted_index.json', 
                        help='Path to output inverted index JSON file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Inverted Index Builder".center(60))
    print("=" * 60)
    
    # Create builder
    builder = InvertedIndexBuilder()
    
    # Build index
    success = builder.build_from_json(args.documents, args.vocab)
    
    if success:
        # Print sample
        builder.print_sample()
        
        # Save to file
        builder.save_to_json(args.output)
        
        print("\nDone!")
    else:
        print("Failed to build inverted index!")

if __name__ == "__main__":
    main()
