import json
import argparse
from typing import Iterable, Dict, List, Any
import os
import time
from collections import Counter

from .tokenizer import RegexMatchTokenizer, Tokenizer, TokenType
from .document import Document
from .preprocess import (
    PreprocessingPipeline, 
    LowercasePreprocessor, 
    StopWordsPreprocessor, 
    NonsenseTokenPreprocessor,
    RemoveDiacriticsPreprocessor
)

# Import stemming and lemmatization
from . import stemming
from . import lemmatization
try:
    from .stem_preprocessor import StemPreprocessor
    STEMMING_AVAILABLE = True
except ImportError:
    STEMMING_AVAILABLE = False
    print("Warning: StemPreprocessor not available")

def build_vocabulary(documents: Iterable[Document]) -> Counter:
    """
    Build vocabulary from documents.
    
    Args:
        documents: Iterable of Document objects
    
    Returns:
        Counter object with word frequencies
    """
    vocab = Counter()
    for doc in documents:
        # Count only tokens with non-empty processed_form (filter out empty tokens after stop words removal)
        vocab.update([token.processed_form for token in doc.tokens if token.processed_form])
    return vocab

def write_weighted_vocab(vocab: Counter, file) -> None:
    """
    Write vocabulary with occurrence counts.
    
    Args:
        vocab: Counter object with word frequencies
        file: File object to write to
    """
    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        file.write(f"{key} {value}\n")

def create_pipeline(config: Dict) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        PreprocessingPipeline object
    """
    preprocessors = []
    pipeline_name = []
    
    # Stop words
    if config['preprocessing']['stop_words']['use']:
        language = config['preprocessing']['stop_words']['language']
        stop_words_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        preprocessors.append(StopWordsPreprocessor(language=language, stop_words_dir=stop_words_dir))
        pipeline_name.append(f"StopWords({language})")
    
    # Nonsense tokens
    if config['preprocessing']['nonsense_tokens']['remove']:
        min_word_length = config['preprocessing']['nonsense_tokens']['min_word_length']
        preprocessors.append(NonsenseTokenPreprocessor(min_word_length=min_word_length))
        pipeline_name.append(f"NonsenseFilter(min={min_word_length})")
    
    # Lowercase
    if config['preprocessing'].get('lowercase', True):
        preprocessors.append(LowercasePreprocessor())
        pipeline_name.append("Lowercase")
    
    # Diacritics removal
    if config['preprocessing'].get('remove_diacritics', True):
        preprocessors.append(RemoveDiacriticsPreprocessor())
        pipeline_name.append("RemoveDiacritics")
    
    # Stemming - add only if configured and available
    if STEMMING_AVAILABLE and config['stemming']['use']:
        language = config['stemming']['language']
        stemmer_path = config['stemming'].get('stemmer_path')
        preprocessors.append(StemPreprocessor(language=language, stemmer_path=stemmer_path))
        pipeline_name.append(f"Stemming({language})")
    
    return PreprocessingPipeline(preprocessors, name="+".join(pipeline_name))

def process_with_pipeline(documents: List[Document], pipeline: PreprocessingPipeline) -> Dict[str, Any]:
    """
    Process documents with a preprocessing pipeline.
    
    Args:
        documents: List of Document objects
        pipeline: PreprocessingPipeline object
    
    Returns:
        Dictionary with results:
        - vocabulary: Counter with word frequencies
        - vocab_size: Size of vocabulary
        - processing_time: Time taken for processing
    """
    start_time = time.time()
    
    # Process each document
    for doc in documents:
        doc.preprocess(pipeline)
    
    # Build vocabulary from processed documents
    vocabulary = build_vocabulary(documents)
    
    processing_time = time.time() - start_time
    
    # Return results
    return {
        'vocabulary': vocabulary,
        'vocab_size': len(vocabulary),
        'processing_time': processing_time
    }

def print_pipeline_stats(pipeline_name: str, results: Dict[str, Any], original_vocab_size: int) -> None:
    """
    Print statistics about preprocessing pipeline results.
    
    Args:
        pipeline_name: Name of the pipeline
        results: Results dictionary from process_with_pipeline
        original_vocab_size: Size of the original vocabulary before preprocessing
    """
    vocab_size = results['vocab_size']
    processing_time = results['processing_time']
    
    reduction = (original_vocab_size - vocab_size) / original_vocab_size * 100 if original_vocab_size > 0 else 0
    
    print(f"Pipeline: '{pipeline_name}'")
    print(f"Vocabulary size after preprocessing: {vocab_size} (reduction of {reduction:.1f}%)")
    print(f"Processing time: {processing_time:.3f} seconds")

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file, handling comments.
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        # Read file content
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Remove line comments (lines starting with //)
        lines = content.splitlines()
        filtered_lines = []
        for line in lines:
            # Remove comment from line if exists
            line_without_comment = line.split("//")[0]
            # Add line without comment if not empty
            if line_without_comment.strip():
                filtered_lines.append(line_without_comment)
        
        # Join lines back into one string
        filtered_content = "\n".join(filtered_lines)
        
        # Load JSON
        try:
            config = json.loads(filtered_content)
            return config
        except json.JSONDecodeError as e:
            print(f"Error loading configuration file: {e}")
            print("Creating standard configuration...")
            # If loading configuration fails, create standard one
            from .create_default_config import create_default_config
            create_default_config("config.standard.json")
            # Load standard configuration
            with open("config.standard.json", "r", encoding="utf-8") as f:
                return json.load(f)
                
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found. Creating default configuration.")
        from .create_default_config import create_default_config
        create_default_config(config_file)
        # Load the newly created configuration
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

def load_documents(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    Load documents from a JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
    
    Returns:
        List of Document objects
    """
    documents = []
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
        # Load documents from the JSON data
        for i, article in enumerate(data):
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            text = article.get("text", "")
            
            # Create document only if it has content
            if title or abstract or text:
                doc = Document(id=str(i), title=title, abstract=abstract, text=text)
                documents.append(doc.tokenize())
    
    return documents

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Text preprocessing for NLP')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load documents
    input_file = config['input']['file']
    encoding = config['input']['encoding']
    
    print(f"Loading documents from {input_file}...")
    documents = load_documents(input_file, encoding)
    print(f"Loaded {len(documents)} documents.")
    
    # Original vocabulary without preprocessing
    original_vocab = build_vocabulary(documents)
    original_vocab_size = len(original_vocab)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Save original vocabulary
    original_vocab_file = os.path.join(output_dir, config['output']['original_vocab'])
    with open(original_vocab_file, "w", encoding=encoding) as f:
        write_weighted_vocab(original_vocab, f)
    
    # Create pipeline according to configuration
    pipeline = create_pipeline(config)
    
    # Process documents
    print(f"\nProcessing pipeline: '{pipeline.name}'...")
    results = process_with_pipeline(documents, pipeline)
    print_pipeline_stats(pipeline.name, results, original_vocab_size)
    
    # Save preprocessed vocabulary
    preprocessed_vocab_file = os.path.join(output_dir, config['output']['preprocessed_vocab'])
    with open(preprocessed_vocab_file, "w", encoding=encoding) as f:
        write_weighted_vocab(results['vocabulary'], f)
    
    # Apply stemming if requested
    if config['stemming']['use']:
        stemmed_vocab_file = os.path.join(output_dir, config['output']['stemmed_vocab'])
        print("\nPerforming stemming...")
        
        # Use stemming module directly
        stemming.process_stemming(
            input_file=preprocessed_vocab_file,
            output_file=stemmed_vocab_file,
            language=config['stemming']['language'],
            output_dir=output_dir,
            stemmer_path=config['stemming']['stemmer_path']
        )
        
        # Get stemmed vocabulary size
        stemmed_word_count = 0
        with open(stemmed_vocab_file, 'r', encoding=encoding) as f:
            for line in f:
                stemmed_word_count += 1
        
        stemmed_reduction = (original_vocab_size - stemmed_word_count) / original_vocab_size * 100
    
    # Apply lemmatization if requested
    if config['lemmatization']['use']:
        lemmatized_vocab_file = os.path.join(output_dir, "vocab_lemmatized.txt")
        print("\nPerforming lemmatization...")
        
        # Determine model path by language
        model_path = config['lemmatization'].get('model_path')
        
        # Use lemmatization module directly
        lemmatization.process_lemmatization(
            input_file=preprocessed_vocab_file,
            output_file=lemmatized_vocab_file,
            language=config['lemmatization']['language'],
            output_dir=output_dir,
            model_path=model_path
        )
        
        # Get lemmatized vocabulary size
        lemmatized_word_count = 0
        with open(lemmatized_vocab_file, 'r', encoding=encoding) as f:
            for line in f:
                lemmatized_word_count += 1
        
        lemmatized_reduction = (original_vocab_size - lemmatized_word_count) / original_vocab_size * 100
    
    # Print summary of results
    print("\n--- Results ---")
    print(f"1. Original vocabulary: {original_vocab_size} words")
    print(f"2. After preprocessing: {results['vocab_size']} words")
    print(f"   - Saved in: {preprocessed_vocab_file}")
    
    if config['stemming']['use']:
        print(f"3. After stemming: {stemmed_word_count} words (reduction of {stemmed_reduction:.1f}%)")
        print(f"   - Saved in: {stemmed_vocab_file}")
        print(f"   - Groups of words with same stem: {os.path.join(output_dir, 'stemming_groups.json')}")
    
    if config['lemmatization']['use']:
        print(f"4. After lemmatization: {lemmatized_word_count} words (reduction of {lemmatized_reduction:.1f}%)")
        print(f"   - Saved in: {lemmatized_vocab_file}")
        print(f"   - Groups of words with same lemma: {os.path.join(output_dir, 'lemmatization_groups.json')}")

if __name__ == '__main__':
    main()
