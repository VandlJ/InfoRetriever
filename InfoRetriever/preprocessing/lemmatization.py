import json
from collections import defaultdict
import argparse
import os
from typing import Dict, List, Tuple
from ufal.udpipe import Model, Pipeline, ProcessingError

def prepare_lemmatization_input(input_vocab_file: str, output_words_file: str) -> Dict[str, int]:
    """
    Prepare input file for lemmatizer and return a dictionary mapping words to their counts.
    
    Args:
        input_vocab_file: Input file with vocabulary (word count)
        output_words_file: Output file with just words for lemmatizer
    
    Returns:
        Dictionary mapping words to their occurrence counts
    """
    word_counts = {}
    
    # Load dictionary with counts
    with open(input_vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Split line into word and count
            parts = line.strip().split()
            if len(parts) >= 2:
                # If the last part is a number, consider it a count and the rest as a word
                try:
                    count = int(parts[-1])
                    word = ' '.join(parts[:-1])  # Join all parts except the last one
                    word_counts[word] = count
                except ValueError:
                    # If the last part is not a number, skip this line
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
    
    # Write only words for lemmatizer
    with open(output_words_file, 'w', encoding='utf-8') as f:
        for word in word_counts.keys():
            f.write(f"{word}\n")
    
    return word_counts

def lemmatize_words(input_file: str, output_file: str, model_path: str, language: str = 'cz') -> bool:
    """
    Lemmatize words using UDPipe.
    
    Args:
        input_file: Input file with words
        output_file: Output file for lemmatized words
        model_path: Path to UDPipe model
        language: Language for lemmatization ('cz', 'en')
    
    Returns:
        True on successful execution, False on error
    """
    # Check model existence
    if not os.path.exists(model_path):
        print(f"Error: UDPipe model not found at path: {model_path}")
        print("Downloading UDPipe model for the language...")
        
        # Try to download the model
        import urllib.request
        
        # Set parameters based on language
        if language == 'cz':
            model_url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/czech-pdt-ud-2.5-191206.udpipe"
            local_model = "czech-pdt-ud-2.5-191206.udpipe"
        elif language == 'en':
            model_url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe"
            local_model = "english-ewt-ud-2.5-191206.udpipe"
        else:
            print(f"Unsupported language: {language}")
            return False
        
        try:
            urllib.request.urlretrieve(model_url, local_model)
            model_path = local_model
            print(f"Model downloaded and saved as {local_model}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    
    # Load UDPipe model
    print(f"Loading UDPipe model: {model_path}")
    model = Model.load(model_path)
    if not model:
        print(f"Error: Failed to load UDPipe model from {model_path}")
        return False
    
    # Create pipeline for lemmatization
    pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    error = ProcessingError()
    
    # Load input words
    with open(input_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f]
    
    # Lemmatize each word and write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in words:
            # Process word using UDPipe
            processed = pipeline.process(word, error)
            if error.occurred():
                print(f"Error processing word '{word}': {error.message}")
                # In case of error, write the original word
                f.write(f"{word}\n")
                continue
            
            # Extract lemma from output
            lemma = word  # Default value if extraction fails
            
            # UDPipe returns output in CONLL-U format
            # Lines have format: ID FORM LEMMA ... (tab-separated)
            for line in processed.split('\n'):
                if line.startswith('#'):
                    continue  # Skip comments
                
                parts = line.split('\t')
                if len(parts) >= 3 and parts[0].isdigit():  # Valid token line
                    lemma = parts[2]  # Lemma is third column
                    break
            
            f.write(f"{lemma}\n")
    
    return True

def process_lemmatization_results(lemmatized_file: str, word_counts: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Process lemmatization results and assign lemmas to original word counts.
    
    Args:
        lemmatized_file: File with lemmatized words
        word_counts: Original dictionary of words with their counts
    
    Returns:
        Tuple containing:
        - Dictionary mapping lemmas to sum of occurrence counts
        - Dictionary mapping lemmas to lists of words that were lemmatized to this lemma
    """
    lemma_to_words = defaultdict(list)
    lemma_counts = defaultdict(int)
    
    # Load lemmas and assign to words
    original_words = list(word_counts.keys())
    
    with open(lemmatized_file, 'r', encoding='utf-8') as f:
        lemmatized_words = [line.strip() for line in f]
    
    # For each word assign its lemma
    for original, lemma in zip(original_words, lemmatized_words):
        lemma_to_words[lemma].append(original)
        lemma_counts[lemma] += word_counts[original]
    
    # Identify which words have the same lemma (for logging)
    lemma_groups = {}
    for lemma, words in lemma_to_words.items():
        if len(words) > 1:
            lemma_groups[lemma] = words
    
    return lemma_counts, lemma_groups

def process_lemmatization(input_file: str, output_file: str, language: str = 'cz', 
                         output_dir: str = 'results', model_path: str = None):
    """
    Perform the complete lemmatization process from input preparation to result saving.
    
    Args:
        input_file: Input file with vocabulary (word count)
        output_file: Output file for lemmatized words and counts
        language: Language for lemmatization ('cz', 'en')
        output_dir: Directory for temporary and output files
        model_path: Path to UDPipe model (if None, default for language is used)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model path based on language, if not provided
    if model_path is None:
        if language == 'cz':
            model_path = "czech-pdt-ud-2.5-191206.udpipe"
        elif language == 'en':
            model_path = "english-ewt-ud-2.5-191206.udpipe"
        else:
            print(f"Unsupported language: {language}")
            return False
    
    # Prepare input file for lemmatizer
    temp_input_file = os.path.join(output_dir, "temp_lemmatization_input.txt")
    temp_output_file = os.path.join(output_dir, "temp_lemmatization_output.txt")
    
    print(f"Preparing input file for lemmatizer from {input_file}...")
    word_counts = prepare_lemmatization_input(input_file, temp_input_file)
    print(f"Number of unique words before lemmatization: {len(word_counts)}")
    
    # Perform lemmatization
    if lemmatize_words(temp_input_file, temp_output_file, model_path, language):
        print("Lemmatization completed, processing results...")
        
        # Process lemmatization results
        lemma_counts, lemma_groups = process_lemmatization_results(temp_output_file, word_counts)
        print(f"Number of unique lemmas after lemmatization: {len(lemma_counts)}")
        print(f"Reduction: {(len(word_counts) - len(lemma_counts)) / len(word_counts) * 100:.1f}%")
        
        # Write result file
        with open(output_file, 'w', encoding='utf-8') as f:
            for lemma, count in sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{lemma} {count}\n")
        
        # Write information about word grouping
        with open(os.path.join(output_dir, 'lemmatization_groups.json'), 'w', encoding='utf-8') as f:
            json.dump(lemma_groups, f, ensure_ascii=False, indent=2)
            
        print(f"Result written to {output_file}")
        print(f"Groups of words with the same lemma were written to {os.path.join(output_dir, 'lemmatization_groups.json')}")
        
        # Remove temporary files
        for file in [temp_input_file, temp_output_file]:
            if os.path.exists(file):
                os.remove(file)
                
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Data preparation for lemmatization and reconstruction of occurrence counts')
    parser.add_argument('-i', '--input', required=True, help='Input file with vocabulary (word count)')
    parser.add_argument('-o', '--output', required=True, help='Output file with lemmatized words and counts')
    parser.add_argument('-l', '--language', default='cz', help='Language for lemmatization (cz, en)')
    parser.add_argument('--output-dir', default='results', help='Directory for temporary and output files')
    parser.add_argument('--model-path', help='Path to UDPipe model')
    args = parser.parse_args()
    
    process_lemmatization(args.input, args.output, args.language, args.output_dir, args.model_path)

if __name__ == '__main__':
    main()
