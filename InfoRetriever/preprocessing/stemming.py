import json
from collections import defaultdict
import argparse
import os
import subprocess
from typing import Dict, List, Tuple

def prepare_stemming_input(input_vocab_file: str, output_words_file: str) -> Dict[str, int]:
    """
    Prepare input file for stemmer and return a dictionary mapping words to their counts.
    
    Args:
        input_vocab_file: Input file with vocabulary (word count)
        output_words_file: Output file with just words for stemmer
    
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
    
    # Write only words for stemmer
    with open(output_words_file, 'w', encoding='utf-8') as f:
        for word in word_counts.keys():
            f.write(f"{word}\n")
    
    return word_counts

def process_stemming_results(stemmed_file: str, word_counts: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Process stemming results and assign stems to original word counts.
    
    Args:
        stemmed_file: File with stemmed words
        word_counts: Original dictionary of words with their counts
    
    Returns:
        Tuple containing:
        - Dictionary mapping stems to sum of occurrence counts
        - Dictionary mapping stems to lists of words that were stemmed to this stem
    """
    stem_to_words = defaultdict(list)
    stem_counts = defaultdict(int)
    
    # Load stems and assign to words
    original_words = list(word_counts.keys())
    
    with open(stemmed_file, 'r', encoding='utf-8') as f:
        stemmed_words = [line.strip() for line in f]
    
    # For each word assign its stem
    for original, stem in zip(original_words, stemmed_words):
        stem_to_words[stem].append(original)
        stem_counts[stem] += word_counts[original]
    
    # Identify which words have the same stem (for logging)
    stem_groups = {}
    for stem, words in stem_to_words.items():
        if len(words) > 1:
            stem_groups[stem] = words
    
    return stem_counts, stem_groups

def run_stemmer(input_file: str, output_file: str, language: str = 'cz', stemmer_path: str = './stemmer') -> bool:
    """
    Run external stemmer with given parameters.
    
    Args:
        input_file: Input file with words
        output_file: Output file for stemmed words
        language: Language for stemming ('cz' for Czech)
        stemmer_path: Path to stemmer binary
    
    Returns:
        True on successful execution, False on error
    """
    # Check stemmer existence
    if not os.path.exists(stemmer_path):
        # Try several possible locations
        potential_paths = [
            "./CzechSnowballStemmer/stemmer",
            "../CzechSnowballStemmer/stemmer",
            "../../CzechSnowballStemmer/stemmer",
            "./stemmer",
            stemmer_path
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                stemmer_path = path
                break
        else:
            print(f"Error: Stemmer not found in any of the locations: {potential_paths}")
            return False
    
    stemmer_cmd = f"{stemmer_path} -l {language} -i {input_file} -o {output_file}"
    
    try:
        print(f"Running stemmer: {stemmer_cmd}")
        subprocess.run(stemmer_cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running stemmer: {e}")
        return False

def process_stemming(input_file: str, output_file: str, language: str = 'cz', 
                     output_dir: str = 'results', stemmer_path: str = './stemmer'):
    """
    Perform the complete stemming process from input preparation to result saving.
    
    Args:
        input_file: Input file with vocabulary (word count)
        output_file: Output file for stemmed words and counts
        language: Language for stemming ('cz', 'en')
        output_dir: Directory for temporary and output files
        stemmer_path: Path to stemmer binary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare input file for stemmer
    temp_input_file = os.path.join(output_dir, "temp_stemming_input.txt")
    temp_output_file = os.path.join(output_dir, "temp_stemming_output.txt")
    
    print(f"Preparing input file for stemmer from {input_file}...")
    word_counts = prepare_stemming_input(input_file, temp_input_file)
    print(f"Number of unique words before stemming: {len(word_counts)}")
    
    # Run stemmer
    if run_stemmer(temp_input_file, temp_output_file, language, stemmer_path):
        print("Stemming completed, processing results...")
        
        # Process stemming results
        stem_counts, stem_groups = process_stemming_results(temp_output_file, word_counts)
        print(f"Number of unique stems after stemming: {len(stem_counts)}")
        print(f"Reduction: {(len(word_counts) - len(stem_counts)) / len(word_counts) * 100:.1f}%")
        
        # Write result file
        with open(output_file, 'w', encoding='utf-8') as f:
            for stem, count in sorted(stem_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{stem} {count}\n")
        
        # Write information about word grouping
        with open(os.path.join(output_dir, 'stemming_groups.json'), 'w', encoding='utf-8') as f:
            json.dump(stem_groups, f, ensure_ascii=False, indent=2)
            
        print(f"Result written to {output_file}")
        print(f"Groups of words with the same stem were written to {os.path.join(output_dir, 'stemming_groups.json')}")
        
        # Remove temporary files
        for file in [temp_input_file, temp_output_file]:
            if os.path.exists(file):
                os.remove(file)
                
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Data preparation for stemming and reconstruction of occurrence counts')
    parser.add_argument('-i', '--input', required=True, help='Input file with vocabulary (word count)')
    parser.add_argument('-o', '--output', required=True, help='Output file with stemmed words and counts')
    parser.add_argument('-l', '--language', default='cz', help='Language for stemming (cz, en)')
    parser.add_argument('--output-dir', default='results', help='Directory for temporary and output files')
    parser.add_argument('--stemmer-path', default='./stemmer', help='Path to stemmer binary')
    args = parser.parse_args()
    
    process_stemming(args.input, args.output, args.language, args.output_dir, args.stemmer_path)

if __name__ == '__main__':
    main()
