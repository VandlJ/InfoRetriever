from typing import List, Dict
import os
import tempfile
import subprocess
from .preprocess import TokenPreprocessor
from .tokenizer import Token

# Import the Czech stemmer
try:
    from .czech_stemmer import cz_stem_word
    CZECH_STEMMER_AVAILABLE = True
    print("Czech stemmer loaded successfully")
except ImportError:
    CZECH_STEMMER_AVAILABLE = False
    print("WARNING: Czech stemmer module not found")

class StemPreprocessor(TokenPreprocessor):
    """
    Preprocessor that stems tokens using either the Czech stemmer or an external stemming tool.
    
    This preprocessor uses the stemming module to apply stemming to tokens.
    """
    
    def __init__(self, language: str = 'cz', stemmer_path: str = None, cache: bool = True):
        """
        Initialize the stemming preprocessor.
        
        Args:
            language: Language code for stemming ('cz' for Czech, 'en' for English)
            stemmer_path: Path to the stemmer binary (if None, will try to locate it)
            cache: Whether to cache stemming results to avoid repeated calls
        """
        self.language = language
        self.cache = cache
        self._stem_cache: Dict[str, str] = {}  # Cache for stemmed words
        
        # Set Czech stemmer as primary method for Czech language
        self.use_czech_stemmer = CZECH_STEMMER_AVAILABLE and language.lower() == 'cz'
        
        if not self.use_czech_stemmer:
            # Try to find the external stemmer binary
            self.stemmer_path = self._find_and_prepare_stemmer(stemmer_path)
            if not self.stemmer_path:
                print("WARNING: No external stemmer found. Stemming will be limited.")
    
    def _find_and_prepare_stemmer(self, stemmer_path: str = None) -> str:
        """
        Find and prepare the stemmer binary.
        
        Args:
            stemmer_path: Optional path to the stemmer binary
            
        Returns:
            Path to the usable stemmer binary or None if not found
        """
        # Define potential locations for the stemmer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            # User-provided path
            stemmer_path,
            # Current directory
            "./stemmer",
            # Preprocessing directory
            os.path.join(current_dir, "stemmer"),
            # CzechSnowballStemmer directory
            os.path.join(current_dir, "CzechSnowballStemmer/stemmer"),
            # Parent directory
            os.path.join(os.path.dirname(current_dir), "stemmer"),
            # Parent's CzechSnowballStemmer directory
            os.path.join(os.path.dirname(current_dir), "CzechSnowballStemmer/stemmer"),
            # Data directory stemmer
            os.path.join(os.path.dirname(current_dir), "data/CzechSnowballStemmer/stemmer"),
        ]
        
        # Filter out None values
        potential_paths = [p for p in potential_paths if p]
        
        for path in potential_paths:
            if os.path.exists(path):
                # Found the stemmer, make sure it's executable
                try:
                    # Make the file executable (chmod +x)
                    os.chmod(path, 0o755)
                    print(f"Found stemmer at: {path}")
                    return path
                except Exception as e:
                    print(f"Found stemmer at {path} but couldn't make it executable: {e}")
        
        # If we get here, we couldn't find the stemmer at any of the expected locations
        # Let's check if we need to build it from source
        czech_snowball_dir = os.path.join(current_dir, "CzechSnowballStemmer")
        if os.path.exists(czech_snowball_dir) and os.path.isdir(czech_snowball_dir):
            print(f"Found CzechSnowballStemmer directory at {czech_snowball_dir}, trying to build the stemmer...")
            try:
                # Try to build the stemmer
                build_cmd = f"cd {czech_snowball_dir} && make"
                subprocess.run(build_cmd, shell=True, check=True)
                
                # Check if build was successful
                built_stemmer_path = os.path.join(czech_snowball_dir, "stemmer")
                if os.path.exists(built_stemmer_path):
                    os.chmod(built_stemmer_path, 0o755)
                    print(f"Successfully built stemmer at: {built_stemmer_path}")
                    return built_stemmer_path
            except Exception as e:
                print(f"Failed to build stemmer: {e}")
        
        print("ERROR: Could not find or build the stemmer. Stemming will be disabled.")
        return None
    
    def preprocess(self, token: Token, document: str) -> Token:
        """
        Apply stemming to a single token.
        
        Args:
            token: Token to preprocess
            document: Original document text (not used in this preprocessor)
            
        Returns:
            Processed token with stemmed form
        """
        # Skip if empty token
        if not token.processed_form:
            return token
            
        # Check cache first if enabled
        if self.cache and token.processed_form in self._stem_cache:
            token.processed_form = self._stem_cache[token.processed_form]
            return token
        
        # Apply stemming to the token
        if self.use_czech_stemmer:
            # Use the Python Czech stemmer implementation
            stemmed_form = cz_stem_word(token.processed_form, aggressive=False)
        elif self.stemmer_path:
            # Use external stemming binary
            stemmed_form = self._stem_single_word(token.processed_form)
        else:
            # No stemming available, return as is
            return token
        
        # Cache the result if enabled
        if self.cache:
            self._stem_cache[token.processed_form] = stemmed_form
            
        # Update the token's processed form
        token.processed_form = stemmed_form
        return token
        
    def _apply_stemming(self, words: List[str]) -> List[str]:
        """
        Apply stemming to a list of words using the external stemmer.
        
        Args:
            words: List of words to stem
            
        Returns:
            List of stemmed words
        """
        if not words or not self.stemmer_path:
            return words
            
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as input_file, \
             tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as output_file:
            
            # Write words to input file
            for word in words:
                input_file.write(f"{word}\n")
            input_file.flush()
            
            # Close files to ensure all data is written
            input_path = input_file.name
            output_path = output_file.name
        
        try:
            # Run stemmer with full path
            stemmer_cmd = f"{self.stemmer_path} -l {self.language} -i {input_path} -o {output_path}"
            # print(f"Executing stemmer command: {stemmer_cmd}")
            
            subprocess.run(
                stemmer_cmd, 
                shell=True, 
                check=True, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Read stemmed words
            with open(output_path, 'r', encoding='utf-8') as f:
                stemmed_words = [line.strip() for line in f]
            
            # If we got results with different length, something went wrong
            if len(stemmed_words) != len(words):
                print(f"Warning: Stemmer returned {len(stemmed_words)} results for {len(words)} input words")
                return words
                
            return stemmed_words
        
        except subprocess.CalledProcessError as e:
            print(f"Error running stemmer: {e}")
            if hasattr(e, 'stderr'):
                print(f"Stderr: {e.stderr}")
            # Return original words on error
            return words
        
        finally:
            # Clean up temporary files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Could not remove temporary file {path}: {e}")
    
    def _stem_single_word(self, word: str) -> str:
        """
        Stem a single word.
        
        Args:
            word: Word to stem
            
        Returns:
            Stemmed word
        """
        if not self.stemmer_path:
            return word
            
        result = self._apply_stemming([word])
        return result[0] if result else word
