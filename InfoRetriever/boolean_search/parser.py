import re
import os
import json
try:
    from ..preprocessing.preprocess import (
        LowercasePreprocessor, 
        RemoveDiacriticsPreprocessor,
        StopWordsPreprocessor,
        NonsenseTokenPreprocessor,
        PreprocessingPipeline
    )
    from ..preprocessing.stem_preprocessor import StemPreprocessor
    from ..preprocessing.tokenizer import Token, TokenType, RegexMatchTokenizer
    
    # Load config to determine preprocessing steps
    def load_config():
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}. Using default settings.")
        
        # Default config if no file exists
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
    
    # Create preprocessing pipeline based on config
    def create_preprocessing_pipeline(config=None):
        if not config:
            config = load_config()
        
        preprocessors = []
        
        # Get preprocessing config
        preproc_config = config.get("preprocessing", {})
        stemming_config = config.get("stemming", {})
        
        # Add preprocessors in the order specified in config
        pipeline_order = config.get("pipeline_order", [])
        
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
        
        return PreprocessingPipeline(preprocessors, name="QueryPipeline")
    
    # Create tokenizer
    tokenizer = RegexMatchTokenizer()
    
    # Create preprocessors pipeline
    pipeline = create_preprocessing_pipeline()
    
    def preprocess_term(term):
        """Normalize a term for consistent lookup using the configured pipeline"""
        # Create a token
        token = Token(
            token_type=TokenType.WORD,
            processed_form=term,
            position=0,
            length=len(term)
        )
        
        # Apply full preprocessing pipeline
        tokens = [token]
        pipeline.preprocess(tokens, "")
        
        return token.processed_form if token.processed_form else ""
        
except ImportError:
    # Fallback if preprocessing components aren't available
    def preprocess_term(term):
        """Simple term normalization without preprocessing components"""
        import unicodedata
        
        # Convert to lowercase
        term = term.lower()
        
        # Remove diacritics
        term = unicodedata.normalize('NFKD', term)
        term = ''.join([c for c in term if not unicodedata.combining(c)])
        
        return term

# AST Node definitions
class Node:
    def evaluate(self, index, all_docs=None):
        raise NotImplementedError()

class TermNode(Node):
    def __init__(self, value):
        # Store original value for debugging
        self.original = value
        # Apply preprocessing to the term value to ensure consistent lookup
        self.value = preprocess_term(value)

    def __repr__(self):
        return f"Term({self.value})"

    def evaluate(self, index, all_docs=None):
        """Return the set of documents containing this term"""
        # Simplified implementation - just do a direct exact match
        if self.value in index:
            return set(index[self.value])
        
        # No match found
        return set()  # Return empty set if term not found

class NotNode(Node):
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"Not({self.child})"

    def evaluate(self, index, all_docs=None):
        """Apply the NOT operation (complement)"""
        # Evaluate the child node
        child_docs = self.child.evaluate(index, all_docs)
        
        # Get the universal set (all document IDs)
        # Use provided all_docs if available
        if all_docs is not None:
            return all_docs - child_docs
            
        # First check if index has a cached all_docs attribute
        if hasattr(index, 'all_docs'):
            all_docs = index.all_docs
        else:
            # Build the universal set from all document IDs in the index
            all_docs = set()
            for docs in index.values():
                all_docs.update(docs)
            
            # Cache it for future use if index is a dictionary-like object
            try:
                index.all_docs = all_docs
            except (AttributeError, TypeError):
                pass  # Can't cache, index doesn't support attribute assignment
        
        # Return the complement: universal set minus child docs
        return all_docs - child_docs

class AndNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"And({self.left}, {self.right})"

    def evaluate(self, index, all_docs=None):
        """Perform the AND operation (intersection)"""
        # Get the left result first
        left_docs = self.left.evaluate(index, all_docs)
        
        # Short-circuit: if left is empty, result will be empty
        if not left_docs:
            return set()
            
        # Get the right result
        # Pass left_docs as the all_docs for the right side (optimization)
        right_docs = self.right.evaluate(index, all_docs)
        
        # Return the intersection
        # Use the smaller set as the basis for more efficient intersection
        if len(left_docs) > len(right_docs):
            return right_docs.intersection(left_docs)
        else:
            return left_docs.intersection(right_docs)

class OrNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Or({self.left}, {self.right})"

    def evaluate(self, index, all_docs=None):
        """Perform the OR operation (union)"""
        # Get left and right results
        left_docs = self.left.evaluate(index, all_docs)
        right_docs = self.right.evaluate(index, all_docs)
        
        # Return the union
        return left_docs.union(right_docs)

class BooleanParser:
    """
    Lexical and syntax analyzer for boolean queries with the following grammar:
    expr: term (OR term)*
    term: factor (AND factor)*
    factor: [NOT] base
    base: LPAREN expr RPAREN | TERM
    """
    def __init__(self, text):
        self.tokens = self.tokenize(text)
        self.pos = 0

    def tokenize(self, text):
        # First, check for balanced parentheses
        if text.count('(') != text.count(')'):
            # Try to auto-correct by adding missing closing parentheses
            if text.count('(') > text.count(')'):
                text = text + ')' * (text.count('(') - text.count(')'))
            else:
                # If we have more closing than opening, add opening parentheses at the start
                text = '(' * (text.count(')') - text.count('(')) + text
        
        token_spec = [
            ('AND', r'\bAND\b'),
            ('OR', r'\bOR\b'),
            ('NOT', r'\bNOT\b'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('TERM', r'[\w\.\-]+'),
            ('SKIP', r'\s+'),
        ]
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
        tokens = []
        for mo in re.finditer(tok_regex, text):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'SKIP':
                continue
            tokens.append((kind, value))
        return tokens

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (None, None)

    def consume(self, expected_kind=None):
        token = self.current_token()
        if token[0] is None:
            return token
        if expected_kind and token[0] != expected_kind:
            raise SyntaxError(f"Expected token {expected_kind} but got {token[0]}")
        self.pos += 1
        return token

    def parse(self):
        result = self.parse_expr()
        if self.current_token()[0] is not None:
            raise SyntaxError("Unexpected token at the end")
        return result

    def parse_expr(self):
        node = self.parse_term()
        while True:
            token = self.current_token()
            if token[0] == 'OR':
                self.consume('OR')
                right = self.parse_term()
                node = OrNode(node, right)
            else:
                break
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            token = self.current_token()
            if token[0] == 'AND':
                self.consume('AND')
                right = self.parse_factor()
                node = AndNode(node, right)
            else:
                break
        return node

    def parse_factor(self):
        token = self.current_token()
        if token[0] == 'NOT':
            self.consume('NOT')
            # Parse only one NOT by parsing a base after consuming NOT.
            child = self.parse_base()
            return NotNode(child)
        else:
            return self.parse_base()

    def parse_base(self):
        token = self.current_token()
        if token[0] == 'LPAREN':
            self.consume('LPAREN')
            node = self.parse_expr()
            self.consume('RPAREN')
            return node
        elif token[0] == 'TERM':
            term = self.consume('TERM')[1]
            return TermNode(term)
        else:
            raise SyntaxError("Unexpected token: " + str(token))
