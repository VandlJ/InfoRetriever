import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    NUMBER = 1
    WORD = 2
    TAG = 3
    PUNCT = 4
    URL = 5    # Type for URL addresses
    DATE = 6   # Type for dates

@dataclass
class Token:
    processed_form: str
    position: int
    length: int
    token_type: TokenType = TokenType.WORD

    def __repr__(self):
        return self.processed_form


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> list[Token]:
        raise NotImplementedError()

class SplitTokenizer(Tokenizer):
    def __init__(self, split_char: str):
        self.split_char = split_char

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        position = 0
        for word in document.split(self.split_char):
            token = Token(word, position, len(word))
            tokens.append(token)
            position += len(word) + 1
        return tokens

class RegexMatchTokenizer(Tokenizer):
    # Patterns for individual token types
    url_pattern = r'https?://[^\s]+|www\.[^\s]+\.[^\s]+'
    date_pattern = r'\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}|\d{1,2}\.\s*(?:ledna|února|března|dubna|května|června|července|srpna|září|října|listopadu|prosince)\s*\d{2,4}'
    num_pattern = r'\d+[.,]?\d*'  # matches numbers like 123, 123.123, 123,123
    word_pattern = r'\w+'  # matches words
    html_tag_pattern = r'<.*?>'  # matches html tags
    punctuation_pattern = r'[^\w\s]+'  # matches punctuation

    def __init__(self):
        # List of pairs (pattern, token type)
        self.patterns = [
            (self.url_pattern, TokenType.URL),
            (self.date_pattern, TokenType.DATE),
            (self.num_pattern, TokenType.NUMBER),
            (self.html_tag_pattern, TokenType.TAG),
            (self.word_pattern, TokenType.WORD),
            (self.punctuation_pattern, TokenType.PUNCT)
        ]
        
        # Building combined regex pattern
        patterns_str = '|'.join(f'({pattern})' for pattern, _ in self.patterns)
        self.regex = re.compile(patterns_str, re.UNICODE | re.IGNORECASE)

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        for match in re.finditer(self.regex, document):
            # Determine which group was captured (first non-empty group)
            group_index = 1
            for i in range(1, len(self.patterns) + 1):
                if match.group(i):
                    group_index = i
                    break
            
            # Token type based on pattern index in self.patterns
            token_type = self.patterns[group_index - 1][1]
            
            token = Token(match.group(0), match.start(), len(match.group(0)), token_type)
            tokens.append(token)
        
        return tokens
