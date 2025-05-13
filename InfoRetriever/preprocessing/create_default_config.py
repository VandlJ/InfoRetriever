import json
import os

def create_default_config(config_file="config.json"):
    """
    Create default configuration file in standard JSON format.
    """
    
    # Standard configuration without comments
    standard_config = {
        "input": {
            "file": "data/documents.json",
            "encoding": "utf-8"
        },
        "output": {
            "directory": "results",
            "original_vocab": "vocab_original.txt",
            "preprocessed_vocab": "vocab_preprocessed.txt",
            "stemmed_vocab": "vocab_stemmed.txt"
        },
        "preprocessing": {
            "lowercase": True,
            "remove_diacritics": True,
            "stop_words": {
                "use": False,
                "language": "both"  # "czech", "english", "both", "none"
            },
            "nonsense_tokens": {
                "remove": True,
                "min_word_length": 2
            }
        },
        "stemming": {
            "use": True,
            "language": "cz",  # "cz", "en"
            "stemmer_path": "./czech_stemmer.py"
        },
        "lemmatization": {
            "use": False,
            "language": "cz",
            "model_path": None
        },
        "pipeline_order": [
            "tokenize",
            "lowercase",
            "remove_diacritics",
            "nonsense_tokens",
            "stemming"
        ]
    }
    
    # If configuration already exists, don't overwrite it
    if os.path.exists(config_file):
        print(f"Configuration file {config_file} already exists, it was not overwritten.")
        return
    
    # Create new standard configuration file
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(standard_config, f, indent=4, ensure_ascii=False)
    
    print(f"Created standard configuration file: {config_file}")
    
    # Create documentation in Markdown
    markdown_doc = """# Text Preprocessing Configuration

This document describes configuration options in the `config.json` file.

## Input data (`input`)

| Key | Description |
|------|-------|
| `file` | Path to the input data file |
| `encoding` | Input file encoding (e.g. "utf-8") |

## Output data (`output`)

| Key | Description |
|------|-------|
| `directory` | Output directory for processed files |
| `original_vocab` | Filename for the original vocabulary |
| `preprocessed_vocab` | Filename for the preprocessed vocabulary |
| `stemmed_vocab` | Filename for the stemmed vocabulary |

## Preprocessing (`preprocessing`)

| Key | Description |
|------|-------|
| `lowercase` | Whether to convert text to lowercase (`true`/`false`) |
| `remove_diacritics` | Whether to remove diacritics (`true`/`false`) |

### Stop words (`stop_words`)

| Key | Description |
|------|-------|
| `use` | Whether to remove stop words (`true`/`false`) |
| `language` | Which stop words languages to use: `"czech"`, `"english"`, `"both"`, `"none"` |

### Nonsense tokens (`nonsense_tokens`)

| Key | Description |
|------|-------|
| `remove` | Whether to remove nonsense tokens (`true`/`false`) |
| `min_word_length` | Minimum word length to preserve (number) |

## Stemming

| Key | Description |
|------|-------|
| `use` | Whether to use stemming (`true`/`false`) |
| `language` | Language for stemming: `"cz"`, `"en"` |
| `stemmer_path` | Path to the stemmer binary |

## Lemmatization (`lemmatization`)

| Key | Description |
|------|-------|
| `use` | Whether to use lemmatization (`true`/`false`) |
| `language` | Language for lemmatization: `"cz"`, `"en"` |
| `model_path` | Path to the UDPipe model (null for automatic download) |
"""
    
    with open("CONFIG.md", "w", encoding="utf-8") as f:
        f.write(markdown_doc)
    
    print("Created configuration documentation: CONFIG.md")

if __name__ == "__main__":
    create_default_config()
