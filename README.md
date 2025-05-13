# InfoRetriever

<div align="center">

![InfoRetriever](https://img.shields.io/badge/InfoRetriever-v1.0-blue)
![Python](https://img.shields.io/badge/Python-3.13-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**A powerful and flexible information retrieval system with Boolean and TF-IDF search capabilities**

</div>

---

## 📚 Overview

InfoRetriever is a comprehensive information retrieval library that combines Boolean and TF-IDF search capabilities into a unified system. It features Czech language support with stemming and lemmatization, powerful preprocessing options, and a user-friendly interface.

<details open>
<summary><strong>🌟 Features</strong></summary>

- **Preprocessing Pipeline**: Text cleaning, tokenization, stop word removal, stemming, and lemmatization
- **Czech Language Support**: Built-in Czech stemmer for better text processing
- **Boolean Search**: Full support for AND, OR, NOT operators and parentheses 
- **TF-IDF Search**: Vector space model with TF-IDF weighting for ranked retrieval
- **Beautiful CLI**: Rich-formatted terminal interface for interactive use
- **Unified API**: Consistent programming interface for both search methods
- **Evaluation Framework**: Comprehensive system for testing retrieval quality

</details>

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/VandlJ/InfoRetriever.git

# Navigate to the project directory
cd InfoRetriever

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Using the Enhanced CLI

The most user-friendly way to use InfoRetriever is through our rich command-line interface:

```bash
# Make the script executable
chmod +x run_cli.sh

# Run the CLI in interactive mode
./run_cli.sh
```

<details>
<summary><strong>CLI Features</strong></summary>

- Beautiful text formatting with colored output
- Progress indicators for long-running operations
- Support for both Boolean and TF-IDF search queries
- Interactive mode with menu-based navigation
- Command-line arguments for scripting and automation
- Formatted result display with tables and highlighting

</details>

---

## 🌐 Project Structure

```
InfoRetriever/
  ├── __init__.py
  ├── main.py
  ├── build_inverted_index.py
  ├── boolean_search/
  │   ├── __init__.py
  │   ├── boolean_search.py
  │   ├── parser.py
  │   └── query_optimizer.py
  ├── preprocessing/
  │   ├── __init__.py
  │   ├── tokenizer.py
  │   ├── preprocess.py
  │   ├── stemming.py
  │   ├── lemmatization.py
  │   ├── stem_preprocessor.py
  │   ├── document.py
  │   ├── czech_stemmer.py
  │   └── create_default_config.py
  └── tfidf_search/
      ├── __init__.py
      └── tfidf_search.py
```

## 📊 Code Examples

### Basic Usage

```python
from InfoRetriever.main import InfoRetriever

# Create a retriever instance
retriever = InfoRetriever()

# Load documents
retriever.load_documents("path/to/documents.json")

# Initialize search engines
retriever.init_boolean_engine()
retriever.init_tfidf_engine()

# Boolean search
boolean_results = retriever.search_boolean("apple AND (banana OR NOT cherry)")

# TF-IDF search
tfidf_results = retriever.search_tfidf("machine learning algorithms")
```

### Command Line Arguments

You can use command-line arguments for quick searches:

```bash
# Boolean search
python cli_app.py --documents InfoRetriever/data/autorevue.json --engine boolean --query "auto AND motor"

# TF-IDF search
python cli_app.py --documents InfoRetriever/data/autorevue.json --engine tfidf --query "electric vehicles"

# Both search engines
python cli_app.py --documents InfoRetriever/data/autorevue.json --engine both --query "hybrid cars"
```

### Document Format

The system expects documents in JSON format:

```json
[
  {
    "title": "Document Title",
    "abstract": "A brief summary of the document",
    "text": "The full content of the document..."
  },
  ...
]
```

---

## 🛠️ System Architecture

<details>
<summary><strong>Preprocessing Module</strong></summary>

- **Tokenization**: Breaking text into individual tokens
- **Normalization**: Converting to lowercase, removing diacritics
- **Stop Words**: Removing common words that don't carry meaning
- **Stemming**: Reducing words to their stems
- **Lemmatization**: Converting words to their base forms

</details>

<details>
<summary><strong>Czech Stemmer</strong></summary>

The Czech stemmer component improves search quality by reducing Czech words to their base forms:

1. It removes case endings (e.g., "-ům", "-ami", "-ách")
2. Handles palatalization specific to Czech
3. Removes possessive endings
4. Applies specific rules for Czech derivational morphology
5. Can be run in aggressive or light mode

The Czech stemmer is based on the algorithm described in:
- Dolamic, L., & Savoy, J. (2009). Indexing and stemming approaches for the Czech language. Information Processing & Management, 45(6), 714-720.

</details>

<details>
<summary><strong>Boolean Search</strong></summary>

- **Query Parser**: Builds abstract syntax trees from boolean expressions
- **Operators**: AND, OR, NOT with proper precedence 
- **Optimization**: De Morgan's laws for efficient NOT operations
- **Inverted Index**: Fast document retrieval by term

</details>

<details>
<summary><strong>TF-IDF Search</strong></summary>

- **Vector Space Model**: Documents represented as term vectors
- **Term Frequency**: Measures how often a term appears in a document
- **Inverse Document Frequency**: Measures term importance across the collection
- **Cosine Similarity**: Ranks documents by similarity to the query vector

</details>

---

## ⚙️ Configuration

InfoRetriever's preprocessing pipeline can be customized through the `config.json` file:

<details>
<summary><strong>Configuration Options</strong></summary>

### Preprocessing

| Key | Description |
|------|-------|
| `lowercase` | Whether to convert text to lowercase (`true`/`false`) |
| `remove_diacritics` | Whether to remove diacritics (`true`/`false`) |

### Stop Words

| Key | Description |
|------|-------|
| `use` | Whether to remove stop words (`true`/`false`) |
| `language` | Which stop words languages to use: `"czech"`, `"english"`, `"both"`, `"none"` |

### Stemming

| Key | Description |
|------|-------|
| `use` | Whether to use stemming (`true`/`false`) |
| `language` | Language for stemming: `"cz"`, `"en"` |

### Pipeline Order

The `pipeline_order` setting controls the sequence of preprocessing operations:

```json
"pipeline_order": [
  "tokenize", "lowercase", "remove_diacritics", 
  "nonsense_tokens", "stemming"
]
```

</details>

---

## 📊 Evaluation System

InfoRetriever includes a comprehensive evaluation framework for assessing retrieval performance.

<details>
<summary><strong>Evaluation Metrics</strong></summary>

### For Boolean Search:
- Result count for each query
- Query execution time

### For Ranked Search:
- Mean Average Precision (MAP)
- Precision at 10 documents (P@10) 
- Query execution time
- Relevance metrics via TREC evaluation

</details>

<details>
<summary><strong>Running Evaluations</strong></summary>

```bash
# Basic usage with default settings
python run_evaluation.py

# Specify query set and evaluation type
python run_evaluation.py --boolean_queries boolean_queries_standard_100.txt --eval_type both
```

The evaluation system generates:
1. CSV files with per-query results
2. JSON summary of overall evaluation performance
3. TREC-format results files for use with the trec_eval tool

</details>

---

## 📋 Project Accomplishments

- Successfully merged Boolean and TF-IDF search capabilities into a unified system
- Implemented Czech stemmer integration for improved language support
- Created a comprehensive demo script showcasing various features
- Fixed compatibility issues across modules
- Enhanced documentation and created a beautiful CLI interface

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p><i>InfoRetriever: Making Information Retrieval Powerful and Accessible</i></p>
</div>
