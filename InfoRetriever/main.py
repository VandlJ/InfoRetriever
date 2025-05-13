import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any

# Import InfoRetriever modules
from InfoRetriever.preprocessing.document import Document
from InfoRetriever.boolean_search.boolean_search import BooleanSearchEngine
from InfoRetriever.build_inverted_index import InvertedIndexBuilder
from InfoRetriever.tfidf_search.tfidf_search import TFIDFSearchEngine


class InfoRetriever:
    """
    Unified interface for InfoRetriever search engines.
    Combines Boolean and TF-IDF search capabilities.
    """
    def __init__(self):
        self.documents = []
        self.boolean_engine = None
        self.tfidf_engine = None
        self.documents_loaded = False
    
    def load_documents(self, documents_path: str) -> bool:
        """
        Load documents from a JSON file.
        
        Args:
            documents_path: Path to the JSON file with documents
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            print(f"Loading documents from: {documents_path}")
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            print(f"Loaded {len(self.documents)} documents.")
            self.documents_loaded = True
            
            return True
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def init_boolean_engine(self, index_path: str = None) -> bool:
        """
        Initialize the Boolean search engine.
        
        Args:
            index_path: Path to a pre-built inverted index (optional)
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            config = None
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load config file: {e}. Using default settings.")
            
            if index_path and os.path.exists(index_path):
                # Load existing index
                print(f"Loading Boolean inverted index from: {index_path}")
                self.boolean_engine = BooleanSearchEngine(index_path, config=config)
                return True
            
            if not self.documents_loaded:
                print("No documents loaded. Load documents first or provide an index file.")
                return False
            
            # Build new index from documents
            print("Building Boolean inverted index from documents...")
            index_builder = InvertedIndexBuilder(config=config)
            
            # Create a temporary file to store the documents
            temp_path = os.path.join(os.getcwd(), "temp_documents.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f)
            
            # Build the index
            index_builder.build_from_json(temp_path)
            
            # Save the index
            index_output_path = os.path.join(os.getcwd(), "inverted_index.json")
            index_builder.save_to_json(index_output_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Initialize the engine with the new index
            self.boolean_engine = BooleanSearchEngine(index_output_path, config=config)
            
            return True
        except Exception as e:
            print(f"Error initializing Boolean engine: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def init_tfidf_engine(self) -> bool:
        """
        Initialize the TF-IDF search engine.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self.documents_loaded:
                print("No documents loaded. Load documents first.")
                return False
            
            # Use the same config as for Boolean search
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            config = None
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load config file: {e}. Using default settings.")
            
            print("Initializing TF-IDF search engine...")
            self.tfidf_engine = TFIDFSearchEngine(
                documents=self.documents,
                config=config
            )
            
            return True
        except Exception as e:
            print(f"Error initializing TF-IDF engine: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_boolean(self, query: str, top_k: int = 5, debug: bool = False) -> List[Document]:
        """
        Perform a Boolean search.
        
        Args:
            query: Boolean query string
            top_k: Maximum number of results to return
            debug: Whether to show debug information
            
        Returns:
            List of matching Document objects
        """
        if not self.boolean_engine:
            print("Boolean engine not initialized.")
            return []
        
        print(f"Executing Boolean search: '{query}'")
        try:
            # Debug query parsing if requested
            if debug:
                from InfoRetriever.boolean_search.parser import BooleanParser, preprocess_term
                print("\nDebugging query terms:")
                simple_terms = [term.strip() for term in re.split(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', query) 
                              if term.strip() and term.strip() not in ['AND', 'OR', 'NOT']]
                for term in simple_terms:
                    processed = preprocess_term(term)
                    print(f"  '{term}' -> '{processed}'")
                
                parser = BooleanParser(query)
                ast = parser.parse()
                print(f"Query structure: {ast}")
            
            # Execute search
            result_set, execution_time = self.boolean_engine.search(query)
            
            print(f"Found {len(result_set)} documents in {execution_time:.6f} seconds")
            
            # Convert results to Document objects and limit to top_k
            documents = []
            for doc_id in sorted(result_set)[:top_k]:
                try:
                    # Try to find the document by ID in the original documents
                    if isinstance(doc_id, int) and doc_id < len(self.documents):
                        doc_data = self.documents[doc_id]
                        doc = Document(
                            id=doc_id,  # Using id instead of doc_id
                            title=doc_data.get('title', ''),
                            content=doc_data.get('text', ''),
                            metadata={
                                'abstract': doc_data.get('abstract', ''),
                                'source': 'boolean_search'
                            }
                        )
                        documents.append(doc)
                except Exception as doc_e:
                    print(f"Error creating Document for doc_id={doc_id}: {doc_e}")
            
            return documents
        except Exception as e:
            print(f"Error during Boolean search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_tfidf(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Perform a TF-IDF search.
        
        Args:
            query: Free-text query string
            top_k: Maximum number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if not self.tfidf_engine:
            print("TF-IDF engine not initialized.")
            return []
        
        print(f"Executing TF-IDF search: '{query}'")
        try:
            results = self.tfidf_engine.search(query, top_k=top_k)
            
            print(f"Found {len(results)} matching documents")
            
            return results
        except Exception as e:
            print(f"Error during TF-IDF search: {e}")
            import traceback
            traceback.print_exc()
            return []


def display_results(results, engine_type):
    """Display search results in a formatted way"""
    if not results:
        print(f"\nNo results found for {engine_type} search.")
        return
    
    print(f"\n{engine_type.upper()} SEARCH RESULTS:")
    print("=" * 60)
    
    if engine_type.lower() == 'boolean':
        # For Boolean search (list of documents)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.title}")
            
            # Print abstract if available
            abstract = doc.metadata.get('abstract', '')
            if abstract:
                print(f"   Abstract: {abstract[:100]}{'...' if len(abstract) > 100 else ''}")
            
            # Print snippet from content
            content_snippet = doc.content[:150].replace('\n', ' ')
            if content_snippet:
                print(f"   Content: {content_snippet}...")
            
            print()
    else:
        # For TF-IDF search (list of document, score tuples)
        for i, (doc, score) in enumerate(results):
            print(f"{i+1}. {doc.title}")
            print(f"   Similarity: {score:.4f}")
            
            # Print abstract if available
            abstract = doc.metadata.get('abstract', '')
            if abstract:
                print(f"   Abstract: {abstract[:100]}{'...' if len(abstract) > 100 else ''}")
            
            # Print snippet from content
            content_snippet = doc.content[:150].replace('\n', ' ')
            if content_snippet:
                print(f"   Content: {content_snippet}...")
            
            print()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='InfoRetriever - Combined Boolean and TF-IDF Search System'
    )
    parser.add_argument('--documents', help='Path to documents JSON file')
    parser.add_argument('--index', help='Path to pre-built inverted index (for Boolean search)')
    parser.add_argument('--engine', choices=['boolean', 'tfidf', 'both'], default='both',
                      help='Which search engine to use')
    parser.add_argument('--query', help='Query string to search for')
    parser.add_argument('--boolean-query', help='Boolean query for Boolean search')
    parser.add_argument('--tfidf-query', help='Free text query for TF-IDF search')
    parser.add_argument('--top', type=int, default=5,
                      help='Number of top results to display')
    parser.add_argument('--debug', action='store_true',
                      help='Show debug information for queries')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    args = parser.parse_args()
    
    # Create retriever
    retriever = InfoRetriever()
    
    # Load documents if provided
    if args.documents:
        if not retriever.load_documents(args.documents):
            sys.exit(1)
    
    # Initialize search engines
    engines_initialized = False
    
    if args.engine in ['boolean', 'both']:
        if retriever.init_boolean_engine(args.index):
            engines_initialized = True
    
    if args.engine in ['tfidf', 'both']:
        if retriever.init_tfidf_engine():
            engines_initialized = True
    
    if not engines_initialized:
        print("Failed to initialize any search engines.")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("\nInfoRetriever Interactive Mode")
        print("Type 'quit' to exit")
        
        while True:
            print("\n" + "=" * 60)
            print("Choose search mode:")
            print("1. Boolean Search")
            print("2. TF-IDF Search")
            print("3. Both")
            print("4. Quit")
            
            choice = input("\nEnter choice (1-4): ")
            
            if choice == '4' or choice.lower() == 'quit':
                break
            
            if choice not in ['1', '2', '3']:
                print("Invalid choice. Please enter a number between 1 and 4.")
                continue
            
            query = input("\nEnter search query: ")
            if not query:
                print("Empty query. Please try again.")
                continue
            
            top_k = args.top
            try:
                top_k_input = input(f"Number of results to show (default: {top_k}): ")
                if top_k_input:
                    top_k = int(top_k_input)
            except ValueError:
                print(f"Invalid number. Using default: {top_k}")
            
            if choice in ['1', '3'] and retriever.boolean_engine:
                boolean_results = retriever.search_boolean(query, top_k, debug=args.debug)
                display_results(boolean_results, "Boolean")
            
            if choice in ['2', '3'] and retriever.tfidf_engine:
                tfidf_results = retriever.search_tfidf(query, top_k)
                display_results(tfidf_results, "TF-IDF")
    
    # Query mode
    else:
        if args.boolean_query and retriever.boolean_engine:
            boolean_results = retriever.search_boolean(args.boolean_query, args.top, debug=args.debug)
            display_results(boolean_results, "Boolean")
        
        if args.tfidf_query and retriever.tfidf_engine:
            tfidf_results = retriever.search_tfidf(args.tfidf_query, args.top)
            display_results(tfidf_results, "TF-IDF")
        
        if args.query:
            if args.engine in ['boolean', 'both'] and retriever.boolean_engine:
                boolean_results = retriever.search_boolean(args.query, args.top, debug=args.debug)
                display_results(boolean_results, "Boolean")
            
            if args.engine in ['tfidf', 'both'] and retriever.tfidf_engine:
                tfidf_results = retriever.search_tfidf(args.query, args.top)
                display_results(tfidf_results, "TF-IDF")


if __name__ == "__main__":
    main()
