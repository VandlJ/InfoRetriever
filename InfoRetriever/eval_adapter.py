#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapter module that allows InfoRetriever to be used through the standard
evaluation interface.
"""

import json
import os
from typing import Iterable, List, Dict, Tuple, Set, Any
import sys

from InfoRetriever.eval_interface.interface import Index, SearchEngine
from InfoRetriever.preprocessing.document import Document
from InfoRetriever.boolean_search.boolean_search import BooleanSearchEngine
from InfoRetriever.build_inverted_index import InvertedIndexBuilder
from InfoRetriever.tfidf_search.tfidf_search import TFIDFSearchEngine


class InfoRetrieverIndex(Index):
    """
    Adapter class that implements the Index interface for InfoRetriever.
    """
    
    def __init__(self, language: str = 'cz'):
        """
        Initialize the InfoRetriever index.
        
        Args:
            language: The language code for document processing
        """
        super().__init__()
        self.documents = {}
        self.inverted_index_builder = InvertedIndexBuilder()
        self.inverted_index_path = None
        self.language = language
        self.indexed = False
    
    def index_documents(self, documents: Iterable[dict[str, str]]) -> None:
        """
        Index documents using both Boolean and TF-IDF indexing.
        
        Args:
            documents: List of documents to index
        """
        # Convert and store documents 
        doc_list = []
        for i, doc_dict in enumerate(documents):
            doc_id = doc_dict.get('id', str(i))
            
            # Create Document object
            doc = Document(
                id=doc_id, 
                title=doc_dict.get('title', ''),
                content=doc_dict.get('text', doc_dict.get('content', '')),
                abstract=doc_dict.get('abstract', ''),
                metadata=doc_dict.get('metadata', {})
            )
            
            # Store the original document for retrieval
            self.documents[doc_id] = doc_dict
            doc_list.append(doc_dict)
        
        # Create a temporary file to index the documents
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join(temp_dir, "temp_documents.json")
        
        try:
            # Load configuration
            config_path = os.path.join(temp_dir, "config.json")
            config = None
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load config file: {e}. Using default settings.")
            
            # Initialize the index builder with the configuration 
            self.inverted_index_builder = InvertedIndexBuilder(config=config)
            
            # Write documents to a temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(doc_list, f)
            
            # Build the boolean index
            self.inverted_index_builder.build_from_json(temp_path)
            
            # Save the index
            self.inverted_index_path = os.path.join(temp_dir, "inverted_index.json")
            self.inverted_index_builder.save_to_json(self.inverted_index_path)
            
            self.indexed = True
            
            print(f"Successfully indexed {len(doc_list)} documents")
        except Exception as e:
            print(f"Error indexing documents: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_document(self, doc_id: str) -> dict[str, str]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document as a dictionary
        """
        return self.documents.get(doc_id, {})


class InfoRetrieverSearchEngine(SearchEngine):
    """
    Adapter class that implements the SearchEngine interface for InfoRetriever.
    """
    
    def __init__(self, index: InfoRetrieverIndex):
        """
        Initialize the InfoRetriever search engine.
        
        Args:
            index: InfoRetrieverIndex instance
        """
        super().__init__(index)
        self.index = index  # The InfoRetrieverIndex instance
        
        # Initialize search engines
        if not self.index.indexed:
            raise ValueError("Index must be built before initializing search engines")
        
        # Initialize Boolean search engine
        self.boolean_engine = BooleanSearchEngine(self.index.inverted_index_path)
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        config = None
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}. Using default settings.")
        
        # Initialize TF-IDF search engine with the same configuration as Boolean search
        # Process documents in batches to avoid memory issues
        print("Initializing TF-IDF search engine (this may take a while)...")
        self.tfidf_engine = TFIDFSearchEngine(config=config)
        
        # Process documents in batches of 1000
        batch_size = 1000
        doc_ids = list(self.index.documents.keys())
        total_docs = len(doc_ids)
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_ids = doc_ids[i:batch_end]
            print(f"Processing TF-IDF batch {i//batch_size + 1}/{(total_docs+batch_size-1)//batch_size}: documents {i+1}-{batch_end} of {total_docs}")
            
            # Get the batch of documents
            batch_docs = [self.index.get_document(doc_id) for doc_id in batch_ids]
            
            # Add documents to the TF-IDF engine
            self.tfidf_engine.add_documents(batch_docs)
            
            # Give feedback on progress
            if (i + batch_size) % (batch_size * 10) == 0 or batch_end == total_docs:
                print(f"Processed {batch_end}/{total_docs} documents ({batch_end/total_docs*100:.1f}%)")
        
        print("TF-IDF initialization complete")
    
    def search(self, query: str) -> list[tuple[str, float]]:
        """
        Perform a TF-IDF search with the given query.
        
        Args:
            query: Search query
            
        Returns:
            List of (document_id, score) tuples
        """
        try:
            # Use TF-IDF search
            results = self.tfidf_engine.search(query, top_k=10000000000000)
            
            # Convert results to the expected format: list of (document_id, score) tuples
            # Use the original_id if available in metadata, otherwise use the document ID
            formatted_results = []
            for doc, score in results:
                if hasattr(doc, 'metadata') and 'original_id' in doc.metadata:
                    doc_id = doc.metadata['original_id']
                else:
                    doc_id = str(doc.id)
                formatted_results.append((doc_id, score))
            return formatted_results
        except Exception as e:
            print(f"Error performing TF-IDF search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def boolean_search(self, query: str) -> set[str]:
        """
        Perform a Boolean search with the given query.
        
        Args:
            query: Boolean query
            
        Returns:
            Set of document IDs
        """
        try:
            # Use Boolean search
            result_set, _ = self.boolean_engine.search(query)
            
            # Convert integer IDs to strings
            return {str(doc_id) for doc_id in result_set}
        except Exception as e:
            print(f"Error performing Boolean search: {e}")
            import traceback
            traceback.print_exc()
            return set()
