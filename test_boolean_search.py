#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Boolean Search functionality
"""

import sys
import json
import os
from InfoRetriever.boolean_search.boolean_search import BooleanSearchEngine
from InfoRetriever.boolean_search.parser import BooleanParser

def test_boolean_search():
    """Test Boolean Search with a simple query"""
    # Load the index
    index_path = "inverted_index.json"
    if not os.path.exists(index_path):
        print(f"Error: Index file {index_path} not found")
        return
    
    # Create the search engine
    engine = BooleanSearchEngine(index_path)
    
    # Test query
    test_query = "Porsche OR Volkswagen"
    print(f"Testing query: '{test_query}'")
    
    # Parse the query
    parser = BooleanParser(test_query)
    ast = parser.parse()
    print(f"Parsed AST: {ast}")
    
    # Debug the terms
    for term in ["Porsche", "Volkswagen"]:
        print(f"\nDebugging term: {term}")
        preprocessed = engine.debug_term(term)
        
        # Check if term is in index
        if preprocessed in engine.index:
            doc_count = len(engine.index[preprocessed])
            print(f"Found {doc_count} documents with term '{preprocessed}'")
            
            # Print sample documents
            print("Sample document IDs:", list(engine.index[preprocessed])[:5])
        else:
            print(f"Term '{preprocessed}' not found in index")
    
    # Execute the search
    result_set, execution_time = engine.search(test_query, debug=True)
    
    # Print results
    print(f"\nSearch Results:")
    print(f"Found {len(result_set)} documents in {execution_time:.6f} seconds")
    
    if result_set:
        print("Sample document IDs:", list(result_set)[:10])
    else:
        print("No results found")
    
    return result_set

if __name__ == "__main__":
    test_boolean_search()
