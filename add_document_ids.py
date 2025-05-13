#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to add sequential document IDs to documents in JSON file

This script adds sequential document IDs (id1, id2, etc.) to documents in the autorevue.json file.
These IDs are required for proper matching with the boolean search engine results.
"""

import json
import os
import sys

def add_document_ids(input_file, output_file=None):
    """
    Add sequential IDs to documents in JSON file
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to overwriting input file)
    
    Returns:
        int: Number of documents processed
    """
    # Set default output file
    if not output_file:
        output_file = input_file
    
    try:
        # Load the documents
        print(f"Loading documents from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Check if it's a list of documents
        if not isinstance(documents, list):
            print("Error: Input file must contain a JSON array of documents")
            return 0
        
        # Count valid documents (non-empty entries)
        valid_docs = 0
        
        # Add IDs to each document
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and doc:  # Skip empty documents
                # Add 'id' field if it doesn't exist
                if 'id' not in doc:
                    doc['id'] = f"id{i+1}"
                valid_docs += 1
        
        # Save the modified documents
        print(f"Saving {valid_docs} documents with IDs to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully added IDs to {valid_docs} documents")
        return valid_docs
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return 0

def main():
    """Main entry point"""
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default paths based on repository structure
        input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "InfoRetriever", "data", "autorevue.json")
        output_file = None
    
    # Process the file
    docs_processed = add_document_ids(input_file, output_file)
    
    if docs_processed > 0:
        print("Done!")
        return 0
    else:
        print("Failed to process documents!")
        return 1

if __name__ == "__main__":
    sys.exit(main())