#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for the InfoRetriever system.
Uses the evaluation framework to test both Boolean and TF-IDF search capabilities
on the Czech evaluation data.
"""

import argparse
import json
import os
import sys
from datetime import datetime
import time

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InfoRetriever.eval_adapter import InfoRetrieverIndex, InfoRetrieverSearchEngine
from InfoRetriever.eval_interface.evaluate import (
    load_queries, 
    load_data_json, 
    evaluate_ranked, 
    evaluate_boolean, 
    run_trec_eval
)


def main():
    """Main function for evaluating the InfoRetriever system."""
    parser = argparse.ArgumentParser(description="Evaluate InfoRetriever with Czech test data")
    parser.add_argument(
        "--data_dir",
        default="InfoRetriever/data/eval_data_cs",
        help="Directory containing evaluation data"
    )
    parser.add_argument(
        "--boolean_queries",
        default="boolean_queries_standard_10.txt",
        help="Boolean queries file to use"
    )
    parser.add_argument(
        "--output_dir", 
        default="evaluation_results", 
        help="Directory for output files"
    )
    parser.add_argument(
        "--eval_type", 
        choices=["boolean", "ranked", "both"], 
        default="both", 
        help="Type of evaluation to perform"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Define paths
    base_dir = os.path.abspath(args.data_dir)
    documents_path = os.path.join(base_dir, "documents.json")
    boolean_queries_path = os.path.join(base_dir, args.boolean_queries)
    ranked_queries_path = os.path.join(base_dir, "full_text_queries.json")
    gold_path = os.path.join(base_dir, "gold_relevancies.txt")
    
    # Check if files exist
    if not os.path.exists(documents_path):
        print(f"Error: Documents file not found at {documents_path}")
        return 1
    
    if not os.path.exists(boolean_queries_path):
        print(f"Error: Boolean queries file not found at {boolean_queries_path}")
        return 1
    
    if not os.path.exists(ranked_queries_path):
        print(f"Error: Ranked queries file not found at {ranked_queries_path}")
        return 1
    
    if not os.path.exists(gold_path):
        print(f"Error: Gold relevancies file not found at {gold_path}")
        return 1
    
    # Load documents
    print(f"Loading documents from {documents_path}...")
    try:
        documents = load_data_json(documents_path)
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return 1
    
    # Create and initialize the index
    print("Creating and initializing the index...")
    start_time = time.time()
    index = InfoRetrieverIndex(language='cz')
    
    # Print intermediate indexing step for better feedback
    print(f"Starting indexing of {len(documents)} documents...")
    index.index_documents(documents)
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.2f} seconds")
    
    # Create the search engine
    print("Creating search engine...")
    try:
        search_engine = InfoRetrieverSearchEngine(index)
        print("Search engine created successfully")
    except Exception as e:
        print(f"Error creating search engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indexing_time": index_time,
        "document_count": len(documents)
    }
    
    # Run evaluations
    if args.eval_type in ["boolean", "both"]:
        print("\n======= BOOLEAN SEARCH EVALUATION =======")
        try:
            boolean_queries = load_queries(boolean_queries_path)
            print(f"Loaded {len(boolean_queries)} boolean queries")
            
            # Evaluate Boolean search
            start_time = time.time()
            boolean_results = []
            
            for i, query in enumerate(boolean_queries):
                print(f"Processing boolean query {i+1}/{len(boolean_queries)}: {query}")
                try:
                    result = search_engine.boolean_search(query)
                    boolean_results.append({
                        "query": query,
                        "result_count": len(result),
                        "doc_ids": list(result)[:10]  # Save only first 10 doc IDs
                    })
                    print(f"Found {len(result)} matching documents")
                except Exception as e:
                    print(f"Error evaluating boolean query '{query}': {e}")
                    boolean_results.append({
                        "query": query,
                        "error": str(e)
                    })
            
            boolean_time = time.time() - start_time
            print(f"Boolean evaluation completed in {boolean_time:.2f} seconds")
            
            results["boolean_evaluation"] = {
                "time": boolean_time,
                "query_count": len(boolean_queries),
                "results": boolean_results
            }
        except Exception as e:
            print(f"Error during Boolean evaluation: {e}")
    
    if args.eval_type in ["ranked", "both"]:
        print("\n======= RANKED SEARCH EVALUATION =======")
        try:
            ranked_queries = load_data_json(ranked_queries_path)
            print(f"Loaded {len(ranked_queries)} ranked queries")
            
            # Create results file for trec_eval
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(args.output_dir, f"ranked_results_{timestamp}.txt")
            
            # Evaluate ranked search
            start_time = time.time()
            with open(results_file, "w", encoding="utf-8") as f:
                for i, query in enumerate(ranked_queries):
                    query_id = query["id"]
                    query_text = query["description"]
                    print(f"Processing ranked query {i+1}/{len(ranked_queries)}: {query_id} - {query_text}")
                    
                    try:
                        # Perform search
                        search_results = search_engine.search(query_text)
                        print(f"Found {len(search_results)} matching documents")
                        
                        # Write results in TREC format: query_id Q0 doc_id rank score run_id
                        for rank, (doc_id, score) in enumerate(search_results, 1):
                            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} InfoRetriever\n")
                    except Exception as e:
                        print(f"Error processing ranked query '{query_id}': {e}")
            
            ranked_time = time.time() - start_time
            print(f"Ranked evaluation completed in {ranked_time:.2f} seconds")
            
            results["ranked_evaluation"] = {
                "time": ranked_time,
                "query_count": len(ranked_queries),
                "results_file": results_file
            }
            
            # Run TREC evaluation
            print("\n======= TREC EVALUATION =======")
            print(f"Running TREC evaluation with gold standard: {gold_path}")
            print(f"Results file: {results_file}")
            
            # Check if the trec_eval tool is available
            trec_eval_path = os.path.join("InfoRetriever/trec_eval/trec_eval")
            if not os.path.exists(trec_eval_path):
                trec_eval_path = "trec_eval"  # Try using system path
            
            try:
                import subprocess
                cmd = [trec_eval_path, "-m", "all_trec", gold_path, results_file]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                # Save TREC eval results
                trec_results_file = os.path.join(args.output_dir, f"trec_eval_results_{timestamp}.txt")
                with open(trec_results_file, "w") as f:
                    f.write(stdout.decode("utf-8"))
                
                if stderr:
                    print(f"TREC Eval stderr: {stderr.decode('utf-8')}")
                
                print(f"TREC evaluation results saved to: {trec_results_file}")
                results["trec_evaluation"] = {
                    "results_file": trec_results_file
                }
            except Exception as e:
                print(f"Error running TREC evaluation: {e}")
                results["trec_evaluation"] = {
                    "error": str(e)
                }
        except Exception as e:
            print(f"Error during ranked evaluation: {e}")
    
    # Save overall results
    results_file = os.path.join(args.output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
