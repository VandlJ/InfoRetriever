import json
import json
import time
import argparse
import re
import os
from .parser import BooleanParser, TermNode, AndNode, OrNode, NotNode

class BooleanSearchEngine:
    def __init__(self, index_file=None, index_data=None, config=None):
        """
        Initialize the search engine with a pre-built inverted index.
        
        Args:
            index_file: Path to the inverted index JSON file (optional if index_data provided)
            index_data: Dictionary containing the inverted index (optional if index_file provided)
            config: Optional configuration dictionary
        """
        self.query_cache = {}
        self.config = config
        
        # Load config if not provided
        if not self.config:
            self._load_config()
        
        if index_file:
            print(f"Loading inverted index from {index_file}...")
            self.load_index_from_file(index_file)
        elif index_data:
            self.load_index_from_dict(index_data)
        else:
            self.index = {}
            self.all_docs = set()
            self.metadata = {}
            print("Warning: No index provided. Search results will be empty.")
    
    def _load_config(self):
        """Load configuration from file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                    print("Loaded search configuration from config.json")
            except Exception as e:
                print(f"Warning: Could not load config: {e}, using default settings")
                self.config = None
        else:
            print("No config.json found, using default settings")
            self.config = None
    
    def load_index_from_file(self, index_file):
        """Load an inverted index from a JSON file"""
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.load_index_from_dict(data)
    
    def load_index_from_dict(self, data):
        """Load an inverted index from a dictionary"""
        # Extract metadata
        self.metadata = data.get('metadata', {})
        
        # Convert document IDs from strings to integers if needed
        self.index = {}
        for term, doc_ids in data['index'].items():
            self.index[term] = set(doc_ids)
        
        # Precompute the universal set of all document IDs
        self.all_docs = set()
        for doc_ids in self.index.values():
            self.all_docs.update(doc_ids)
        
        print(f"Loaded index with {len(self.index)} terms and {len(self.all_docs)} documents")
    
    def optimize_query(self, query_ast):
        """Optimize the query AST to minimize NOT operations"""
        # Apply De Morgan's laws to push NOT operators down the tree
        return self._apply_de_morgan(query_ast)
    
    def _apply_de_morgan(self, node):
        """Apply De Morgan's laws to push NOT operators down"""
        if isinstance(node, NotNode):
            child = node.child
            if isinstance(child, AndNode):
                # NOT (A AND B) => NOT A OR NOT B
                return OrNode(
                    self._apply_de_morgan(NotNode(child.left)),
                    self._apply_de_morgan(NotNode(child.right))
                )
            elif isinstance(child, OrNode):
                # NOT (A OR B) => NOT A AND NOT B
                return AndNode(
                    self._apply_de_morgan(NotNode(child.left)),
                    self._apply_de_morgan(NotNode(child.right))
                )
            elif isinstance(child, NotNode):
                # NOT NOT A => A (double negation elimination)
                return self._apply_de_morgan(child.child)
        
        # Recursively optimize children
        if isinstance(node, AndNode) or isinstance(node, OrNode):
            node.left = self._apply_de_morgan(node.left)
            node.right = self._apply_de_morgan(node.right)
        
        return node

    def evaluate_optimized(self, query_ast):
        """Evaluate a query AST with optimizations for NOT operations"""
        # First, rewrite the AST to minimize NOT operations
        optimized_ast = self.optimize_query(query_ast)
        
        # Apply query reordering optimization to put smaller sets first
        from .query_optimizer import reorder_query_ast
        reordered_ast = reorder_query_ast(optimized_ast, self.index)
        
        # Evaluate the optimized and reordered AST
        return self._evaluate_node(reordered_ast)
    
    def _evaluate_node(self, node):
        """Evaluate a query node, returning a set of document IDs"""
        if isinstance(node, TermNode):
            # Look up the term in the index
            return self.index.get(node.value, set())
        
        elif isinstance(node, NotNode):
            # NOT operation: compute the complement
            child_result = self._evaluate_node(node.child)
            return self.all_docs - child_result
        
        elif isinstance(node, AndNode):
            # AND operation: intersect the results
            # Optimize by evaluating smaller set first
            left_result = self._evaluate_node(node.left)
            if not left_result:
                return set()  # Short circuit if left is empty
            
            right_result = self._evaluate_node(node.right)
            return left_result.intersection(right_result)
        
        elif isinstance(node, OrNode):
            # OR operation: union the results
            left_result = self._evaluate_node(node.left)
            right_result = self._evaluate_node(node.right)
            return left_result.union(right_result)
        
        return set()
    
    def debug_term(self, term):
        """Print information about a term's preprocessing and presence in the index"""
        # Import preprocessing from parser
        from .parser import preprocess_term
        
        # Preprocess the term using the same function as parser
        preprocessed = preprocess_term(term)
        
        # Check if in index (exact match)
        exact_match = preprocessed in self.index
        
        # Try case-insensitive match if exact match fails
        case_insensitive_match = False
        if not exact_match:
            for key in self.index.keys():
                if key.lower() == preprocessed.lower():
                    case_insensitive_match = True
                    break
        
        # Get document count
        doc_count = len(self.index.get(preprocessed, set()))
        if not exact_match and case_insensitive_match:
            # Find the case-insensitive matching key to get doc count
            for key in self.index.keys():
                if key.lower() == preprocessed.lower():
                    doc_count = len(self.index.get(key, set()))
                    break
        
        print("\nTerm debugging:")
        print(f"  Original: '{term}'")
        print(f"  Preprocessed: '{preprocessed}'")
        print(f"  Exact match in index: {exact_match}")
        if not exact_match:
            print(f"  Case-insensitive match: {case_insensitive_match}")
        print(f"  Matching documents: {doc_count}")
        
        # If not found but similar terms exist, suggest similar terms
        if not exact_match and not case_insensitive_match:
            # Find similar terms (prefix match)
            similar = [t for t in self.index.keys() if len(preprocessed) >= 3 and 
                      t.lower().startswith(preprocessed.lower()[:3])][:5]
            if similar:
                print(f"  Similar terms in index: {similar}")
        
        return preprocessed

    def search(self, query_string, debug=False):
        """
        Execute a boolean search query
        
        Args:
            query_string: Boolean query string (AND, OR, NOT, parentheses)
            debug: Enable debug output
            
        Returns:
            tuple: (result_set, execution_time)
        """
        # Check cache
        if query_string in self.query_cache:
            results, execution_time = self.query_cache[query_string]
            if debug:
                print(f"Cache hit for query: '{query_string}'")
                print(f"Cached results: {len(results)} documents")
            return results, execution_time
        
        start_time = time.time()
        
        try:
            # Debug individual terms in the query
            if debug:
                print(f"\nDebugging query: '{query_string}'")
                # Extract simple terms for debugging
                simple_terms = [term.strip() for term in re.split(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', query_string) 
                              if term.strip() and term.strip() not in ['AND', 'OR', 'NOT']]
                print(f"Extracted terms: {simple_terms}")
                
                for term in simple_terms:
                    self.debug_term(term)
            
            # Parse the query
            parser = BooleanParser(query_string)
            query_ast = parser.parse()
            
            if debug:
                print(f"\nQuery AST: {query_ast}")
            
            # Evaluate the query
            results = self.evaluate_optimized(query_ast)
            
            execution_time = time.time() - start_time
            
            # Cache the results
            self.query_cache[query_string] = (results, execution_time)
            return results, execution_time
        
        except Exception as e:
            print(f"Error processing query '{query_string}': {e}")
            import traceback
            traceback.print_exc()
            return set(), time.time() - start_time
    
    def batch_search(self, query_file, output_file=None):
        """Process multiple queries from a file"""
        print(f"Processing queries from {query_file}")
        
        # Read queries from file
        with open(query_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out comments and empty lines
        queries = [line.strip() for line in lines if line.strip() and not line.strip().startswith('//')]
        
        results = []
        total_time = 0
        total_docs = 0
        
        print(f"Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            try:
                result_set, execution_time = self.search(query)
                results.append((query, result_set, execution_time))
                
                total_time += execution_time
                total_docs += len(result_set)
                
                print(f"Query {i+1}: '{query}' - Found {len(result_set)} documents in {execution_time:.6f} seconds")
            
            except Exception as e:
                print(f"Error processing query {i+1}: '{query}' - {e}")
                results.append((query, set(), 0))
        
        # Print summary
        print("\nSearch Results Summary:")
        print(f"Total queries: {len(queries)}")
        print(f"Total execution time: {total_time:.6f} seconds")
        print(f"Average time per query: {total_time/len(queries) if queries else 0:.6f} seconds")
        print(f"Total documents found: {total_docs}")
        print(f"Average documents per query: {total_docs/len(queries) if queries else 0:.2f}")
        
        # Count queries with no results
        zero_results = sum(1 for _, docs, _ in results if not docs)
        print(f"Queries with no results: {zero_results} ({zero_results/len(queries)*100:.1f}%)")
        
        # Write results to output file if provided
        if output_file:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Boolean Search Results\n")
                f.write("=====================\n\n")
                
                for query, docs, time_taken in results:
                    f.write(f"Query: {query}\n")
                    f.write(f"Matching documents: {len(docs)}\n")
                    f.write(f"Time: {time_taken:.6f} seconds\n")
                    
                    if docs:
                        # Sort and limit document IDs in output for readability
                        sorted_docs = sorted(docs)
                        if len(sorted_docs) > 20:
                            f.write(f"Documents: {sorted_docs[:20]} ... ({len(sorted_docs)-20} more)\n")
                        else:
                            f.write(f"Documents: {sorted_docs}\n")
                    else:
                        f.write("Documents: []\n")
                    
                    f.write("\n")
                
                # Add summary
                f.write("\nSearch Results Summary:\n")
                f.write(f"Total queries: {len(queries)}\n")
                f.write(f"Total execution time: {total_time:.6f} seconds\n")
                f.write(f"Average time per query: {total_time/len(queries) if queries else 0:.6f} seconds\n")
                f.write(f"Total documents found: {total_docs}\n")
                f.write(f"Average documents per query: {total_docs/len(queries) if queries else 0:.2f}\n")
                f.write(f"Queries with no results: {zero_results} ({zero_results/len(queries)*100:.1f}%)\n")
            
            print(f"\nResults written to {output_file}")
        
        return results, total_time


def main():
    parser = argparse.ArgumentParser(description='Boolean Search Engine')
    parser.add_argument('index', help='Path to inverted index JSON file')
    parser.add_argument('--query', help='Single query to execute')
    parser.add_argument('--queries', help='Path to file with multiple queries')
    parser.add_argument('--output', help='Path to output results file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create search engine
    engine = BooleanSearchEngine(args.index)
    
    if args.query:
        # Execute a single query
        result_set, execution_time = engine.search(args.query, debug=args.debug)
        print(f"\nQuery: '{args.query}'")
        print(f"Found {len(result_set)} documents in {execution_time:.6f} seconds")
        
        if result_set:
            sorted_docs = sorted(result_set)
            if len(sorted_docs) > 20:
                print(f"Documents: {sorted_docs[:20]} ... ({len(sorted_docs)-20} more)")
            else:
                print(f"Documents: {sorted_docs}")
        else:
            print("No matching documents found")
    
    elif args.queries:
        # Process multiple queries from file
        engine.batch_search(args.queries, args.output)
    
    else:
        # Interactive mode
        print("\nInteractive search mode (type 'quit' to exit)")
        
        while True:
            query = input("\nEnter query: ")
            if query.lower() == 'quit':
                break
                
            result_set, execution_time = engine.search(query, debug=args.debug)
            print(f"Found {len(result_set)} documents in {execution_time:.6f} seconds")
            
            if result_set:
                sorted_docs = sorted(result_set)
                if len(sorted_docs) > 20:
                    print(f"Documents: {sorted_docs[:20]} ... ({len(sorted_docs)-20} more)")
                else:
                    print(f"Documents: {sorted_docs}")
            else:
                print("No matching documents found")


if __name__ == "__main__":
    main()
