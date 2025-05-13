#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfoRetriever - Enhanced CLI Interface
A powerful command-line interface for the InfoRetriever search system
"""

import os
import sys
import json
import time
from typing import Optional, List, Dict, Any
import argparse
from datetime import datetime

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import box
except ImportError:
    print("Please install required packages: pip install rich")
    sys.exit(1)

# Import InfoRetriever modules
from InfoRetriever.preprocessing.document import Document
from InfoRetriever.boolean_search.boolean_search import BooleanSearchEngine
from InfoRetriever.build_inverted_index import InvertedIndexBuilder
from InfoRetriever.tfidf_search.tfidf_search import TFIDFSearchEngine

# Initialize rich console
console = Console()

class InfoRetrieverCLI:
    def __init__(self):
        """Initialize the CLI interface"""
        self.documents = []
        self.boolean_engine = None
        self.tfidf_engine = None
        self.documents_loaded = False
        self.index_file = None
    
    def print_header(self):
        """Display the application header"""
        console.print(Panel.fit(
            "[bold blue]InfoRetriever[/bold blue] [yellow]Search Engine[/yellow]",
            border_style="blue",
            subtitle="A powerful information retrieval system"
        ))
    
    def load_documents(self, documents_path: str) -> bool:
        """Load documents from a JSON file"""
        try:
            console.print(f"Loading documents from: [cyan]{documents_path}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Loading documents...", total=None)
                
                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                
                progress.update(task, completed=True)
            
            self.documents_loaded = True
            console.print(f"[green]Successfully loaded [bold]{len(self.documents)}[/bold] documents[/green]")
            return True
        
        except Exception as e:
            console.print(f"[bold red]Error loading documents:[/bold red] {str(e)}")
            return False
    
    def init_boolean_engine(self, index_path: str = None) -> bool:
        """Initialize the Boolean search engine"""
        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "InfoRetriever", "config.json")
            config = None
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load config file: {e}. Using default settings.[/yellow]")
            
            if index_path and os.path.exists(index_path):
                # Load existing index
                console.print(f"Loading Boolean inverted index from: [cyan]{index_path}[/cyan]")
                self.boolean_engine = BooleanSearchEngine(index_path, config=config)
                self.index_file = index_path
                return True
            
            if not self.documents_loaded:
                console.print("[bold red]No documents loaded. Load documents first or provide an index file.[/bold red]")
                return False
            
            # Build new index from documents
            console.print("[cyan]Building Boolean inverted index from documents...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Building index...", total=None)
                
                index_builder = InvertedIndexBuilder(config=config)
                
                # Create a temporary file to store the documents
                temp_path = os.path.join(os.getcwd(), "temp_documents.json")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.documents, f)
                
                # Build the index
                index_builder.build_from_json(temp_path)
                
                # Save the index
                self.index_file = os.path.join(os.getcwd(), "inverted_index.json")
                index_builder.save_to_json(self.index_file)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Initialize the engine with the new index
                self.boolean_engine = BooleanSearchEngine(self.index_file, config=config)
                
                progress.update(task, completed=True)
            
            console.print("[green]Boolean search engine initialized successfully[/green]")
            return True
        
        except Exception as e:
            console.print(f"[bold red]Error initializing Boolean engine:[/bold red] {str(e)}")
            import traceback
            console.print(Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True))
            return False
    
    def init_tfidf_engine(self) -> bool:
        """Initialize the TF-IDF search engine"""
        try:
            if not self.documents_loaded:
                console.print("[bold red]No documents loaded. Load documents first.[/bold red]")
                return False
            
            # Use the same config as for Boolean search
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "InfoRetriever", "config.json")
            config = None
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load config file: {e}. Using default settings.[/yellow]")
            
            console.print("[cyan]Initializing TF-IDF search engine...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Building TF-IDF model...", total=None)
                
                self.tfidf_engine = TFIDFSearchEngine(
                    documents=self.documents,
                    config=config
                )
                
                progress.update(task, completed=True)
            
            console.print("[green]TF-IDF search engine initialized successfully[/green]")
            return True
        
        except Exception as e:
            console.print(f"[bold red]Error initializing TF-IDF engine:[/bold red] {str(e)}")
            import traceback
            console.print(Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True))
            return False
    
    def search_boolean(self, query: str, top_k: int = 5, debug: bool = False) -> List[Document]:
        """Perform a Boolean search"""
        if not self.boolean_engine:
            console.print("[bold red]Boolean engine not initialized.[/bold red]")
            return []
        
        console.print(f"Executing Boolean search: '[cyan]{query}[/cyan]'")
        
        try:
            # Debug query parsing if requested
            if debug:
                from InfoRetriever.boolean_search.parser import BooleanParser, preprocess_term
                console.print("\n[yellow]Debugging query terms:[/yellow]")
                simple_terms = [term.strip() for term in re.split(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', query) 
                              if term.strip() and term.strip() not in ['AND', 'OR', 'NOT']]
                
                for term in simple_terms:
                    processed = preprocess_term(term)
                    console.print(f"  '[yellow]{term}[/yellow]' -> '[green]{processed}[/green]'")
                
                parser = BooleanParser(query)
                ast = parser.parse()
                console.print(f"Query structure: [cyan]{ast}[/cyan]")
            
            # Execute search with timing
            start_time = time.time()
            result_set, execution_time = self.boolean_engine.search(query)
            
            console.print(f"[green]Found {len(result_set)} documents in {execution_time:.6f} seconds[/green]")
            
            # Convert results to Document objects and limit to top_k
            documents = []
            for doc_id in sorted(result_set)[:top_k]:
                try:
                    # Try to find the document by ID in the original documents
                    if isinstance(doc_id, int) and doc_id < len(self.documents):
                        doc_data = self.documents[doc_id]
                        doc = Document(
                            id=doc_id,
                            title=doc_data.get('title', ''),
                            content=doc_data.get('text', doc_data.get('content', '')),
                            abstract=doc_data.get('abstract', ''),
                            metadata={'source': 'boolean_search'}
                        )
                        documents.append(doc)
                except Exception as doc_e:
                    console.print(f"[yellow]Error creating Document for doc_id={doc_id}: {doc_e}[/yellow]")
            
            return documents
        
        except Exception as e:
            console.print(f"[bold red]Error during Boolean search:[/bold red] {str(e)}")
            import traceback
            console.print(Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True))
            return []
    
    def search_tfidf(self, query: str, top_k: int = 5) -> List[tuple]:
        """Perform a TF-IDF search"""
        if not self.tfidf_engine:
            console.print("[bold red]TF-IDF engine not initialized.[/bold red]")
            return []
        
        console.print(f"Executing TF-IDF search: '[cyan]{query}[/cyan]'")
        
        try:
            # Execute search
            start_time = time.time()
            results = self.tfidf_engine.search(query, top_k=top_k)
            execution_time = time.time() - start_time
            
            console.print(f"[green]Found {len(results)} documents in {execution_time:.6f} seconds[/green]")
            return results
        
        except Exception as e:
            console.print(f"[bold red]Error during TF-IDF search:[/bold red] {str(e)}")
            import traceback
            console.print(Syntax(traceback.format_exc(), "python", theme="monokai", line_numbers=True))
            return []
    
    def display_results(self, results, engine_type):
        """Display search results in a formatted way"""
        if not results:
            console.print(f"[yellow]No results found for {engine_type} search.[/yellow]")
            return
        
        console.print(f"\n[bold cyan]{engine_type.upper()} SEARCH RESULTS:[/bold cyan]")
        
        if engine_type.lower() == 'boolean':
            # For Boolean search (list of documents)
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=4)
            table.add_column("Title", style="cyan")
            table.add_column("Content", style="green", no_wrap=False)
            
            for i, doc in enumerate(results):
                # Truncate content for display
                content_snippet = doc.content[:150].replace('\n', ' ')
                if len(content_snippet) == 150:
                    content_snippet += "..."
                    
                table.add_row(
                    str(i+1),
                    doc.title,
                    content_snippet
                )
            
            console.print(table)
        
        else:
            # For TF-IDF search (list of document, score tuples)
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim", width=4)
            table.add_column("Title", style="cyan")
            table.add_column("Score", style="yellow", width=10)
            table.add_column("Content", style="green", no_wrap=False)
            
            for i, (doc, score) in enumerate(results):
                # Truncate content for display
                content_snippet = doc.content[:150].replace('\n', ' ')
                if len(content_snippet) == 150:
                    content_snippet += "..."
                
                table.add_row(
                    str(i+1),
                    doc.title,
                    f"{score:.4f}",
                    content_snippet
                )
            
            console.print(table)
    
    def interactive_mode(self):
        """Run the application in interactive mode"""
        while True:
            console.rule("[bold blue]InfoRetriever[/bold blue]")
            
            # Step 1: Load data if not loaded
            if not self.documents_loaded:
                console.print("[bold yellow]First, let's load some documents.[/bold yellow]")
                documents_path = input("Enter path to documents JSON file: ")
                if not documents_path:
                    console.print("[bold red]No document path provided. Exiting.[/bold red]")
                    return
                
                if not self.load_documents(documents_path):
                    continue
                
                # Initialize engines
                self.init_boolean_engine()
                self.init_tfidf_engine()
            
            # Show menu
            console.print("\n[bold cyan]Choose search mode:[/bold cyan]")
            console.print("1. [yellow]Boolean Search[/yellow]")
            console.print("2. [yellow]TF-IDF Search[/yellow]")
            console.print("3. [yellow]Both[/yellow]")
            console.print("4. [yellow]Quit[/yellow]")
            
            choice = input("\nEnter choice (1-4): ")
            
            if choice == '4' or choice.lower() == 'quit':
                break
            
            if choice not in ['1', '2', '3']:
                console.print("[bold red]Invalid choice. Please enter a number between 1 and 4.[/bold red]")
                continue
            
            query = input("\nEnter search query: ")
            if not query:
                console.print("[bold red]Empty query. Please try again.[/bold red]")
                continue
            
            top_k = 5
            try:
                top_k_input = input(f"Number of results to show (default: {top_k}): ")
                if top_k_input:
                    top_k = int(top_k_input)
            except ValueError:
                console.print(f"[yellow]Invalid number. Using default: {top_k}[/yellow]")
            
            if choice in ['1', '3'] and self.boolean_engine:
                boolean_results = self.search_boolean(query, top_k)
                self.display_results(boolean_results, "Boolean")
            
            if choice in ['2', '3'] and self.tfidf_engine:
                tfidf_results = self.search_tfidf(query, top_k)
                self.display_results(tfidf_results, "TF-IDF")


def main():
    """Main entry point for the CLI application"""
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
    
    # Create CLI interface
    retriever = InfoRetrieverCLI()
    retriever.print_header()
    
    # Run in interactive mode if specified or if no command-line arguments are provided
    if args.interactive or (not args.documents and not args.query and not args.boolean_query and not args.tfidf_query):
        retriever.interactive_mode()
        return
    
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
        console.print("[bold red]Failed to initialize any search engines.[/bold red]")
        sys.exit(1)
    
    # Query mode
    if args.boolean_query and retriever.boolean_engine:
        boolean_results = retriever.search_boolean(args.boolean_query, args.top, debug=args.debug)
        retriever.display_results(boolean_results, "Boolean")
    
    if args.tfidf_query and retriever.tfidf_engine:
        tfidf_results = retriever.search_tfidf(args.tfidf_query, args.top)
        retriever.display_results(tfidf_results, "TF-IDF")
    
    if args.query:
        if args.engine in ['boolean', 'both'] and retriever.boolean_engine:
            boolean_results = retriever.search_boolean(args.query, args.top, debug=args.debug)
            retriever.display_results(boolean_results, "Boolean")
        
        if args.engine in ['tfidf', 'both'] and retriever.tfidf_engine:
            tfidf_results = retriever.search_tfidf(args.query, args.top)
            retriever.display_results(tfidf_results, "TF-IDF")


if __name__ == "__main__":
    # Import here to avoid circular imports
    import re
    main()
