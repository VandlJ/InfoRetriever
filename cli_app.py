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
import argparse
import re
import threading
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import List

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.layout import Layout
    from rich import box
except ImportError:
    # Install rich silently
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rich'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
    # Now import after installation
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.layout import Layout
    from rich import box

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
        console.print(Panel(
            "[bold blue]InfoRetriever[/bold blue] [yellow]Search Engine[/yellow]",
            border_style="blue",
            subtitle="A powerful information retrieval system",
            width=80
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
            
            # Execute search (execution time is returned by the engine)
            result_set, execution_time = self.boolean_engine.search(query)
            
            console.print(f"[green]Found {len(result_set)} documents in {execution_time:.6f} seconds[/green]")
            
            # Convert results to Document objects and limit to top_k
            documents = []
            for doc_id in sorted(result_set)[:top_k]:
                try:
                    # Try to find the document by ID in the original documents
                    doc_data = None
                    
                    # Handle both numeric and string IDs (like "d1", "d2", etc)
                    if isinstance(doc_id, int) and doc_id < len(self.documents):
                        # Numeric ID as index
                        doc_data = self.documents[doc_id]
                    elif isinstance(doc_id, str):
                        # String ID (e.g., "d1")
                        for doc in self.documents:
                            if isinstance(doc, dict) and doc.get('id') == doc_id:
                                doc_data = doc
                                break
                    
                    if doc_data:
                        doc = Document(
                            id=doc_id,
                            title=doc_data.get('title', ''),
                            content=doc_data.get('text', doc_data.get('content', '')),
                            abstract=doc_data.get('abstract', ''),
                            metadata={'source': 'boolean_search'}
                        )
                        documents.append(doc)
                    else:
                        console.print(f"[yellow]Could not find document with ID {doc_id}[/yellow]")
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
        
        timestamp = time.strftime("%H:%M:%S")
        console.print(f"\n[bold cyan]📊 {engine_type.upper()} SEARCH RESULTS [dim]({timestamp})[/dim]:[/bold cyan]")
        
        if engine_type.lower() == 'boolean':
            # For Boolean search (list of documents)
            table = Table(
                box=box.HEAVY_EDGE, 
                show_header=True, 
                header_style="bold magenta",
                title=f"[bold]Found {len(results)} document(s)[/bold]",
                title_style="yellow"
            )
            table.add_column("📌", style="dim", width=4)
            table.add_column("📑 Title", style="cyan bold")
            table.add_column("📝 Content", style="green", no_wrap=False)
            
            for i, doc in enumerate(results):
                # Truncate content for display
                content_snippet = doc.content[:150].replace('\n', ' ')
                if len(content_snippet) == 150:
                    content_snippet += "..."
                
                # Highlight the row for the top result
                row_style = "on blue" if i == 0 else ""
                    
                table.add_row(
                    str(i+1),
                    doc.title or "[dim]<No title>[/dim]",
                    content_snippet,
                    style=row_style
                )
            
            console.print(table)
            console.print("[dim]Tip: You can select a different search method from the menu.[/dim]")
        
        else:
            # For TF-IDF search (list of document, score tuples)
            table = Table(
                box=box.HEAVY_EDGE, 
                show_header=True, 
                header_style="bold magenta",
                title=f"[bold]Found {len(results)} document(s) ranked by relevance[/bold]",
                title_style="yellow"
            )
            table.add_column("📌", style="dim", width=4)
            table.add_column("📑 Title", style="cyan bold")
            table.add_column("🔢 Score", style="yellow", width=10)
            table.add_column("📝 Content", style="green", no_wrap=False)
            
            for i, (doc, score) in enumerate(results):
                # Truncate content for display
                content_snippet = doc.content[:150].replace('\n', ' ')
                if len(content_snippet) == 150:
                    content_snippet += "..."
                
                # Highlight the row for the top result
                row_style = "on blue" if i == 0 else ""
                
                # Format score with visual indicator based on value
                score_str = f"{score:.4f}"
                if score > 0.7:
                    score_display = f"[bold green]{score_str}[/bold green]"
                elif score > 0.4:
                    score_display = f"[yellow]{score_str}[/yellow]"
                else:
                    score_display = f"[dim]{score_str}[/dim]"
                
                table.add_row(
                    str(i+1),
                    doc.title or "[dim]<No title>[/dim]",
                    score_display,
                    content_snippet,
                    style=row_style
                )
            
            console.print(table)
            console.print("[dim]Tip: Higher scores indicate more relevant results.[/dim]")
    
    def get_available_json_files(self, data_dir="InfoRetriever/data"):
        """List available JSON files in the data directory"""
        json_files = []
        
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))
        
        return json_files
    
    def run_evaluation(self):
        """Run evaluation script with user-defined parameters"""
        console.print(Panel(
            "[bold yellow]Running InfoRetriever Evaluation[/bold yellow]",
            border_style="yellow",
            subtitle="Testing system performance"
        ))
        
        data_dir_options = [
            "InfoRetriever/data/eval_data_cs",
            "InfoRetriever/data/eval_data_en"
        ]
        
        # Create selection table for data directory
        table = Table(title="[bold]Available Evaluation Datasets[/bold]", box=box.ROUNDED)
        table.add_column("#", style="dim")
        table.add_column("Dataset", style="cyan")
        
        for i, dataset in enumerate(data_dir_options, 1):
            if os.path.exists(dataset):
                table.add_row(str(i), dataset)
        
        console.print(table)
        
        # Get user input for parameters
        data_dir_choice = console.input("[bold cyan]Select dataset number (default: 1): [/bold cyan]")
        data_dir = data_dir_options[int(data_dir_choice) - 1 if data_dir_choice.isdigit() and 1 <= int(data_dir_choice) <= len(data_dir_options) else 0]
        
        # Always use "ranked" evaluation
        eval_type = "ranked"
        
        output_dir = "evaluation_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Prepare command-line arguments for the evaluation script
        args = [
            "--data_dir=" + data_dir,
            "--eval_type=" + eval_type,
            "--output_dir=" + output_dir
        ]
        
        console.print("\n[bold cyan]🔍 Starting ranked evaluation...[/bold cyan]")
        
        # Set up a simpler status display with animated spinners
        # Using imported modules
        
        # Create a string buffer to capture output
        output_buffer = io.StringIO()
        result_data = {"success": False, "output": "", "result_file": ""}
        
        # Define evaluation steps with corresponding patterns to detect in output
        eval_steps = [
            {
                "id": "indexing",
                "description": "[bold yellow]⚙️ Indexing documents...[/bold yellow]",
                "status": "waiting",  # can be "waiting", "running", "completed"
                "patterns": [
                    "Starting indexing of",
                    "documents in",
                    "Total terms in inverted index",
                    "Inverted index saved to",
                    "Successfully indexed"
                ],
                "completion_patterns": [
                    "Creating search engine",
                    "Creating the search engine"
                ]
            },
            {
                "id": "engine",
                "description": "[bold yellow]🔍 Creating search engine...[/bold yellow]",
                "status": "waiting",
                "patterns": [
                    "TF-IDF batch",
                    "Calculating document vectors",
                    "Loading inverted index"
                ],
                "completion_patterns": [
                    "Search engine created successfully"
                ]
            },
            {
                "id": "query",
                "description": "[bold yellow]📊 Processing queries...[/bold yellow]",
                "status": "waiting",
                "patterns": [
                    "Processing ranked query",
                    "Processing query"
                ],
                "completion_patterns": [
                    "Ranked evaluation completed",
                    "evaluation completed",
                    "perform trec_eval"
                ]
            },
            {
                "id": "trec",
                "description": "[bold yellow]📈 Running TREC evaluation...[/bold yellow]",
                "status": "waiting",
                "patterns": [
                    "Running trec_eval",
                    "TREC evaluation"
                ],
                "completion_patterns": [
                    "TREC evaluation results saved",
                    "Evaluation results saved to",
                    "completed successfully"
                ]
            }
        ]
        
        # Function to get current status panel
        def get_status_panel():
            content = []
            # Add a header
            content.append("[bold cyan]Evaluation Progress:[/bold cyan]\n")
            
            # Show current status of each step
            for step in eval_steps:
                if step["status"] == "waiting":
                    icon = "⬜"
                    status = "[dim]Waiting...[/dim]"
                elif step["status"] == "running":
                    icon = "🔄"
                    status = "[bold blue]Running[/bold blue]"
                elif step["status"] == "completed":
                    icon = "✅"
                    status = "[bold green]Completed[/bold green]"
                
                content.append(f"{icon} {step['description']} {status}")
                
            # Add seconds elapsed
            elapsed = time.time() - start_time
            content.append(f"\n[dim]Time elapsed: {elapsed:.1f} seconds[/dim]")
            
            return Panel("\n".join(content), title="[bold]Evaluation Status[/bold]", border_style="blue")
        
        # Start evaluation thread
        def run_eval():
            try:
                with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                    import run_evaluation
                    sys.argv = ["run_evaluation.py"] + args
                    
                    # Monitor the output to update progress
                    original_print = print
                    
                    def progress_monitor(*args, **kwargs):
                        text = " ".join(str(arg) for arg in args)
                        original_print(*args, **kwargs)
                        
                        # Update the result file path when found
                        result_file_match = re.search(r"Evaluation results saved to: (\S+)", text)
                        if result_file_match:
                            result_data["result_file"] = result_file_match.group(1)
                    
                    # Replace print function to monitor progress
                    import builtins
                    builtins.print = progress_monitor
                    
                    # Run evaluation
                    run_evaluation.main()
                    
                    # Restore print function
                    builtins.print = original_print
                    
                    result_data["success"] = True
            except Exception as e:
                result_data["error"] = str(e)
                import traceback
                result_data["traceback"] = traceback.format_exc()
        
        # Start evaluation thread
        eval_thread = threading.Thread(target=run_eval)
        eval_thread.daemon = True
        
        # Track timing and state
        start_time = time.time()
        detected_operation = ""
        last_output_check = ""
        
        # Initialize first step as running
        eval_steps[0]["status"] = "running"
        
        # Run evaluation with live status display
        with Live(get_status_panel(), refresh_per_second=10) as live:
            eval_thread.start()
            
            # Monitor thread and update status
            while eval_thread.is_alive():
                # Get current output
                output = output_buffer.getvalue()
                
                # Skip if no new output
                if output == last_output_check:
                    time.sleep(0.1)
                    # Still need to refresh to update elapsed time
                    live.update(get_status_panel())
                    continue
                
                last_output_check = output
                
                # Check for pattern matches in output and update status
                for i, step in enumerate(eval_steps):
                    # Check completion patterns for the current active step
                    if step["status"] == "running":
                        for pattern in step["completion_patterns"]:
                            if pattern in output:
                                step["status"] = "completed"
                                # If there are more steps, start the next one
                                if i + 1 < len(eval_steps):
                                    eval_steps[i + 1]["status"] = "running"
                                break
                    
                    # Find specific operations for the detailed status
                    for pattern in step["patterns"]:
                        if pattern in output:
                            # Extract more specific information if available
                            if "Processing ranked query" in output:
                                query_match = re.search(r"Processing ranked query (\d+)/(\d+)", output)
                                if query_match:
                                    curr, total = query_match.groups()
                                    detected_operation = f"Processing query {curr} of {total}"
                            else:
                                detected_operation = pattern
                
                # Update the live display
                live.update(get_status_panel())
                time.sleep(0.1)
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.1)
        
        # Check result
        if result_data["success"]:
            console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]")
            
            # Parse the results file to get MAP score
            try:
                if result_data["result_file"] and os.path.exists(result_data["result_file"]):
                    with open(result_data["result_file"], 'r') as f:
                        eval_results = json.load(f)
                    
                    # Find corresponding TREC eval file and extract MAP score
                    trec_file = None
                    if "trec_evaluation" in eval_results and "results_file" in eval_results["trec_evaluation"]:
                        trec_file = eval_results["trec_evaluation"]["results_file"]
                    
                    map_score = None
                    if trec_file and os.path.exists(trec_file):
                        with open(trec_file, 'r') as f:
                            for line in f:
                                if line.startswith("map"):
                                    parts = line.strip().split()
                                    if len(parts) >= 3:
                                        map_score = float(parts[2])
                                    break
                    
                    # Display results in a nice panel
                    if map_score is not None:
                        # Create a nice visualization with stars based on MAP score
                        # When score is 0.15 or higher, display 4 stars
                        if map_score >= 0.15:
                            stars = "★" * 4
                            empty_stars = "☆"
                        else:
                            stars = "★" * min(5, max(1, int(map_score * 10)))
                            empty_stars = "☆" * (5 - min(5, max(1, int(map_score * 10))))
                        
                        map_rating = f"[yellow]{stars}[dim]{empty_stars}[/dim][/yellow]"
                        
                        # Add color coding based on score
                        if map_score > 0.3:
                            score_display = f"[bold green]{map_score:.4f}[/bold green]"
                        elif map_score > 0.15:
                            score_display = f"[bold yellow]{map_score:.4f}[/bold yellow]"
                        else:
                            score_display = f"[bold red]{map_score:.4f}[/bold red]"
                        
                        performance_panel = Panel(
                            f"[bold cyan]Mean Average Precision (MAP):[/bold cyan] {score_display}\n"
                            f"[dim]Rating:[/dim] {map_rating}",
                            title="[bold]🏆 Performance Results[/bold]",
                            border_style="green",
                            expand=False,
                            padding=(1, 2)
                        )
                        console.print(performance_panel)
                    
                    console.print(f"[dim]Detailed results saved to [cyan]{result_data['result_file']}[/cyan][/dim]")
                
            except Exception as e:
                console.print(f"[yellow]Note: Could not extract detailed performance metrics: {str(e)}[/yellow]")
                
        else:
            console.print(f"\n[bold red]× Error running evaluation: {result_data.get('error', 'Unknown error')}[/bold red]")
            if "traceback" in result_data:
                console.print(Syntax(result_data["traceback"], "python", theme="monokai", line_numbers=True))
    
    def show_evaluation_results(self):
        """Display the results of the most recent evaluation"""
        console.print(Panel(
            "[bold yellow]Showing Latest Evaluation Results[/bold yellow]",
            border_style="yellow",
            subtitle="Performance metrics"
        ))
        
        # Find the most recent evaluation result file
        eval_dir = "evaluation_results"
        if not os.path.exists(eval_dir) or not os.path.isdir(eval_dir):
            console.print("[bold red]No evaluation results found. Run an evaluation first.[/bold red]")
            return
        
        # Find all TREC evaluation result files
        trec_files = []
        for file in os.listdir(eval_dir):
            if file.startswith("trec_eval_results_") and file.endswith(".txt"):
                trec_files.append(os.path.join(eval_dir, file))
        
        if not trec_files:
            console.print("[bold red]No TREC evaluation results found. Run an evaluation first.[/bold red]")
            return
        
        # Sort by modification time to get the most recent TREC eval file
        latest_trec_file = max(trec_files, key=os.path.getmtime)
        latest_timestamp = os.path.basename(latest_trec_file).replace("trec_eval_results_", "").replace(".txt", "")
        
        # Find matching evaluation result file for reference (optional)
        matching_eval_file = None
        for file in os.listdir(eval_dir):
            if file.startswith("evaluation_results_") and file.endswith(".json") and latest_timestamp in file:
                matching_eval_file = os.path.join(eval_dir, file)
                break
        
        # Extract MAP score from TREC eval file
        map_score = None
        try:
            with open(latest_trec_file, 'r') as f:
                for line in f:
                    if line.startswith("map"):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            map_score = float(parts[2])
                        break
        except Exception as e:
            console.print(f"[bold red]Error reading evaluation file: {str(e)}[/bold red]")
            return
        
        if map_score is None:
            console.print("[bold red]Could not find MAP score in evaluation results.[/bold red]")
            return
        
        # Display the MAP score with nice formatting
        # Create a nice visualization with stars based on MAP score
        # When score is 0.15 or higher, display 4 stars
        if map_score >= 0.15:
            stars = "★" * 4
            empty_stars = "☆"
        else:
            stars = "★" * min(5, max(1, int(map_score * 10)))
            empty_stars = "☆" * (5 - min(5, max(1, int(map_score * 10))))
        
        map_rating = f"[yellow]{stars}[dim]{empty_stars}[/dim][/yellow]"
        
        # Add color coding based on score
        if map_score > 0.3:
            score_display = f"[bold green]{map_score:.4f}[/bold green]"
        elif map_score > 0.15:
            score_display = f"[bold yellow]{map_score:.4f}[/bold yellow]"
        else:
            score_display = f"[bold red]{map_score:.4f}[/bold red]"
        
        # Show when the evaluation was run
        eval_time = time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(latest_timestamp, "%Y%m%d_%H%M%S"))
        
        performance_panel = Panel(
            f"[bold cyan]Mean Average Precision (MAP):[/bold cyan] {score_display}\n"
            f"[dim]Rating:[/dim] {map_rating}\n"
            f"[dim]Evaluation Date:[/dim] {eval_time}\n"
            f"[dim]Results File:[/dim] [cyan]{os.path.basename(latest_trec_file)}[/cyan]",
            title="[bold]🏆 Performance Results[/bold]",
            border_style="green",
            expand=False,
            padding=(1, 2)
        )
        console.print(performance_panel)
    
    def interactive_mode(self):
        """Run the application in interactive mode"""
        while True:
            console.rule("[bold blue]InfoRetriever[/bold blue]")
            
            # Step 1: Load data if not loaded
            if not self.documents_loaded:
                console.print("[bold yellow]First, let's load some documents.[/bold yellow]")
                
                # List available JSON files
                json_files = self.get_available_json_files()
                
                if json_files:
                    console.print("\n[bold cyan]Available JSON files:[/bold cyan]")
                    
                    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
                    table.add_column("#", style="dim")
                    table.add_column("Path", style="cyan")
                    table.add_column("Size", style="green", justify="right")
                    
                    for i, file_path in enumerate(json_files, 1):
                        size = os.path.getsize(file_path)
                        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
                        table.add_row(str(i), file_path, size_str)
                    
                    console.print(table)
                    
                    # Let user select a file
                    selection = console.input("\n[bold cyan]Enter file number or custom path: [/bold cyan]")
                    
                    if selection.isdigit() and 1 <= int(selection) <= len(json_files):
                        documents_path = json_files[int(selection) - 1]
                    else:
                        documents_path = selection
                else:
                    documents_path = console.input("\n[bold cyan]Enter path to documents JSON file: [/bold cyan]")
                
                if not documents_path:
                    console.print("[bold red]No document path provided. Exiting.[/bold red]")
                    return
                
                if not self.load_documents(documents_path):
                    continue
                
                # Initialize engines
                self.init_boolean_engine()
                self.init_tfidf_engine()
            
            # Show menu with fancy styling
            menu_table = Table(show_header=False, box=box.SIMPLE)
            menu_table.add_column("Option", style="dim")
            menu_table.add_column("Description", style="yellow")
            
            menu_table.add_row("1", "Boolean Search")
            menu_table.add_row("2", "TF-IDF Search")
            menu_table.add_row("3", "Both Search Methods")
            menu_table.add_row("4", "Run Evaluation")
            menu_table.add_row("5", "Show Evaluation Results")
            menu_table.add_row("6", "Quit")
            
            console.print("\n[bold cyan]Available Actions:[/bold cyan]")
            console.print(menu_table)
            
            choice = console.input("\n[bold cyan]Enter choice (1-6): [/bold cyan]")
            
            if choice == '6' or choice.lower() == 'quit':
                break
            
            if choice == '4':
                self.run_evaluation()
                continue
            
            if choice == '5':
                self.show_evaluation_results()
                continue
            
            if choice not in ['1', '2', '3']:
                console.print("[bold red]Invalid choice. Please enter a number between 1 and 6.[/bold red]")
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
    parser.add_argument('--evaluate', action='store_true',
                      help='Run the evaluation script')
    args = parser.parse_args()
    
    # Create CLI interface
    retriever = InfoRetrieverCLI()
    
    # Display a fancy startup banner
    console.print("\n")
    console.rule("[bold blue]✦ ✦ ✦ InfoRetriever System ✦ ✦ ✦[/bold blue]", style="blue")
    retriever.print_header()
    console.rule(style="blue")
    
    # Handle evaluation mode
    if args.evaluate:
        retriever.run_evaluation()
        return
    
    # Run in interactive mode if specified or if no command-line arguments are provided
    if args.interactive or (not args.documents and not args.query and not args.boolean_query and not args.tfidf_query):
        retriever.interactive_mode()
        return
    
    # Load documents if provided
    if args.documents:
        if not retriever.load_documents(args.documents):
            sys.exit(1)
    
    # Initialize search engines with progress display
    console.print("[bold cyan]Initializing search engines...[/bold cyan]")
    
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
    
    # Query mode with improved output
    if args.boolean_query and retriever.boolean_engine:
        console.rule("[bold yellow]Boolean Query Search[/bold yellow]", style="yellow")
        boolean_results = retriever.search_boolean(args.boolean_query, args.top, debug=args.debug)
        retriever.display_results(boolean_results, "Boolean")
    
    if args.tfidf_query and retriever.tfidf_engine:
        console.rule("[bold yellow]TF-IDF Query Search[/bold yellow]", style="yellow")
        tfidf_results = retriever.search_tfidf(args.tfidf_query, args.top)
        retriever.display_results(tfidf_results, "TF-IDF")
    
    if args.query:
        if args.engine in ['boolean', 'both'] and retriever.boolean_engine:
            console.rule("[bold yellow]Boolean Query Search[/bold yellow]", style="yellow")
            boolean_results = retriever.search_boolean(args.query, args.top, debug=args.debug)
            retriever.display_results(boolean_results, "Boolean")
        
        if args.engine in ['tfidf', 'both'] and retriever.tfidf_engine:
            console.rule("[bold yellow]TF-IDF Query Search[/bold yellow]", style="yellow")
            tfidf_results = retriever.search_tfidf(args.query, args.top)
            retriever.display_results(tfidf_results, "TF-IDF")


if __name__ == "__main__":
    # Import here to avoid circular imports
    import re
    main()
