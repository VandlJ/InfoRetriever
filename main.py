import os
import sys
import argparse

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main module
from InfoRetriever.main import main

if __name__ == "__main__":
    main()
