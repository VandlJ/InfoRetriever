#!/bin/zsh

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install required packages if they don't exist
pip install rich

# Run the CLI application
python cli_app.py --interactive
