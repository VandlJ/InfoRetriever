#!/bin/zsh

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install required packages silently (no "already satisfied" messages)
pip install -q rich

# Clear screen for a clean start
clear

# Run the CLI application with interactive mode
python cli_app.py --interactive
