#!/bin/bash
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Setting up virtual environment (first run — takes a minute)…"
  python3 -m venv .venv
fi

PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"

# Install / upgrade dependencies
echo "Checking dependencies…"
$PIP install --quiet --upgrade flask anthropic pillow rawpy

# Start the server
echo "Starting Film Batch Sorter…"
$PYTHON film_batch_sorter.py &
SERVER_PID=$!

# Wait for server to be ready
sleep 3
open http://localhost:5174

echo "Server running (PID $SERVER_PID). Close this window to stop."
wait $SERVER_PID
