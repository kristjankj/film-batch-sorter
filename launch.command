#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Film Batch Sorter — double-click to launch
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# Check for Python 3
if ! command -v python3 &>/dev/null; then
  osascript -e 'display alert "Python 3 not found" message "Install Python 3 from https://python.org and try again."'
  exit 1
fi

# Install / upgrade dependencies (silent after first run)
python3 -m pip install flask anthropic --quiet 2>&1 | grep -v WARNING

python3 film_batch_sorter.py
