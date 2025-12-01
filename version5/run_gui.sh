#!/bin/bash
# ROSHN GUI Launcher for Git Bash / Linux / macOS
# This script ensures the GUI runs with the correct Python environment

echo "============================================================"
echo "ROSHN Demarcation Plan Validation System"
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f ".venv/Scripts/python.exe" ] && [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    echo "Then: .venv/Scripts/pip install -r requirements.txt"
    exit 1
fi

# Determine which Python to use
if [ -f ".venv/Scripts/python.exe" ]; then
    # Windows (Git Bash)
    PYTHON=".venv/Scripts/python.exe"
else
    # Linux/macOS
    PYTHON=".venv/bin/python"
fi

# Run the GUI
echo "Starting GUI..."
echo ""
"$PYTHON" gui.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] GUI exited with an error"
    read -p "Press Enter to exit..."
fi
