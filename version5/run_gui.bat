@echo off
REM ROSHN GUI Launcher for Windows
REM This batch file ensures the GUI runs with the correct Python environment

echo ============================================================
echo ROSHN Demarcation Plan Validation System
echo ============================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Run the GUI with virtual environment Python
echo Starting GUI...
echo.
".venv\Scripts\python.exe" gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error
    pause
)
