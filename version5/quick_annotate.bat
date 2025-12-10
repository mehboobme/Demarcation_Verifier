@echo off
echo ========================================
echo LABEL STUDIO - Quick Start
echo ========================================
echo.
echo Opening Label Studio at http://localhost:8080
echo.
echo Login with:
echo Username: admin@localhost
echo Password: admin123
echo.
echo Press Ctrl+C to stop when done.
echo.

cd /d "C:\MEHBOOB HD\Roshn Backup\Demarcation_Verifier\version5\label_studio_data"
start http://localhost:8080
label-studio start . --port 8080
