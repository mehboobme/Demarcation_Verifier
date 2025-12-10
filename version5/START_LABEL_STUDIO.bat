@echo off
title Label Studio Server
echo ============================================
echo Label Studio Server - Running on port 8080
echo ============================================
echo.
echo Server URL: http://localhost:8080
echo.
echo IMPORTANT: Keep this window OPEN while annotating
echo Press Ctrl+C to stop the server when done
echo.
echo ============================================
echo.

:start
label-studio start --port 8080 --log-level ERROR
if errorlevel 1 (
    echo.
    echo Server stopped. Restarting in 3 seconds...
    timeout /t 3 /nobreak
    goto start
)
