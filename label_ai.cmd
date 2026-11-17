@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 tools\ai_labels.py
set "tech1_result=%errorlevel%"
pause
exit /b %tech1_result%
