@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 tools\check_release.py
set "CHECK_RESULT=%ERRORLEVEL%"
echo Report: outputs\release_check\report.json
pause
exit /b %CHECK_RESULT%
