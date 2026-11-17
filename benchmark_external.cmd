@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 tools\benchmark_external.py --download
set "tech1_result=%errorlevel%"
echo Report: outputs\external_benchmark\report.md
echo This measures team correctness at pass starts, not full possession accuracy.
pause
exit /b %tech1_result%
