@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 tools\ai_labels.py --config config\exp_hmm_v2.toml --out-dir outputs\ai_labels_v2
set "tech1_result=%errorlevel%"
pause
exit /b %tech1_result%
