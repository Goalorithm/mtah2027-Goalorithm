@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
echo Same-game human validation is retired under updated rules.
echo AI pseudo-label agreement cannot select a model by accuracy.
echo See docs\RULE_UPDATE.md. Use label_ai.cmd for pseudo-label diagnostics.
pause
exit /b 2
