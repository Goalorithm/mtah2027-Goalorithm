@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "outputs\supervised_joint\model.json" goto missing
".venv\Scripts\python.exe" -c "import torch"
if errorlevel 1 goto torch_missing
".venv\Scripts\python.exe" -X utf8 tools\train_temporal.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\evaluate_temporal.py --mode test
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\evaluate_temporal.py --mode anchors
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\evaluate_temporal.py --mode competition
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\check_release.py --with-artifacts
if errorlevel 1 goto failure
echo Temporal experiment reproduced. Read outputs\temporal\test_report.json for actual performance.
pause
exit /b 0
:torch_missing
echo Install optional PyTorch first, then rerun:
echo .venv\Scripts\python.exe -m pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cu128
pause
exit /b 1
:missing
echo Run run_all_tech.cmd first to prepare the frozen parent and local data.
pause
exit /b 1
:failure
echo A temporal step failed. Read the error above.
pause
exit /b 1
