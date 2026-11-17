@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "outputs\temporal\model.json" goto missing
".venv\Scripts\python.exe" -c "import torch"
if errorlevel 1 goto missing
".venv\Scripts\python.exe" -X utf8 tools\train_temporal_v2.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\select_temporal_ensemble.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\prepare_skillcorner.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_temporal_domain.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\select_domain_ensemble.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_temporal_domain_v1.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\refit_temporal.py
if errorlevel 1 goto failure
for %%M in (temporal_v2 temporal_domain temporal_domain_v1 temporal_refit) do (
    for %%P in (test anchors competition) do (
        ".venv\Scripts\python.exe" -X utf8 tools\evaluate_temporal.py --model-dir outputs\%%M --mode %%P
        if errorlevel 1 goto failure
    )
)
".venv\Scripts\python.exe" -X utf8 tools\evaluate_skillcorner.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\check_release.py --with-artifacts
if errorlevel 1 goto failure
echo Domain experiments reproduced. Read each report; partial ownership accuracy is not spellF1.
pause
exit /b 0
:missing
echo Prepare run_all_tech.cmd and run_temporal.cmd first, including optional PyTorch.
pause
exit /b 1
:failure
echo A domain step failed. Read the error above.
pause
exit /b 1
