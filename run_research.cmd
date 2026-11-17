@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "outputs\supervised_v3\model.json" goto missing
if not exist "outputs\supervised_balanced\model.json" goto missing
if not exist "outputs\supervised_boundary\model.json" goto missing
".venv\Scripts\python.exe" -X utf8 tools\select_ensemble.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_dense_boundaries.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_ensemble.toml --name exp_supervised_ensemble --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_boundary_dense.toml --name exp_supervised_boundary_dense --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py --config config\exp_supervised_ensemble.toml --plan config\idsse_benchmark_research.toml --out-dir outputs\idsse_benchmark_research
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\audit_spells.py --benchmark outputs\idsse_benchmark_research\report.json --out outputs\idsse_benchmark_research\duration_audit.json
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\check_release.py --with-artifacts --supervised-config config\exp_supervised_ensemble.toml --supervised-plan config\idsse_benchmark_research.toml --supervised-benchmark-dir outputs\idsse_benchmark_research --out-dir outputs\release_check_research
if errorlevel 1 goto failure
echo Research artifacts reproduced. Inspect validation and transfer before choosing a model.
pause
exit /b 0
:missing
echo Run run_all_tech.cmd first to prepare the frozen state and handover components.
pause
exit /b 1
:failure
echo A research step failed. Read the error above.
pause
exit /b 1
