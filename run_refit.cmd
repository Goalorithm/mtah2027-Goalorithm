@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
if not exist "outputs\supervised_joint\model.json" goto missing
".venv\Scripts\python.exe" -X utf8 tools\refit_external.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_refit.toml --name exp_supervised_refit --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\run_experiments.py --config config\refit_experiments.toml --out-dir outputs\refit_experiments
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py --config config\exp_supervised_refit.toml --plan config\idsse_benchmark_refit.toml --out-dir outputs\idsse_benchmark_refit
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\audit_spells.py --benchmark outputs\idsse_benchmark_refit\report.json --out outputs\idsse_benchmark_refit\duration_audit.json
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_external.py --config config\external_benchmark_refit.toml --out-dir outputs\external_benchmark_refit
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\check_release.py --with-artifacts --supervised-config config\exp_supervised_refit.toml --supervised-plan config\idsse_benchmark_refit.toml --supervised-benchmark-dir outputs\idsse_benchmark_refit --out-dir outputs\release_check_refit
if errorlevel 1 goto failure
echo Refit artifacts reproduced. Check performance flags; successful checks do not imply a score of 0.8.
pause
exit /b 0
:missing
echo Run run_all_tech.cmd first to prepare the frozen selected model and training lineage.
pause
exit /b 1
:failure
echo A refit step failed. Read the error above.
pause
exit /b 1
