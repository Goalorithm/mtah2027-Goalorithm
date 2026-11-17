@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow docs\SETUP.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 tools\prepare_idsse.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_supervised.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_supervised.py --config config\exp_supervised_v2.toml --out-dir outputs\supervised_v2
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_supervised.py --config config\exp_supervised_v3.toml --out-dir outputs\supervised_v3
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_supervised.py --config config\exp_supervised_v4.toml --out-dir outputs\supervised_v4
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_boundaries.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\train_supervised.py --config config\exp_supervised_balanced.toml --out-dir outputs\supervised_balanced
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\select_joint.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_v1.toml --name exp_supervised_v1 --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_v2.toml --name exp_supervised_v2 --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_v3.toml --name exp_supervised_v3 --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_v4.toml --name exp_supervised_v4 --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_boundary.toml --name exp_supervised_boundary --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_balanced.toml --name exp_supervised_balanced --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 -m mtah.predict --config config\exp_supervised_joint.toml --name exp_supervised_joint --no-score
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\run_experiments.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_external.py --download
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py --config config\exp_supervised_v2.toml --plan config\idsse_benchmark_v2.toml --out-dir outputs\idsse_benchmark_v2
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py --config config\exp_supervised_v3.toml --plan config\idsse_benchmark_v3.toml --out-dir outputs\idsse_benchmark_v3
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\audit_spells.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\benchmark_idsse.py --config config\exp_supervised_joint.toml --plan config\idsse_benchmark_joint.toml --out-dir outputs\idsse_benchmark_joint
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\audit_spells.py --benchmark outputs\idsse_benchmark_joint\report.json --out outputs\idsse_benchmark_joint\duration_audit.json
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\stress_idsse.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\baseline_diagnostics.py
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -X utf8 tools\check_release.py --with-artifacts
if errorlevel 1 goto failure
echo All local Tech 1/2/3 steps completed. No final model selected or submitted.
echo Results: outputs\team_experiments and outputs\release_check
pause
exit /b 0
:failure
echo A step failed. Read the error above; existing artifacts are not automatically repaired.
pause
exit /b 1
