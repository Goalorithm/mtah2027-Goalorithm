@echo off
rem ATD-realistic TCN, fitted only on external matches (IDSSE, Metrica samples, SoccerTrack v2).
rem No video ball and no broadcast cutaways. Needs requirements-temporal.txt and
rem requirements-validation.txt, the IDSSE and Metrica downloads (run_all_tech.cmd) and SoccerTrack
rem v2 (accept its terms on Hugging Face, then "hf auth login").
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -c "import torch, ijson, huggingface_hub"
if errorlevel 1 goto missing
set PY=".venv\Scripts\python.exe" -X utf8
%PY% tools\prepare_metrica_epts.py
if errorlevel 1 goto failure
%PY% tools\prepare_soccertrack.py --no-video
if errorlevel 1 goto failure
%PY% tools\prepare_atd.py --splits train validation test
if errorlevel 1 goto failure
%PY% tools\prepare_atd_soccertrack.py
if errorlevel 1 goto failure
%PY% tools\train_atd.py --out-dir outputs\atds --set epochs=32
if errorlevel 1 goto failure
%PY% tools\train_atd.py --out-dir outputs\atds_s2 --set epochs=32 --set training_seed=20261121
if errorlevel 1 goto failure
%PY% tools\train_atd.py --out-dir outputs\atds_s3 --set epochs=32 --set training_seed=20261122
if errorlevel 1 goto failure
%PY% tools\ensemble_atd.py atds atds_s2 atds_s3 --out-dir outputs\atds_ensemble --soccertrack-validation
if errorlevel 1 goto failure
%PY% tools\evaluate_atd.py --model-dir outputs\atds_ensemble --mode test --legacy-dir ""
if errorlevel 1 goto failure
%PY% tools\evaluate_atd.py --model-dir outputs\atds_ensemble --mode competition --name Goalorithm
if errorlevel 1 goto failure
echo Done: outputs\Goalorithm.csv
pause
exit /b 0
:missing
echo Create .venv and install requirements-team.txt, requirements-temporal.txt and
echo requirements-validation.txt first (docs\SETUP.md).
pause
exit /b 1
:failure
echo A step failed. Read the error above.
pause
exit /b 1
