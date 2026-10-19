@echo off
echo ======================================================================
echo  NeuroScope AI — Meningioma Cohort Volumetric Evaluation Launcher
echo ======================================================================
cd /d "%~dp0"

if exist ".venv_gpu\Scripts\python.exe" (
    set "PYTHON_EXE=.venv_gpu\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" evaluate_meningioma_cases.py --num-cases 5
pause
