@echo off
echo ===================================================================
echo   NEUROSCOPE AI: MULTIPLE SCLEROSIS 3D GPU TRAINING (RTX 2050)
echo ===================================================================
cd /d "%~dp0"
.venv_gpu\Scripts\python.exe train_multiple_sclerosis.py --epochs 10 --batch-size 2 --device cuda
pause
