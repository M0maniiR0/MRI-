@echo off
echo ===================================================================
echo   NEUROSCOPE AI: BRAIN METASTASIS 3D GPU TRAINING (RTX 2050)
echo ===================================================================
cd /d "%~dp0"
.venv_gpu\Scripts\python.exe train_brain_metastasis_transfer.py --epochs 12 --batch-size 2 --device cuda
pause
