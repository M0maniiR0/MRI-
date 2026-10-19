@echo off
echo ===================================================================
echo   NEUROSCOPE AI: ALZHEIMER'S HIPPOCAMPUS LOCAL GPU TRAINING (RTX 2050)
echo ===================================================================
.venv_gpu\Scripts\python.exe train_alzheimer_hippocampus.py --epochs 20 --batch-size 4 --device cuda
pause
