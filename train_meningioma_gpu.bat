@echo off
echo ===================================================================
echo   NEUROSCOPE AI: MENINGIOMA LOCAL GPU TRAINING (RTX 2050 - 200 CASES)
echo ===================================================================
cd /d "%~dp0"
.venv_gpu\Scripts\python.exe train_meningioma_transfer.py --data Task_Meningioma/BraTS_MEN --pretrained models/segresnet_fullvolume_best.pt --output models/segresnet_meningioma_best.pt --epochs 6 --batch-size 2 --device cuda
pause
