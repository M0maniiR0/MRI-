@echo off
echo ===================================================================
echo   NEUROSCOPE AI: MENINGIOMA LOCAL GPU TRAINING (RTX 2050)
echo ===================================================================
.venv_gpu\Scripts\python.exe train_meningioma_transfer.py --data Task_Meningioma/BraTS_MEN --pretrained models/segresnet_fullvolume_best.pt --output models/segresnet_meningioma_best.pt --epochs 10 --device cuda
pause
