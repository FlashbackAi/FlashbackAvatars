@echo off
echo ========================================
echo Quick Fix for Voice Chat
echo ========================================
echo.

echo Creating clean virtual environment...
python -m venv venv

echo Activating environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================
echo Testing GPU...
echo ========================================
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Now run:
echo   venv\Scripts\activate
echo   python server.py
echo.
pause
