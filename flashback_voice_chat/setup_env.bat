@echo off
echo ========================================
echo Voice Chat Environment Setup (Windows)
echo ========================================
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (for RTX 4070)
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install requirements
echo Installing voice chat requirements...
pip install -r requirements.txt

REM Fix numpy version conflict
echo Fixing numpy version...
pip install "numpy>=1.26.0" --upgrade

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate environment:
echo   venv\Scripts\activate
echo.
echo To run server:
echo   python server.py
echo.
echo To deactivate:
echo   deactivate
echo.
pause
