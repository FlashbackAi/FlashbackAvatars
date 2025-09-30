@echo off
REM 🚀 Windows Avatar System Deployment Script
REM Deploys real-time avatar system on Windows with PowerShell

echo 🚀 Starting Avatar System Deployment on Windows...
echo ========================================

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Please run as Administrator!
    echo Right-click and "Run as Administrator"
    pause
    exit /b 1
)

REM 1. Install Chocolatey (package manager)
echo 📦 Installing Chocolatey package manager...
powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

REM Refresh environment
call refreshenv.cmd

REM 2. Install Python 3.11
echo 🐍 Installing Python 3.11...
choco install python311 -y

REM 3. Install Git
echo 📂 Installing Git...
choco install git -y

REM 4. Install Visual Studio Build Tools
echo 🔨 Installing Visual Studio Build Tools...
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools" -y

REM 5. Install FFmpeg
echo 🎬 Installing FFmpeg...
choco install ffmpeg -y

REM 6. Install Ollama
echo 🦙 Installing Ollama...
powershell -Command "Invoke-WebRequest -Uri 'https://ollama.ai/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'"
start /wait OllamaSetup.exe /S
del OllamaSetup.exe

REM Wait for Ollama to start
echo ⏳ Waiting for Ollama to initialize...
timeout /t 10

REM 7. Download Llama model
echo 🧠 Downloading Llama 3.2 3B model...
ollama pull llama3.2:3b

REM 8. Create virtual environment
echo 🌍 Creating Python virtual environment...
python -m venv avatar_env
call avatar_env\Scripts\activate.bat

REM 9. Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM 10. Install PyTorch with CUDA
echo ⚡ Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM 11. Install dependencies
echo 📋 Installing Python dependencies...
pip install -r requirements.txt

REM 12. Run setup
echo ⚙️ Running system setup...
python setup_realtime_avatar.py

REM 13. Create startup batch file
echo 📝 Creating startup script...
echo @echo off > start_avatar.bat
echo call avatar_env\Scripts\activate.bat >> start_avatar.bat
echo python realtime_avatar_server.py >> start_avatar.bat
echo pause >> start_avatar.bat

echo ========================================
echo ✅ Windows Avatar System Deployment Complete!
echo ========================================
echo.
echo 🚀 To start your avatar:
echo    • Run: start_avatar.bat
echo    • Or: call avatar_env\Scripts\activate.bat ^&^& python realtime_avatar_server.py
echo.
echo 🌐 Web interface will be at: http://localhost:8000
echo.
echo 💡 Your avatar is ready for real-time conversations!
pause