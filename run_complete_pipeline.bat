@echo off
REM Complete Real-Time Avatar Pipeline for Windows
REM Run this to go from video to live avatar

echo ========================================
echo FlashbackAvatars - Complete Pipeline
echo ========================================

REM Step 1: Check video exists
set VIDEO_PATH=avatar_input\vinay_intro_shoulders_up.mp4
if not exist "%VIDEO_PATH%" (
    echo ❌ Video not found: %VIDEO_PATH%
    echo Please ensure your video is at: %VIDEO_PATH%
    exit /b 1
)

echo ✅ Video found: %VIDEO_PATH%

REM Step 2: Install dependencies
echo.
echo 📦 Step 1: Installing dependencies...
pip install mediapipe plyfile huggingface_hub

REM Step 3: Download models
echo.
echo 📥 Step 2: Downloading models...
python download_models.py

REM Step 4: Preprocess video
echo.
echo 🎬 Step 3: Preprocessing video...
python preprocess_avatar_video.py "%VIDEO_PATH%" --fps 30

REM Step 5: Train avatar (THIS TAKES 1-2 HOURS)
echo.
echo 🎨 Step 4: Training SplattingAvatar (1-2 hours on H200)...
echo ⏳ This will take a while. Go get coffee! ☕
python train_avatar.py third_party\SplattingAvatar\data\vinay_intro_shoulders_up --export

REM Step 6: Verify output
set PLY_PATH=third_party\SplattingAvatar\output\vinay_intro_shoulders_up\vinay_avatar.ply
if exist "%PLY_PATH%" (
    echo.
    echo ✅ Avatar model created successfully!
    echo 📂 Location: %PLY_PATH%
) else (
    echo.
    echo ❌ Avatar model not found. Training may have failed.
    exit /b 1
)

REM Step 7: Start server
echo.
echo ==========================================
echo 🎉 Setup complete! Starting server...
echo ==========================================
echo.
echo 📱 Web interface: http://localhost:8000
echo 🎤 Microphone will auto-start for voice input
echo 💬 Or type messages in the chat
echo.

python realtime_avatar_server.py
