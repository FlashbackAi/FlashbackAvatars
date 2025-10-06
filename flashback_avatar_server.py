#!/usr/bin/env python3
"""
Flashback Avatar - Real-Time Interactive Avatar Server
Complete integration: LLM + TTS + MuseTalk Avatar Animation
"""

import os
import sys
import subprocess
from pathlib import Path
import asyncio
import json
import tempfile
import shutil
from typing import Optional
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Flashback Avatar Server")


class FlashbackAvatarEngine:
    """Real-time avatar engine with LLM + TTS + MuseTalk"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.musetalk_dir = self.base_dir / "third_party" / "MuseTalk"

        # Avatar settings
        self.avatar_name = "vinay_avatar"
        self.avatar_cache = self.musetalk_dir / "results" / "v15" / "avatars" / self.avatar_name

        # Model paths
        self.unet_path = self.musetalk_dir / "models" / "musetalkV15" / "unet.pth"
        self.unet_config = self.musetalk_dir / "models" / "musetalkV15" / "musetalk.json"

        print("🎭 Initializing Flashback Avatar Engine...")
        self._check_ollama()
        self._check_musetalk()
        self._load_tts()

    def _check_ollama(self):
        """Check if Ollama LLM is available"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.llm_available = True
                models = response.json().get("models", [])
                print(f"✅ Ollama LLM connected ({len(models)} models)")
            else:
                self.llm_available = False
                print("⚠️  Ollama not responding")
        except Exception as e:
            self.llm_available = False
            print(f"⚠️  Ollama not available: {e}")

    def _check_musetalk(self):
        """Check if MuseTalk is ready"""
        if not self.musetalk_dir.exists():
            print("❌ MuseTalk directory not found!")
            self.musetalk_ready = False
            return

        if not self.avatar_cache.exists():
            print(f"⚠️  Avatar not prepared. Run preparation first.")
            self.musetalk_ready = False
            return

        self.musetalk_ready = True
        print(f"✅ MuseTalk ready (avatar: {self.avatar_name})")

    def _load_tts(self):
        """Load Text-to-Speech engine"""
        try:
            import edge_tts
            self.tts = edge_tts
            print("✅ TTS (edge-tts) loaded")
        except ImportError:
            print("⚠️  edge-tts not installed. Run: pip install edge-tts")
            self.tts = None

    async def generate_llm_response(self, user_message: str, context: str = "") -> str:
        """Generate LLM response using Ollama"""
        if not self.llm_available:
            return "I'm sorry, the language model is currently unavailable."

        try:
            import requests

            # Build prompt with context
            system_prompt = """You are Vinay Thadem. You are a helpful, friendly person who speaks naturally and conversationally.
Keep responses under 2-3 sentences unless asked for more detail. Respond as yourself, not as an AI or avatar."""

            full_prompt = f"{system_prompt}\n\nUser: {user_message}\nVinay:"

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "I apologize, I'm having trouble generating a response."

        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "I encountered an error while processing your request."

    async def text_to_speech(self, text: str) -> Optional[Path]:
        """Convert text to speech using edge-tts"""
        if not self.tts:
            print("⚠️  TTS not available")
            return None

        try:
            # Create temp audio file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp")
            output_path = Path(temp_audio.name)

            # Generate speech
            communicate = self.tts.Communicate(text, voice="en-US-GuyNeural")
            await communicate.save(str(output_path))

            print(f"✅ Generated speech: {output_path.name}")
            return output_path

        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None

    def animate_avatar(self, audio_path: Path) -> Optional[Path]:
        """Animate avatar with audio using MuseTalk"""
        if not self.musetalk_ready:
            print("⚠️  MuseTalk not ready")
            return None

        try:
            print(f"🎬 Animating avatar with: {audio_path.name}")
            start_time = time.time()

            # Create temp config for this audio
            temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml', dir="/tmp")
            config_data = {
                self.avatar_name: {
                    'preparation': False,  # Use cached avatar
                    'bbox_shift': 0,
                    'video_path': 'data/video/vinay_small.mp4',
                    'audio_clips': {
                        'temp_audio': str(audio_path)
                    }
                }
            }

            import yaml
            yaml.dump(config_data, temp_config)
            temp_config.close()

            # Run MuseTalk inference
            cmd = [
                sys.executable, "-m", "scripts.realtime_inference",
                "--inference_config", temp_config.name,
                "--result_dir", "results",
                "--unet_model_path", str(self.unet_path),
                "--unet_config", str(self.unet_config),
                "--version", "v15",
                "--fps", "25"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.musetalk_dir,
                capture_output=True,
                text=True,
                timeout=120  # Increased from 30 to 120 seconds
            )

            if result.returncode != 0:
                print(f"❌ MuseTalk failed: {result.stderr}")
                return None

            # Find generated video
            output_video = self.avatar_cache / "vid_output" / "temp_audio.mp4"

            if not output_video.exists():
                print(f"❌ Output video not found: {output_video}")
                return None

            # Copy to temp location for serving
            final_video = Path(f"/tmp/avatar_{int(time.time())}.mp4")
            shutil.copy(output_video, final_video)

            elapsed = time.time() - start_time
            print(f"✅ Avatar animated in {elapsed:.2f}s: {final_video.name}")

            # Cleanup
            os.unlink(temp_config.name)

            return final_video

        except Exception as e:
            print(f"❌ Animation error: {e}")
            return None

    async def process_message(self, user_message: str) -> dict:
        """Process user message and generate avatar response"""

        print(f"\n{'='*60}")
        print(f"💬 User: {user_message}")

        # 1. Generate LLM response
        llm_response = await self.generate_llm_response(user_message)
        print(f"🤖 Avatar: {llm_response}")

        # 2. Convert to speech
        audio_path = await self.text_to_speech(llm_response)
        if not audio_path:
            return {"error": "TTS failed"}

        # 3. Animate avatar
        video_path = self.animate_avatar(audio_path)

        # Cleanup audio
        audio_path.unlink()

        if not video_path:
            return {"error": "Animation failed"}

        # 4. Return results
        return {
            "text": llm_response,
            "video": str(video_path),
            "video_filename": video_path.name
        }


# Global engine instance
engine = FlashbackAvatarEngine()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    print("🔗 Client connected")

    try:
        while True:
            # Receive user message
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            # Send "thinking" status
            await websocket.send_json({"status": "thinking"})

            # Process message
            result = await engine.process_message(user_message)

            if "error" in result:
                await websocket.send_json({"error": result["error"]})
                continue

            # Send result
            await websocket.send_json({
                "text": result["text"],
                "video_url": f"/videos/{result['video_filename']}"
            })

    except WebSocketDisconnect:
        print("🔌 Client disconnected")


@app.get("/videos/{filename}")
async def serve_video(filename: str):
    """Serve generated video files"""
    video_path = Path(f"/tmp/{filename}")
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


@app.get("/")
async def get_home():
    """Serve web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flashback Avatar - Interactive AI Avatar</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .header {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 20px;
                text-align: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }

            .header h1 {
                color: white;
                font-size: 32px;
                font-weight: 700;
                margin-bottom: 5px;
            }

            .header p {
                color: rgba(255, 255, 255, 0.8);
                font-size: 14px;
            }

            .container {
                flex: 1;
                max-width: 1400px;
                width: 100%;
                margin: 0 auto;
                padding: 30px 20px;
                display: flex;
                gap: 30px;
            }

            .avatar-panel {
                flex: 1;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }

            #status {
                text-align: center;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 20px;
                font-weight: 600;
                font-size: 14px;
            }

            .connected {
                background: #10b981;
                color: white;
            }

            .disconnected {
                background: #ef4444;
                color: white;
            }

            .thinking {
                background: #f59e0b;
                color: white;
            }

            #avatar-container {
                width: 100%;
                aspect-ratio: 1;
                background: #000;
                border-radius: 15px;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            }

            #avatar-video {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }

            .placeholder {
                color: #666;
                text-align: center;
                padding: 40px;
            }

            .chat-panel {
                flex: 1;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                display: flex;
                flex-direction: column;
            }

            .chat-panel h2 {
                margin-bottom: 20px;
                color: #1f2937;
            }

            #messages {
                flex: 1;
                overflow-y: auto;
                background: #f9fafb;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                min-height: 400px;
            }

            .message {
                margin: 15px 0;
                padding: 12px 18px;
                border-radius: 12px;
                max-width: 80%;
                word-wrap: break-word;
            }

            .user-message {
                background: #667eea;
                color: white;
                margin-left: auto;
                text-align: right;
            }

            .avatar-message {
                background: #10b981;
                color: white;
            }

            #input-container {
                display: flex;
                gap: 10px;
            }

            #user-input {
                flex: 1;
                padding: 15px;
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
            }

            #user-input:focus {
                outline: none;
                border-color: #667eea;
            }

            #send-btn {
                padding: 15px 30px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background 0.3s;
            }

            #send-btn:hover {
                background: #5568d3;
            }

            #send-btn:active {
                transform: scale(0.98);
            }

            @media (max-width: 1024px) {
                .container {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎭 Flashback Avatar</h1>
            <p>Real-Time Interactive AI Avatar powered by Flashback Dev</p>
        </div>

        <div class="container">
            <div class="avatar-panel">
                <div id="status" class="disconnected">Connecting...</div>
                <div id="avatar-container">
                    <video id="avatar-video" autoplay muted>
                        <source src="" type="video/mp4">
                    </video>
                    <div id="placeholder" class="placeholder">
                        <h3>👤 Avatar Ready</h3>
                        <p>Start a conversation to see the avatar in action</p>
                    </div>
                </div>
            </div>

            <div class="chat-panel">
                <h2>💬 Conversation</h2>
                <div id="messages"></div>
                <div id="input-container">
                    <input type="text" id="user-input" placeholder="Type your message..." />
                    <button id="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            let ws;
            const statusDiv = document.getElementById('status');
            const messagesDiv = document.getElementById('messages');
            const userInput = document.getElementById('user-input');
            const avatarVideo = document.getElementById('avatar-video');
            const placeholder = document.getElementById('placeholder');

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

                ws.onopen = () => {
                    statusDiv.textContent = '✅ Connected';
                    statusDiv.className = 'connected';
                };

                ws.onclose = () => {
                    statusDiv.textContent = '❌ Disconnected - Reconnecting...';
                    statusDiv.className = 'disconnected';
                    setTimeout(connect, 3000);
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);

                    if (data.status === 'thinking') {
                        statusDiv.textContent = '🤔 Thinking...';
                        statusDiv.className = 'thinking';
                        return;
                    }

                    if (data.error) {
                        alert('Error: ' + data.error);
                        statusDiv.textContent = '✅ Connected';
                        statusDiv.className = 'connected';
                        return;
                    }

                    // Add avatar message
                    addMessage(data.text, 'avatar');

                    // Play avatar video
                    if (data.video_url) {
                        placeholder.style.display = 'none';
                        avatarVideo.src = data.video_url + '?t=' + Date.now();
                        avatarVideo.style.display = 'block';
                        avatarVideo.play();
                    }

                    statusDiv.textContent = '✅ Connected';
                    statusDiv.className = 'connected';
                };
            }

            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'avatar-message');
                messageDiv.textContent = (sender === 'user' ? 'You: ' : 'Vinay: ') + text;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function sendMessage() {
                const message = userInput.value.trim();
                if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

                addMessage(message, 'user');
                ws.send(JSON.stringify({ message: message }));
                userInput.value = '';
            }

            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            // Initial connection
            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "llm": engine.llm_available,
        "tts": engine.tts is not None,
        "musetalk": engine.musetalk_ready
    }


def main():
    """Start server"""
    print("\n" + "="*60)
    print("🚀 Flashback Avatar Server")
    print("="*60)
    print(f"📱 Web interface: http://localhost:8000")
    print(f"🔧 Health check: http://localhost:8000/health")
    print(f"🎭 Avatar: {engine.avatar_name}")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
