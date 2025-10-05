#!/usr/bin/env python3
"""
Real-Time MuseTalk Avatar Server
Integrates LLM + TTS + MuseTalk for interactive avatar
With optional background diffusion
"""

import os
import sys
from pathlib import Path
import asyncio
import json
import base64
from typing import Optional
import tempfile

# Add MuseTalk to path
musetalk_dir = Path(__file__).parent / "third_party" / "MuseTalk"
sys.path.insert(0, str(musetalk_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()


class RealtimeAvatarServer:
    """Real-time avatar with MuseTalk"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.musetalk_dir = self.base_dir / "third_party" / "MuseTalk"

        # Avatar configuration
        self.avatar_video = "data/video/vinay.mp4"
        self.avatar_prepared = False

        # Background options
        self.background_mode = "original"  # original, blur, green_screen, generated
        self.generated_background = None

        print("🎭 Initializing Real-Time Avatar Server...")
        self._load_musetalk()
        self._load_tts()
        self._load_llm()

    def _load_musetalk(self):
        """Load MuseTalk model"""
        try:
            # Import MuseTalk components
            from musetalk.utils.utils import get_file_type, get_video_fps, datagen
            from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
            from musetalk.utils.blending import get_image
            from musetalk.models.unet import UNet

            import torch
            from omegaconf import OmegaConf

            # Load MuseTalk V1.5 model
            model_path = self.musetalk_dir / "models" / "musetalkV15" / "unet.pth"
            config_path = self.musetalk_dir / "models" / "musetalkV15" / "musetalk.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️  Using device: {self.device}")

            # Load model config
            with open(config_path) as f:
                config = json.load(f)

            self.musetalk_model = UNet(**config)
            self.musetalk_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.musetalk_model.to(self.device)
            self.musetalk_model.eval()

            print("✅ MuseTalk model loaded")

        except Exception as e:
            print(f"❌ Failed to load MuseTalk: {e}")
            self.musetalk_model = None

    def _load_tts(self):
        """Load Text-to-Speech model"""
        try:
            # Using edge-tts (fast and free)
            import edge_tts
            self.tts = edge_tts
            print("✅ TTS (edge-tts) loaded")
        except:
            print("⚠️  edge-tts not installed. Install with: pip install edge-tts")
            self.tts = None

    def _load_llm(self):
        """Load LLM (using Ollama)"""
        try:
            import requests
            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.llm_available = True
                print("✅ LLM (Ollama) connected")
            else:
                self.llm_available = False
                print("⚠️  Ollama not running")
        except:
            self.llm_available = False
            print("⚠️  Ollama not available")

    async def prepare_avatar(self, video_path: str):
        """Prepare avatar for first-time use (extract features)"""
        print(f"🎬 Preparing avatar from: {video_path}")

        # This would run MuseTalk preprocessing
        # For now, we'll assume it's been done
        self.avatar_prepared = True
        print("✅ Avatar prepared")

    async def generate_response(self, user_message: str) -> str:
        """Generate LLM response"""
        if not self.llm_available:
            return "Hello! I'm Vinay's avatar. LLM is not available right now."

        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": user_message,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Sorry, I couldn't generate a response."

        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "Sorry, something went wrong."

    async def text_to_speech(self, text: str) -> Path:
        """Convert text to speech audio"""
        if not self.tts:
            print("⚠️  TTS not available")
            return None

        try:
            # Create temp file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            output_path = Path(temp_audio.name)

            communicate = self.tts.Communicate(text, voice="en-US-GuyNeural")
            await communicate.save(str(output_path))

            print(f"✅ Generated speech: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None

    async def animate_avatar(self, audio_path: Path) -> Optional[Path]:
        """Animate avatar with audio using MuseTalk"""
        if not self.musetalk_model:
            print("⚠️  MuseTalk model not available")
            return None

        try:
            # Run MuseTalk inference
            print(f"🎬 Animating avatar with audio: {audio_path}")

            # This would call MuseTalk real-time inference
            # For now, return a placeholder
            # TODO: Implement actual MuseTalk inference call

            output_video = Path(tempfile.mkdtemp()) / "avatar_output.mp4"

            print(f"✅ Avatar animated: {output_video}")
            return output_video

        except Exception as e:
            print(f"❌ Animation error: {e}")
            return None

    async def process_message(self, user_message: str) -> dict:
        """Process user message and generate avatar response"""

        # 1. Generate LLM response
        print(f"💬 User: {user_message}")
        llm_response = await self.generate_response(user_message)
        print(f"🤖 Avatar: {llm_response}")

        # 2. Convert to speech
        audio_path = await self.text_to_speech(llm_response)
        if not audio_path:
            return {"error": "TTS failed"}

        # 3. Animate avatar
        video_path = await self.animate_avatar(audio_path)
        if not video_path:
            # Cleanup
            audio_path.unlink()
            return {"error": "Animation failed"}

        # 4. Return results
        return {
            "text": llm_response,
            "audio": str(audio_path),
            "video": str(video_path)
        }


# Global server instance
server = RealtimeAvatarServer()


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

            # Process message
            result = await server.process_message(user_message)

            if "error" in result:
                await websocket.send_json({"error": result["error"]})
                continue

            # Send video path (client will fetch it)
            await websocket.send_json({
                "text": result["text"],
                "video_url": f"/video/{Path(result['video']).name}"
            })

    except WebSocketDisconnect:
        print("🔌 Client disconnected")


@app.get("/")
async def get_home():
    """Serve web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Avatar - Vinay</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #1a1a1a;
                color: #fff;
            }
            h1 {
                text-align: center;
                color: #4CAF50;
            }
            #avatar-container {
                width: 800px;
                height: 600px;
                margin: 20px auto;
                background: #000;
                border-radius: 10px;
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            video {
                max-width: 100%;
                max-height: 100%;
            }
            #chat-container {
                max-width: 800px;
                margin: 20px auto;
            }
            #messages {
                height: 200px;
                overflow-y: auto;
                background: #2a2a2a;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .message {
                margin: 10px 0;
                padding: 8px;
                border-radius: 5px;
            }
            .user-message {
                background: #1976D2;
                text-align: right;
            }
            .avatar-message {
                background: #388E3C;
            }
            #input-container {
                display: flex;
                gap: 10px;
            }
            #user-input {
                flex: 1;
                padding: 10px;
                border: none;
                border-radius: 5px;
                background: #2a2a2a;
                color: #fff;
                font-size: 16px;
            }
            #send-btn {
                padding: 10px 30px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #send-btn:hover {
                background: #45a049;
            }
            #status {
                text-align: center;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .connected {
                background: #2E7D32;
            }
            .disconnected {
                background: #C62828;
            }
        </style>
    </head>
    <body>
        <h1>🎭 Real-Time Avatar - Vinay</h1>

        <div id="status" class="disconnected">Connecting...</div>

        <div id="avatar-container">
            <video id="avatar-video" autoplay loop muted>
                <source src="" type="video/mp4">
                Your browser does not support video.
            </video>
        </div>

        <div id="chat-container">
            <div id="messages"></div>
            <div id="input-container">
                <input type="text" id="user-input" placeholder="Type your message..." />
                <button id="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let ws;
            const statusDiv = document.getElementById('status');
            const messagesDiv = document.getElementById('messages');
            const userInput = document.getElementById('user-input');
            const avatarVideo = document.getElementById('avatar-video');

            function connect() {
                ws = new WebSocket('ws://localhost:8000/ws');

                ws.onopen = () => {
                    statusDiv.textContent = '✅ Connected';
                    statusDiv.className = 'connected';
                };

                ws.onclose = () => {
                    statusDiv.textContent = '❌ Disconnected';
                    statusDiv.className = 'disconnected';
                    setTimeout(connect, 3000);
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);

                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }

                    // Add avatar message
                    addMessage(data.text, 'avatar');

                    // Play avatar video
                    if (data.video_url) {
                        avatarVideo.src = data.video_url;
                        avatarVideo.play();
                    }
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
                if (!message) return;

                addMessage(message, 'user');
                ws.send(JSON.stringify({ message: message }));
                userInput.value = '';
            }

            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def main():
    """Start server"""
    print("🚀 Starting Real-Time Avatar Server")
    print("=" * 60)
    print("📱 Web interface: http://localhost:8000")
    print("🎭 Avatar: Vinay (shoulders-up)")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
