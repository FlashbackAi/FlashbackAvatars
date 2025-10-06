#!/usr/bin/env python3
"""
Flashback Avatar - Complete RAG + Voice-Enabled Server
Features: RAG knowledge base, Voice input/output, Real-time avatar
"""

import os
import sys
import subprocess
from pathlib import Path
import asyncio
import json
import tempfile
import shutil
from typing import Optional, List
import time

# RAG and embeddings
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Flashback Avatar - RAG + Voice")


class RAGKnowledgeBase:
    """Vector database for Vinay's knowledge"""

    def __init__(self):
        if not CHROMA_AVAILABLE:
            print("⚠️  ChromaDB not available. RAG disabled.")
            self.client = None
            return

        self.client = chromadb.PersistentClient(path="./rag_db")

        # Use sentence transformers for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name="vinay_knowledge",
                embedding_function=self.embedding_fn
            )
            print(f"✅ Loaded RAG knowledge base ({self.collection.count()} documents)")
        except:
            self.collection = self.client.create_collection(
                name="vinay_knowledge",
                embedding_function=self.embedding_fn
            )
            self._load_default_knowledge()
            print(f"✅ Created RAG knowledge base ({self.collection.count()} documents)")

    def _load_default_knowledge(self):
        """Load default knowledge about Vinay and Flashback Labs"""
        knowledge = [
            {
                "id": "bio_1",
                "text": "Vinay Thadem is the Co-Founder of Flashback Labs, a company focused on AI-powered avatar technology and real-time interactive systems.",
                "metadata": {"category": "bio", "topic": "founder"}
            },
            {
                "id": "company_1",
                "text": "Flashback Labs specializes in creating photorealistic digital avatars using advanced AI technologies including MuseTalk for lip-sync and 3D Gaussian Splatting for rendering.",
                "metadata": {"category": "company", "topic": "technology"}
            },
            # {
            #     "id": "tech_1",
            #     "text": "Our avatar technology uses MuseTalk for real-time lip synchronization, achieving 30+ FPS on modern GPUs with sub-second latency.",
            #     "metadata": {"category": "technology", "topic": "musetalk"}
            # },
            {
                "id": "mission_1",
                "text": "Flashback Labs' mission is to make human-AI interaction more natural and engaging through lifelike digital avatars.",
                "metadata": {"category": "company", "topic": "mission"}
            }
        ]

        for doc in knowledge:
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[doc["metadata"]]
            )

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """Search knowledge base for relevant context"""
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if results['documents']:
            return results['documents'][0]
        return []

    def add_knowledge(self, text: str, category: str = "general"):
        """Add new knowledge to the database"""
        if not self.collection:
            return

        doc_id = f"{category}_{int(time.time())}"
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"category": category, "timestamp": time.time()}]
        )


class FlashbackAvatarRAG:
    """Complete avatar engine with RAG and voice"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.musetalk_dir = self.base_dir / "third_party" / "MuseTalk"

        # Avatar settings
        self.avatar_name = "vinay_avatar"
        self.avatar_cache = self.musetalk_dir / "results" / "v15" / "avatars" / self.avatar_name

        # Model paths
        self.unet_path = self.musetalk_dir / "models" / "musetalkV15" / "unet.pth"
        self.unet_config = self.musetalk_dir / "models" / "musetalkV15" / "musetalk.json"

        print("🎭 Initializing Flashback Avatar (RAG + Voice)...")

        # Initialize RAG
        self.rag = RAGKnowledgeBase()

        # Check services
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
                print("✅ Ollama LLM connected")
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

    async def generate_rag_response(self, user_message: str) -> tuple[str, List[str]]:
        """Generate LLM response with RAG context"""
        if not self.llm_available:
            return "I'm sorry, the language model is currently unavailable.", []

        try:
            import requests

            # Get relevant context from RAG
            context_docs = self.rag.search(user_message, n_results=3)

            # Build context string
            context_str = "\n".join([f"- {doc}" for doc in context_docs]) if context_docs else ""

            # Build prompt with RAG context
            system_prompt = """You are Vinay Thadem, Co-Founder of Flashback Labs. You are helpful, friendly, and speak naturally.
Keep responses under 2-3 sentences unless asked for more detail. Use the provided context when relevant."""

            if context_str:
                full_prompt = f"""{system_prompt}

Context information:
{context_str}

User: {user_message}
Vinay:"""
            else:
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
                return response.json()["response"], context_docs
            else:
                return "I apologize, I'm having trouble generating a response.", []

        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "I encountered an error while processing your request.", []

    async def text_to_speech(self, text: str) -> Optional[Path]:
        """Convert text to speech using edge-tts"""
        if not self.tts:
            print("⚠️  TTS not available")
            return None

        try:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp")
            output_path = Path(temp_audio.name)

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
            import yaml
            config_data = {
                self.avatar_name: {
                    'preparation': False,
                    'bbox_shift': 0,
                    'video_path': 'data/video/vinay_small.mp4',
                    'audio_clips': {
                        'temp_audio': str(audio_path)
                    }
                }
            }

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
                timeout=120
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
        """Process user message with RAG and generate avatar response"""

        print(f"\n{'='*60}")
        print(f"💬 User: {user_message}")

        # 1. Generate LLM response with RAG
        llm_response, context = await self.generate_rag_response(user_message)
        print(f"🤖 Avatar: {llm_response}")
        if context:
            print(f"📚 Used context: {len(context)} documents")

        # 2. Convert to speech
        audio_path = await self.text_to_speech(llm_response)
        if not audio_path:
            return {"error": "TTS failed"}

        # 3. Animate avatar
        video_path = self.animate_avatar(audio_path)

        # Return audio path for voice output
        audio_url = f"/audio/{audio_path.name}" if audio_path else None

        if not video_path:
            if audio_path:
                audio_path.unlink()
            return {"error": "Animation failed"}

        # 4. Return results
        return {
            "text": llm_response,
            "video": str(video_path),
            "video_filename": video_path.name,
            "audio": str(audio_path),
            "audio_filename": audio_path.name,
            "context_used": context
        }


# Global engine instance
engine = FlashbackAvatarRAG()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    print("🔗 Client connected")

    try:
        # Send initial welcome message
        welcome_text = "Hi! I'm Vinay Thadem, Co-Founder of Flashback Labs. How can I help you today?"

        await websocket.send_json({"status": "initializing"})

        audio_path = await engine.text_to_speech(welcome_text)
        if audio_path:
            video_path = engine.animate_avatar(audio_path)

            if video_path:
                await websocket.send_json({
                    "type": "welcome",
                    "text": welcome_text,
                    "video_url": f"/videos/{video_path.name}",
                    "audio_url": f"/audio/{audio_path.name}"
                })
            else:
                if audio_path.exists():
                    audio_path.unlink()

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

            # Send result with audio
            await websocket.send_json({
                "type": "response",
                "text": result["text"],
                "video_url": f"/videos/{result['video_filename']}",
                "audio_url": f"/audio/{result['audio_filename']}",
                "context": result.get("context_used", [])
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


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    audio_path = Path(f"/tmp/{filename}")
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    return {"error": "Audio not found"}


@app.post("/add_knowledge")
async def add_knowledge(text: str, category: str = "general"):
    """Add new knowledge to RAG database"""
    engine.rag.add_knowledge(text, category)
    return {"status": "added", "category": category}


@app.get("/")
async def get_home():
    """Serve web interface with voice input/output"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flashback Avatar - Voice Enabled</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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

            .connected { background: #10b981; color: white; }
            .disconnected { background: #ef4444; color: white; }
            .thinking { background: #f59e0b; color: white; }
            .listening { background: #8b5cf6; color: white; }

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

            #avatar-audio {
                display: none;
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

            .context-info {
                font-size: 11px;
                margin-top: 5px;
                opacity: 0.8;
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
            }

            #user-input:focus {
                outline: none;
                border-color: #667eea;
            }

            .btn {
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
            }

            #send-btn {
                background: #667eea;
                color: white;
            }

            #send-btn:hover { background: #5568d3; }

            #voice-btn {
                background: #8b5cf6;
                color: white;
                min-width: 60px;
            }

            #voice-btn.listening {
                background: #ef4444;
                animation: pulse 1s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            @media (max-width: 1024px) {
                .container { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Flashback Labs</h1>
            <p>Real-Time Voice-Enabled AI Avatar</p>
        </div>

        <div class="container">
            <div class="avatar-panel">
                <div id="status" class="disconnected">Connecting...</div>
                <div id="avatar-container">
                    <video id="avatar-video" autoplay muted></video>
                    <audio id="avatar-audio" autoplay></audio>
                    <div id="placeholder" class="placeholder">
                        <h3>👤 Avatar Ready</h3>
                        <p>Click connect to begin</p>
                    </div>
                </div>
            </div>

            <div class="chat-panel">
                <h2>💬 Conversation</h2>
                <div id="messages"></div>
                <div id="input-container">
                    <button id="voice-btn" class="btn" onclick="toggleVoice()">🎤</button>
                    <input type="text" id="user-input" placeholder="Type or speak..." />
                    <button id="send-btn" class="btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            let ws;
            let recognition;
            let isListening = false;

            const statusDiv = document.getElementById('status');
            const messagesDiv = document.getElementById('messages');
            const userInput = document.getElementById('user-input');
            const avatarVideo = document.getElementById('avatar-video');
            const avatarAudio = document.getElementById('avatar-audio');
            const placeholder = document.getElementById('placeholder');
            const voiceBtn = document.getElementById('voice-btn');

            // Initialize Speech Recognition
            if ('webkitSpeechRecognition' in window) {
                recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    userInput.value = transcript;
                    sendMessage();
                };

                recognition.onend = () => {
                    isListening = false;
                    voiceBtn.classList.remove('listening');
                    voiceBtn.textContent = '🎤';
                };
            }

            function toggleVoice() {
                if (!recognition) {
                    alert('Speech recognition not supported in this browser');
                    return;
                }

                if (isListening) {
                    recognition.stop();
                    isListening = false;
                    voiceBtn.classList.remove('listening');
                    voiceBtn.textContent = '🎤';
                    statusDiv.textContent = '✅ Ready';
                    statusDiv.className = 'connected';
                } else {
                    recognition.start();
                    isListening = true;
                    voiceBtn.classList.add('listening');
                    voiceBtn.textContent = '⏹️';
                    statusDiv.textContent = '🎤 Listening...';
                    statusDiv.className = 'listening';
                }
            }

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

                    if (data.status === 'initializing') {
                        statusDiv.textContent = '🎬 Initializing Avatar...';
                        statusDiv.className = 'thinking';
                        return;
                    }

                    if (data.status === 'thinking') {
                        statusDiv.textContent = '🤔 Thinking...';
                        statusDiv.className = 'thinking';
                        return;
                    }

                    if (data.error) {
                        alert('Error: ' + data.error);
                        statusDiv.textContent = '✅ Ready';
                        statusDiv.className = 'connected';
                        return;
                    }

                    // Handle welcome
                    if (data.type === 'welcome') {
                        addMessage(data.text, 'avatar');

                        if (data.video_url) {
                            placeholder.style.display = 'none';
                            avatarVideo.src = data.video_url + '?t=' + Date.now();
                            avatarVideo.style.display = 'block';
                            avatarVideo.play();
                        }

                        if (data.audio_url) {
                            avatarAudio.src = data.audio_url + '?t=' + Date.now();
                            avatarAudio.play();
                        }

                        statusDiv.textContent = '✅ Ready';
                        statusDiv.className = 'connected';
                        return;
                    }

                    // Handle response
                    if (data.type === 'response') {
                        let contextInfo = '';
                        if (data.context && data.context.length > 0) {
                            contextInfo = `<div class="context-info">📚 Used ${data.context.length} knowledge source(s)</div>`;
                        }
                        addMessage(data.text + contextInfo, 'avatar');

                        if (data.video_url) {
                            placeholder.style.display = 'none';
                            avatarVideo.src = data.video_url + '?t=' + Date.now();
                            avatarVideo.style.display = 'block';
                            avatarVideo.play();
                        }

                        if (data.audio_url) {
                            avatarAudio.src = data.audio_url + '?t=' + Date.now();
                            avatarAudio.play();
                        }

                        statusDiv.textContent = '✅ Ready';
                        statusDiv.className = 'connected';
                    }
                };
            }

            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'avatar-message');
                messageDiv.innerHTML = (sender === 'user' ? 'You: ' : 'Vinay: ') + text;
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
        "musetalk": engine.musetalk_ready,
        "rag": engine.rag.client is not None,
        "knowledge_count": engine.rag.collection.count() if engine.rag.collection else 0
    }


def main():
    """Start server"""
    print("\n" + "="*60)
    print("🚀 Flashback Avatar - RAG + Voice Server")
    print("="*60)
    print(f"📱 Web interface: http://localhost:8000")
    print(f"🔧 Health check: http://localhost:8000/health")
    print(f"🎭 Avatar: {engine.avatar_name}")
    print(f"📚 RAG: {'Enabled' if CHROMA_AVAILABLE else 'Disabled'}")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
