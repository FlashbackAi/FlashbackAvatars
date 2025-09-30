#!/usr/bin/env python3
"""
Real-Time Avatar Streaming Server
Handles WebSocket connections, voice processing, LLM integration, and frame streaming
"""

import asyncio
import json
import base64
import io
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image
import torch
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Voice Activity Detection
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("⚠️  webrtcvad not available. Install with: pip install webrtcvad")

# Speech Recognition
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("⚠️  speech_recognition not available. Install with: pip install SpeechRecognition")

# Text-to-Speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")

from frame_bank_generator import FrameBankPlayer
from local_llm_rag import EnhancedLLMProcessor

class AvatarState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"

@dataclass
class ConversationContext:
    user_input: str
    llm_response: str
    timestamp: float
    duration: float

class VoiceActivityDetector:
    """Real-time voice activity detection"""

    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)

        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        else:
            self.vad = None

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Detect if audio frame contains speech"""
        if not VAD_AVAILABLE or self.vad is None:
            # Fallback: simple energy-based detection
            return np.mean(np.abs(audio_frame)) > 0.01

        # Convert to 16-bit PCM
        audio_int16 = (audio_frame * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        try:
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except Exception:
            return False

class LLMProcessor:
    """Handles LLM conversation processing - Legacy fallback"""

    def __init__(self):
        self.conversation_history = []

    async def process_query(self, user_input: str) -> str:
        """Fallback processor when EnhancedLLMProcessor is not available"""
        responses = {
            "hello": "Hello! I'm Vinay Thadem, Co-founder of Flashback Labs. How can I help you today?",
            "how are you": "I'm doing great! Working on exciting AI projects at Flashback Labs.",
            "what can you do": "I can discuss AI, technology, business strategy, and our innovative solutions at Flashback Labs.",
            "who are you": "I'm Vinay Thadem, Co-founder of Flashback Labs. We develop cutting-edge AI solutions.",
            "goodbye": "Great talking with you! Feel free to reach out about AI or Flashback Labs anytime.",
        }

        user_lower = user_input.lower()
        for keyword, response in responses.items():
            if keyword in user_lower:
                return response

        return f"That's an interesting question about '{user_input}'. Let me think about that..."

class TTSProcessor:
    """Text-to-Speech with phoneme timing"""

    def __init__(self):
        if TTS_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Words per minute
            self.engine.setProperty('volume', 0.8)
        else:
            self.engine = None

    def text_to_phonemes(self, text: str) -> List[Dict]:
        """Convert text to phoneme sequence with timing"""
        # Simplified phoneme mapping for demo
        # In production, use proper phoneme analysis library

        phoneme_map = {
            'a': 'phoneme_A', 'e': 'phoneme_E', 'i': 'phoneme_I',
            'o': 'phoneme_O', 'u': 'phoneme_U', 'm': 'consonant_M',
            'b': 'consonant_B', 'p': 'consonant_P'
        }

        phonemes = []
        duration_per_char = 0.15  # 150ms per character average

        for i, char in enumerate(text.lower()):
            if char.isalpha():
                phoneme = phoneme_map.get(char, 'phoneme_A')
                phonemes.append({
                    'phoneme': phoneme,
                    'start_time': i * duration_per_char,
                    'duration': duration_per_char
                })

        return phonemes

    async def synthesize_speech(self, text: str) -> bytes:
        """Generate speech audio"""
        if not TTS_AVAILABLE or not self.engine:
            return b''  # Return empty audio

        # This is a placeholder - in production, generate actual audio file
        return b''  # TTS audio bytes

class RealTimeAvatarServer:
    """Main real-time avatar streaming server"""

    def __init__(self):
        self.app = FastAPI()
        self.frame_player = None
        self.vad = VoiceActivityDetector()

        # Try to use enhanced LLM with RAG, fallback to simple LLM
        try:
            self.llm = EnhancedLLMProcessor()
            print("✅ Enhanced LLM with RAG initialized")
        except Exception as e:
            print(f"⚠️  Enhanced LLM failed, using fallback: {e}")
            self.llm = LLMProcessor()

        self.tts = TTSProcessor()

        # Avatar state management
        self.current_state = AvatarState.IDLE
        self.frame_counter = 0
        self.connected_clients: List[WebSocket] = []

        # Conversation tracking
        self.is_user_speaking = False
        self.last_speech_time = 0
        self.silence_threshold = 2.0  # Seconds of silence before switching to idle

        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def get_homepage():
            return HTMLResponse(self.get_web_interface())

        @self.app.websocket("/avatar")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time communication"""
        await websocket.accept()
        self.connected_clients.append(websocket)

        try:
            # Initialize frame bank player
            if not self.frame_player:
                self.frame_player = FrameBankPlayer("frame_banks")

            # Start streaming avatar frames
            frame_task = asyncio.create_task(self.stream_avatar_frames(websocket))

            # Handle incoming audio/messages
            async for message in websocket.iter_text():
                await self.process_websocket_message(websocket, message)

        except WebSocketDisconnect:
            self.connected_clients.remove(websocket)
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)

    async def stream_avatar_frames(self, websocket: WebSocket):
        """Stream avatar frames based on current state"""
        fps = 25
        frame_interval = 1.0 / fps

        while websocket in self.connected_clients:
            try:
                # Get current expression based on state
                expression_key = self.get_current_expression()

                # Get frame from bank
                if self.frame_player:
                    frame = self.frame_player.get_frame(expression_key, self.frame_counter)

                    if frame:
                        # Convert frame to base64 for streaming
                        frame_data = self.frame_to_base64(frame)

                        await websocket.send_json({
                            "type": "frame",
                            "data": frame_data,
                            "state": self.current_state.value,
                            "frame_index": self.frame_counter
                        })

                self.frame_counter += 1
                await asyncio.sleep(frame_interval)

            except Exception as e:
                print(f"Frame streaming error: {e}")
                break

    async def process_websocket_message(self, websocket: WebSocket, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "audio":
                await self.process_audio_input(websocket, data)
            elif message_type == "speech_complete":
                await self.process_complete_speech(websocket, data)
            elif message_type == "text":
                await self.process_text_input(websocket, data)
            elif message_type == "state_change":
                await self.handle_state_change(data)

        except json.JSONDecodeError:
            await websocket.send_json({"error": "Invalid JSON message"})

    async def process_audio_input(self, websocket: WebSocket, data: Dict):
        """Process real-time audio input for voice activity detection"""
        try:
            # Decode audio data
            audio_b64 = data.get("audio", "")
            audio_bytes = base64.b64decode(audio_b64)

            # Convert to numpy array (assuming 16kHz, 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

            # Voice activity detection
            is_speech = self.vad.is_speech(audio_array)

            if is_speech:
                self.is_user_speaking = True
                self.last_speech_time = time.time()

                if self.current_state != AvatarState.LISTENING:
                    await self.change_avatar_state(AvatarState.LISTENING)

            else:
                # Check for end of speech
                if self.is_user_speaking and time.time() - self.last_speech_time > self.silence_threshold:
                    self.is_user_speaking = False

                    # Process accumulated speech
                    await self.process_speech_end(websocket)

        except Exception as e:
            print(f"Audio processing error: {e}")

    async def process_complete_speech(self, websocket: WebSocket, data: Dict):
        """Process complete speech from client"""
        try:
            # Decode complete speech audio
            audio_b64 = data.get("audio", "")
            duration = data.get("duration", 0)

            if not audio_b64:
                return

            # In a real implementation, convert audio to text here
            # For now, simulate STT result
            await self.change_avatar_state(AvatarState.THINKING)

            # Simulate speech-to-text processing
            await asyncio.sleep(0.5)  # Brief processing delay

            # Mock STT result - replace with actual STT
            user_text = "Hello, how are you doing today?"  # Replace with real STT

            # Generate response
            await self.generate_response(websocket, user_text)

        except Exception as e:
            print(f"Complete speech processing error: {e}")

    async def process_text_input(self, websocket: WebSocket, data: Dict):
        """Process text input from user"""
        user_text = data.get("text", "")

        if user_text.strip():
            await self.generate_response(websocket, user_text)

    async def process_speech_end(self, websocket: WebSocket):
        """Process when user stops speaking"""
        await self.change_avatar_state(AvatarState.THINKING)

        # In a real implementation, convert audio to text here
        # For demo, simulate STT result
        user_text = "Hello, how are you?"

        await self.generate_response(websocket, user_text)

    async def generate_response(self, websocket: WebSocket, user_input: str):
        """Generate LLM response and start speaking"""
        # Get LLM response
        if hasattr(self.llm, 'process_avatar_query'):
            # Enhanced LLM with structured response
            response_data = await self.llm.process_avatar_query(user_input)
            response = response_data["text"]
            emotion = response_data.get("emotion", "neutral")
            duration = response_data.get("estimated_duration", len(response) * 0.15)
        else:
            # Fallback LLM
            response = await self.llm.process_query(user_input)
            emotion = "neutral"
            duration = len(response) * 0.15

        # Change to speaking state
        await self.change_avatar_state(AvatarState.SPEAKING)

        # Convert response to phonemes for lip-sync
        phonemes = self.tts.text_to_phonemes(response)

        # Send response to client
        await websocket.send_json({
            "type": "response",
            "text": response,
            "phonemes": phonemes,
            "duration": duration,
            "emotion": emotion
        })

        # Schedule return to idle state
        asyncio.create_task(self.schedule_idle_return(duration))

    async def schedule_idle_return(self, delay: float):
        """Return to idle state after speaking"""
        await asyncio.sleep(delay)
        await self.change_avatar_state(AvatarState.IDLE)

    async def change_avatar_state(self, new_state: AvatarState):
        """Change avatar state and reset frame counter"""
        if self.current_state != new_state:
            self.current_state = new_state
            self.frame_counter = 0

            # Notify all connected clients
            for client in self.connected_clients:
                try:
                    await client.send_json({
                        "type": "state_change",
                        "state": new_state.value
                    })
                except:
                    pass

    def get_current_expression(self) -> str:
        """Get current expression key based on avatar state"""
        expressions = {
            AvatarState.IDLE: "idle_breathing",
            AvatarState.LISTENING: "listening_attentive",
            AvatarState.THINKING: "thinking_contemplative",
            AvatarState.SPEAKING: "speaking_phoneme_A"  # This would be dynamic based on phonemes
        }

        return expressions.get(self.current_state, "idle_breathing")

    def frame_to_base64(self, frame: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        frame.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def get_web_interface(self) -> str:
        """Return HTML for web interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Avatar Interface</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
                .container { max-width: 1200px; margin: 0 auto; }
                .avatar-section { display: flex; gap: 20px; }
                .video-container { flex: 1; background: black; border-radius: 10px; overflow: hidden; }
                .avatar-video { width: 100%; height: 400px; object-fit: cover; }
                .controls { flex: 1; }
                .chat-area { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .input-area { background: white; padding: 20px; border-radius: 10px; }
                .status { padding: 10px; background: #e0e0e0; border-radius: 5px; margin: 10px 0; }
                button { padding: 10px 20px; margin: 5px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .speaking { background: #ffeb3b; }
                .listening { background: #4caf50; color: white; }
                .thinking { background: #ff9800; color: white; }
                .idle { background: #e0e0e0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Real-Time Avatar Interface</h1>

                <div class="avatar-section">
                    <div class="video-container">
                        <canvas id="avatarCanvas" class="avatar-video" width="512" height="512"></canvas>
                    </div>

                    <div class="controls">
                        <div class="status" id="statusBar">
                            Status: <span id="currentState">Idle</span>
                        </div>

                        <div class="chat-area">
                            <h3>💬 Conversation</h3>
                            <div id="chatHistory" style="height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background: #f9f9f9;">
                                <p><strong>Vinay:</strong> Hello! I'm Vinay Thadem from Flashback Labs. Ready to discuss AI and technology!</p>
                            </div>
                        </div>

                        <div class="input-area">
                            <h3>🎤 Input</h3>
                            <button id="toggleMic">Start Listening</button>
                            <button id="stopMic" disabled>Stop Listening</button>

                            <div style="margin-top: 15px;">
                                <input type="text" id="textInput" placeholder="Or type your message here...">
                                <button onclick="sendTextMessage()">Send</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // WebSocket connection
                const ws = new WebSocket(`ws://${window.location.host}/avatar`);
                const canvas = document.getElementById('avatarCanvas');
                const ctx = canvas.getContext('2d');
                const statusBar = document.getElementById('statusBar');
                const currentState = document.getElementById('currentState');
                const chatHistory = document.getElementById('chatHistory');

                let isRecording = false;
                let mediaRecorder;
                let audioChunks = [];

                // Handle WebSocket messages
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);

                    switch(data.type) {
                        case 'frame':
                            displayFrame(data.data);
                            updateStatus(data.state);
                            break;
                        case 'response':
                            addToChat('Avatar', data.text);
                            break;
                        case 'state_change':
                            updateStatus(data.state);
                            break;
                    }
                };

                // Display avatar frame
                function displayFrame(frameData) {
                    const img = new Image();
                    img.onload = function() {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = 'data:image/jpeg;base64,' + frameData;
                }

                // Update status bar
                function updateStatus(state) {
                    currentState.textContent = state.charAt(0).toUpperCase() + state.slice(1);
                    statusBar.className = 'status ' + state;
                }

                // Add message to chat
                function addToChat(sender, message) {
                    const chatDiv = document.createElement('div');
                    // Replace "Avatar" with "Vinay" in sender name
                    const displaySender = sender === 'Avatar' ? 'Vinay' : sender;
                    chatDiv.innerHTML = `<strong>${displaySender}:</strong> ${message}`;
                    chatHistory.appendChild(chatDiv);
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }

                // Send text message
                function sendTextMessage() {
                    const input = document.getElementById('textInput');
                    const message = input.value.trim();

                    if (message) {
                        addToChat('You', message);
                        ws.send(JSON.stringify({
                            type: 'text',
                            text: message
                        }));
                        input.value = '';
                    }
                }

                // Handle Enter key in text input
                document.getElementById('textInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendTextMessage();
                    }
                });

                // Real-time microphone with Voice Activity Detection
                let mediaStream = null;
                let audioContext = null;
                let processor = null;
                let isListening = false;

                // Auto-start continuous listening on page load
                window.addEventListener('load', function() {
                    setTimeout(startContinuousListening, 1000);
                });

                document.getElementById('toggleMic').onclick = function() {
                    if (!isListening) {
                        startContinuousListening();
                    } else {
                        stopContinuousListening();
                    }
                };

                document.getElementById('stopMic').onclick = function() {
                    stopContinuousListening();
                };

                async function startContinuousListening() {
                    try {
                        // Request microphone permission
                        mediaStream = await navigator.mediaDevices.getUserMedia({
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                sampleRate: 16000
                            }
                        });

                        // Set up Web Audio API for real-time processing
                        audioContext = new (window.AudioContext || window.webkitAudioContext)({
                            sampleRate: 16000
                        });

                        const source = audioContext.createMediaStreamSource(mediaStream);
                        processor = audioContext.createScriptProcessor(4096, 1, 1);

                        let silenceCounter = 0;
                        let speechBuffer = [];
                        let isSpeaking = false;
                        const silenceThreshold = 30; // ~600ms of silence to stop
                        const volumeThreshold = 0.01; // Minimum volume for speech

                        processor.onaudioprocess = function(e) {
                            const inputData = e.inputBuffer.getChannelData(0);

                            // Calculate RMS volume
                            let sum = 0;
                            for (let i = 0; i < inputData.length; i++) {
                                sum += inputData[i] * inputData[i];
                            }
                            const rms = Math.sqrt(sum / inputData.length);

                            // Voice Activity Detection
                            if (rms > volumeThreshold) {
                                // Speech detected
                                if (!isSpeaking) {
                                    isSpeaking = true;
                                    speechBuffer = [];
                                    addToChat('System', '🎤 Listening...');
                                }

                                speechBuffer.push(new Float32Array(inputData));
                                silenceCounter = 0;
                            } else {
                                // Silence detected
                                if (isSpeaking) {
                                    silenceCounter++;

                                    if (silenceCounter > silenceThreshold) {
                                        // End of speech detected
                                        isSpeaking = false;
                                        processSpeech(speechBuffer);
                                        speechBuffer = [];
                                    }
                                }
                            }

                            // Send real-time audio data for VAD
                            if (speechBuffer.length > 0) {
                                sendAudioChunk(inputData);
                            }
                        };

                        source.connect(processor);
                        processor.connect(audioContext.destination);

                        isListening = true;
                        document.getElementById('toggleMic').textContent = 'Stop Listening';
                        document.getElementById('toggleMic').style.backgroundColor = '#dc3545';
                        document.getElementById('stopMic').disabled = false;

                        addToChat('System', '✅ Continuous voice detection started');

                    } catch (err) {
                        console.error('Microphone access denied:', err);
                        addToChat('System', '❌ Microphone permission denied. Please allow microphone access and refresh.');
                    }
                }

                function stopContinuousListening() {
                    if (mediaStream) {
                        mediaStream.getTracks().forEach(track => track.stop());
                    }
                    if (processor) {
                        processor.disconnect();
                    }
                    if (audioContext) {
                        audioContext.close();
                    }

                    isListening = false;
                    document.getElementById('toggleMic').textContent = 'Start Listening';
                    document.getElementById('toggleMic').style.backgroundColor = '#007bff';
                    document.getElementById('stopMic').disabled = true;

                    addToChat('System', '🔇 Voice detection stopped');
                }

                function sendAudioChunk(audioData) {
                    // Convert Float32Array to base64 for transmission
                    const buffer = new ArrayBuffer(audioData.length * 2);
                    const view = new DataView(buffer);

                    for (let i = 0; i < audioData.length; i++) {
                        const sample = Math.max(-1, Math.min(1, audioData[i]));
                        view.setInt16(i * 2, sample * 0x7FFF, true);
                    }

                    const base64 = arrayBufferToBase64(buffer);

                    // Send to WebSocket
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'audio',
                            audio: base64,
                            timestamp: Date.now()
                        }));
                    }
                }

                function processSpeech(speechBuffer) {
                    // Combine all speech chunks
                    let totalLength = 0;
                    speechBuffer.forEach(chunk => totalLength += chunk.length);

                    const combinedAudio = new Float32Array(totalLength);
                    let offset = 0;
                    speechBuffer.forEach(chunk => {
                        combinedAudio.set(chunk, offset);
                        offset += chunk.length;
                    });

                    // Convert to base64 and send complete speech
                    const buffer = new ArrayBuffer(combinedAudio.length * 2);
                    const view = new DataView(buffer);

                    for (let i = 0; i < combinedAudio.length; i++) {
                        const sample = Math.max(-1, Math.min(1, combinedAudio[i]));
                        view.setInt16(i * 2, sample * 0x7FFF, true);
                    }

                    const base64 = arrayBufferToBase64(buffer);

                    // Send complete speech to server
                    ws.send(JSON.stringify({
                        type: 'speech_complete',
                        audio: base64,
                        duration: combinedAudio.length / 16000
                    }));

                    addToChat('You', '[Speech processed - waiting for response...]');
                }

                function arrayBufferToBase64(buffer) {
                    let binary = '';
                    const bytes = new Uint8Array(buffer);
                    for (let i = 0; i < bytes.byteLength; i++) {
                        binary += String.fromCharCode(bytes[i]);
                    }
                    return window.btoa(binary);
                }
            </script>
        </body>
        </html>
        """

def main():
    """Run the real-time avatar server"""
    server = RealTimeAvatarServer()

    print("🚀 Starting Real-Time Avatar Server...")
    print("📱 Web interface will be available at: http://localhost:8000")
    print("🎭 Avatar streaming ready!")

    uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()