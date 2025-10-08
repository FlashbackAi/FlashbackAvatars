"""
Flashback Voice Chat Server
WhatsApp-style voice interaction with Vinay
"""

import asyncio
import json
import uuid
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

from voice_cloning import VoiceCloner


class VoiceChatServer:
    """Voice-only chat server with API integration"""

    def __init__(
        self,
        api_url: str = "http://13.234.246.123:8188/answer/generate",
        reference_audio: str = None
    ):
        """
        Initialize voice chat server

        Args:
            api_url: RAG API endpoint
            reference_audio: Path to Vinay's voice sample for cloning
        """
        print("🚀 Initializing Flashback Voice Chat Server...")

        self.api_url = api_url

        # Setup directories
        self.audio_dir = Path("static/audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        # Initialize voice cloner
        print("🎙️ Loading voice synthesis...")
        self.voice_cloner = VoiceCloner(reference_audio=reference_audio)

        print("✅ Voice Chat Server ready!")

    async def query_api(self, question: str) -> dict:
        """
        Query the RAG API

        Args:
            question: User's question

        Returns:
            API response dict
        """
        try:
            response = requests.post(
                self.api_url,
                json={"q": question},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "query": question,
                    "answer": "I'm having trouble processing that. Could you try again?",
                    "confidence": 0.0
                }

        except Exception as e:
            print(f"❌ API error: {e}")
            return {
                "query": question,
                "answer": "I'm currently experiencing technical difficulties. Please try again.",
                "confidence": 0.0
            }

    async def process_message(self, message: str) -> dict:
        """
        Process user message and generate voice response

        Args:
            message: User's text message

        Returns:
            Response dict with text and audio
        """
        print(f"💬 User: {message}")

        # Query API
        api_response = await self.query_api(message)
        answer_text = api_response.get("answer", "I didn't understand that.")

        print(f"🤖 Vinay: {answer_text}")

        # Generate voice
        audio_filename = f"audio_{uuid.uuid4().hex}.wav"
        audio_path = self.audio_dir / audio_filename

        await self.voice_cloner.synthesize(
            text=answer_text,
            output_path=str(audio_path)
        )

        print(f"🎙️ Audio: {audio_filename}")

        return {
            "type": "response",
            "text": answer_text,
            "audio_url": f"/audio/{audio_filename}",
            "confidence": api_response.get("confidence", 1.0),
            "model": api_response.get("model", {}),
            "citations": api_response.get("citations", [])
        }


# FastAPI app
app = FastAPI(title="Flashback Voice Chat")

# CORS middleware for external website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directories
Path("static/audio").mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")

# Initialize server
voice_server = None


@app.on_event("startup")
async def startup():
    global voice_server

    print("=" * 70)
    print("Flashback Voice Chat Server")
    print("=" * 70)

    # REQUIRED: Reference audio for voice cloning
    reference_audio = "../avatar_input/vinay_audio.wav"

    if not Path(reference_audio).exists():
        print(f"❌ FATAL: Reference audio not found: {reference_audio}")
        print("   Voice cloning REQUIRES Vinay's audio sample!")
        print("   Please provide audio file at: avatar_input/vinay_audio.wav")
        raise FileNotFoundError(f"Reference audio required: {reference_audio}")

    voice_server = VoiceChatServer(
        api_url="http://13.234.246.123:8188/answer/generate",
        reference_audio=reference_audio
    )

    print("=" * 70)
    print("✅ Server ready!")
    print("=" * 70)


@app.get("/")
async def get_ui():
    """Serve the web UI"""
    html_path = Path("static/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return {"error": "UI not found"}


@app.post("/api/text-message")
async def text_message(data: dict):
    """
    Handle text message from user (keyboard input)

    POST body: {"message": "your text here"}
    """
    message = data.get("message", "")
    if not message:
        return {"error": "No message provided"}

    response = await voice_server.process_message(message)
    return response


@app.post("/api/voice-message")
async def voice_message(audio: UploadFile = File(...)):
    """
    Handle voice message from user (speech-to-text then process)

    POST with audio file (for external website integration)

    Returns:
        {
            "text": "response text",
            "audio_url": "/audio/filename.wav",
            "confidence": 1.0
        }
    """
    import tempfile
    import os

    # Save uploaded audio temporarily
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    content = await audio.read()
    temp_audio.write(content)
    temp_audio.close()

    try:
        # Convert webm to wav using pydub
        from pydub import AudioSegment
        print("🎤 Converting audio...")

        # Load webm and convert to wav
        audio_segment = AudioSegment.from_file(temp_audio.name, format="webm")
        wav_path = temp_audio.name.replace(".webm", ".wav")
        audio_segment.export(wav_path, format="wav")

        # Speech-to-text with Whisper
        print("🎤 Transcribing audio...")

        # Import whisper here after cleanup
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path)
        user_text = result["text"]

        print(f"   Transcribed: {user_text}")

        # Process message
        response = await voice_server.process_message(user_text)

        # Add transcribed text to response
        response["transcribed_text"] = user_text

        # Cleanup temp files
        os.unlink(temp_audio.name)
        os.unlink(wav_path)

        return response

    except Exception as e:
        print(f"❌ Voice processing error: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        try:
            os.unlink(temp_audio.name)
            if 'wav_path' in locals():
                os.unlink(wav_path)
        except:
            pass

        return {
            "error": str(e),
            "text": "",
            "audio_url": ""
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice chat"""
    await websocket.accept()

    try:
        print("🔌 Client connected")

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Vinay"
        })

        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            message_type = message_data.get("type", "text")
            message_content = message_data.get("message", "")

            if not message_content:
                continue

            # Process message
            response = await voice_server.process_message(message_content)

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "flashback-voice-chat"
    }


if __name__ == "__main__":
    print("Starting Flashback Voice Chat Server...")
    print("Access at: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
