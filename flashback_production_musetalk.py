"""
Flashback Avatar - Production Server (MuseTalk + Heavy Diffusion)
No LivePortrait required - uses MuseTalk with comprehensive enhancement pipeline

Features:
- RAG with Flashback knowledge
- Coqui XTTS voice cloning
- MuseTalk avatar generation
- Heavy diffusion pipeline (GFPGAN + Real-ESRGAN + Background blur)
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional
import subprocess

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
from voice_cloning_vinay import VinayVoiceCloner
from avatar_diffusion_pipeline import AvatarDiffusionPipeline

# RAG imports
import chromadb
from chromadb.utils import embedding_functions
import requests


class RAGKnowledgeBase:
    """Vector database for Vinay's knowledge."""

    def __init__(self, db_path: str = "./rag_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        try:
            self.collection = self.client.get_collection(
                name="vinay_knowledge",
                embedding_function=self.embedding_fn
            )
            print(f"✅ Loaded RAG database with {self.collection.count()} documents")
        except:
            print("⚠️  RAG collection not found. Run: python extract_flashback_knowledge.py")
            self.collection = None

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """Search knowledge base for relevant context."""
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []


class MuseTalkAvatarEngine:
    """
    Production avatar engine using MuseTalk + Heavy Enhancement.
    No LivePortrait required.
    """

    def __init__(
        self,
        reference_audio: str = "avatar_input/vinay_audio.wav",
        musetalk_video: str = "third_party/MuseTalk/data/video/vinay_small.mp4",
        upscale_factor: int = 2,
        enhancement_strength: float = 0.8,
        background_blur: bool = True
    ):
        """
        Initialize MuseTalk avatar engine with heavy enhancement.

        Args:
            reference_audio: Vinay's voice sample for cloning
            musetalk_video: Prepared video for MuseTalk
            upscale_factor: Resolution multiplier (2 or 4)
            enhancement_strength: Face enhancement strength (0-1)
            background_blur: Apply professional background blur
        """
        print("🚀 Initializing MuseTalk Avatar Engine with Heavy Enhancement...")

        self.reference_audio = reference_audio
        self.musetalk_video = musetalk_video
        self.musetalk_dir = Path("third_party/MuseTalk")

        # Output directories
        self.audio_dir = Path("static/audio")
        self.video_dir = Path("static/videos")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_voice_cloner()
        self._init_diffusion_pipeline(upscale_factor, enhancement_strength, background_blur)

        print("✅ MuseTalk Avatar Engine ready!")

    def _init_voice_cloner(self):
        """Initialize Vinay's voice cloner with Coqui XTTS."""
        print("🎙️ Initializing voice cloner...")
        try:
            self.voice_cloner = VinayVoiceCloner(
                reference_audio_path=self.reference_audio
            )
        except Exception as e:
            print(f"⚠️  Voice cloner error: {e}")
            print("   Falling back to edge-tts")
            self.voice_cloner = None

    def _init_diffusion_pipeline(
        self,
        upscale_factor: int,
        enhancement_strength: float,
        background_blur: bool
    ):
        """Initialize comprehensive diffusion enhancement pipeline."""
        print("🎨 Initializing diffusion enhancement pipeline...")
        try:
            self.diffusion = AvatarDiffusionPipeline(
                upscale_factor=upscale_factor,
                face_enhancement_strength=enhancement_strength,
                background_blur=background_blur,
                use_gpu=True
            )
        except Exception as e:
            print(f"⚠️  Diffusion pipeline error: {e}")
            print("   Continuing without enhancement")
            self.diffusion = None

    async def text_to_speech(self, text: str) -> Path:
        """
        Convert text to speech using Vinay's cloned voice.

        Args:
            text: Text to synthesize

        Returns:
            Path to generated audio file
        """
        audio_filename = f"audio_{uuid.uuid4().hex}.wav"
        audio_path = self.audio_dir / audio_filename

        if self.voice_cloner:
            # Use Coqui XTTS with Vinay's voice
            self.voice_cloner.clone_voice(
                text=text,
                output_path=str(audio_path),
                language="en"
            )
        else:
            # Fallback to edge-tts
            import edge_tts
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            await communicate.save(str(audio_path))

        return audio_path

    def animate_avatar(self, audio_path: Path) -> Path:
        """
        Animate avatar from audio using MuseTalk + Heavy Enhancement.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to enhanced video
        """
        print(f"🎬 Generating avatar with MuseTalk...")

        # Step 1: Run MuseTalk to generate raw video
        raw_video_path = self._run_musetalk(audio_path)

        # Step 2: Apply heavy diffusion enhancement
        if self.diffusion and raw_video_path.exists():
            print("🎨 Applying diffusion enhancement...")

            enhanced_filename = f"video_{uuid.uuid4().hex}.mp4"
            enhanced_path = self.video_dir / enhanced_filename

            self.diffusion.enhance_video(
                input_video_path=str(raw_video_path),
                output_video_path=str(enhanced_path),
                apply_bg_blur=True,
                show_progress=True
            )

            # Clean up raw video
            raw_video_path.unlink()

            return enhanced_path
        else:
            # Return raw video if enhancement fails
            if not raw_video_path.exists():
                raise RuntimeError("MuseTalk failed to generate video")
            return raw_video_path

    def _run_musetalk(self, audio_path: Path) -> Path:
        """
        Run MuseTalk for avatar animation.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to raw (unenhanced) video
        """
        # Create unique output filename
        output_filename = f"raw_{uuid.uuid4().hex}.mp4"
        output_path = self.video_dir / output_filename

        print(f"   Running MuseTalk inference...")
        print(f"   Audio: {audio_path.name}")
        print(f"   Video: {self.musetalk_video}")

        # Run MuseTalk
        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--audio_path", str(audio_path.absolute()),
            "--video_path", str(Path(self.musetalk_video).absolute()),
            "--bbox_shift", "0",
            "--result_dir", str(self.video_dir.absolute())
        ]

        result = subprocess.run(
            cmd,
            cwd=self.musetalk_dir,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"❌ MuseTalk error: {result.stderr}")
            raise RuntimeError(f"MuseTalk failed: {result.stderr}")

        # Find the generated video (MuseTalk creates output with timestamp)
        result_files = sorted(
            self.video_dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Get the most recent file (excluding our target output)
        for candidate in result_files:
            if candidate != output_path and not candidate.name.startswith("raw_"):
                candidate.rename(output_path)
                print(f"   ✅ MuseTalk generated: {output_path.name}")
                return output_path

        raise RuntimeError("MuseTalk output video not found")


class FlashbackAvatarProduction:
    """Main production server: RAG + Voice + MuseTalk + Heavy Diffusion."""

    def __init__(self):
        # Initialize RAG
        self.rag = RAGKnowledgeBase()

        # Initialize MuseTalk avatar engine with heavy enhancement
        self.engine = MuseTalkAvatarEngine(
            reference_audio="avatar_input/vinay_audio.wav",
            musetalk_video="third_party/MuseTalk/data/video/vinay_small.mp4",
            upscale_factor=2,  # 2x resolution boost (512 → 1024)
            enhancement_strength=0.8,  # Strong face enhancement
            background_blur=True  # Professional background blur
        )

        # LLM endpoint
        self.llm_url = "http://localhost:11434/api/generate"

    async def generate_rag_response(self, user_message: str) -> tuple[str, List[str]]:
        """Generate LLM response with RAG context."""

        # Search knowledge base
        context_docs = self.rag.search(user_message, n_results=3)

        # System prompt
        system_prompt = """You are Vinay Thadem, Co-Founder of Flashback Labs.
You are helpful, friendly, and speak naturally about your work on private AI avatars and TEEPIN.
Keep responses under 2-3 sentences unless asked for more detail.
Respond as yourself, not as an AI or avatar."""

        # Build prompt with context
        if context_docs:
            context_text = "\n".join([f"- {doc}" for doc in context_docs])
            full_prompt = f"""{system_prompt}

Context information:
{context_text}

User: {user_message}
Vinay:"""
        else:
            full_prompt = f"""{system_prompt}

User: {user_message}
Vinay:"""

        # Call Ollama LLM
        response = requests.post(
            self.llm_url,
            json={
                "model": "llama3.2:3b",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            llm_response = response.json()["response"].strip()
            return llm_response, context_docs
        else:
            raise RuntimeError(f"LLM error: {response.status_code}")

    async def process_message(self, message: str) -> dict:
        """Process user message and generate avatar response."""

        print(f"\n💬 User: {message}")

        # Generate response with RAG
        llm_response, context_used = await self.generate_rag_response(message)
        print(f"🤖 Vinay: {llm_response}")

        # Generate audio with Vinay's voice
        audio_path = await self.engine.text_to_speech(llm_response)
        print(f"🎙️ Audio: {audio_path.name}")

        # Generate avatar video with MuseTalk + Heavy Enhancement
        video_path = self.engine.animate_avatar(audio_path)
        print(f"🎬 Video (enhanced): {video_path.name}")

        return {
            "type": "response",
            "text": llm_response,
            "audio_url": f"/audio/{audio_path.name}",
            "video_url": f"/videos/{video_path.name}",
            "context_used": len(context_used) > 0
        }


# FastAPI app
app = FastAPI(title="Flashback Avatar Production (MuseTalk)")

# Create static directories if they don't exist
Path("static/audio").mkdir(parents=True, exist_ok=True)
Path("static/videos").mkdir(parents=True, exist_ok=True)

# Mount static directories
app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")
app.mount("/videos", StaticFiles(directory="static/videos"), name="videos")

# Initialize avatar system
avatar_system = None


@app.on_event("startup")
async def startup():
    global avatar_system
    print("=" * 70)
    print("Flashback Avatar - Production Server (MuseTalk + Heavy Diffusion)")
    print("=" * 70)
    avatar_system = FlashbackAvatarProduction()
    print("=" * 70)
    print("✅ Server ready!")
    print("=" * 70)


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the web UI."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return "<h1>Flashback Avatar Production (MuseTalk)</h1><p>UI coming soon</p>"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time avatar interaction."""
    await websocket.accept()

    try:
        # Send welcome message
        welcome_text = "Hi! I'm Vinay Thadem, Co-Founder of Flashback Labs. How can I help you today?"
        welcome_response = await avatar_system.process_message(welcome_text)
        welcome_response["type"] = "welcome"
        await websocket.send_json(welcome_response)

        # Handle user messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            if not user_message:
                continue

            # Process message
            response = await avatar_system.process_message(user_message)
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


if __name__ == "__main__":
    print("Starting Flashback Avatar Production Server (MuseTalk + Heavy Diffusion)...")
    print("Access at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
