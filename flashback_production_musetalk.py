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
try:
    from voice_cloning_vinay import VinayVoiceCloner
    VOICE_CLONING_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Voice cloning unavailable: {e}")
    print("   Will use edge-tts instead")
    VinayVoiceCloner = None
    VOICE_CLONING_AVAILABLE = False

try:
    from avatar_diffusion_pipeline import AvatarDiffusionPipeline
    DIFFUSION_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Diffusion pipeline unavailable: {e}")
    print("   Will skip enhancement")
    AvatarDiffusionPipeline = None
    DIFFUSION_AVAILABLE = False

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
        use_enhancement: bool = False  # Disabled by default - enhance reference video once offline
    ):
        """
        Initialize MuseTalk avatar engine.

        Args:
            reference_audio: Vinay's voice sample for cloning
            musetalk_video: Prepared video for MuseTalk (use pre-enhanced version)
            use_enhancement: Enable real-time enhancement (slow, not recommended)
        """
        print("🚀 Initializing MuseTalk Avatar Engine...")

        self.reference_audio = reference_audio
        self.musetalk_video = musetalk_video
        self.musetalk_dir = Path("third_party/MuseTalk")
        self.use_enhancement = use_enhancement

        # Output directories
        self.audio_dir = Path("static/audio")
        self.video_dir = Path("static/videos")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_voice_cloner()

        # Only load diffusion if explicitly enabled
        if use_enhancement:
            print("⚠️  Real-time enhancement enabled (slow)")
            self._init_diffusion_pipeline()
        else:
            print("ℹ️  Using pre-enhanced reference video (recommended)")
            self.diffusion = None

        print("✅ MuseTalk Avatar Engine ready!")

    def _init_voice_cloner(self):
        """Initialize Vinay's voice cloner with Coqui XTTS."""
        print("🎙️ Initializing voice cloner...")

        if not VOICE_CLONING_AVAILABLE or VinayVoiceCloner is None:
            print("   Voice cloning not available, using edge-tts")
            self.voice_cloner = None
            return

        # Accept Coqui XTTS license automatically (free for non-commercial use)
        os.environ["COQUI_TOS_AGREED"] = "1"

        try:
            self.voice_cloner = VinayVoiceCloner(
                reference_audio_path=self.reference_audio
            )
        except Exception as e:
            print(f"⚠️  Voice cloner error: {e}")
            print("   Falling back to edge-tts")
            self.voice_cloner = None

    def _init_diffusion_pipeline(self):
        """Initialize comprehensive diffusion enhancement pipeline."""
        print("🎨 Initializing diffusion enhancement pipeline...")

        if not DIFFUSION_AVAILABLE or AvatarDiffusionPipeline is None:
            print("   Diffusion pipeline not available, skipping enhancement")
            self.diffusion = None
            return

        try:
            self.diffusion = AvatarDiffusionPipeline(
                upscale_factor=2,
                face_enhancement_strength=0.8,
                background_blur=True,
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
        Animate avatar from audio using MuseTalk.
        Returns video directly (no real-time enhancement).

        Args:
            audio_path: Path to audio file

        Returns:
            Path to generated video
        """
        print(f"🎬 Generating avatar with MuseTalk...")

        # Run MuseTalk to generate video
        video_path = self._run_musetalk(audio_path)

        # Optional: Apply enhancement if enabled (not recommended for production)
        if self.use_enhancement and self.diffusion and video_path.exists():
            print("🎨 Applying real-time enhancement (slow)...")

            enhanced_filename = f"video_{uuid.uuid4().hex}.mp4"
            enhanced_path = self.video_dir / enhanced_filename

            self.diffusion.enhance_video(
                input_video_path=str(video_path),
                output_video_path=str(enhanced_path),
                apply_bg_blur=True,
                show_progress=False  # Don't spam logs
            )

            # Clean up unenhanced video
            video_path.unlink()
            return enhanced_path

        return video_path

    def _run_musetalk(self, audio_path: Path) -> Path:
        """
        Run MuseTalk for avatar animation.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to raw (unenhanced) video
        """
        import tempfile
        import yaml
        import shutil

        print(f"   Running MuseTalk inference...")
        print(f"   Audio: {audio_path.name}")
        print(f"   Video: {self.musetalk_video}")

        # Setup avatar name and paths
        # Use absolute paths for command-line arguments
        avatar_name = "vinay_avatar"
        avatar_cache = self.musetalk_dir / "results" / "v15" / "avatars" / avatar_name
        unet_path = self.musetalk_dir / "models" / "musetalkV15" / "unet.pth"
        unet_config = self.musetalk_dir / "models" / "musetalkV15" / "musetalk.json"

        # Create temporary YAML config file
        temp_config = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.yaml',
            dir=str(self.musetalk_dir / "configs")
        )

        config_data = {
            avatar_name: {
                'preparation': False,
                'bbox_shift': 0,
                'video_path': 'data/video/vinay_small.mp4',
                'audio_clips': {
                    'temp_audio': str(audio_path.absolute())
                }
            }
        }

        yaml.dump(config_data, temp_config)
        temp_config.close()

        # Run MuseTalk with correct command structure (use absolute paths)
        cmd = [
            sys.executable, "-m", "scripts.realtime_inference",
            "--inference_config", temp_config.name,
            "--result_dir", "results",
            "--unet_model_path", str(unet_path.absolute()),
            "--unet_config", str(unet_config.absolute()),
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

        # Cleanup temp config
        try:
            os.unlink(temp_config.name)
        except:
            pass

        if result.returncode != 0:
            print(f"❌ MuseTalk error: {result.stderr}")
            raise RuntimeError(f"MuseTalk failed: {result.stderr}")

        # Find generated video (MuseTalk creates it in results/v15/avatars/vinay_avatar/vid_output/temp_audio.mp4)
        output_video = avatar_cache / "vid_output" / "temp_audio.mp4"

        if not output_video.exists():
            print(f"❌ Output video not found at: {output_video}")
            raise RuntimeError("MuseTalk output video not found")

        # Copy to our video directory with unique name
        output_filename = f"video_{uuid.uuid4().hex}.mp4"
        output_path = self.video_dir / output_filename
        shutil.copy(output_video, output_path)

        print(f"   ✅ MuseTalk generated: {output_path.name}")
        return output_path


class FlashbackAvatarProduction:
    """Main production server: RAG + Voice + MuseTalk + Heavy Diffusion."""

    def __init__(self):
        # Initialize RAG
        self.rag = RAGKnowledgeBase()

        # Initialize MuseTalk avatar engine (no real-time enhancement)
        self.engine = MuseTalkAvatarEngine(
            reference_audio="avatar_input/vinay_audio.wav",
            musetalk_video="third_party/MuseTalk/data/video/vinay_small.mp4",
            use_enhancement=False  # Disabled - use pre-enhanced video instead
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
        # Send welcome message (direct TTS, no LLM processing)
        welcome_text = "Hi! I'm Vinay Thadem, Co-Founder of Flashback Labs."

        print(f"\n💬 Vinay (welcome): {welcome_text}")

        # Generate audio directly
        audio_path = await avatar_system.engine.text_to_speech(welcome_text)
        print(f"🎙️ Audio: {audio_path.name}")

        # Generate avatar video
        video_path = avatar_system.engine.animate_avatar(audio_path)
        print(f"🎬 Video: {video_path.name}")

        # Send welcome response
        await websocket.send_json({
            "type": "welcome",
            "text": welcome_text,
            "audio_url": f"/audio/{audio_path.name}",
            "video_url": f"/videos/{video_path.name}",
            "context_used": False
        })

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
