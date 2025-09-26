import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import librosa

from echomimic_engine import EchoMimicEngine
from streaming_buffer import StreamingBuffer

app = FastAPI(title="Flashback Renderer")

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "pretrained", "Wan2.1-Fun-V1.1-1.3B-InP")
WAV2VEC_DIR = os.path.join(os.path.dirname(__file__), "models", "pretrained", "wav2vec2-base-960h")

# Initialize engine with proper paths
engine = EchoMimicEngine(model_dir=MODEL_DIR, wav2vec_dir=WAV2VEC_DIR)
buffer = StreamingBuffer()


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


# Request/Response models
class RenderRequest(BaseModel):
    video: List[List[List[List[int]]]]  # RGB video as nested list (frames, height, width, channels)
    audio_file: str  # Path to audio file

class RenderResponse(BaseModel):
    frame: str  # Base64 encoded image
    status: str
    message: str

@app.post("/render", response_model=RenderResponse)
async def render_frame(request: RenderRequest):
    """Generate avatar frame using EchoMimic with lazy loading."""
    try:
        print(f"🎬 Received render request")

        # Convert video array to the expected format
        video_array = np.array(request.video, dtype=np.uint8)
        print(f"📸 Video shape: {video_array.shape}")

        # Get first frame as reference image for clip_image
        first_frame = Image.fromarray(video_array[0])

        # Load and process audio
        if not os.path.exists(request.audio_file):
            raise HTTPException(status_code=400, detail=f"Audio file not found: {request.audio_file}")

        print(f"🎵 Loading audio: {request.audio_file}")
        audio_array, sample_rate = librosa.load(request.audio_file, sr=16000)
        print(f"🎵 Audio shape: {audio_array.shape}, Sample rate: {sample_rate}")

        # Generate frame using lazy loading engine
        print(f"🚀 Starting EchoMimic inference...")
        generated_video = engine.generate_video(video_array, first_frame, audio_array)

        # Take first frame from generated video
        if isinstance(generated_video, np.ndarray):
            if len(generated_video.shape) == 4:  # (frames, H, W, C)
                generated_frame = generated_video[0]
            else:  # assume it's already a single frame
                generated_frame = generated_video
            generated_image = Image.fromarray((generated_frame * 255).astype(np.uint8))
        else:
            # If it's already a PIL image or tensor, handle accordingly
            generated_image = generated_video

        # Encode to base64
        buffer = BytesIO()
        generated_image.save(buffer, format="PNG")
        frame_base64 = base64.b64encode(buffer.getvalue()).decode()

        print(f"✅ Frame generated successfully!")

        return RenderResponse(
            frame=frame_base64,
            status="success",
            message="Video frame generated successfully with lazy loading"
        )

    except Exception as e:
        print(f"❌ Error during rendering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")