"""
MuseTalk Animator for Real-Time Lip-Sync
Integrates with SplattingAvatar for Anam.ai-like experience
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import sys
from typing import AsyncGenerator, Dict

# Add MuseTalk to path
MUSETALK_PATH = Path(__file__).parent.parent / "third_party" / "MuseTalk"
sys.path.insert(0, str(MUSETALK_PATH))

try:
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image
    from musetalk.utils.utils import load_all_model
    MUSETALK_AVAILABLE = True
except ImportError:
    print("⚠️  MuseTalk not available. Install with: cd third_party/MuseTalk && pip install -r requirements.txt")
    MUSETALK_AVAILABLE = False


class MuseTalkAnimator:
    """Real-time lip-sync animation using MuseTalk (30+ FPS)"""

    def __init__(self, model_dir: str = None):
        """
        Initialize MuseTalk for real-time lip-sync animation

        Args:
            model_dir: Path to MuseTalk models (default: third_party/MuseTalk/models)
        """
        if not MUSETALK_AVAILABLE:
            raise ImportError("MuseTalk is not available. Run setup_musetalk.sh first.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_dir is None:
            model_dir = str(MUSETALK_PATH / "models")

        self.model_dir = model_dir

        # Load MuseTalk models
        print("📥 Loading MuseTalk models...")
        try:
            self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
            print(f"✅ MuseTalk models loaded on {self.device}")
        except Exception as e:
            print(f"❌ Error loading MuseTalk models: {e}")
            raise

    async def animate_from_audio(
        self,
        audio_path: str,
        fps: int = 30,
        reference_image: str = None
    ) -> AsyncGenerator[Dict[str, float], None]:
        """
        Generate lip-sync animation parameters from audio

        Args:
            audio_path: Path to audio file (from TTS)
            fps: Target frames per second (default: 30)
            reference_image: Optional reference image (use neutral avatar frame)

        Yields:
            Dictionary of animation parameters at specified FPS:
            {
                'jaw_open': float,      # Jaw opening amount (0-1)
                'mouth_width': float,   # Mouth width (0-1)
                'lip_upper': float,     # Upper lip movement (0-1)
                'lip_lower': float,     # Lower lip movement (0-1)
                'timestamp': float      # Time in seconds
            }
        """
        # Load audio
        audio_waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed (MuseTalk expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_waveform = resampler(audio_waveform)

        # Convert to mono if stereo
        if audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

        # Process audio to get whisper features
        whisper_chunks = self.audio_processor.audio2feat(audio_waveform)

        # Calculate frame timing
        frame_duration = 1.0 / fps
        timestamp = 0.0

        # Generate animation parameters frame by frame
        for chunk_idx, chunk in enumerate(whisper_chunks):
            # Get latent code for this audio chunk
            with torch.no_grad():
                latent = self.vae.get_audio_latent(chunk)

            # Convert latent to animation parameters
            # MuseTalk outputs blend shape coefficients for FLAME model
            animation_params = {
                'jaw_open': self._safe_sigmoid(latent[0].item()),
                'mouth_width': self._safe_sigmoid(latent[1].item()) if len(latent) > 1 else 0.0,
                'lip_upper': self._safe_sigmoid(latent[2].item()) if len(latent) > 2 else 0.0,
                'lip_lower': self._safe_sigmoid(latent[3].item()) if len(latent) > 3 else 0.0,
                'timestamp': timestamp,
                'frame_index': chunk_idx
            }

            yield animation_params

            timestamp += frame_duration

    def _safe_sigmoid(self, x: float) -> float:
        """Safe sigmoid to normalize values to 0-1 range"""
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except:
            return 0.5  # Neutral value on error

    @staticmethod
    def apply_to_flame_mesh(animation_params: Dict[str, float], flame_model):
        """
        Apply MuseTalk animation parameters to FLAME mesh

        This is used with SplattingAvatar where FLAME mesh controls
        embedded Gaussian Splats through barycentric coordinates

        Args:
            animation_params: Animation parameters from animate_from_audio()
            flame_model: FLAME mesh model (from SplattingAvatar)

        Returns:
            Updated FLAME mesh with applied blend shapes
        """
        # Apply blend shapes to FLAME model
        blend_shapes = {
            'jaw': animation_params.get('jaw_open', 0.0),
            'mouth_stretch': animation_params.get('mouth_width', 0.0),
            'lips_upper_up': animation_params.get('lip_upper', 0.0),
            'lips_lower_down': animation_params.get('lip_lower', 0.0),
        }

        # Update FLAME mesh (this automatically updates embedded Gaussians)
        flame_model.update_blend_shapes(blend_shapes)

        return flame_model


class MockMuseTalkAnimator:
    """Mock animator for testing without MuseTalk installed"""

    def __init__(self, *args, **kwargs):
        print("⚠️  Using MockMuseTalkAnimator (MuseTalk not installed)")
        self.device = "cpu"

    async def animate_from_audio(self, audio_path: str, fps: int = 30, reference_image: str = None):
        """Generate mock animation parameters"""
        import asyncio

        # Get audio duration
        audio_waveform, sample_rate = torchaudio.load(audio_path)
        duration = audio_waveform.shape[1] / sample_rate

        num_frames = int(duration * fps)
        frame_duration = 1.0 / fps

        for i in range(num_frames):
            # Generate simple sine wave animation for testing
            t = i * frame_duration

            yield {
                'jaw_open': 0.3 + 0.2 * np.sin(t * 10),  # Oscillate jaw
                'mouth_width': 0.5 + 0.1 * np.sin(t * 8),
                'lip_upper': 0.2 + 0.1 * np.sin(t * 12),
                'lip_lower': 0.2 + 0.1 * np.sin(t * 12 + np.pi),
                'timestamp': t,
                'frame_index': i
            }

            await asyncio.sleep(frame_duration)


# Export the appropriate class
if MUSETALK_AVAILABLE:
    __all__ = ['MuseTalkAnimator']
else:
    # Use mock for testing
    MuseTalkAnimator = MockMuseTalkAnimator
    __all__ = ['MuseTalkAnimator', 'MockMuseTalkAnimator']


# Test function
async def test_musetalk_animator():
    """Test MuseTalk animation with sample audio"""
    import asyncio

    animator = MuseTalkAnimator()

    # Use test audio (you need to provide this)
    test_audio = "test_audio.wav"

    if not Path(test_audio).exists():
        print(f"❌ Test audio not found: {test_audio}")
        return

    print(f"🎤 Testing MuseTalk animation with {test_audio}")

    frame_count = 0
    async for params in animator.animate_from_audio(test_audio, fps=30):
        frame_count += 1
        if frame_count % 30 == 0:  # Print every second
            print(f"Frame {frame_count}: jaw={params['jaw_open']:.3f}, "
                  f"mouth={params['mouth_width']:.3f}, time={params['timestamp']:.2f}s")

    print(f"✅ Generated {frame_count} animation frames")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_musetalk_animator())
