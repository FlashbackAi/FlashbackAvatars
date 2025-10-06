"""
Vinay's Voice Cloning using Coqui XTTS (Open Source)
Replaces generic edge-tts with cloned voice from vinay_audio.wav
"""

import os
from pathlib import Path
from TTS.api import TTS
import torch

class VinayVoiceCloner:
    def __init__(self, reference_audio_path: str = "avatar_input/vinay_audio.wav"):
        """
        Initialize Coqui XTTS voice cloner with Vinay's reference audio.

        Args:
            reference_audio_path: Path to Vinay's audio sample (minimum 6 seconds recommended)
        """
        self.reference_audio = reference_audio_path

        if not os.path.exists(self.reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")

        print("🔧 Initializing Coqui XTTS v2 for voice cloning...")

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")

        # Initialize XTTS v2 model (multilingual, multi-speaker)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

        print(f"✅ Voice cloner initialized with reference: {self.reference_audio}")

    def clone_voice(self, text: str, output_path: str = "output_audio.wav", language: str = "en") -> str:
        """
        Generate speech using Vinay's cloned voice.

        Args:
            text: Text to synthesize
            output_path: Where to save the audio
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)

        Returns:
            Path to generated audio file
        """
        print(f"🎙️ Generating speech with Vinay's voice...")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate speech with voice cloning
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.reference_audio,
            language=language,
            file_path=str(output_path)
        )

        print(f"✅ Speech generated: {output_path}")
        return str(output_path)

    def clone_voice_streaming(self, text: str, language: str = "en"):
        """
        Generate speech with streaming output (for real-time playback).

        Args:
            text: Text to synthesize
            language: Language code

        Yields:
            Audio chunks as they are generated
        """
        # XTTS v2 supports streaming but requires advanced setup
        # For now, using file-based approach
        # TODO: Implement streaming for lower latency
        output_path = "temp_audio.wav"
        self.clone_voice(text, output_path, language)

        # Read and yield the audio file
        with open(output_path, 'rb') as f:
            yield f.read()


def test_voice_cloning():
    """Test Vinay's voice cloning."""
    print("=" * 60)
    print("Testing Vinay's Voice Cloning with Coqui XTTS")
    print("=" * 60)

    # Initialize cloner
    cloner = VinayVoiceCloner(reference_audio_path="avatar_input/vinay_audio.wav")

    # Test sentences
    test_texts = [
        "Hi! I'm Vinay Thadem, Co-Founder of Flashback Labs.",
        "TEEPIN is a decentralized framework for training and deploying Private AI Avatars.",
        "Our mission is to deliver privacy-first AI for personal memories.",
    ]

    output_dir = Path("test_voice_outputs")
    output_dir.mkdir(exist_ok=True)

    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        output_path = output_dir / f"vinay_test_{i+1}.wav"
        cloner.clone_voice(text, str(output_path))
        print(f"✅ Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("✅ Voice cloning test complete!")
    print(f"📁 Check audio files in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    test_voice_cloning()
