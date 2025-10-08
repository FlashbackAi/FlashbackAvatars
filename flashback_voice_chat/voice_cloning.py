"""
Voice Cloning for Vinay using XTTS or Edge-TTS
Clones voice from reference audio and synthesizes speech
"""

import os
import asyncio
from pathlib import Path
import torch
from config import VOICE_SETTINGS, LANGUAGE


class VoiceCloner:
    """Clone and synthesize Vinay's voice"""

    def __init__(self, reference_audio: str = None, use_gpu: bool = True, enable_cache: bool = True):
        """
        Initialize voice cloner

        Args:
            reference_audio: Path to Vinay's voice sample (5-30 seconds)
            use_gpu: Use GPU if available
            enable_cache: Enable KV cache for faster generation
        """
        self.reference_audio = reference_audio
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tts_engine = None
        self.enable_cache = enable_cache

        # Cache speaker embeddings for faster generation
        self.speaker_embedding = None

        # Try to load XTTS, fallback to edge-tts
        self._init_tts()

    def _init_tts(self):
        """Initialize TTS engine (XTTS voice cloning - REQUIRED)"""

        # Validate reference audio exists
        if not self.reference_audio:
            raise ValueError("❌ Reference audio path is required for voice cloning!")

        if not Path(self.reference_audio).exists():
            raise FileNotFoundError(f"❌ Reference audio not found: {self.reference_audio}")

        # Load XTTS (no fallback)
        try:
            from TTS.api import TTS

            print("🎙️ Initializing XTTS voice cloning...")
            os.environ["COQUI_TOS_AGREED"] = "1"

            # NOTE: TTS 0.22.0 has issues loading from custom paths
            # For now, it will use/download to system cache (~2GB download if not cached)
            # Models at flashback_voice_chat/models/ are for deployment via HuggingFace
            print("   Loading XTTS v2 (will download if not in system cache)...")
            self.tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

            # Move to device
            if self.device == "cuda":
                self.tts_engine.to(self.device)
                print(f"✅ XTTS loaded on GPU ({self.device})")
            else:
                print("✅ XTTS loaded on CPU")

            # Pre-compute speaker embeddings for faster generation
            if self.enable_cache and self.reference_audio:
                print("   Pre-computing speaker embeddings (this speeds up generation)...")
                try:
                    # Compute and cache speaker embedding
                    gpt_cond_latent, speaker_embedding = self.tts_engine.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=[self.reference_audio]
                    )
                    self.speaker_embedding = {
                        "gpt_cond_latent": gpt_cond_latent,
                        "speaker_embedding": speaker_embedding
                    }
                    print("   ✅ Speaker embeddings cached!")
                except Exception as e:
                    print(f"   ⚠️  Could not cache embeddings: {e}")
                    self.speaker_embedding = None

            self.mode = "xtts"

        except ImportError:
            raise ImportError(
                "❌ TTS library not installed. Install with: pip install TTS"
            )
        except Exception as e:
            raise RuntimeError(f"❌ XTTS initialization failed: {e}")

    async def synthesize(self, text: str, output_path: str, language: str = "en"):
        """
        Synthesize speech from text using XTTS voice cloning

        Args:
            text: Text to speak
            output_path: Where to save audio file
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko)
                      Note: For Indian English accent, reference audio accent will be cloned

        Returns:
            Path to generated audio file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"   Generating with XTTS (cloning Vinay's voice)...")

        # Use cached embeddings for faster generation (30-50% faster)
        if self.speaker_embedding:
            try:
                # Direct inference with cached embeddings (faster)
                # Using settings from config.py for best accent preservation
                wav = self.tts_engine.synthesizer.tts_model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.speaker_embedding["gpt_cond_latent"],
                    speaker_embedding=self.speaker_embedding["speaker_embedding"],
                    temperature=VOICE_SETTINGS["temperature"],
                    length_penalty=VOICE_SETTINGS["length_penalty"],
                    repetition_penalty=VOICE_SETTINGS["repetition_penalty"],
                    top_k=VOICE_SETTINGS["top_k"],
                    top_p=VOICE_SETTINGS["top_p"],
                    speed=VOICE_SETTINGS["speed"],
                    enable_text_splitting=VOICE_SETTINGS["enable_text_splitting"],
                    do_sample=VOICE_SETTINGS["do_sample"]
                )

                # Save audio
                import torchaudio
                torchaudio.save(
                    str(output_path),
                    torch.tensor(wav["wav"]).unsqueeze(0),
                    24000
                )
                print(f"   ✅ Generated using cached embeddings (faster)")
            except Exception as e:
                print(f"   ⚠️  Cached generation failed, using standard method: {e}")
                # Fallback to standard generation using config settings
                self.tts_engine.tts_to_file(
                    text=text,
                    speaker_wav=self.reference_audio,
                    language=language,
                    file_path=str(output_path),
                    temperature=VOICE_SETTINGS["temperature"],
                    length_penalty=VOICE_SETTINGS["length_penalty"],
                    repetition_penalty=VOICE_SETTINGS["repetition_penalty"],
                    top_k=VOICE_SETTINGS["top_k"],
                    top_p=VOICE_SETTINGS["top_p"],
                    speed=VOICE_SETTINGS["speed"]
                )
        else:
            # Standard generation (slower, computes embeddings each time)
            # Using settings from config.py for best accent preservation
            self.tts_engine.tts_to_file(
                text=text,
                speaker_wav=self.reference_audio,
                language=language,
                file_path=str(output_path),
                temperature=VOICE_SETTINGS["temperature"],
                length_penalty=VOICE_SETTINGS["length_penalty"],
                repetition_penalty=VOICE_SETTINGS["repetition_penalty"],
                top_k=VOICE_SETTINGS["top_k"],
                top_p=VOICE_SETTINGS["top_p"],
                speed=VOICE_SETTINGS["speed"]
            )

        return output_path


# Test function
async def test_voice_cloning():
    """Test voice cloning with sample text"""

    # Option 1: With voice cloning (if you have reference audio)
    reference_audio = "avatar_input/vinay_audio.wav"

    if Path(reference_audio).exists():
        print("Testing with voice cloning...")
        cloner = VoiceCloner(reference_audio=reference_audio)
    else:
        print("No reference audio found, using edge-tts...")
        cloner = VoiceCloner()

    # Synthesize test
    text = "Hi! I'm Vinay Thadem, Co-Founder of Flashback Labs. How can I help you today?"
    output = await cloner.synthesize(text, "test_output.wav")

    print(f"✅ Generated: {output}")


if __name__ == "__main__":
    asyncio.run(test_voice_cloning())
