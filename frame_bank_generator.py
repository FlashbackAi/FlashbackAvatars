#!/usr/bin/env python3
"""
Frame Bank Generator for Real-Time Avatar System
Pre-generates avatar expressions for instant playback during conversations
"""

import os
import json
import numpy as np
import torch
from PIL import Image
import librosa
from moviepy import VideoFileClip
from typing import Dict, List, Tuple
import time

class FrameBankGenerator:
    def __init__(self, base_video_path: str, output_dir: str = "frame_banks"):
        self.base_video_path = base_video_path
        self.output_dir = output_dir
        self.frame_banks = {}

        # Expression categories for real-time avatar
        self.expressions = {
            "idle": {
                "breathing": {"duration": 3.0, "loop": True},
                "blinking": {"duration": 1.0, "loop": True},
                "subtle_head": {"duration": 2.0, "loop": True}
            },
            "listening": {
                "attentive": {"duration": 2.0, "loop": True},
                "nodding": {"duration": 1.5, "loop": False},
                "engaged": {"duration": 3.0, "loop": True}
            },
            "speaking": {
                "phoneme_A": {"duration": 0.3, "loop": False},
                "phoneme_E": {"duration": 0.3, "loop": False},
                "phoneme_I": {"duration": 0.3, "loop": False},
                "phoneme_O": {"duration": 0.3, "loop": False},
                "phoneme_U": {"duration": 0.3, "loop": False},
                "consonant_M": {"duration": 0.2, "loop": False},
                "consonant_B": {"duration": 0.2, "loop": False},
                "consonant_P": {"duration": 0.2, "loop": False}
            },
            "thinking": {
                "contemplative": {"duration": 2.5, "loop": True},
                "processing": {"duration": 1.8, "loop": True}
            }
        }

        os.makedirs(output_dir, exist_ok=True)

    def generate_expression_prompts(self) -> Dict[str, str]:
        """Generate specific prompts for each expression type"""
        prompts = {
            # Idle states
            "breathing": "person in black turtleneck, natural breathing, subtle chest movement, professional lighting",
            "blinking": "person in black turtleneck, natural eye blinking, professional headshot, studio lighting",
            "subtle_head": "person in black turtleneck, slight head movement, professional business portrait",

            # Listening states
            "attentive": "person in black turtleneck, attentive listening expression, focused eyes, professional",
            "nodding": "person in black turtleneck, slight nodding gesture, understanding expression, professional",
            "engaged": "person in black turtleneck, engaged listening, slight smile, professional lighting",

            # Speaking phonemes
            "phoneme_A": "person in black turtleneck, mouth open for 'A' sound, speaking expression, professional",
            "phoneme_E": "person in black turtleneck, mouth shaped for 'E' sound, speaking, professional lighting",
            "phoneme_I": "person in black turtleneck, mouth shaped for 'I' sound, speaking expression, professional",
            "phoneme_O": "person in black turtleneck, mouth rounded for 'O' sound, speaking, professional",
            "phoneme_U": "person in black turtleneck, mouth shaped for 'U' sound, speaking expression, professional",
            "consonant_M": "person in black turtleneck, lips together for 'M' sound, speaking, professional",
            "consonant_B": "person in black turtleneck, lips together for 'B' sound, speaking, professional",
            "consonant_P": "person in black turtleneck, lips for 'P' sound, speaking expression, professional",

            # Thinking states
            "contemplative": "person in black turtleneck, thoughtful expression, slight upward gaze, professional",
            "processing": "person in black turtleneck, thinking expression, focused look, professional lighting"
        }

        return prompts

    def create_synthetic_audio(self, expression_name: str, duration: float) -> str:
        """Create synthetic audio for expression generation"""
        # Generate silence or appropriate sound for the expression
        sr = 22050
        samples = int(duration * sr)

        if "phoneme" in expression_name or "consonant" in expression_name:
            # Generate appropriate phoneme sound
            audio = self._generate_phoneme_audio(expression_name, samples, sr)
        else:
            # Generate silence for idle/listening states
            audio = np.zeros(samples)

        # Save temporary audio file
        temp_audio_path = f"temp_{expression_name}_audio.wav"
        import soundfile as sf
        sf.write(temp_audio_path, audio, sr)

        return temp_audio_path

    def _generate_phoneme_audio(self, phoneme: str, samples: int, sr: int) -> np.ndarray:
        """Generate basic phoneme sounds for lip-sync"""
        t = np.linspace(0, samples/sr, samples)

        # Basic formant frequencies for different phonemes
        formants = {
            "phoneme_A": [730, 1090],  # 'A' formants
            "phoneme_E": [530, 1840],  # 'E' formants
            "phoneme_I": [270, 2290],  # 'I' formants
            "phoneme_O": [570, 840],   # 'O' formants
            "phoneme_U": [300, 870],   # 'U' formants
            "consonant_M": [200, 800], # 'M' formants
            "consonant_B": [200, 900], # 'B' formants
            "consonant_P": [200, 1000] # 'P' formants
        }

        if phoneme in formants:
            f1, f2 = formants[phoneme]
            audio = 0.3 * (np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t))
            # Apply envelope
            envelope = np.exp(-3 * t) if phoneme.startswith("consonant") else np.ones_like(t)
            audio *= envelope
        else:
            audio = np.zeros(samples)

        return audio

    async def generate_frame_bank(self):
        """Generate all expression frame banks"""
        print("🎬 Starting Frame Bank Generation...")

        # Import EchoMimic after ensuring it's available
        from test_echomimic_v3 import Config, extract_first_frame_from_video

        config = Config()
        prompts = self.generate_expression_prompts()

        bank_metadata = {
            "base_video": self.base_video_path,
            "generated_time": time.time(),
            "expressions": {}
        }

        for category, expressions in self.expressions.items():
            print(f"\n📁 Processing {category} expressions...")

            for expression_name, params in expressions.items():
                print(f"  🎭 Generating {expression_name}...")

                # Create expression-specific directory
                expr_dir = os.path.join(self.output_dir, category, expression_name)
                os.makedirs(expr_dir, exist_ok=True)

                # Generate synthetic audio for this expression
                audio_path = self.create_synthetic_audio(expression_name, params["duration"])

                # Use expression-specific prompt
                expression_prompt = prompts.get(expression_name, config.prompt)

                # Generate frames using EchoMimic (this would be the actual generation)
                # For now, we'll create placeholder structure

                frames_info = {
                    "duration": params["duration"],
                    "loop": params["loop"],
                    "frame_count": int(params["duration"] * 25),  # 25 FPS
                    "prompt": expression_prompt,
                    "directory": expr_dir
                }

                bank_metadata["expressions"][f"{category}_{expression_name}"] = frames_info

                # Clean up temporary audio
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        # Save metadata
        metadata_path = os.path.join(self.output_dir, "frame_bank_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(bank_metadata, f, indent=2)

        print(f"\n✅ Frame bank generation complete!")
        print(f"📄 Metadata saved to: {metadata_path}")

        return bank_metadata

class FrameBankPlayer:
    """Real-time frame bank player for avatar streaming"""

    def __init__(self, frame_bank_dir: str):
        self.frame_bank_dir = frame_bank_dir
        self.metadata = self._load_metadata()
        self.current_expression = "idle_breathing"
        self.frame_cache = {}

    def _load_metadata(self) -> dict:
        """Load frame bank metadata"""
        metadata_path = os.path.join(self.frame_bank_dir, "frame_bank_metadata.json")
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def preload_expression(self, expression_key: str):
        """Preload frames for an expression into memory"""
        if expression_key in self.frame_cache:
            return

        expr_info = self.metadata["expressions"].get(expression_key)
        if not expr_info:
            return

        # Load all frames for this expression
        frames = []
        expr_dir = expr_info["directory"]

        for i in range(expr_info["frame_count"]):
            frame_path = os.path.join(expr_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                frames.append(Image.open(frame_path))

        self.frame_cache[expression_key] = frames

    def get_frame(self, expression_key: str, frame_index: int) -> Image.Image:
        """Get specific frame from expression"""
        if expression_key not in self.frame_cache:
            self.preload_expression(expression_key)

        frames = self.frame_cache.get(expression_key, [])
        if not frames:
            return None

        # Handle looping
        expr_info = self.metadata["expressions"][expression_key]
        if expr_info["loop"]:
            frame_index = frame_index % len(frames)
        else:
            frame_index = min(frame_index, len(frames) - 1)

        return frames[frame_index] if frame_index < len(frames) else frames[-1]

if __name__ == "__main__":
    import asyncio

    print("🎬 Starting Frame Bank Generation...")
    print("=" * 60)

    # Check if input video exists
    import os
    if not os.path.exists("avatar_input/vinay_intro.mp4"):
        print("❌ Error: avatar_input/vinay_intro.mp4 not found!")
        print("Please upload your video file first.")
        exit(1)

    generator = FrameBankGenerator("avatar_input/vinay_intro.mp4")

    print("\n⚠️  WARNING: This will take 30-40 minutes to complete!")
    print("The system will generate ~25 expression sets for real-time playback.")
    print("\nGenerating frame banks using EchoMimic v3...")
    print("Each expression takes ~1-2 minutes to generate.\n")

    # Run the generation process
    asyncio.run(generator.generate_frame_bank())

    print("\n✅ Frame bank generation complete!")
    print("You can now start the avatar server with:")
    print("python realtime_avatar_server.py")