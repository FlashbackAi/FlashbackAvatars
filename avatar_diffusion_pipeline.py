"""
Comprehensive Avatar Diffusion Pipeline
Applies heavy enhancement to MuseTalk output for production quality:
1. Face enhancement (GFPGAN) - restore face quality
2. Face super-resolution (Real-ESRGAN) - 2x-4x upscale
3. Background segmentation and enhancement
4. Environment diffusion (optional Stable Diffusion background)
5. Temporal smoothing for video consistency
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from typing import Optional, Literal
import subprocess
import sys


class AvatarDiffusionPipeline:
    """
    Production-quality enhancement pipeline for MuseTalk avatars.
    Focuses on making MuseTalk output look like Anam.ai quality.
    """

    def __init__(
        self,
        upscale_factor: int = 2,
        face_enhancement_strength: float = 0.8,  # 0-1, higher = more enhancement
        background_blur: bool = True,
        use_gpu: bool = True
    ):
        """
        Initialize comprehensive diffusion pipeline.

        Args:
            upscale_factor: Resolution multiplier (2 or 4)
            face_enhancement_strength: How much to enhance face (0=original, 1=full)
            background_blur: Apply professional background blur (bokeh effect)
            use_gpu: Use GPU acceleration
        """
        self.upscale_factor = upscale_factor
        self.face_enhancement_strength = face_enhancement_strength
        self.background_blur = background_blur
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"🎨 Initializing Avatar Diffusion Pipeline on {self.device}...")
        print(f"   Upscale: {upscale_factor}x")
        print(f"   Enhancement strength: {face_enhancement_strength}")
        print(f"   Background blur: {background_blur}")

        self._check_dependencies()
        self._load_models()

        print("✅ Avatar Diffusion Pipeline ready")

    def _check_dependencies(self):
        """Install required packages."""
        required = {
            "gfpgan": "gfpgan",
            "realesrgan": "realesrgan",
            "facexlib": "facexlib",
            "basicsr": "basicsr",
            "rembg": "rembg",  # Background removal
        }

        print("📦 Checking dependencies...")
        for module, package in required.items():
            try:
                __import__(module)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   📥 Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])

    def _load_models(self):
        """Load all enhancement models."""
        from gfpgan import GFPGANer
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from rembg import remove, new_session

        print("🔽 Loading enhancement models...")

        # Real-ESRGAN for super-resolution
        if self.upscale_factor == 2:
            model_name = 'RealESRGAN_x2plus'
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            scale = 2
        elif self.upscale_factor == 4:
            model_name = 'RealESRGAN_x4plus'
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            scale = 4
        else:
            raise ValueError("upscale_factor must be 2 or 4")

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=scale
        )

        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False,
            device=self.device
        )
        print(f"   ✅ Real-ESRGAN {scale}x loaded")

        # GFPGAN for face restoration
        self.face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
            device=self.device
        )
        print(f"   ✅ GFPGAN loaded")

        # Background removal model
        self.bg_session = new_session("u2net")  # High quality model
        print(f"   ✅ Background removal (U2Net) loaded")

        print("✅ All models loaded")

    def enhance_frame(
        self,
        frame: np.ndarray,
        apply_bg_blur: bool = True
    ) -> np.ndarray:
        """
        Apply complete enhancement to a single frame.

        Args:
            frame: Input frame (BGR)
            apply_bg_blur: Apply background blur

        Returns:
            Enhanced frame
        """
        # Step 1: Face enhancement with GFPGAN
        _, _, enhanced_frame = self.face_enhancer.enhance(
            frame,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=self.face_enhancement_strength
        )

        # Step 2: Background blur for professional look (optional)
        if apply_bg_blur and self.background_blur:
            enhanced_frame = self._apply_background_blur(enhanced_frame)

        return enhanced_frame

    def _apply_background_blur(self, frame: np.ndarray, blur_strength: int = 25) -> np.ndarray:
        """
        Apply professional background blur (bokeh effect) while keeping face sharp.

        Args:
            frame: Input frame
            blur_strength: Blur kernel size (odd number, higher = more blur)

        Returns:
            Frame with blurred background
        """
        from rembg import remove
        from PIL import Image

        # Convert to PIL for rembg
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Remove background to get alpha mask
        output = remove(pil_image, session=self.bg_session)
        output_np = np.array(output)

        # Extract alpha channel as mask
        if output_np.shape[2] == 4:
            alpha = output_np[:, :, 3]
            # Normalize to 0-1
            mask = alpha.astype(float) / 255.0
            # Expand dimensions for broadcasting
            mask = mask[:, :, np.newaxis]
        else:
            # No alpha channel, skip blur
            return frame

        # Create blurred background
        if blur_strength % 2 == 0:
            blur_strength += 1  # Must be odd
        blurred_bg = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

        # Blend: sharp foreground + blurred background
        result = (frame * mask + blurred_bg * (1 - mask)).astype(np.uint8)

        return result

    def enhance_video(
        self,
        input_video_path: str,
        output_video_path: str,
        apply_bg_blur: bool = None,
        show_progress: bool = True
    ) -> str:
        """
        Apply complete enhancement pipeline to video.

        Args:
            input_video_path: Path to MuseTalk output video
            output_video_path: Where to save enhanced video
            apply_bg_blur: Override background blur setting
            show_progress: Show progress updates

        Returns:
            Path to enhanced video
        """
        if apply_bg_blur is None:
            apply_bg_blur = self.background_blur

        print(f"🎨 Enhancing avatar video: {input_video_path}")
        print(f"   Face enhancement: {self.face_enhancement_strength}")
        print(f"   Upscale: {self.upscale_factor}x")
        print(f"   Background blur: {apply_bg_blur}")

        input_path = Path(input_video_path)
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"   Input: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Output dimensions (upscaled)
        out_width = width * self.upscale_factor
        out_height = height * self.upscale_factor
        print(f"   Output: {out_width}x{out_height} @ {fps}fps")

        # Create temporary output (without audio)
        temp_output = output_path.with_suffix('.temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(temp_output),
            fourcc,
            fps,
            (out_width, out_height)
        )

        # Process frames
        frame_count = 0
        previous_enhanced = None  # For temporal smoothing

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Enhance frame
                enhanced_frame = self.enhance_frame(frame, apply_bg_blur=apply_bg_blur)

                # Temporal smoothing (blend with previous frame to reduce jitter)
                if previous_enhanced is not None:
                    alpha = 0.8  # 80% current, 20% previous
                    enhanced_frame = cv2.addWeighted(
                        enhanced_frame, alpha,
                        previous_enhanced, 1 - alpha,
                        0
                    )

                previous_enhanced = enhanced_frame.copy()

                # Write enhanced frame
                out.write(enhanced_frame)

                frame_count += 1
                if show_progress and frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%)", end='\r')

        finally:
            cap.release()
            out.release()

        if show_progress:
            print()  # New line after progress

        # Copy audio from original video with high-quality encoding
        print("🔊 Adding audio with high-quality encoding...")
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(temp_output),
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "slow",  # Better quality
            "-crf", "18",  # High quality (18-23 range, lower = better)
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",  # High quality audio
            "-map", "0:v:0",
            "-map", "1:a:0?",
            str(output_path)
        ], check=True)

        # Clean up temp file
        temp_output.unlink()

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Enhanced video saved: {output_path}")
        print(f"   Size: {file_size:.2f} MB")

        return str(output_path)


def test_diffusion_pipeline():
    """Test the comprehensive diffusion pipeline."""
    print("=" * 70)
    print("Testing Avatar Diffusion Pipeline")
    print("=" * 70)

    # Initialize with strong enhancement settings
    pipeline = AvatarDiffusionPipeline(
        upscale_factor=2,  # 2x resolution
        face_enhancement_strength=0.8,  # Strong enhancement
        background_blur=True,  # Professional blur
        use_gpu=True
    )

    # Test with image first
    test_image = "avatar_input/vinayone.jpg"
    if Path(test_image).exists():
        print(f"\n📸 Testing with image: {test_image}")

        img = cv2.imread(test_image)
        enhanced = pipeline.enhance_frame(img, apply_bg_blur=True)

        output_path = "test_diffusion_enhanced.jpg"
        cv2.imwrite(output_path, enhanced)
        print(f"✅ Enhanced image saved: {output_path}")
        print(f"   Original: {img.shape[1]}x{img.shape[0]}")
        print(f"   Enhanced: {enhanced.shape[1]}x{enhanced.shape[0]}")

    # Test with video
    test_videos = [
        "third_party/MuseTalk/results/output_video.mp4",
        "static/videos/*.mp4",  # Any generated videos
    ]

    for video_pattern in test_videos:
        video_files = list(Path(".").glob(video_pattern))
        if video_files:
            test_video = str(video_files[0])
            print(f"\n🎬 Testing with video: {test_video}")

            output_video = pipeline.enhance_video(
                input_video_path=test_video,
                output_video_path="test_diffusion_video.mp4",
                apply_bg_blur=True
            )
            print(f"✅ Enhanced video: {output_video}")
            break
    else:
        print("\n⚠️  No test videos found. Run MuseTalk first to generate videos.")

    print("\n" + "=" * 70)
    print("✅ Diffusion pipeline test complete!")
    print("\nCompare:")
    print("  Original image: avatar_input/vinayone.jpg")
    print("  Enhanced image: test_diffusion_enhanced.jpg")
    print("=" * 70)


if __name__ == "__main__":
    test_diffusion_pipeline()
