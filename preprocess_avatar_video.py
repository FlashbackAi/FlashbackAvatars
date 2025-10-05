#!/usr/bin/env python3
"""
Video Preprocessing for SplattingAvatar
Prepares Vinay's shoulders-up video for training
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
import argparse
from tqdm import tqdm

# Add MediaPipe for face detection and landmarks
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False


class AvatarVideoPreprocessor:
    """Preprocess video for SplattingAvatar training"""

    def __init__(self, video_path, output_dir, target_fps=30, max_frames=None):
        """
        Initialize preprocessor

        Args:
            video_path: Path to input video
            output_dir: Directory to save preprocessed data
            target_fps: Target frames per second (default: 30)
            max_frames: Maximum frames to process (None = all)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.max_frames = max_frames

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # General model
            )
        else:
            self.face_mesh = None
            self.segmenter = None

    def preprocess(self):
        """Run preprocessing pipeline"""
        print(f"🎬 Preprocessing video: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {original_fps}")
        print(f"   Total frames: {total_frames}")

        # Calculate frame skip for target FPS
        frame_skip = int(original_fps / self.target_fps) if original_fps > self.target_fps else 1

        # Limit frames if specified
        if self.max_frames:
            total_frames = min(total_frames, self.max_frames * frame_skip)

        print(f"🎯 Target FPS: {self.target_fps} (processing every {frame_skip} frame)")

        # Process frames
        frame_idx = 0
        output_idx = 0
        landmarks_data = []
        camera_params = []

        pbar = tqdm(total=total_frames // frame_skip, desc="Processing frames")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames to match target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # Process frame
            result = self.process_frame(frame, output_idx)

            if result:
                landmarks_data.append(result['landmarks'])
                camera_params.append(result['camera'])

            output_idx += 1
            pbar.update(1)

            if self.max_frames and output_idx >= self.max_frames:
                break

            frame_idx += 1

        pbar.close()
        cap.release()

        # Save metadata
        metadata = {
            'video_path': str(self.video_path),
            'total_frames': output_idx,
            'fps': self.target_fps,
            'resolution': [width, height],
            'original_fps': original_fps
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save landmarks
        np.save(self.output_dir / "landmarks.npy", np.array(landmarks_data))

        # Save camera parameters
        with open(self.output_dir / "cameras.json", 'w') as f:
            json.dump(camera_params, f, indent=2)

        print(f"✅ Preprocessing complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Processed {output_idx} frames")

        return True

    def process_frame(self, frame, frame_idx):
        """Process a single frame"""
        height, width = frame.shape[:2]

        # Save original image
        image_path = self.images_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(image_path), frame)

        # Extract face landmarks
        landmarks = None
        if self.face_mesh:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]

        # Create mask (shoulders-up region)
        mask = self.create_shoulders_mask(frame)

        # Save mask
        mask_path = self.masks_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(mask_path), mask)

        # Estimate camera parameters (simple projection)
        camera_params = {
            'frame_idx': frame_idx,
            'focal_length': width,  # Approximate
            'principal_point': [width / 2, height / 2],
            'rotation': [0, 0, 0],  # Identity for frontal view
            'translation': [0, 0, 5]  # Approximate distance
        }

        return {
            'landmarks': landmarks if landmarks else [],
            'camera': camera_params
        }

    def create_shoulders_mask(self, frame):
        """Create mask for shoulders-up region"""
        height, width = frame.shape[:2]

        if self.segmenter:
            # Use MediaPipe segmentation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.segmenter.process(rgb_frame)

            # Get segmentation mask
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

            # Remove below shoulders (bottom 20% of image)
            shoulder_cutoff = int(height * 0.8)
            mask[shoulder_cutoff:, :] = 0

            return mask
        else:
            # Simple center-focused mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Create elliptical mask for shoulders-up
            center_x = width // 2
            center_y = height // 3  # Upper third for shoulders-up

            radius_x = width // 3
            radius_y = int(height * 0.6)

            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                       0, 0, 360, 255, -1)

            return mask


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Preprocess video for SplattingAvatar")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        video_path = Path(args.video_path)
        output_dir = Path(__file__).parent / "third_party" / "SplattingAvatar" / "data" / video_path.stem
    else:
        output_dir = Path(args.output)

    print("🚀 Avatar Video Preprocessor")
    print("=" * 50)

    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  MediaPipe not available. Installing...")
        os.system(f"{sys.executable} -m pip install mediapipe")
        print("Please re-run this script after installation.")
        return

    # Run preprocessing
    preprocessor = AvatarVideoPreprocessor(
        video_path=args.video_path,
        output_dir=output_dir,
        target_fps=args.fps,
        max_frames=args.max_frames
    )

    preprocessor.preprocess()

    print("\n📋 Next steps:")
    print(f"1. Train SplattingAvatar:")
    print(f"   cd third_party/SplattingAvatar")
    print(f"   python train_splatting_avatar.py --config configs/head_avatar.yaml --dat_dir data/{Path(args.video_path).stem}")
    print("\n2. This will take 1-2 hours on H200 GPU")


if __name__ == "__main__":
    main()
