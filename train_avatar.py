#!/usr/bin/env python3
"""
SplattingAvatar Training Wrapper
Integrates with existing FlashbackAvatars pipeline
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import json


class SplattingAvatarTrainer:
    """Wrapper for SplattingAvatar training"""

    def __init__(self, data_dir, output_dir=None, config="head_avatar"):
        """
        Initialize trainer

        Args:
            data_dir: Preprocessed data directory
            output_dir: Output directory for trained model
            config: Config name (head_avatar or full_body)
        """
        self.base_dir = Path(__file__).parent
        self.splatting_dir = self.base_dir / "third_party" / "SplattingAvatar"
        self.data_dir = Path(data_dir)

        if output_dir is None:
            self.output_dir = self.splatting_dir / "output" / self.data_dir.name
        else:
            self.output_dir = Path(output_dir)

        self.config_name = config
        self.config_path = self.splatting_dir / "configs" / f"{config}.yaml"

        # Check if SplattingAvatar exists
        if not self.splatting_dir.exists():
            raise FileNotFoundError(
                f"SplattingAvatar not found at {self.splatting_dir}\n"
                "Run: bash setup_splatting_avatar.sh"
            )

        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Run: python preprocess_avatar_video.py <video_path>"
            )

    def create_config(self):
        """Create or update config file for head avatar"""

        config_content = f"""
# SplattingAvatar Head Avatar Configuration
# For shoulders-up avatar like Anam.ai

model:
  type: SplattingAvatar
  mode: head  # head or full_body
  max_n_gauss: 200000  # Reduced for head-only (vs 300k for full body)
  use_flame: true
  flame_model: data/FLAME2020/generic_model.pkl

training:
  batch_size: 1
  num_iterations: 30000
  learning_rate: 0.0001
  lr_decay: 0.98
  checkpoint_interval: 5000

data:
  data_dir: {self.data_dir.absolute()}
  output_dir: {self.output_dir.absolute()}
  image_scale: 1.0
  use_mask: true

optimization:
  walking_on_triangle: true  # SplattingAvatar's unique feature
  point_cloud_optimization: true
  blend_shape_optimization: true

rendering:
  image_size: [512, 512]
  near: 0.1
  far: 100.0
"""

        # Save config
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            f.write(config_content)

        print(f"✅ Created config: {self.config_path}")

    def train(self, resume=False):
        """Run training"""

        print("🎬 Starting SplattingAvatar Training")
        print("=" * 50)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚙️  Config: {self.config_name}")
        print("=" * 50)

        # Create config if it doesn't exist
        if not self.config_path.exists():
            self.create_config()

        # Change to SplattingAvatar directory
        os.chdir(self.splatting_dir)

        # Build training command
        # Use relative path from splatting_dir
        relative_data_dir = os.path.relpath(self.data_dir.absolute(), self.splatting_dir)

        cmd = [
            sys.executable,
            "train_splatting_avatar.py",
            "--config", str(self.config_path),
            "--dat_dir", relative_data_dir
        ]

        if resume:
            cmd.append("--resume")

        print(f"🚀 Running: {' '.join(cmd)}")
        print()

        # Run training
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ Training complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            return False

    def export_for_web(self):
        """Export trained model for web rendering"""

        print("\n📦 Exporting model for web...")

        # Find latest checkpoint
        checkpoints = list(self.output_dir.glob("*.pth"))
        if not checkpoints:
            print("❌ No checkpoints found. Train the model first.")
            return False

        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"📂 Using checkpoint: {latest_checkpoint}")

        # Export to PLY format
        export_script = self.base_dir / "export_avatar_for_web.py"

        if not export_script.exists():
            self.create_export_script(export_script)

        cmd = [
            sys.executable,
            str(export_script),
            str(latest_checkpoint),
            str(self.output_dir / "vinay_avatar.ply")
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Exported to: {self.output_dir / 'vinay_avatar.ply'}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Export failed: {e}")
            return False

    def create_export_script(self, export_path):
        """Create export script for web"""

        export_code = '''#!/usr/bin/env python3
"""Export SplattingAvatar to PLY for web rendering"""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import sys
from pathlib import Path

def export_to_ply(checkpoint_path, output_path):
    """Export trained model to PLY format"""

    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract Gaussian parameters
    if 'gaussian_params' in checkpoint:
        params = checkpoint['gaussian_params']
    elif 'model_state_dict' in checkpoint:
        params = checkpoint['model_state_dict']
    else:
        params = checkpoint

    # Extract positions, colors, scales, rotations
    positions = params.get('positions', params.get('_xyz')).cpu().numpy()

    # SH coefficients (spherical harmonics for color)
    if 'sh_coeffs' in params:
        sh_coeffs = params['sh_coeffs'].cpu().numpy()
        colors = sh_coeffs[:, :3]  # First 3 channels (RGB)
    elif 'colors' in params:
        colors = params['colors'].cpu().numpy()
    else:
        colors = np.ones((len(positions), 3)) * 0.5

    scales = params.get('scales', params.get('_scaling')).cpu().numpy()
    rotations = params.get('rotations', params.get('_rotation')).cpu().numpy()
    opacities = params.get('opacities', params.get('_opacity')).cpu().numpy()

    print(f"📊 Gaussian count: {len(positions)}")

    # Create PLY structure
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    vertices = np.zeros(len(positions), dtype=vertex_dtype)

    vertices['x'] = positions[:, 0]
    vertices['y'] = positions[:, 1]
    vertices['z'] = positions[:, 2]

    # Normals (compute from nearby points or use default)
    vertices['nx'] = 0
    vertices['ny'] = 0
    vertices['nz'] = 1

    # Colors (convert to 0-1 range)
    vertices['f_dc_0'] = colors[:, 0]
    vertices['f_dc_1'] = colors[:, 1]
    vertices['f_dc_2'] = colors[:, 2]

    # Opacity
    if opacities.ndim == 2:
        opacities = opacities.squeeze()
    vertices['opacity'] = opacities

    # Scales
    vertices['scale_0'] = scales[:, 0]
    vertices['scale_1'] = scales[:, 1]
    vertices['scale_2'] = scales[:, 2]

    # Rotations (quaternions)
    vertices['rot_0'] = rotations[:, 0]
    vertices['rot_1'] = rotations[:, 1]
    vertices['rot_2'] = rotations[:, 2]
    vertices['rot_3'] = rotations[:, 3]

    # Save PLY
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)

    print(f"✅ Exported to: {output_path}")
    print(f"📊 File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_avatar_for_web.py <checkpoint.pth> <output.ply>")
        sys.exit(1)

    export_to_ply(sys.argv[1], sys.argv[2])
'''

        with open(export_path, 'w') as f:
            f.write(export_code)

        os.chmod(export_path, 0o755)
        print(f"✅ Created export script: {export_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train SplattingAvatar")
    parser.add_argument("data_dir", help="Preprocessed data directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--config", "-c", default="head_avatar", help="Config name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--export", action="store_true", help="Export to web format after training")

    args = parser.parse_args()

    print("🎬 SplattingAvatar Training Pipeline")
    print("=" * 50)

    # Initialize trainer
    trainer = SplattingAvatarTrainer(
        data_dir=args.data_dir,
        output_dir=args.output,
        config=args.config
    )

    # Train
    success = trainer.train(resume=args.resume)

    # Export if requested
    if success and args.export:
        trainer.export_for_web()

    if success:
        print("\n📋 Next steps:")
        print("1. Test avatar rendering:")
        print("   python realtime_avatar_server.py")
        print("\n2. Access UI at: http://localhost:8000")


if __name__ == "__main__":
    main()
