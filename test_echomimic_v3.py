#!/usr/bin/env python3
"""
Simple test script for EchoMimic v3 with custom video/audio inputs
Uses vinay_intro.mp4 and vinay_audio.wav from avatar_input folder
"""

import os
import sys
import datetime
import math
import cv2
import numpy as np
import torch
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from omegaconf import OmegaConf

# Add third_party path for imports
sys.path.append('third_party/echomimic_v3')

from diffusers import FlowMatchEulerDiscreteScheduler
from src.dist import set_multi_gpus_devices
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs, get_image_to_video_latent3, save_videos_grid
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.face_detect import get_mask_coord


class Config:
    def __init__(self):
        # Paths
        self.config_path = "third_party/echomimic_v3/config/config.yaml"
        self.model_name = "third_party/echomimic_v3/models/Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = "third_party/echomimic_v3/models/transformer/diffusion_pytorch_model.safetensors"
        self.wav2vec_model_dir = "third_party/echomimic_v3/models/wav2vec2-base-960h"

        # Input files (your video and audio)
        self.input_video = "avatar_input/vinay_intro.mp4"
        self.input_audio = "avatar_input/vinay_audio.wav"

        # Output
        self.save_path = "outputs"

        # Generation settings (optimized for fast testing)
        self.num_inference_steps = 8  # Reduced for speed
        self.partial_video_length = 25  # Short test (1 second at 25fps)
        self.guidance_scale = 4.0
        self.audio_guidance_scale = 2.5
        self.seed = 42

        # Professional styling prompt - black turtle neck in robotic lab
        self.prompt = "professional speaker wearing black turtle neck sweater, standing in modern robotic laboratory background, high-tech lab environment with robotic equipment, professional studio lighting, clean cinematic portrait, sharp focus, premium quality"
        self.negative_prompt = "blurry, low quality, distorted, amateur, bad lighting, casual clothing, messy background, unprofessional, poor posture, bad composition"

        # Model settings
        self.weight_dtype = torch.bfloat16
        self.sample_size = [512, 512]  # Smaller for speed
        self.fps = 25

        # Pipeline settings
        self.sampler_name = "Flow"
        self.audio_scale = 1.0
        self.shift = 5.0
        self.use_dynamic_cfg = True
        self.use_dynamic_acfg = True


def extract_first_frame_from_video(video_path):
    """Extract the first frame from video as reference image."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError(f"Could not read first frame from {video_path}")


def load_wav2vec_models(wav2vec_model_dir):
    """Load Wav2Vec models for audio feature extraction."""
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    model = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).eval().to("cuda")
    model.requires_grad_(False)
    return processor, model


def extract_audio_features(audio_path, processor, model):
    """Extract audio features using Wav2Vec."""
    sr = 16000
    audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
    input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values = input_values.to(model.device)
    features = model(input_values).last_hidden_state
    return features.squeeze(0)


def get_sample_size(image, default_size):
    """Calculate the sample size based on the input image dimensions."""
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]

    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16

    return int(height), int(width)


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    mask = mask.reshape(-1)
    return mask.float()


def main():
    print("🎬 Starting EchoMimic v3 test with your video/audio...")

    config = Config()

    # Check input files exist
    if not os.path.exists(config.input_video):
        print(f"❌ Video file not found: {config.input_video}")
        return

    if not os.path.exists(config.input_audio):
        print(f"❌ Audio file not found: {config.input_audio}")
        return

    print(f"✅ Found video: {config.input_video}")
    print(f"✅ Found audio: {config.input_audio}")

    # Set up device
    device = set_multi_gpus_devices(1, 1)
    print(f"🖥️ Using device: {device}")

    # Load configuration
    cfg = OmegaConf.load(config.config_path)

    # Load models
    print("📥 Loading EchoMimic v3 models...")

    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        os.path.join(config.model_name, cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        torch_dtype=config.weight_dtype,
    )

    if config.transformer_path and os.path.exists(config.transformer_path):
        from safetensors.torch import load_file
        state_dict = load_file(config.transformer_path)
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        print(f"Loaded transformer: {len(missing)} missing, {len(unexpected)} unexpected keys")

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
    ).to(config.weight_dtype)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        torch_dtype=config.weight_dtype,
    ).eval()

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(config.weight_dtype).eval()

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs'])))

    # Create pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    pipeline.to(device=device)

    # Load Wav2Vec models
    print("📥 Loading Wav2Vec models...")
    wav2vec_processor, wav2vec_model = load_wav2vec_models(config.wav2vec_model_dir)

    # Extract reference image from video
    print("🖼️ Extracting reference image from video...")
    ref_img = extract_first_frame_from_video(config.input_video)

    # Save reference image temporarily for face detection
    temp_image_path = "temp_reference_frame.jpg"
    ref_img.save(temp_image_path)

    # Get face mask coordinates from the extracted frame
    y1, y2, x1, x2, h_, w_ = get_mask_coord(temp_image_path)

    # Extract audio features
    print("🎵 Processing audio...")
    audio_features = extract_audio_features(config.input_audio, wav2vec_processor, wav2vec_model)
    audio_embeds = audio_features.unsqueeze(0).to(device=device, dtype=config.weight_dtype)

    # Calculate video parameters
    audio_clip = AudioFileClip(config.input_audio)
    video_length = min(config.partial_video_length, int(audio_clip.duration * config.fps))
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1

    print(f"🎬 Generating {video_length} frames ({video_length/config.fps:.1f}s) at {config.fps} FPS")

    # Setup generation parameters
    sample_height, sample_width = get_sample_size(ref_img, config.sample_size)
    downratio = math.sqrt(sample_height * sample_width / h_ / w_)
    coords = (
        y1 * downratio // 16, y2 * downratio // 16,
        x1 * downratio // 16, x2 * downratio // 16,
        sample_height // 16, sample_width // 16,
    )
    ip_mask = get_ip_mask(coords).unsqueeze(0)
    ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=config.weight_dtype)

    # Prepare input
    input_video, input_video_mask, clip_image = get_image_to_video_latent3(
        ref_img, None, video_length=video_length, sample_size=[sample_height, sample_width]
    )

    # Crop audio to match video length
    partial_audio_embeds = audio_embeds[:, :video_length * 2]

    # Generate video
    print("🚀 Generating avatar video...")
    generator = torch.Generator(device=device).manual_seed(config.seed)

    with torch.no_grad():
        sample = pipeline(
            config.prompt,
            num_frames=video_length,
            negative_prompt=config.negative_prompt,
            audio_embeds=partial_audio_embeds,
            audio_scale=config.audio_scale,
            ip_mask=ip_mask,
            height=sample_height,
            width=sample_width,
            generator=generator,
            use_dynamic_cfg=config.use_dynamic_cfg,
            use_dynamic_acfg=config.use_dynamic_acfg,
            guidance_scale=config.guidance_scale,
            audio_guidance_scale=config.audio_guidance_scale,
            num_inference_steps=config.num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            shift=config.shift,
        ).videos

    # Save results
    os.makedirs(config.save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    video_path = os.path.join(config.save_path, f"vinay_avatar_{timestamp}.mp4")
    video_audio_path = os.path.join(config.save_path, f"vinay_avatar_{timestamp}_with_audio.mp4")

    # Save video without audio
    save_videos_grid(sample, video_path, fps=config.fps)

    # Add original audio to video
    video_clip = VideoFileClip(video_path)
    audio_clip = audio_clip.subclipped(0, video_length / config.fps)
    final_clip = video_clip.with_audio(audio_clip)
    final_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac")

    # Cleanup temporary files
    if os.path.exists("temp_reference_frame.jpg"):
        os.remove("temp_reference_frame.jpg")

    print(f"✅ Generated avatar video: {video_audio_path}")
    print(f"📊 Video info: {video_length} frames, {sample_height}x{sample_width}, {config.fps} FPS")
    print("🎉 Test completed successfully!")


if __name__ == "__main__":
    main()