import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from diffusers import FlowMatchEulerDiscreteScheduler

# Import EchoMimic v3 components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'echomimic_v3'))

from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs, get_image_to_video_latent3
from src.face_detect import get_mask_coord

class EchoMimicEngine:
    def __init__(self, model_dir, wav2vec_dir, device="auto"):
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
            else:
                self.device = "cpu"
                print("⚠️  CUDA not available, using CPU")
        else:
            self.device = device

        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"📱 Device: {self.device}, Data type: {self.weight_dtype}")

        # Set memory optimization for large models
        if self.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            print(f"🧠 Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'echomimic_v3', 'config', 'config.yaml')
        try:
            cfg = OmegaConf.load(config_path)
            print(f"✅ Config loaded from: {config_path}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            # Fallback to minimal config
            cfg = OmegaConf.create({
                'transformer_additional_kwargs': {},
                'vae_kwargs': {'vae_subpath': 'Wan2.1_VAE.pth'},
                'text_encoder_kwargs': {'tokenizer_subpath': 'google/umt5-xxl', 'text_encoder_subpath': 'models_t5_umt5-xxl-enc-bf16.pth'},
                'image_encoder_kwargs': {'image_encoder_subpath': 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'},
                'scheduler_kwargs': {}
            })

        # Load transformer
        print("🔄 Loading transformer...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.transformer = WanTransformerAudioMask3DModel.from_pretrained(
            model_dir,
            transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
            torch_dtype=self.weight_dtype,
        ).eval().to(self.device)
        if self.device == "cuda":
            print(f"🧠 After transformer: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            torch.cuda.empty_cache()

        # Load VAE - the path points directly to the .pth file
        print("🔄 Loading VAE...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        vae_path = os.path.join(model_dir, cfg['vae_kwargs'].get('vae_subpath', 'Wan2.1_VAE.pth'))
        self.vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
        ).to(self.weight_dtype).to(self.device).eval()
        if self.device == "cuda":
            print(f"🧠 After VAE: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            torch.cuda.empty_cache()

        # Load tokenizer and text encoder
        print("🔄 Loading text encoder...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        tokenizer_path = cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'google/umt5-xxl')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        text_encoder_path = os.path.join(model_dir, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'models_t5_umt5-xxl-enc-bf16.pth'))
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            text_encoder_path,
            additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
            torch_dtype=self.weight_dtype,
        ).eval().to(self.device)
        if self.device == "cuda":
            print(f"🧠 After text encoder: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            torch.cuda.empty_cache()

        # Load CLIP image encoder - the path points directly to the .pth file
        print("🔄 Loading CLIP image encoder...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        image_encoder_path = os.path.join(model_dir, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'))
        self.clip_image_encoder = CLIPModel.from_pretrained(
            image_encoder_path,
        ).to(self.weight_dtype).to(self.device).eval()
        if self.device == "cuda":
            print(f"🧠 After CLIP: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            torch.cuda.empty_cache()

        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs'])))

        # Create pipeline
        self.pipe = WanFunInpaintAudioPipeline(
            transformer=self.transformer,
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            clip_image_encoder=self.clip_image_encoder,
        )
        self.pipe.to(device=self.device)

        # Load Wav2Vec models
        print("🔄 Loading Wav2Vec...")
        self.audio_proc = Wav2Vec2Processor.from_pretrained(wav2vec_dir)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_dir).eval().to(self.device)
        self.wav2vec_model.requires_grad_(False)

        print("✅ EchoMimic engine initialized successfully!")

    def generate_frame(self, image, audio_array, steps=8, cfg=2.5):
        # Process audio with Wav2Vec model to get embeddings
        audio_feats = self.audio_proc(audio_array, sampling_rate=16000,
                                      return_tensors="pt").input_values.to(self.device)

        # Get audio embeddings from Wav2Vec model
        with torch.no_grad():
            audio_embeds = self.wav2vec_model(audio_feats).last_hidden_state

        prompt = "cinematic portrait, soft studio key light, shallow depth of field, filmic color"
        with torch.no_grad():
            out = self.pipe(
                prompt=prompt,
                clip_image=image,
                audio_embeds=audio_embeds,
                num_inference_steps=steps,
                guidance_scale=cfg,
                audio_guidance_scale=cfg,
                num_frames=1,  # Single frame
                output_type="numpy"
            )
        return out.images[0] if hasattr(out, 'images') else out.frames[0]

    def generate_video(self, video_frames, reference_image, audio_array, steps=8, cfg=2.5):
        """Generate video with audio-driven animation using EchoMimic v3"""
        print(f"🎬 Generating video from reference image with audio")

        # Process audio with Wav2Vec model to get embeddings
        audio_feats = self.audio_proc(audio_array, sampling_rate=16000,
                                      return_tensors="pt").input_values.to(self.device)

        # Get audio embeddings from Wav2Vec model
        with torch.no_grad():
            audio_embeds = self.wav2vec_model(audio_feats).last_hidden_state

        # EchoMimic v3 generates video from single reference image + audio
        prompt = "cinematic portrait, soft studio key light, shallow depth of field, filmic color"

        # Skip face detection and use simple approach
        print("🎭 Using simplified pipeline without face detection")
        img_height = reference_image.height
        img_width = reference_image.width
        sample_size = [img_height, img_width]

        # Use EchoMimic's proper image processing
        video_length = 12  # Reduced for fast sample (12 frames ≈ 0.8 seconds at 15 FPS)

        # Process the reference image using EchoMimic's utility function
        _, _, clip_image = get_image_to_video_latent3(
            reference_image, None,
            video_length=video_length,
            sample_size=sample_size
        )

        # Use the actual video from video_frames parameter
        print(f"🎬 Using actual video with {len(video_frames)} frames")

        # Convert video frames to tensor format expected by EchoMimic v3
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])

        # Convert video frames to tensor: [batch, channels, frames, height, width]
        frame_tensors = []
        max_frames = min(len(video_frames), 12)  # Limit to 12 frames for fast sample

        for i in range(max_frames):
            # Convert numpy array to PIL Image, then to tensor
            frame_pil = Image.fromarray(video_frames[i])
            frame_tensor = transform(frame_pil)
            frame_tensors.append(frame_tensor)

        # Stack frames: [frames, channels, height, width] -> [1, channels, frames, height, width]
        initial_video = torch.stack(frame_tensors, dim=1).unsqueeze(0)  # [1, 3, frames, H, W]
        print(f"📹 Video tensor shape: {initial_video.shape}")

        # Optimize for speed: lower resolution and fewer steps
        fast_height = min(img_height, 512)  # Limit height to 512
        fast_width = min(img_width, 512)   # Limit width to 512
        fast_steps = 4  # Reduced from 8 to 4 steps for speed

        with torch.no_grad():
            out = self.pipe(
                prompt,  # Style prompt for video generation
                num_frames=12,  # Reduced frame count for fast sample
                negative_prompt="bad quality, distorted face, blurry",
                audio_embeds=audio_embeds,  # Processed audio embeddings for lip-sync
                audio_scale=cfg,  # Audio guidance scale
                clip_image=clip_image,  # Processed reference image for identity
                video=initial_video,  # Initial video tensor to avoid 'y' variable error
                height=fast_height,
                width=fast_width,
                num_inference_steps=fast_steps,  # Reduced steps for speed
                guidance_scale=cfg,
                generator=torch.Generator(device=self.device).manual_seed(42),
                output_type="numpy"
            )

        # Return the generated video frames
        if hasattr(out, 'frames') and out.frames is not None:
            print(f"✅ Generated {len(out.frames)} video frames")
            return out.frames
        elif hasattr(out, 'images') and out.images is not None:
            print(f"✅ Generated {len(out.images)} images")
            return out.images
        else:
            # Fallback: generate single frame
            print("⚠️ Falling back to single frame generation")
            return [self.generate_frame(reference_image, audio_array, steps, cfg)]