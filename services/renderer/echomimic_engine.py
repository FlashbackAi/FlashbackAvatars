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
from src.utils import filter_kwargs

class EchoMimicEngine:
    def __init__(self, model_dir, wav2vec_dir, device="auto", low_mem_mode=True):
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
                
                # Enable low memory mode for GPUs with < 12GB
                if gpu_memory < 12 and low_mem_mode:
                    print("⚡ Enabling low memory optimizations for <12GB GPU")
                    self.low_mem_mode = True
                else:
                    self.low_mem_mode = False
                    
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.low_mem_mode = False
                print("⚠️  CUDA not available, using CPU")
        else:
            self.device = device
            self.low_mem_mode = low_mem_mode
            
        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"📱 Device: {self.device}, Data type: {self.weight_dtype}, Low mem: {self.low_mem_mode}")
        
        # Set memory optimization environment variables
        if self.low_mem_mode and self.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()  # Clear any existing cache
        
        # Store paths and configs for lazy loading
        self.model_dir = model_dir
        self.wav2vec_dir = wav2vec_dir
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'echomimic_v3', 'config', 'config.yaml')
        self.cfg = OmegaConf.load(config_path)
        
        # Initialize model containers (will be loaded on demand)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.clip_image_encoder = None
        self.wav2vec_model = None
        self.audio_proc = None
        self.scheduler = None
        self.pipe = None
        
        print("✅ EchoMimic engine initialized with lazy loading!")
        print("🔄 Models will be loaded on-demand during inference")
    
    def load_transformer(self):
        """Load transformer model on demand"""
        if self.transformer is None:
            print("🔄 Loading transformer...")
            if self.low_mem_mode and self.device == "cuda":
                torch.cuda.empty_cache()
                
            self.transformer = WanTransformerAudioMask3DModel.from_pretrained(
                self.model_dir,
                transformer_additional_kwargs=OmegaConf.to_container(self.cfg['transformer_additional_kwargs']),
                torch_dtype=self.weight_dtype,
            ).eval()
            
            if self.device == "cuda":
                self.transformer = self.transformer.to(self.device)
                if self.low_mem_mode:
                    torch.cuda.empty_cache()
                    print(f"   GPU memory after transformer: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def load_vae(self):
        """Load VAE model on demand"""
        if self.vae is None:
            print("🔄 Loading VAE...")
            vae_path = os.path.join(self.model_dir, self.cfg['vae_kwargs'].get('vae_subpath', 'Wan2.1_VAE.pth'))
            self.vae = AutoencoderKLWan.from_pretrained(
                vae_path,
                additional_kwargs=OmegaConf.to_container(self.cfg['vae_kwargs']),
            ).to(self.weight_dtype).eval()
            
            if self.device == "cuda":
                self.vae = self.vae.to(self.device)
                if self.low_mem_mode:
                    torch.cuda.empty_cache()
                    print(f"   GPU memory after VAE: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def load_text_encoder(self):
        """Load text encoder on demand"""
        if self.text_encoder is None or self.tokenizer is None:
            print("🔄 Loading text encoder...")
            tokenizer_path = self.cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'google/umt5-xxl')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            text_encoder_path = os.path.join(self.model_dir, self.cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'models_t5_umt5-xxl-enc-bf16.pth'))
            self.text_encoder = WanT5EncoderModel.from_pretrained(
                text_encoder_path,
                additional_kwargs=OmegaConf.to_container(self.cfg['text_encoder_kwargs']),
                torch_dtype=self.weight_dtype,
            ).eval()
            
            if self.device == "cuda":
                self.text_encoder = self.text_encoder.to(self.device)
                if self.low_mem_mode:
                    torch.cuda.empty_cache()
                    print(f"   GPU memory after text encoder: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def load_clip_encoder(self):
        """Load CLIP image encoder on demand"""
        if self.clip_image_encoder is None:
            print("🔄 Loading CLIP image encoder...")
            image_encoder_path = os.path.join(self.model_dir, self.cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'))
            self.clip_image_encoder = CLIPModel.from_pretrained(
                image_encoder_path,
            ).to(self.weight_dtype).eval()
            
            if self.device == "cuda":
                self.clip_image_encoder = self.clip_image_encoder.to(self.device)
                if self.low_mem_mode:
                    torch.cuda.empty_cache()
                    print(f"   GPU memory after CLIP: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def load_wav2vec(self):
        """Load Wav2Vec model on demand"""
        if self.wav2vec_model is None or self.audio_proc is None:
            print("🔄 Loading Wav2Vec models...")
            self.audio_proc = Wav2Vec2Processor.from_pretrained(self.wav2vec_dir)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(self.wav2vec_dir).eval()
            
            if self.device == "cuda":
                self.wav2vec_model = self.wav2vec_model.to(self.device)
                self.wav2vec_model.requires_grad_(False)
    
    def unload_model(self, model_name):
        """Unload a specific model to free VRAM"""
        if model_name == "text_encoder" and self.text_encoder is not None:
            del self.text_encoder
            del self.tokenizer
            self.text_encoder = None
            self.tokenizer = None
            print(f"🗑️  Unloaded text encoder")
        elif model_name == "clip" and self.clip_image_encoder is not None:
            del self.clip_image_encoder
            self.clip_image_encoder = None
            print(f"🗑️  Unloaded CLIP encoder")
        elif model_name == "wav2vec" and self.wav2vec_model is not None:
            del self.wav2vec_model
            del self.audio_proc
            self.wav2vec_model = None
            self.audio_proc = None
            print(f"🗑️  Unloaded Wav2Vec model")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print(f"   GPU memory after unload: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def create_pipeline(self):
        """Create pipeline with loaded models"""
        if self.scheduler is None:
            self.scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(self.cfg['scheduler_kwargs'])))
        
        if self.pipe is None:
            print("🔄 Creating pipeline...")
            self.pipe = WanFunInpaintAudioPipeline(
                transformer=self.transformer,
                vae=self.vae,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                scheduler=self.scheduler,
                clip_image_encoder=self.clip_image_encoder,
            )
            if self.device == "cuda":
                self.pipe.to(device=self.device)

    def generate_frame(self, image, audio_array, steps=8, cfg=2.5):
        """Generate frame with lazy loading and memory management"""
        print("🚀 Starting inference with lazy loading...")
        
        # Step 1: Load and process audio
        self.load_wav2vec()
        audio_feats = self.audio_proc(audio_array, sampling_rate=16000,
                                      return_tensors="pt").input_values.to(self.device)
        # Unload wav2vec after processing
        self.unload_model("wav2vec")
        
        # Step 2: Load text encoder (if needed for conditioning)
        self.load_text_encoder()
        
        # Step 3: Load CLIP encoder for image processing
        self.load_clip_encoder()
        
        # Step 4: Load core models for generation
        self.load_transformer()
        self.load_vae()
        
        # Step 5: Create pipeline with all loaded models
        self.create_pipeline()
        
        # Step 6: Run inference
        print("🎬 Running inference...")
        with torch.no_grad():
            out = self.pipe(
                image=image,
                audio_features=audio_feats,
                num_inference_steps=steps,
                guidance_scale=cfg
            )
        
        # Step 7: Clean up non-essential models after inference
        self.unload_model("text_encoder")
        self.unload_model("clip")
        
        print("✅ Inference complete!")
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"🔥 Final GPU Memory: {memory_allocated:.2f}GB")
        
        return out.videos[0]

    def generate_video(self, video_array, clip_image, audio_array, steps=8, cfg=2.5):
        """Generate video with lazy loading and memory management"""
        print("🚀 Starting video inference with lazy loading...")

        # Step 1: Load and process audio
        self.load_wav2vec()
        audio_embeds = self.audio_proc(audio_array, sampling_rate=16000,
                                       return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            audio_embeds = self.wav2vec_model(audio_embeds).last_hidden_state
        # Unload wav2vec after processing
        self.unload_model("wav2vec")

        # Step 2: Load text encoder (if needed for conditioning)
        self.load_text_encoder()

        # Step 3: Load CLIP encoder for image processing
        self.load_clip_encoder()

        # Step 4: Load core models for generation
        self.load_transformer()
        self.load_vae()

        # Step 5: Create pipeline with all loaded models
        self.create_pipeline()

        # Step 6: Convert video array to tensor format
        import torch
        import torchvision.transforms.functional as TF

        # Convert video array (frames, H, W, C) to tensor (1, C, frames, H, W)
        video_tensor = torch.from_numpy(video_array).float() / 255.0  # normalize to [0,1]
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, frames, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # (1, C, frames, H, W)

        # Step 7: Run inference
        print("🎬 Running video inference...")
        with torch.no_grad():
            out = self.pipe(
                video=video_tensor,
                clip_image=clip_image,
                audio_embeds=audio_embeds,
                num_inference_steps=steps,
                guidance_scale=cfg,
                num_frames=video_array.shape[0]  # use actual number of frames
            )

        # Step 8: Clean up non-essential models after inference
        self.unload_model("text_encoder")
        self.unload_model("clip")

        print("✅ Video inference complete!")
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"🔥 Final GPU Memory: {memory_allocated:.2f}GB")

        return out.videos[0]
