import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor

from ..pipeline import StreamDiffusion
from .config import ControlNetConfig, StreamDiffusionControlNetConfig
from .preprocessors import get_preprocessor


class SDXLTurboControlNetPipeline:
    """
    SD-XL Turbo ControlNet pipeline for real-time image-to-image generation
    
    This class implements SD-XL Turbo with ControlNet support, optimized for 
    real-time img2img generation with minimal latency and high resolution.
    """
    
    def __init__(self, 
                 base_model: str = "stabilityai/sdxl-turbo",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 use_taesd: bool = True,
                 safety_checker: bool = False):
        """
        Initialize SD-XL Turbo ControlNet pipeline
        
        Args:
            base_model: SD-XL Turbo model ID or path
            device: Device to run on
            dtype: Data type for models
            use_taesd: Use Tiny AutoEncoder for faster decoding
            safety_checker: Enable safety checker
        """
        self.device = device
        self.dtype = dtype
        self.base_model = base_model
        
        # ControlNet storage
        self.controlnets: List[ControlNetModel] = []
        self.controlnet_configs: List[Dict] = []
        self.preprocessors: List[Optional[Any]] = []
        
        # Pipeline will be created when first ControlNet is added
        self.pipe = None
        self.is_prepared = False
        
        # SD-XL Turbo specific parameters
        self.default_steps = 2  # SD-XL Turbo typically uses 2-4 steps
        self.default_guidance_scale = 0.0  # SD-XL Turbo typically uses no guidance
        self.default_strength = 0.5  # SD-XL Turbo default strength is lower
        
        # Cache for preprocessed images and optimization
        self._preprocessed_cache = {}
        self._last_input_frame = None
        
        # Store initialization parameters
        self._use_taesd = use_taesd
        self._safety_checker = safety_checker
    
    def add_controlnet(self, 
                      controlnet_config: ControlNetConfig,
                      control_image: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None) -> int:
        """
        Add a ControlNet to the pipeline
        
        Args:
            controlnet_config: ControlNet configuration
            control_image: Control image (optional)
            
        Returns:
            Index of the added ControlNet
        """
        if not controlnet_config.enabled:
            print(f"ControlNet {controlnet_config.model_id} is disabled, skipping")
            return -1
        
        # Load ControlNet model
        print(f"Loading SD-XL ControlNet: {controlnet_config.model_id}")
        controlnet = self._load_controlnet_model(controlnet_config.model_id)
        
        # Load preprocessor if specified
        preprocessor = None
        if controlnet_config.preprocessor:
            preprocessor = get_preprocessor(controlnet_config.preprocessor)
            # Set preprocessor parameters including device and dtype
            preprocessor_params = {
                'device': self.device,
                'dtype': self.dtype
            }
            if controlnet_config.preprocessor_params:
                preprocessor_params.update(controlnet_config.preprocessor_params)
            preprocessor.params.update(preprocessor_params)
            # Update device and dtype directly
            if hasattr(preprocessor, 'device'):
                preprocessor.device = self.device
            if hasattr(preprocessor, 'dtype'):
                preprocessor.dtype = self.dtype
        
        # Store configuration for pipeline creation
        controlnet_info = {
            'model': controlnet,
            'conditioning_scale': controlnet_config.conditioning_scale,
            'control_guidance_start': getattr(controlnet_config, 'control_guidance_start', 0.0),
            'control_guidance_end': getattr(controlnet_config, 'control_guidance_end', 1.0),
            'preprocessor_params': controlnet_config.preprocessor_params or {}
        }
        
        self.controlnets.append(controlnet)
        self.controlnet_configs.append(controlnet_info)
        self.preprocessors.append(preprocessor)
        
        # Create or recreate pipeline with new ControlNet
        self._create_pipeline()
        
        print(f"Added SD-XL ControlNet {len(self.controlnets) - 1}: {controlnet_config.model_id}")
        return len(self.controlnets) - 1
    
    def _create_pipeline(self) -> None:
        """Create or recreate the SD-XL Turbo ControlNet pipeline"""
        if not self.controlnets:
            return
        
        # Use the first ControlNet for single ControlNet pipeline
        # For multiple ControlNets, we'll use MultiControlNet
        if len(self.controlnets) == 1:
            controlnet = self.controlnets[0]
        else:
            from diffusers import MultiControlNetModel
            controlnet = MultiControlNetModel(self.controlnets)
        
        # Load improved VAE for SD-XL
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=self.dtype
        )
        
        # Check if base model is a local file path
        model_path = Path(self.base_model)
        is_local_file = model_path.exists() and model_path.is_file()
        is_local_dir = model_path.exists() and model_path.is_dir()
        
        # Create SD-XL Turbo ControlNet Img2Img pipeline
        if is_local_file:
            # Local model file (e.g., .safetensors)
            print(f"Loading from local file: {model_path}")
            if self._safety_checker:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                    str(model_path),
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=self.dtype
                )
            else:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                    str(model_path),
                    controlnet=controlnet,
                    vae=vae,
                    safety_checker=None,
                    torch_dtype=self.dtype
                )
        elif is_local_dir:
            # Local model directory
            print(f"Loading from local directory: {model_path}")
            if self._safety_checker:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    str(model_path),
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=self.dtype,
                    local_files_only=True
                )
            else:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    str(model_path),
                    controlnet=controlnet,
                    vae=vae,
                    safety_checker=None,
                    torch_dtype=self.dtype,
                    local_files_only=True
                )
        else:
            # HuggingFace model ID
            print(f"Loading from HuggingFace: {self.base_model}")
            if self._safety_checker:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    self.base_model,
                    controlnet=controlnet,
                    vae=vae,
                    torch_dtype=self.dtype
                )
            else:
                self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    self.base_model,
                    controlnet=controlnet,
                    vae=vae,
                    safety_checker=None,
                    torch_dtype=self.dtype
                )
        
        # Use Tiny AutoEncoder XL for faster decoding
        if self._use_taesd:
            taesd_model = "madebyollin/taesdxl"
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, 
                torch_dtype=self.dtype, 
                use_safetensors=True
            )
        
        # Configure scheduler and progress bar
        self.pipe.set_progress_bar_config(disable=True)
        
        # Move to device
        self.pipe = self.pipe.to(device=self.device, dtype=self.dtype)
        
        # Optimize for inference
        if self.device != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)
        
        # Create image processor
        self.image_processor = VaeImageProcessor(self.pipe.vae_scale_factor)
        
        # Setup Compel for better prompt processing if available
        try:
            from compel import Compel, ReturnedEmbeddingsType
            self.pipe.compel_proc = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
            print("✓ Compel prompt processing enabled")
        except ImportError:
            print("⚠️  Compel not available, using standard prompt processing")
            self.pipe.compel_proc = None
        
        print(f"Created SD-XL Turbo ControlNet pipeline with {len(self.controlnets)} ControlNet(s)")
    
    def prepare(self,
                prompt: str,
                negative_prompt: str = "",
                width: int = 1024,
                height: int = 1024) -> None:
        """
        Prepare the pipeline for generation
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Output width (typically 1024 for SD-XL)
            height: Output height (typically 1024 for SD-XL)
        """
        if not self.pipe:
            raise RuntimeError("No ControlNets added. Add at least one ControlNet before preparing.")
        
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.is_prepared = True
        
        print(f"Prepared SD-XL Turbo ControlNet pipeline")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}")
    
    def _load_controlnet_model(self, model_id: str) -> ControlNetModel:
        """Load a ControlNet model from HuggingFace or local path"""
        try:
            if Path(model_id).exists():
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    local_files_only=True
                )
            else:
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype
                )
            
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            return controlnet
            
        except Exception as e:
            raise ValueError(f"Failed to load SD-XL ControlNet model '{model_id}': {e}")
    
    def _prepare_control_image(self, 
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              preprocessor: Optional[Any] = None) -> Union[Image.Image, List[Image.Image]]:
        """
        Prepare control image(s) for ControlNet input
        
        Args:
            control_image: Input control image
            preprocessor: Optional preprocessor to apply
            
        Returns:
            Processed control image(s)
        """
        # Load image if path
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        # Convert to PIL if needed
        if isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image)
        elif isinstance(control_image, torch.Tensor):
            # Convert tensor to PIL
            if control_image.dim() == 4:
                control_image = control_image[0]
            if control_image.dim() == 3 and control_image.shape[0] in [1, 3]:
                control_image = control_image.permute(1, 2, 0)
            
            if control_image.is_cuda:
                control_image = control_image.cpu()
            control_image = control_image.numpy()
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image.astype(np.uint8))
        
        # Apply preprocessor if available
        if preprocessor is not None:
            control_image = preprocessor.process(control_image)
        
        # Resize to target size (SD-XL typically uses 1024x1024)
        if hasattr(self, 'width') and hasattr(self, 'height'):
            target_size = (self.width, self.height)
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.LANCZOS)
        
        return control_image
    
    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> None:
        """
        Efficiently update control images for all ControlNets
        
        Args:
            control_image: New control image to apply to all ControlNets
        """
        # Check if we need to reprocess
        if self._last_input_frame is not None:
            if isinstance(control_image, (torch.Tensor, np.ndarray)) and isinstance(self._last_input_frame, type(control_image)):
                if control_image is self._last_input_frame:
                    return  # Same object, use cached results
            elif isinstance(control_image, Image.Image) and isinstance(self._last_input_frame, Image.Image):
                if control_image is self._last_input_frame:
                    return  # Same object, use cached results
        
        self._last_input_frame = control_image
        self._preprocessed_cache.clear()
        
        # Process for each ControlNet
        self.processed_control_images = []
        for i, preprocessor in enumerate(self.preprocessors):
            processed_image = self._prepare_control_image(control_image, preprocessor)
            self.processed_control_images.append(processed_image)
    
    def __call__(self,
                 image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                 control_image: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None,
                 strength: float = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 controlnet_conditioning_scale: Union[float, List[float]] = None,
                 control_guidance_start: float = None,
                 control_guidance_end: float = None,
                 generator: Optional[torch.Generator] = None,
                 **kwargs) -> Image.Image:
        """
        Generate image using SD-XL Turbo with ControlNet
        
        Args:
            image: Input image for img2img
            control_image: Control image (if None, uses input image)
            strength: Denoising strength (0.1-1.0)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            controlnet_conditioning_scale: ControlNet conditioning scale(s)
            control_guidance_start: Control guidance start
            control_guidance_end: Control guidance end
            generator: Random generator
            
        Returns:
            Generated PIL Image
        """
        if not self.is_prepared:
            raise RuntimeError("Pipeline not prepared. Call prepare() first.")
        
        # Use input image as control image if not provided
        if control_image is None:
            control_image = image
        
        # Update control images
        self.update_control_image_efficient(control_image)
        
        # Use defaults for SD-XL Turbo if not specified
        strength = strength or self.default_strength
        num_inference_steps = num_inference_steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance_scale
        
        # Set ControlNet parameters
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = [config['conditioning_scale'] for config in self.controlnet_configs]
        
        if control_guidance_start is None:
            control_guidance_start = [config['control_guidance_start'] for config in self.controlnet_configs]
        
        if control_guidance_end is None:
            control_guidance_end = [config['control_guidance_end'] for config in self.controlnet_configs]
        
        # Handle single vs multiple ControlNets
        if len(self.processed_control_images) == 1:
            control_images = self.processed_control_images[0]
            if isinstance(controlnet_conditioning_scale, list):
                controlnet_conditioning_scale = controlnet_conditioning_scale[0]
            if isinstance(control_guidance_start, list):
                control_guidance_start = control_guidance_start[0]
            if isinstance(control_guidance_end, list):
                control_guidance_end = control_guidance_end[0]
        else:
            control_images = self.processed_control_images
        
        # Prepare input image
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, (np.ndarray, torch.Tensor)):
            image = self._prepare_control_image(image, None)
        
        # Ensure input image is correct size
        if hasattr(self, 'width') and hasattr(self, 'height'):
            target_size = (self.width, self.height)
            if image.size != target_size:
                image = image.resize(target_size, Image.LANCZOS)
        
        # Prepare prompts with Compel if available
        prompt = self.prompt
        negative_prompt = self.negative_prompt
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        
        if hasattr(self.pipe, "compel_proc") and self.pipe.compel_proc is not None:
            try:
                _prompt_embeds, pooled_prompt_embeds = self.pipe.compel_proc(
                    [self.prompt, self.negative_prompt]
                )
                prompt = None
                negative_prompt = None
                prompt_embeds = _prompt_embeds[0:1]
                pooled_prompt_embeds = pooled_prompt_embeds[0:1]
                negative_prompt_embeds = _prompt_embeds[1:2]
                negative_pooled_prompt_embeds = pooled_prompt_embeds[1:2]
            except Exception as e:
                print(f"⚠️  Compel processing failed: {e}, using standard prompts")
        
        # Generate image
        try:
            result = self.pipe(
                image=image,
                control_image=control_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                generator=generator,
                output_type="pil",
                **kwargs
            )
            
            # Handle NSFW detection
            if hasattr(result, 'nsfw_content_detected') and result.nsfw_content_detected[0]:
                print("NSFW content detected, returning black image")
                return Image.new('RGB', (self.width, self.height), (0, 0, 0))
            
            return result.images[0]
            
        except Exception as e:
            print(f"SD-XL Turbo generation failed: {e}")
            # Return black image as fallback
            return Image.new('RGB', (self.width, self.height), (0, 0, 0))
    
    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """Update the conditioning scale for a specific ControlNet"""
        if 0 <= index < len(self.controlnet_configs):
            self.controlnet_configs[index]['conditioning_scale'] = scale
        else:
            raise IndexError(f"ControlNet index {index} out of range")
    
    def update_prompt(self, prompt: str) -> None:
        """Update the generation prompt"""
        self.prompt = prompt
    
    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Get the last processed control image for display purposes"""
        if 0 <= index < len(self.processed_control_images):
            return self.processed_control_images[index]
        return None


def create_sdxlturbo_controlnet_pipeline(config: StreamDiffusionControlNetConfig) -> SDXLTurboControlNetPipeline:
    """
    Create an SD-XL Turbo ControlNet pipeline from configuration
    
    Args:
        config: Configuration object
        
    Returns:
        SDXLTurboControlNetPipeline instance
    """
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)
    
    # Create SD-XL Turbo ControlNet pipeline
    pipeline = SDXLTurboControlNetPipeline(
        base_model=config.model_id,
        device=config.device,
        dtype=dtype,
        use_taesd=getattr(config, 'use_taesd', True),
        safety_checker=getattr(config, 'safety_checker', False)
    )
    
    # Add ControlNets
    for cn_config in config.controlnets:
        pipeline.add_controlnet(cn_config)
    
    # Prepare with prompt
    if config.prompt:
        pipeline.prepare(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height
        )
    
    return pipeline 