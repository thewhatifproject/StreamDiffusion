import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from diffusers.models import ControlNetModel
from diffusers.utils import load_image

from ..pipeline import StreamDiffusion
from .config import ControlNetConfig, StreamDiffusionControlNetConfig
from .preprocessors import get_preprocessor


class ControlNetPipeline:
    """
    ControlNet-enabled StreamDiffusion pipeline
    
    This class extends StreamDiffusion with ControlNet support, allowing for
    conditioning the generation process with multiple ControlNet models.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize ControlNet pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run ControlNets on
            dtype: Data type for ControlNet models
        """
        self.stream = stream_diffusion
        self.device = device
        self.dtype = dtype
        
        # ControlNet storage
        self.controlnets: List[ControlNetModel] = []
        self.controlnet_images: List[Optional[torch.Tensor]] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Optional[Any]] = []
        
        # Store original unet_step method for patching
        self._original_unet_step = None
        self._is_patched = False
        
        # Optimization: Cache transforms and reusable tensors
        import torchvision.transforms as transforms
        self._cached_transform = transforms.ToTensor()
        self._temp_tensor_cache = {}  # For reusing tensors
        
        # Optimization: Cache preprocessed images to eliminate redundant processing
        self._preprocessed_cache = {}  # Maps preprocessor_type -> processed_image
        self._last_input_frame = None  # Track if we need to reprocess
    
    def add_controlnet(self, 
                      controlnet_config: ControlNetConfig,
                      control_image: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None) -> int:
        """
        Add a ControlNet to the pipeline
        
        Args:
            controlnet_config: ControlNet configuration
            control_image: Control image (optional, can be set later)
            
        Returns:
            Index of the added ControlNet
        """
        if not controlnet_config.enabled:
            print(f"ControlNet {controlnet_config.model_id} is disabled, skipping")
            return -1
        
        # Load ControlNet model
        print(f"Loading ControlNet: {controlnet_config.model_id}")
        controlnet = self._load_controlnet_model(controlnet_config.model_id)
        
        # Load preprocessor if specified
        preprocessor = None
        if controlnet_config.preprocessor:
            preprocessor = get_preprocessor(controlnet_config.preprocessor)
            # Set preprocessor parameters
            if controlnet_config.preprocessor_params:
                preprocessor.params.update(controlnet_config.preprocessor_params)
        
        # Process control image if provided
        processed_image = None
        if control_image is not None:
            processed_image = self._prepare_control_image(control_image, preprocessor)
        elif controlnet_config.control_image_path:
            # Load from configured path
            control_image = load_image(controlnet_config.control_image_path)
            processed_image = self._prepare_control_image(control_image, preprocessor)
        
        # Add to collections
        self.controlnets.append(controlnet)
        self.controlnet_images.append(processed_image)
        self.controlnet_scales.append(controlnet_config.conditioning_scale)
        self.preprocessors.append(preprocessor)
        
        # Patch the StreamDiffusion pipeline if this is the first ControlNet
        if len(self.controlnets) == 1:
            self._patch_stream_diffusion()
        
        print(f"Added ControlNet {len(self.controlnets) - 1}: {controlnet_config.model_id}")
        return len(self.controlnets) - 1
    
    def remove_controlnet(self, index: int) -> None:
        """
        Remove a ControlNet by index
        
        Args:
            index: Index of the ControlNet to remove
        """
        if 0 <= index < len(self.controlnets):
            self.controlnets.pop(index)
            self.controlnet_images.pop(index)
            self.controlnet_scales.pop(index)
            self.preprocessors.pop(index)
            
            # Unpatch if no ControlNets remain
            if len(self.controlnets) == 0:
                self._unpatch_stream_diffusion()
            
            print(f"Removed ControlNet {index}")
        else:
            raise IndexError(f"ControlNet index {index} out of range")
    
    def clear_controlnets(self) -> None:
        """Remove all ControlNets"""
        self.controlnets.clear()
        self.controlnet_images.clear()
        self.controlnet_scales.clear()
        self.preprocessors.clear()
        
        self._unpatch_stream_diffusion()
        print("Cleared all ControlNets")
    
    def update_control_image(self, 
                           index: int, 
                           control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> None:
        """
        Update the control image for a specific ControlNet (optimized version)
        
        Args:
            index: Index of the ControlNet
            control_image: New control image
        """
        if not (0 <= index < len(self.controlnets)):
            raise IndexError(f"ControlNet index {index} out of range")
        
        # Skip processing if scale is 0 (optimization)
        if self.controlnet_scales[index] == 0:
            return
            
        preprocessor = self.preprocessors[index]
        processed_image = self._prepare_control_image(control_image, preprocessor)
        self.controlnet_images[index] = processed_image
    
    def update_control_image_batch(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> None:
        """
        Update all ControlNets with the same control image (optimized for webcam use)
        
        Args:
            control_image: New control image to apply to all ControlNets
        """
        # Process once, reuse for all active ControlNets
        for i in range(len(self.controlnets)):
            if self.controlnet_scales[i] > 0:  # Only update active ones
                preprocessor = self.preprocessors[i]
                processed_image = self._prepare_control_image(control_image, preprocessor)
                self.controlnet_images[i] = processed_image
    
    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """
        Get the last processed control image for display purposes (avoids reprocessing)
        
        Args:
            index: Index of the ControlNet
            
        Returns:
            Last processed PIL Image, or None if not available
        """
        if not (0 <= index < len(self.controlnets)):
            return None
        
        preprocessor = self.preprocessors[index]
        if preprocessor is None:
            return None
            
        preprocessor_type = type(preprocessor).__name__
        return self._preprocessed_cache.get(preprocessor_type)
    
    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> None:
        """
        Efficiently update all ControlNets with cache-aware preprocessing
        
        Args:
            control_image: New control image to apply to all ControlNets
        """
        # Check if we need to reprocess (simple frame comparison)
        if self._last_input_frame is control_image:
            return  # No change, use cached results
        
        self._last_input_frame = control_image
        
        # Clear cache for new frame
        self._preprocessed_cache.clear()
        
        # Group ControlNets by preprocessor type to avoid duplicate processing
        preprocessor_groups = {}
        for i in range(len(self.controlnets)):
            if self.controlnet_scales[i] > 0:  # Only process active ones
                preprocessor = self.preprocessors[i]
                if preprocessor is not None:
                    preprocessor_type = type(preprocessor).__name__
                    if preprocessor_type not in preprocessor_groups:
                        preprocessor_groups[preprocessor_type] = {
                            'preprocessor': preprocessor,
                            'indices': []
                        }
                    preprocessor_groups[preprocessor_type]['indices'].append(i)
        
        # Process once per preprocessor type
        for preprocessor_type, group in preprocessor_groups.items():
            preprocessor = group['preprocessor']
            processed_image = self._prepare_control_image(control_image, preprocessor)
            
            # Cache the result
            self._preprocessed_cache[preprocessor_type] = processed_image
            
            # Apply to all ControlNets using this preprocessor
            for index in group['indices']:
                self.controlnet_images[index] = processed_image
    
    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """
        Update the conditioning scale for a specific ControlNet
        
        Args:
            index: Index of the ControlNet
            scale: New conditioning scale
        """
        if 0 <= index < len(self.controlnets):
            self.controlnet_scales[index] = scale
        else:
            raise IndexError(f"ControlNet index {index} out of range")
    
    def _load_controlnet_model(self, model_id: str) -> ControlNetModel:
        """
        Load a ControlNet model from HuggingFace or local path
        
        Args:
            model_id: Model ID or path
            
        Returns:
            Loaded ControlNet model
        """
        try:
            # Check if it's a local path
            if Path(model_id).exists():
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    local_files_only=True
                )
            else:
                # Try as HuggingFace model ID
                if "/" in model_id and model_id.count("/") > 1:
                    # Handle subfolder case (e.g., "repo/model/subfolder")
                    parts = model_id.split("/")
                    repo_id = "/".join(parts[:2])
                    subfolder = "/".join(parts[2:])
                    controlnet = ControlNetModel.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        torch_dtype=self.dtype
                    )
                else:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype
                    )
            
            # Move to device
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            return controlnet
            
        except Exception as e:
            raise ValueError(f"Failed to load ControlNet model '{model_id}': {e}")
    
    def _prepare_control_image(self, 
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              preprocessor: Optional[Any] = None) -> torch.Tensor:
        """
        Prepare a control image for ControlNet input (optimized version)
        
        Args:
            control_image: Input control image
            preprocessor: Optional preprocessor to apply
            
        Returns:
            Processed control image tensor
        """
        # Load image if path
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        # Apply preprocessor if available
        if preprocessor is not None:
            control_image = preprocessor.process(control_image)
        
        # Fast path for PIL Images (most common case)
        if isinstance(control_image, Image.Image):
            # Resize first if needed
            target_size = (self.stream.width, self.stream.height)
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.LANCZOS)
            
            # Use cached transform
            control_tensor = self._cached_transform(control_image).unsqueeze(0)
            control_tensor = control_tensor.to(device=self.device, dtype=self.dtype)
            return control_tensor
        
        # Handle other types (less common, keep existing logic but optimize)
        if isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image)
        elif isinstance(control_image, torch.Tensor):
            # Optimize tensor to PIL conversion
            if control_image.dim() == 4:
                control_image = control_image[0]
            if control_image.dim() == 3 and control_image.shape[0] in [1, 3]:
                control_image = control_image.permute(1, 2, 0)
            
            # Keep on GPU if possible, convert to numpy only when needed
            if control_image.is_cuda:
                control_image = control_image.cpu()
            control_image = control_image.numpy()
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image.astype(np.uint8))
        
        # Resize and convert (recursive call will hit fast path)
        return self._prepare_control_image(control_image, None)
    
    def _get_controlnet_conditioning(self, 
                                   x_t_latent: torch.Tensor,
                                   timestep: torch.Tensor,
                                   encoder_hidden_states: torch.Tensor) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Get combined conditioning from all active ControlNets
        
        Args:
            x_t_latent: Latent input
            timestep: Current timestep
            encoder_hidden_states: Text embeddings
            
        Returns:
            Tuple of (down_block_res_samples, mid_block_res_sample)
        """
        if not self.controlnets:
            return None, None
        
        # Quick check: any active ControlNets?
        active_indices = [
            i for i, (controlnet, control_image, scale) in enumerate(
                zip(self.controlnets, self.controlnet_images, self.controlnet_scales)
            ) if controlnet is not None and control_image is not None and scale > 0
        ]
        
        if not active_indices:
            return None, None
        
        down_block_res_samples = None
        mid_block_res_sample = None
        
        # Only process active ControlNets
        for i in active_indices:
            controlnet = self.controlnets[i]
            control_image = self.controlnet_images[i]
            scale = self.controlnet_scales[i]
            
            # Forward pass through ControlNet
            try:
                down_samples, mid_sample = controlnet(
                    x_t_latent,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_image,
                    conditioning_scale=scale,
                    return_dict=False,
                )
                
                # Combine outputs
                if down_block_res_samples is None:
                    down_block_res_samples = down_samples
                    mid_block_res_sample = mid_sample
                else:
                    # Add contributions from this ControlNet
                    for j in range(len(down_block_res_samples)):
                        down_block_res_samples[j] += down_samples[j]
                    mid_block_res_sample += mid_sample
                    
            except Exception as e:
                print(f"Warning: ControlNet {i} failed: {e}")
                continue
        
        return down_block_res_samples, mid_block_res_sample
    
    def _patch_stream_diffusion(self) -> None:
        """Patch StreamDiffusion's unet_step method to include ControlNet conditioning"""
        if self._is_patched:
            return
        
        # Store original method
        self._original_unet_step = self.stream.unet_step
        
        # Create patched method
        def patched_unet_step(x_t_latent, t_list, idx=None):
            # Handle CFG expansion
            if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list[0:1], t_list], dim=0)
            elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
                t_list_expanded = torch.concat([t_list, t_list], dim=0)
            else:
                x_t_latent_plus_uc = x_t_latent
                t_list_expanded = t_list
            
            # Get ControlNet conditioning
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds
            )
            
            # Call UNet with ControlNet conditioning
            model_pred = self.stream.unet(
                x_t_latent_plus_uc,
                t_list_expanded,
                encoder_hidden_states=self.stream.prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            # Continue with original CFG logic
            if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                noise_pred_text = model_pred[1:]
                self.stream.stock_noise = torch.concat(
                    [model_pred[0:1], self.stream.stock_noise[1:]], dim=0
                )
            elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            else:
                noise_pred_text = model_pred
            
            if self.stream.guidance_scale > 1.0 and (
                self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize"
            ):
                noise_pred_uncond = self.stream.stock_noise * self.stream.delta
            
            if self.stream.guidance_scale > 1.0 and self.stream.cfg_type != "none":
                model_pred = noise_pred_uncond + self.stream.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                model_pred = noise_pred_text
            
            # Compute the previous noisy sample x_t -> x_t-1
            if self.stream.use_denoising_batch:
                denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
                if self.stream.cfg_type == "self" or self.stream.cfg_type == "initialize":
                    scaled_noise = self.stream.beta_prod_t_sqrt * self.stream.stock_noise
                    delta_x = self.stream.scheduler_step_batch(model_pred, scaled_noise, idx)
                    alpha_next = torch.concat(
                        [
                            self.stream.alpha_prod_t_sqrt[1:],
                            torch.ones_like(self.stream.alpha_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = alpha_next * delta_x
                    beta_next = torch.concat(
                        [
                            self.stream.beta_prod_t_sqrt[1:],
                            torch.ones_like(self.stream.beta_prod_t_sqrt[0:1]),
                        ],
                        dim=0,
                    )
                    delta_x = delta_x / beta_next
                    init_noise = torch.concat(
                        [self.stream.init_noise[1:], self.stream.init_noise[0:1]], dim=0
                    )
                    self.stream.stock_noise = init_noise + delta_x
            else:
                denoised_batch = self.stream.scheduler_step_batch(model_pred, x_t_latent, idx)
            
            return denoised_batch, model_pred
        
        # Replace the method
        self.stream.unet_step = patched_unet_step
        self._is_patched = True
        print("Patched StreamDiffusion with ControlNet support")
    
    def _unpatch_stream_diffusion(self) -> None:
        """Restore original StreamDiffusion unet_step method"""
        if self._is_patched and self._original_unet_step is not None:
            self.stream.unet_step = self._original_unet_step
            self._is_patched = False
            print("Unpatched StreamDiffusion")
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying StreamDiffusion instance"""
        return self.stream(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying StreamDiffusion instance"""
        return getattr(self.stream, name)


def create_controlnet_pipeline(config: StreamDiffusionControlNetConfig) -> ControlNetPipeline:
    """
    Create a ControlNet-enabled StreamDiffusion pipeline from configuration
    
    Args:
        config: Configuration object
        
    Returns:
        ControlNetPipeline instance
    """
    from diffusers import StableDiffusionPipeline
    
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)
    
    # Load base pipeline
    print(f"Loading base model: {config.model_id}")
    
    # Check if it's a local file path
    model_path = Path(config.model_id)
    if model_path.exists() and model_path.is_file():
        # Local model file
        print(f"Loading from local file: {model_path}")
        pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype
        )
    elif model_path.exists() and model_path.is_dir():
        # Local model directory
        print(f"Loading from local directory: {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True
        )
    elif "/" in config.model_id:
        # HuggingFace model ID
        print(f"Loading from HuggingFace: {config.model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id, 
            torch_dtype=dtype
        )
    else:
        raise ValueError(f"Invalid model path or ID: {config.model_id}")
    
    pipe = pipe.to(device=config.device, dtype=dtype)
    
    # Create StreamDiffusion instance
    stream = StreamDiffusion(
        pipe,
        t_index_list=config.t_index_list,
        torch_dtype=dtype,
        width=config.width,
        height=config.height,
        cfg_type=config.cfg_type,
    )
    
    # Load LCM LoRA if requested
    if config.use_lcm_lora:
        stream.load_lcm_lora()
        stream.fuse_lora()
    
    # Enable optimizations
    if config.acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    elif config.acceleration == "tensorrt":
        # TensorRT acceleration would need additional setup
        print("TensorRT acceleration requested but not implemented in this example")
    
    # Create ControlNet pipeline
    controlnet_pipeline = ControlNetPipeline(stream, config.device, dtype)
    
    # Add ControlNets
    for cn_config in config.controlnets:
        controlnet_pipeline.add_controlnet(cn_config)
    
    # Prepare with prompt
    if config.prompt:
        stream.prepare(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            seed=config.seed,
        )
    
    return controlnet_pipeline 