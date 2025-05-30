import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from diffusers.models import ControlNetModel
from diffusers.utils import load_image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, AutoencoderTiny

from ..pipeline import StreamDiffusion
from .config import ControlNetConfig, StreamDiffusionControlNetConfig
from .preprocessors import get_preprocessor


class SDXLTurboControlNetPipeline:
    """
    SDXL Turbo ControlNet pipeline using StreamDiffusion
    
    This class extends StreamDiffusion with ControlNet support for SDXL Turbo,
    using t_index_list for efficient real-time generation at high resolution.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize SDXL Turbo ControlNet pipeline
        
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
        self._temp_tensor_cache = {}
        
        # Cache for preprocessed images to eliminate redundant processing
        self._preprocessed_cache = {}
        self._last_input_frame = None
    
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
        print(f"Loading SDXL Turbo ControlNet: {controlnet_config.model_id}")
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
        
        print(f"Added SDXL Turbo ControlNet {len(self.controlnets) - 1}: {controlnet_config.model_id}")
        return len(self.controlnets) - 1
    
    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> None:
        """
        Efficiently update all ControlNets with cache-aware preprocessing
        
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
        
        # Convert to tensor early for GPU processing
        try:
            control_tensor = self._convert_to_tensor_early(control_image)
            use_tensor_processing = True
        except Exception as e:
            use_tensor_processing = False
            print(f"⚠️  Tensor conversion failed: {e}, falling back to PIL processing")
        
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
            
            # Use tensor processing if available and input is tensor
            has_tensor_processing = hasattr(preprocessor, 'process_tensor')
            using_tensor = use_tensor_processing and has_tensor_processing
            
            if using_tensor:
                processed_image = self._prepare_control_image(control_tensor, preprocessor)
            else:
                processed_image = self._prepare_control_image(control_image, preprocessor)
            
            # Cache the result
            self._preprocessed_cache[preprocessor_type] = processed_image
            
            # Apply to all ControlNets using this preprocessor
            for index in group['indices']:
                self.controlnet_images[index] = processed_image
    
    def update_controlnet_scale(self, index: int, scale: float) -> None:
        """Update the conditioning scale for a specific ControlNet"""
        if 0 <= index < len(self.controlnets):
            self.controlnet_scales[index] = scale
        else:
            raise IndexError(f"ControlNet index {index} out of range")
    
    @property
    def controlnet_configs(self) -> List[Dict[str, Any]]:
        """
        Get ControlNet configurations for compatibility with demos
        
        Returns:
            List of dictionaries containing ControlNet configuration info
        """
        configs = []
        for i in range(len(self.controlnets)):
            configs.append({
                'conditioning_scale': self.controlnet_scales[i] if i < len(self.controlnet_scales) else 1.0,
                'enabled': self.controlnet_scales[i] > 0 if i < len(self.controlnet_scales) else False,
                'preprocessor': type(self.preprocessors[i]).__name__ if i < len(self.preprocessors) and self.preprocessors[i] is not None else None
            })
        return configs
    
    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Get the last processed control image for display purposes"""
        if not (0 <= index < len(self.controlnets)):
            return None
        
        preprocessor = self.preprocessors[index]
        if preprocessor is None:
            return None
            
        preprocessor_type = type(preprocessor).__name__
        cached_result = self._preprocessed_cache.get(preprocessor_type)
        
        if cached_result is None:
            return None
        
        # Handle tensor results from GPU processing
        if isinstance(cached_result, torch.Tensor):
            if hasattr(preprocessor, 'tensor_to_pil'):
                return preprocessor.tensor_to_pil(cached_result)
            else:
                return self._tensor_to_pil_fallback(cached_result)
        
        return cached_result
    
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
                if "/" in model_id and model_id.count("/") > 1:
                    # Handle subfolder case
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
            
            controlnet = controlnet.to(device=self.device, dtype=self.dtype)
            return controlnet
            
        except Exception as e:
            raise ValueError(f"Failed to load SDXL Turbo ControlNet model '{model_id}': {e}")
    
    def _prepare_control_image(self, 
                              control_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
                              preprocessor: Optional[Any] = None) -> torch.Tensor:
        """Prepare a control image for ControlNet input"""
        # Load image if path
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        # Check if we can use tensor processing
        if preprocessor is not None and isinstance(control_image, torch.Tensor) and hasattr(preprocessor, 'process_tensor'):
            try:
                processed_tensor = preprocessor.process_tensor(control_image)
                if processed_tensor.dim() == 3:
                    processed_tensor = processed_tensor.unsqueeze(0)
                processed_tensor = processed_tensor.to(device=self.device, dtype=self.dtype)
                
                # SDXL: Ensure control image matches pipeline resolution
                target_size = (self.stream.height, self.stream.width)  # (H, W)
                current_size = processed_tensor.shape[-2:]  # (H, W)
                if current_size != target_size:
                    import torch.nn.functional as F
                    processed_tensor = F.interpolate(
                        processed_tensor,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    print(f"SDXL ControlNet: Resized control image from {current_size} to {target_size}")
                
                return processed_tensor
            except Exception as e:
                print(f"⚠️  Tensor processing failed for {type(preprocessor).__name__}: {e}")
        
        # Direct tensor path for passthrough
        if isinstance(control_image, torch.Tensor) and preprocessor is None:
            target_size = (self.stream.width, self.stream.height)
            
            if control_image.dim() == 4:
                control_image = control_image[0]
            if control_image.dim() == 3 and control_image.shape[0] not in [1, 3]:
                control_image = control_image.permute(2, 0, 1)
            
            current_size = control_image.shape[-2:]
            if current_size != target_size:
                import torch.nn.functional as F
                if control_image.dim() == 3:
                    control_image = control_image.unsqueeze(0)
                
                control_image = F.interpolate(
                    control_image,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                
                if control_image.shape[0] == 1:
                    control_image = control_image.squeeze(0)
            
            if control_image.dim() == 3:
                control_image = control_image.unsqueeze(0)
            
            control_image = control_image.to(device=self.device, dtype=self.dtype)
            return control_image
        
        # Apply preprocessor if available
        if preprocessor is not None:
            control_image = preprocessor.process(control_image)
        
        # Fast path for PIL Images
        if isinstance(control_image, Image.Image):
            target_size = (self.stream.width, self.stream.height)
            if control_image.size != target_size:
                control_image = control_image.resize(target_size, Image.LANCZOS)
                print(f"SDXL ControlNet: Resized PIL control image to {target_size}")
            
            control_tensor = self._cached_transform(control_image).unsqueeze(0)
            control_tensor = control_tensor.to(device=self.device, dtype=self.dtype)
            return control_tensor
        
        # Handle other types
        if isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                control_image = (control_image * 255).astype(np.uint8)
            control_image = Image.fromarray(control_image)
        elif isinstance(control_image, torch.Tensor):
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
        
        # Ensure final control image matches SDXL resolution
        final_result = self._prepare_control_image(control_image, None)
        
        # Final safety check for SDXL dimensions
        target_size = (self.stream.height, self.stream.width)  # (H, W)
        if final_result.shape[-2:] != target_size:
            import torch.nn.functional as F
            final_result = F.interpolate(
                final_result,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            print(f"SDXL ControlNet: Final resize from {final_result.shape[-2:]} to {target_size}")
        
        return final_result
    
    def _get_controlnet_conditioning(self, 
                                   x_t_latent: torch.Tensor,
                                   timestep: torch.Tensor,
                                   encoder_hidden_states: torch.Tensor,
                                   added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """Get combined conditioning from all active ControlNets"""
        if not self.controlnets:
            return None, None
        
        active_indices = [
            i for i, (controlnet, control_image, scale) in enumerate(
                zip(self.controlnets, self.controlnet_images, self.controlnet_scales)
            ) if controlnet is not None and control_image is not None and scale > 0
        ]
        
        if not active_indices:
            return None, None
        
        down_block_res_samples = None
        mid_block_res_sample = None
        
        for i in active_indices:
            controlnet = self.controlnets[i]
            control_image = self.controlnet_images[i]
            scale = self.controlnet_scales[i]
            
            try:
                # SDXL ControlNet call with added_cond_kwargs for time embeddings
                controlnet_kwargs = {
                    'sample': x_t_latent,
                    'timestep': timestep,
                    'encoder_hidden_states': encoder_hidden_states,
                    'controlnet_cond': control_image,
                    'conditioning_scale': scale,
                    'return_dict': False,
                }
                
                # Add SDXL-specific conditioning through added_cond_kwargs parameter
                if added_cond_kwargs and ('text_embeds' in added_cond_kwargs or 'time_ids' in added_cond_kwargs):
                    controlnet_kwargs['added_cond_kwargs'] = added_cond_kwargs
                
                down_samples, mid_sample = controlnet(**controlnet_kwargs)
                
                if down_block_res_samples is None:
                    down_block_res_samples = down_samples
                    mid_block_res_sample = mid_sample
                else:
                    for j in range(len(down_block_res_samples)):
                        down_block_res_samples[j] += down_samples[j]
                    mid_block_res_sample += mid_sample
                    
            except Exception as e:
                print(f"Warning: SDXL Turbo ControlNet {i} failed: {e}")
                continue
        
        return down_block_res_samples, mid_block_res_sample
    
    def _patch_stream_diffusion(self) -> None:
        """Patch StreamDiffusion's unet_step method to include ControlNet conditioning"""
        if self._is_patched:
            return
        
        self._original_unet_step = self.stream.unet_step
        
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
            
            # Always initialize added_cond_kwargs for SDXL compatibility
            added_cond_kwargs = {}
            
            # Try to get SDXL-specific conditioning from the StreamDiffusion instance
            if hasattr(self.stream, 'add_text_embeds') and hasattr(self.stream, 'add_time_ids'):
                # Handle text embeds expansion for CFG
                if self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "initialize"):
                    add_text_embeds = torch.concat([self.stream.add_text_embeds[0:1], self.stream.add_text_embeds], dim=0)
                    add_time_ids = torch.concat([self.stream.add_time_ids[0:1], self.stream.add_time_ids], dim=0)
                elif self.stream.guidance_scale > 1.0 and (self.stream.cfg_type == "full"):
                    add_text_embeds = torch.concat([self.stream.add_text_embeds, self.stream.add_text_embeds], dim=0)
                    add_time_ids = torch.concat([self.stream.add_time_ids, self.stream.add_time_ids], dim=0)
                else:
                    add_text_embeds = self.stream.add_text_embeds
                    add_time_ids = self.stream.add_time_ids
                
                added_cond_kwargs = {
                    'text_embeds': add_text_embeds,
                    'time_ids': add_time_ids
                }
            elif hasattr(self.stream.pipe, 'text_encoder_2'):
                # SDXL pipeline detected but embeddings not extracted yet
                # Create default SDXL conditioning
                batch_size = x_t_latent_plus_uc.shape[0]
                
                # Create default pooled embeddings (will be overridden by proper prompt preparation)
                device = self.stream.device
                dtype = self.stream.dtype
                
                # Default time_ids for SDXL (original_size, crops_coords_top_left, target_size)
                time_ids = torch.tensor([
                    [self.stream.height, self.stream.width, 0, 0, self.stream.height, self.stream.width]
                ], dtype=dtype, device=device)
                time_ids = time_ids.repeat(batch_size, 1)
                
                # Use dummy text embeds if not available
                text_embeds = torch.zeros((batch_size, 1280), dtype=dtype, device=device)
                
                added_cond_kwargs = {
                    'text_embeds': text_embeds,
                    'time_ids': time_ids
                }
            
            # Get ControlNet conditioning
            down_block_res_samples, mid_block_res_sample = self._get_controlnet_conditioning(
                x_t_latent_plus_uc, t_list_expanded, self.stream.prompt_embeds, added_cond_kwargs
            )
            
            # Prepare UNet kwargs
            unet_kwargs = {
                'sample': x_t_latent_plus_uc,
                'timestep': t_list_expanded,
                'encoder_hidden_states': self.stream.prompt_embeds,
                'return_dict': False,
            }
            
            # Add ControlNet conditioning
            if down_block_res_samples is not None:
                unet_kwargs['down_block_additional_residuals'] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs['mid_block_additional_residual'] = mid_block_res_sample
            
            # Add SDXL-specific conditioning through added_cond_kwargs parameter
            if added_cond_kwargs:
                unet_kwargs['added_cond_kwargs'] = added_cond_kwargs
            
            # Call UNet
            model_pred = self.stream.unet(**unet_kwargs)[0]
            
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
        
        self.stream.unet_step = patched_unet_step
        self._is_patched = True
        print("Patched StreamDiffusion with SDXL Turbo ControlNet support")
    
    def _unpatch_stream_diffusion(self) -> None:
        """Restore original StreamDiffusion unet_step method"""
        if self._is_patched and self._original_unet_step is not None:
            self.stream.unet_step = self._original_unet_step
            self._is_patched = False
            print("Unpatched StreamDiffusion")
    
    def _convert_to_tensor_early(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor as early as possible"""
        if isinstance(control_image, str):
            control_image = load_image(control_image)
        
        if isinstance(control_image, Image.Image):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            tensor = to_tensor(control_image)
            return tensor.to(device=self.device, dtype=self.dtype)
        
        elif isinstance(control_image, np.ndarray):
            if control_image.max() <= 1.0:
                tensor = torch.from_numpy(control_image).float()
            else:
                tensor = torch.from_numpy(control_image).float() / 255.0
            
            if len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3]:
                tensor = tensor.permute(2, 0, 1)
            elif len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)
                
            return tensor.to(device=self.device, dtype=self.dtype)
        
        elif isinstance(control_image, torch.Tensor):
            return control_image.to(device=self.device, dtype=self.dtype)
        
        else:
            raise ValueError(f"Unsupported control image type: {type(control_image)}")
    
    def _tensor_to_pil_fallback(self, tensor: torch.Tensor) -> Image.Image:
        """Fallback method to convert tensor to PIL"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        else:
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        
        array = tensor.numpy()
        
        if array.shape[-1] == 3:
            return Image.fromarray(array, 'RGB')
        elif array.shape[-1] == 1:
            return Image.fromarray(array.squeeze(-1), 'L')
        else:
            return Image.fromarray(array)
    
    def _setup_sdxl_embeddings(self) -> None:
        """Extract and store SDXL-specific embeddings if using SDXL pipeline"""
        if not hasattr(self.stream.pipe, 'text_encoder_2'):
            return  # Not an SDXL pipeline
        
        # Check if embeddings are already set up
        if hasattr(self.stream, 'add_text_embeds') and hasattr(self.stream, 'add_time_ids'):
            return
        
        try:
            # Get the current prompt from the stream
            prompt = getattr(self.stream, '_current_prompt', "")
            negative_prompt = getattr(self.stream, '_current_negative_prompt', "")
            
            # Use a simple prompt if we don't have one
            if not prompt:
                prompt = "a photo"
            
            # Get SDXL embeddings using the pipeline's encode_prompt method
            do_classifier_free_guidance = self.stream.guidance_scale > 1.0
            
            # Call the SDXL pipeline's encode_prompt method which returns all necessary embeddings
            encoder_output = self.stream.pipe.encode_prompt(
                prompt=prompt,
                device=self.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            
            # Extract embeddings from the output
            if len(encoder_output) >= 4:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoder_output[:4]
                
                # Store the pooled embeddings as add_text_embeds
                if do_classifier_free_guidance:
                    # For CFG, we need both negative and positive pooled embeddings
                    self.stream.add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    self.stream.add_text_embeds = pooled_prompt_embeds
                
                # Create default time_ids for SDXL
                original_size = (self.stream.height, self.stream.width)
                target_size = (self.stream.height, self.stream.width)
                crops_coords_top_left = (0, 0)
                
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], dtype=self.stream.dtype, device=self.stream.device)
                
                if do_classifier_free_guidance:
                    # For CFG, we need time_ids for both uncond and cond
                    self.stream.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                else:
                    self.stream.add_time_ids = add_time_ids
                
                print("Set up SDXL embeddings for ControlNet pipeline")
                
        except Exception as e:
            print(f"Warning: Failed to set up SDXL embeddings: {e}")
            # Create minimal fallback embeddings
            batch_size = 2 if self.stream.guidance_scale > 1.0 else 1
            self.stream.add_text_embeds = torch.zeros((batch_size, 1280), dtype=self.stream.dtype, device=self.stream.device)
            
            add_time_ids = torch.tensor([
                [self.stream.height, self.stream.width, 0, 0, self.stream.height, self.stream.width]
            ], dtype=self.stream.dtype, device=self.stream.device)
            self.stream.add_time_ids = add_time_ids.repeat(batch_size, 1)
    
    def __call__(self, 
                 image: Union[str, Image.Image, np.ndarray, torch.Tensor] = None,
                 strength: float = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 **kwargs) -> Image.Image:
        """
        Generate image using SDXL Turbo with ControlNet
        
        Args:
            image: Input image for img2img
            strength: Ignored (StreamDiffusion doesn't use this)
            num_inference_steps: Ignored (StreamDiffusion doesn't use this)
            guidance_scale: Ignored (StreamDiffusion doesn't use this)
            **kwargs: Additional arguments
            
        Returns:
            Generated PIL Image
        """
        # Ensure SDXL embeddings are set up
        self._setup_sdxl_embeddings()
        
        if image is not None:
            # Update control image for ControlNet
            self.update_control_image_efficient(image)
            
            # Call StreamDiffusion with the image as positional argument 'x'
            x_output = self.stream(image)  # StreamDiffusion expects 'image' as positional 'x'
        else:
            # Text-to-image generation
            x_output = self.stream()
        
        # Convert tensor output to PIL Image
        from ..image_utils import postprocess_image
        output_image = postprocess_image(x_output, output_type="pil")[0]
        return output_image
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying StreamDiffusion instance"""
        return getattr(self.stream, name)


def create_sdxlturbo_controlnet_pipeline(config: StreamDiffusionControlNetConfig) -> SDXLTurboControlNetPipeline:
    """
    Create an SDXL Turbo ControlNet pipeline from configuration using StreamDiffusion
    
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
    
    # Load base pipeline
    print(f"Loading SDXL Turbo base model: {config.model_id}")
    
    # Check if it's a local file path
    model_path = Path(config.model_id)
    if model_path.exists() and model_path.is_file():
        print(f"Loading from local file: {model_path}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype
        )
    elif model_path.exists() and model_path.is_dir():
        print(f"Loading from local directory: {model_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True
        )
    elif "/" in config.model_id:
        print(f"Loading from HuggingFace: {config.model_id}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            config.model_id, 
            torch_dtype=dtype
        )
    else:
        raise ValueError(f"Invalid model path or ID: {config.model_id}")
    
    pipe = pipe.to(device=config.device, dtype=dtype)
    
    # Load improved VAE for SDXL
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=dtype
    ).to(config.device)
    pipe.vae = vae
    
    # Use Tiny AutoEncoder XL if requested
    if getattr(config, 'use_taesd', True):
        taesd_model = "madebyollin/taesdxl"
        pipe.vae = AutoencoderTiny.from_pretrained(
            taesd_model, 
            torch_dtype=dtype, 
            use_safetensors=True
        ).to(config.device)
    
    # SDXL Turbo uses its default scheduler - don't override it
    
    # Create StreamDiffusion instance
    stream = StreamDiffusion(
        pipe,
        t_index_list=config.t_index_list,
        torch_dtype=dtype,
        width=config.width,
        height=config.height,
        cfg_type=config.cfg_type,
    )
    
    # Enable optimizations
    if config.acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    
    # Create ControlNet pipeline
    controlnet_pipeline = SDXLTurboControlNetPipeline(stream, config.device, dtype)
    
    # Add ControlNets
    for cn_config in config.controlnets:
        controlnet_pipeline.add_controlnet(cn_config)
    
    # Prepare with prompt
    if config.prompt:
        # Store prompt info for SDXL embeddings setup
        stream._current_prompt = config.prompt
        stream._current_negative_prompt = config.negative_prompt or ""
        
        stream.prepare(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            seed=config.seed,
        )
        
        # Set up SDXL embeddings after prepare
        controlnet_pipeline._setup_sdxl_embeddings()
    
    return controlnet_pipeline 