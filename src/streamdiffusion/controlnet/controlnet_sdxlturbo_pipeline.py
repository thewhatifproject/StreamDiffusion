import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np

from ..pipeline import StreamDiffusion
from .base_controlnet_pipeline import BaseControlNetPipeline

# TODO: This broke during refactor. See commit 8c7aff1200e7eaa01a2c02c3e65680ed13c9275f for original implementation and see what went wrong. 
class SDXLTurboControlNetPipeline(BaseControlNetPipeline):
    """
    SDXL Turbo ControlNet pipeline using StreamDiffusion
    
    This class extends StreamDiffusion with ControlNet support for SDXL Turbo,
    using t_index_list for efficient real-time generation at high resolution.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SDXL Turbo"):
        """
        Initialize SDXL Turbo ControlNet pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run ControlNets on
            dtype: Data type for ControlNet models
            model_type: Type of model being used (e.g., "SDXL Turbo")
        """
        super().__init__(stream_diffusion, device, dtype)
        self.model_type = model_type

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

    def _post_process_control_image(self, control_tensor: torch.Tensor) -> torch.Tensor:
        """
        Post-process control image tensor with SDXL-specific handling
        
        Args:
            control_tensor: Processed control image tensor
            
        Returns:
            Final control image tensor
        """
        # SDXL: Ensure control image matches pipeline resolution
        target_size = (self.stream.height, self.stream.width)  # (H, W)
        current_size = control_tensor.shape[-2:]  # (H, W)
        if current_size != target_size:
            import torch.nn.functional as F
            control_tensor = F.interpolate(
                control_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return control_tensor

    def _get_conditioning_context(self, x_t_latent: torch.Tensor, t_list: torch.Tensor) -> Dict[str, Any]:
        """
        Get SDXL-specific conditioning context
        
        Args:
            x_t_latent: Latent input
            t_list: Timestep list
            
        Returns:
            Dictionary of SDXL conditioning context
        """
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
            batch_size = x_t_latent.shape[0]
            
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
        
        return added_cond_kwargs

    def _get_additional_controlnet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Get SDXL-specific additional kwargs for ControlNet calls
        
        Returns:
            Dictionary of additional kwargs
        """
        # Add SDXL-specific conditioning through added_cond_kwargs parameter
        if 'text_embeds' in kwargs or 'time_ids' in kwargs:
            return {'added_cond_kwargs': kwargs}
        return {}

    def _get_additional_unet_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Get SDXL-specific additional kwargs for UNet calls
        
        Returns:
            Dictionary of additional kwargs
        """
        # Add SDXL-specific conditioning through added_cond_kwargs parameter
        if kwargs:
            return {'added_cond_kwargs': kwargs}
        return {}

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
            # Create minimal embeddings
            batch_size = 2 if self.stream.guidance_scale > 1.0 else 1
            self.stream.add_text_embeds = torch.zeros((batch_size, 1280), dtype=self.stream.dtype, device=self.stream.device)
            
            add_time_ids = torch.tensor([
                [self.stream.height, self.stream.width, 0, 0, self.stream.height, self.stream.width]
            ], dtype=self.stream.dtype, device=self.stream.device)
            self.stream.add_time_ids = add_time_ids.repeat(batch_size, 1)

    def __call__(self, 
                 image: Union[str, Image.Image, np.ndarray, torch.Tensor] = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 **kwargs) -> torch.Tensor:
        """
        Generate image using SDXL Turbo with ControlNet
        
        Args:
            image: Input image for img2img
            num_inference_steps: Ignored (StreamDiffusion doesn't use this)
            guidance_scale: Ignored (StreamDiffusion doesn't use this)
            **kwargs: Additional arguments
            
        Returns:
            Generated tensor (raw output from StreamDiffusion)
        """
        # Ensure SDXL embeddings are set up
        self._setup_sdxl_embeddings()
        
        if image is not None:
            # Update control image for ControlNet
            self.update_control_image_efficient(image)
            
            # Call StreamDiffusion with the image as positional argument 'x'
            return self.stream(image)
        else:
            # Text-to-image generation
            return self.stream() 