import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from diffusers import StableDiffusionPipeline, LCMScheduler, AutoencoderTiny

from ..pipeline import StreamDiffusion
from .config import ControlNetConfig, StreamDiffusionControlNetConfig
from .base_controlnet_pipeline import BaseControlNetPipeline


class ControlNetPipeline(BaseControlNetPipeline):
    """
    ControlNet-enabled StreamDiffusion pipeline for SD1.5 and SD Turbo
    
    This class extends StreamDiffusion with ControlNet support, allowing for
    conditioning the generation process with multiple ControlNet models.
    Supports both SD1.5 and SD Turbo models.
    """
    
    def __init__(self, 
                 stream_diffusion: StreamDiffusion,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 model_type: str = "SD1.5"):
        """
        Initialize ControlNet pipeline
        
        Args:
            stream_diffusion: Base StreamDiffusion instance
            device: Device to run ControlNets on
            dtype: Data type for ControlNet models
            model_type: Type of model being used (e.g., "SD1.5", "SD Turbo")
        """
        super().__init__(stream_diffusion, device, dtype)
        self.model_type = model_type


def create_controlnet_pipeline(config: StreamDiffusionControlNetConfig) -> ControlNetPipeline:
    """
    Create a ControlNet-enabled StreamDiffusion pipeline from configuration
    
    Args:
        config: Configuration object
        
    Returns:
        ControlNetPipeline instance
    """
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)
    
    # Determine model type from pipeline_type
    model_type = "SD1.5" if config.pipeline_type == "sd1.5" else "SD Turbo"
    
    # Load base pipeline
    print(f"Loading {model_type} base model: {config.model_id}")
    
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
    
    # Apply model-specific configurations
    if config.pipeline_type == "sdturbo":
        # SD Turbo specific setup
        # Use Tiny AutoEncoder if requested
        if getattr(config, 'use_taesd', True):
            taesd_model = "madebyollin/taesd"
            pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, 
                torch_dtype=dtype, 
                use_safetensors=True
            ).to(config.device)
        
        # Set LCM scheduler for SD Turbo
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        # SD1.5 specific setup
        # Load LCM LoRA if requested
        if config.use_lcm_lora:
            stream = StreamDiffusion(
                pipe,
                t_index_list=config.t_index_list,
                torch_dtype=dtype,
                width=config.width,
                height=config.height,
                cfg_type=config.cfg_type,
            )
            stream.load_lcm_lora()
            stream.fuse_lora()
            # Recreate pipe after LoRA fusion
            pipe = stream.pipe
    
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
    elif config.acceleration == "tensorrt":
        # TensorRT acceleration would need additional setup
        print("TensorRT acceleration requested but not implemented in this example")
    
    # Create ControlNet pipeline with model type
    controlnet_pipeline = ControlNetPipeline(stream, config.device, dtype, model_type)
    
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


def create_sdturbo_controlnet_pipeline(config: StreamDiffusionControlNetConfig) -> ControlNetPipeline:
    """
    Create an SD Turbo ControlNet pipeline from configuration using StreamDiffusion
    
    Args:
        config: Configuration object
        
    Returns:
        ControlNetPipeline instance configured for SD Turbo
    """
    # Ensure pipeline_type is set to sdturbo
    config.pipeline_type = "sdturbo"
    
    # Use the unified creation function
    return create_controlnet_pipeline(config) 