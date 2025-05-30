import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path

from diffusers import StableDiffusionPipeline

from ..pipeline import StreamDiffusion
from .config import ControlNetConfig, StreamDiffusionControlNetConfig
from .base_controlnet_pipeline import BaseControlNetPipeline


class ControlNetPipeline(BaseControlNetPipeline):
    """
    ControlNet-enabled StreamDiffusion pipeline for SD1.5
    
    This class extends StreamDiffusion with ControlNet support, allowing for
    conditioning the generation process with multiple ControlNet models.
    """
    
    def _get_model_type(self) -> str:
        """Return the model type string for logging purposes"""
        return "SD1.5"


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