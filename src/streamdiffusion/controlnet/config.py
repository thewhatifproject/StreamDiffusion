import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ControlNetConfig:
    """Configuration class for ControlNet models and preprocessing"""
    
    model_id: str
    """ControlNet model ID (HuggingFace) or local path"""
    
    conditioning_scale: float = 1.0
    """Strength of the ControlNet conditioning (0.0 to 5.0)"""
    
    preprocessor: Optional[str] = None
    """Name of the preprocessor to use (e.g., 'canny', 'depth', 'openpose')"""
    
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    """Parameters for the preprocessor"""
    
    control_image_path: Optional[str] = None
    """Path to control image (if using static image)"""
    
    enabled: bool = True
    """Whether this ControlNet is enabled"""


@dataclass
class StreamDiffusionControlNetConfig:
    """Complete configuration for StreamDiffusion with ControlNet support"""
    
    # Base StreamDiffusion parameters
    model_id: str
    """Base model ID or path"""
    
    t_index_list: List[int] = field(default_factory=lambda: [32, 45])
    """Time step indices for denoising"""
    
    width: int = 512
    height: int = 512
    
    device: str = "cuda"
    dtype: str = "float16"
    
    # ControlNet configurations
    controlnets: List[ControlNetConfig] = field(default_factory=list)
    """List of ControlNet configurations"""
    
    # Generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 1.2
    num_inference_steps: int = 50
    
    # Advanced parameters
    use_lcm_lora: bool = True
    use_tiny_vae: bool = True
    acceleration: str = "tensorrt"
    cfg_type: str = "self"
    seed: int = 2


def load_controlnet_config(config_path: Union[str, Path]) -> StreamDiffusionControlNetConfig:
    """
    Load ControlNet configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        StreamDiffusionControlNetConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Convert ControlNet configurations
    controlnets = []
    for cn_config in config_data.get('controlnets', []):
        controlnets.append(ControlNetConfig(**cn_config))
    
    config_data['controlnets'] = controlnets
    
    return StreamDiffusionControlNetConfig(**config_data)


def save_controlnet_config(config: StreamDiffusionControlNetConfig, 
                          config_path: Union[str, Path]) -> None:
    """
    Save ControlNet configuration to YAML or JSON file
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    
    # Convert to dictionary
    config_dict = {
        'model_id': config.model_id,
        't_index_list': config.t_index_list,
        'width': config.width,
        'height': config.height,
        'device': config.device,
        'dtype': config.dtype,
        'prompt': config.prompt,
        'negative_prompt': config.negative_prompt,
        'guidance_scale': config.guidance_scale,
        'num_inference_steps': config.num_inference_steps,
        'use_lcm_lora': config.use_lcm_lora,
        'use_tiny_vae': config.use_tiny_vae,
        'acceleration': config.acceleration,
        'cfg_type': config.cfg_type,
        'seed': config.seed,
        'controlnets': [
            {
                'model_id': cn.model_id,
                'conditioning_scale': cn.conditioning_scale,
                'preprocessor': cn.preprocessor,
                'preprocessor_params': cn.preprocessor_params,
                'control_image_path': cn.control_image_path,
                'enabled': cn.enabled,
            }
            for cn in config.controlnets
        ]
    }
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_example_configs(output_dir: Union[str, Path]) -> None:
    """
    Create example configuration files for common ControlNet setups
    
    Args:
        output_dir: Directory to save example configurations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single Canny ControlNet example
    canny_config = StreamDiffusionControlNetConfig(
        model_id="stabilityai/sd-turbo",
        prompt="a beautiful landscape, highly detailed",
        controlnets=[
            ControlNetConfig(
                model_id="lllyasviel/control_v11p_sd15_canny",
                conditioning_scale=1.0,
                preprocessor="canny",
                preprocessor_params={"low_threshold": 100, "high_threshold": 200}
            )
        ]
    )
    save_controlnet_config(canny_config, output_dir / "canny_example.yaml")
    
    # Multi-ControlNet example
    multi_config = StreamDiffusionControlNetConfig(
        model_id="stabilityai/sd-turbo",
        prompt="a person standing in a room, photorealistic",
        controlnets=[
            ControlNetConfig(
                model_id="lllyasviel/control_v11p_sd15_openpose",
                conditioning_scale=0.8,
                preprocessor="openpose"
            ),
            ControlNetConfig(
                model_id="lllyasviel/control_v11f1p_sd15_depth",
                conditioning_scale=0.6,
                preprocessor="depth"
            )
        ]
    )
    save_controlnet_config(multi_config, output_dir / "multi_controlnet_example.yaml")
    
    # Depth-based ControlNet example
    depth_config = StreamDiffusionControlNetConfig(
        model_id="stabilityai/sd-turbo",
        prompt="a futuristic cityscape, cyberpunk style",
        guidance_scale=1.5,
        controlnets=[
            ControlNetConfig(
                model_id="lllyasviel/control_v11f1p_sd15_depth",
                conditioning_scale=1.2,
                preprocessor="depth",
                preprocessor_params={"detect_resolution": 512, "image_resolution": 512}
            )
        ]
    )
    save_controlnet_config(depth_config, output_dir / "depth_example.yaml")
    
    print(f"Example configurations saved to {output_dir}") 