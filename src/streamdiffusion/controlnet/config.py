import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


def load_controlnet_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load ControlNet configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"load_controlnet_config: Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"load_controlnet_config: Unsupported configuration file format: {config_path.suffix}")
    
    # Basic validation
    _validate_config(config_data)
    
    return config_data


def save_controlnet_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save ControlNet configuration to YAML or JSON file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    
    # Basic validation before saving
    _validate_config(config)
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"save_controlnet_config: Unsupported configuration file format: {config_path.suffix}")


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Basic validation of configuration dictionary
    
    Args:
        config: Configuration to validate
    """
    if not isinstance(config, dict):
        raise ValueError("_validate_config: Configuration must be a dictionary")
    
    # Check required fields
    required_fields = ['model_id', 'controlnets']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"_validate_config: Missing required field: {field}")
    
    # Validate controlnets list
    if not isinstance(config['controlnets'], list):
        raise ValueError("_validate_config: 'controlnets' must be a list")
    
    for i, controlnet in enumerate(config['controlnets']):
        if not isinstance(controlnet, dict):
            raise ValueError(f"_validate_config: ControlNet {i} must be a dictionary")
        
        if 'model_id' not in controlnet:
            raise ValueError(f"_validate_config: ControlNet {i} missing required 'model_id'")


def create_example_configs(output_dir: Union[str, Path]) -> None:
    """
    Create example configuration files for common ControlNet setups
    
    Args:
        output_dir: Directory to save example configurations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single Canny ControlNet example (SD 1.5)
    canny_config = {
        'model_id': "runwayml/stable-diffusion-v1-5",
        't_index_list': [32, 45],
        'width': 512,
        'height': 512,
        'device': "cuda",
        'dtype': "float16",
        'pipeline_type': "sd1.5",
        'prompt': "a beautiful landscape, highly detailed",
        'negative_prompt': "",
        'guidance_scale': 1.2,
        'num_inference_steps': 50,
        'frame_buffer_size': 1,
        'delta': 1.0,
        'use_taesd': True,
        'safety_checker': False,
        'use_lcm_lora': True,
        'use_tiny_vae': True,
        'acceleration': "tensorrt",
        'cfg_type': "self",
        'seed': 2,
        'use_denoising_batch': True,
        'controlnets': [
            {
                'model_id': "lllyasviel/control_v11p_sd15_canny",
                'conditioning_scale': 1.0,
                'preprocessor': "canny",
                'preprocessor_params': {"low_threshold": 100, "high_threshold": 200},
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0
            }
        ]
    }
    save_controlnet_config(canny_config, output_dir / "canny_sd15_example.yaml")
    
    # SD Turbo Canny ControlNet example
    sdturbo_canny_config = {
        'model_id': "stabilityai/sd-turbo",
        't_index_list': [32, 45],
        'width': 512,
        'height': 512,
        'device': "cuda",
        'dtype': "float16",
        'pipeline_type': "sdturbo",
        'prompt': "a futuristic robot, highly detailed, cyberpunk style",
        'negative_prompt': "",
        'guidance_scale': 0.0,
        'num_inference_steps': 1,
        'frame_buffer_size': 1,
        'delta': 1.0,
        'use_taesd': True,
        'safety_checker': False,
        'use_lcm_lora': True,
        'use_tiny_vae': True,
        'acceleration': "tensorrt",
        'cfg_type': "self",
        'seed': 2,
        'use_denoising_batch': True,
        'controlnets': [
            {
                'model_id': "lllyasviel/control_v11p_sd15_canny",
                'conditioning_scale': 0.8,
                'preprocessor': "canny",
                'preprocessor_params': {"low_threshold": 50, "high_threshold": 100},
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0
            }
        ]
    }
    save_controlnet_config(sdturbo_canny_config, output_dir / "sdturbo_canny_example.yaml")
    
    # SD Turbo Depth ControlNet example
    sdturbo_depth_config = {
        'model_id': "stabilityai/sd-turbo",
        't_index_list': [32, 45],
        'width': 512,
        'height': 512,
        'device': "cuda",
        'dtype': "float16",
        'pipeline_type': "sdturbo",
        'prompt': "a modern living room with sleek furniture",
        'negative_prompt': "",
        'guidance_scale': 0.0,
        'num_inference_steps': 1,
        'frame_buffer_size': 1,
        'delta': 1.0,
        'use_taesd': True,
        'safety_checker': False,
        'use_lcm_lora': True,
        'use_tiny_vae': True,
        'acceleration': "tensorrt",
        'cfg_type': "self",
        'seed': 2,
        'use_denoising_batch': True,
        'controlnets': [
            {
                'model_id': "lllyasviel/control_v11f1p_sd15_depth",
                'conditioning_scale': 0.9,
                'preprocessor': "depth",
                'preprocessor_params': {"detect_resolution": 384, "image_resolution": 512},
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 0.8
            }
        ]
    }
    save_controlnet_config(sdturbo_depth_config, output_dir / "sdturbo_depth_example.yaml")
    
    # Multi-ControlNet example (SD 1.5)
    multi_config = {
        'model_id': "runwayml/stable-diffusion-v1-5",
        't_index_list': [32, 45],
        'width': 512,
        'height': 512,
        'device': "cuda",
        'dtype': "float16",
        'pipeline_type': "sd1.5",
        'prompt': "a person standing in a room, photorealistic",
        'negative_prompt': "",
        'guidance_scale': 1.2,
        'num_inference_steps': 50,
        'frame_buffer_size': 1,
        'delta': 1.0,
        'use_taesd': True,
        'safety_checker': False,
        'use_lcm_lora': True,
        'use_tiny_vae': True,
        'acceleration': "tensorrt",
        'cfg_type': "self",
        'seed': 2,
        'use_denoising_batch': True,
        'controlnets': [
            {
                'model_id': "lllyasviel/control_v11p_sd15_openpose",
                'conditioning_scale': 0.8,
                'preprocessor': "openpose",
                'preprocessor_params': {},
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0
            },
            {
                'model_id': "lllyasviel/control_v11f1p_sd15_depth",
                'conditioning_scale': 0.6,
                'preprocessor': "depth",
                'preprocessor_params': {},
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0
            }
        ]
    }
    save_controlnet_config(multi_config, output_dir / "multi_controlnet_sd15_example.yaml")
    
    print(f"create_example_configs: Example configurations saved to {output_dir}")
    print("Available configurations:")
    print("  - canny_sd15_example.yaml: SD 1.5 with Canny ControlNet")
    print("  - sdturbo_canny_example.yaml: SD Turbo with Canny ControlNet")
    print("  - sdturbo_depth_example.yaml: SD Turbo with Depth ControlNet")
    print("  - multi_controlnet_sd15_example.yaml: SD 1.5 with multiple ControlNets")


# For backwards compatibility, provide simple functions that match expected usage patterns
def get_controlnet_config(config_dict: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Get a specific ControlNet configuration by index
    
    Args:
        config_dict: Full configuration dictionary
        index: Index of the ControlNet to get
        
    Returns:
        ControlNet configuration dictionary
    """
    if 'controlnets' not in config_dict or index >= len(config_dict['controlnets']):
        raise IndexError(f"get_controlnet_config: ControlNet index {index} out of range")
    
    return config_dict['controlnets'][index]


def get_pipeline_type(config_dict: Dict[str, Any]) -> str:
    """
    Get pipeline type from configuration, with fallback to SD 1.5
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Pipeline type string
    """
    return config_dict.get('pipeline_type', 'sd1.5') 