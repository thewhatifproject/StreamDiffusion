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