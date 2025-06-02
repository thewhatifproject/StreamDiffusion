from .base_controlnet_pipeline import BaseControlNetPipeline
from .controlnet_pipeline import ControlNetPipeline, create_controlnet_pipeline, create_sdturbo_controlnet_pipeline
from .controlnet_sdxlturbo_pipeline import SDXLTurboControlNetPipeline, create_sdxlturbo_controlnet_pipeline
from .config import ControlNetConfig, StreamDiffusionControlNetConfig, load_controlnet_config
from .preprocessors import (
    BasePreprocessor,
    CannyPreprocessor,
    DepthPreprocessor,
    OpenPosePreprocessor,
    LineartPreprocessor,
    get_preprocessor,
)


def create_controlnet_pipeline_auto(config: StreamDiffusionControlNetConfig):
    """
    Automatically create the appropriate ControlNet pipeline based on configuration
    
    Args:
        config: Configuration object with pipeline_type specified
        
    Returns:
        ControlNet pipeline instance (ControlNetPipeline or SDXLTurboControlNetPipeline)
    """
    pipeline_type = getattr(config, 'pipeline_type', 'sd1.5').lower()
    
    if pipeline_type == 'sdxlturbo':
        print("Creating SD-XL Turbo ControlNet pipeline")
        return create_sdxlturbo_controlnet_pipeline(config)
    elif pipeline_type == 'sdturbo':
        print("Creating SD Turbo ControlNet pipeline")
        return create_sdturbo_controlnet_pipeline(config)
    elif pipeline_type == 'sd1.5':
        print("Creating SD 1.5 ControlNet pipeline")
        return create_controlnet_pipeline(config)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}. Supported types: 'sd1.5', 'sdturbo', 'sdxlturbo'")


__all__ = [
    # Base pipeline class  
    "BaseControlNetPipeline",
    
    # Pipeline classes
    "ControlNetPipeline",
    "SDXLTurboControlNetPipeline",
    
    # Pipeline creation functions
    "create_controlnet_pipeline",
    "create_sdturbo_controlnet_pipeline",
    "create_sdxlturbo_controlnet_pipeline", 
    "create_controlnet_pipeline_auto",
    
    # Configuration classes
    "ControlNetConfig",
    "StreamDiffusionControlNetConfig", 
    "load_controlnet_config",
    
    # Preprocessors
    "BasePreprocessor",
    "CannyPreprocessor", 
    "DepthPreprocessor",
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "get_preprocessor",
] 