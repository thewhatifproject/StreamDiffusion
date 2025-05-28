from .base import BasePreprocessor
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor
from .openpose import OpenPosePreprocessor
from .lineart import LineartPreprocessor
from .passthrough import PassthroughPreprocessor

# Try to import TensorRT preprocessor - might not be available on all systems
try:
    from .depth_tensorrt import DepthAnythingTensorrtPreprocessor
    TENSORRT_AVAILABLE = True
except ImportError:
    DepthAnythingTensorrtPreprocessor = None
    TENSORRT_AVAILABLE = False

# Registry for easy lookup
_preprocessor_registry = {
    "canny": CannyPreprocessor,
    "depth": DepthPreprocessor,
    "openpose": OpenPosePreprocessor,
    "lineart": LineartPreprocessor,
    "passthrough": PassthroughPreprocessor,
}

# Add TensorRT preprocessor if available
if TENSORRT_AVAILABLE:
    _preprocessor_registry["depth_tensorrt"] = DepthAnythingTensorrtPreprocessor


def get_preprocessor(name: str) -> BasePreprocessor:
    """
    Get a preprocessor by name
    
    Args:
        name: Name of the preprocessor
        
    Returns:
        Preprocessor instance
        
    Raises:
        ValueError: If preprocessor name is not found
    """
    if name not in _preprocessor_registry:
        available = ", ".join(_preprocessor_registry.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")
    
    return _preprocessor_registry[name]()


def register_preprocessor(name: str, preprocessor_class):
    """
    Register a new preprocessor
    
    Args:
        name: Name to register under
        preprocessor_class: Preprocessor class
    """
    _preprocessor_registry[name] = preprocessor_class


def list_preprocessors():
    """List all available preprocessors"""
    return list(_preprocessor_registry.keys())


__all__ = [
    "BasePreprocessor",
    "CannyPreprocessor",
    "DepthPreprocessor", 
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "PassthroughPreprocessor",
    "get_preprocessor",
    "register_preprocessor",
    "list_preprocessors",
]

if TENSORRT_AVAILABLE:
    __all__.append("DepthAnythingTensorrtPreprocessor") 