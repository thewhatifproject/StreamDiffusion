import numpy as np
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DepthPreprocessor(BasePreprocessor):
    """
    Depth estimation preprocessor for ControlNet using MiDaS
    
    Estimates depth maps from input images using the MiDaS depth estimation model.
    """
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-large",
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize depth preprocessor
        
        Args:
            model_name: Name of the depth estimation model to use
            detect_resolution: Resolution for depth detection
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for depth preprocessing. "
                "Install it with: pip install transformers"
            )
        
        super().__init__(
            model_name=model_name,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            **kwargs
        )
        
        self._depth_estimator = None
    
    @property
    def depth_estimator(self):
        """Lazy loading of the depth estimation model"""
        if self._depth_estimator is None:
            model_name = self.params.get('model_name', 'Intel/dpt-large')
            print(f"Loading depth estimation model: {model_name}")
            self._depth_estimator = pipeline(
                'depth-estimation', 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        return self._depth_estimator
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply depth estimation to the input image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with depth map (grayscale)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize for detection
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        # Estimate depth
        depth_result = self.depth_estimator(image_resized)
        depth_map = depth_result['depth']
        
        # Convert depth map to numpy array
        if hasattr(depth_map, 'cpu'):
            depth_np = depth_map.cpu().numpy()
        else:
            depth_np = np.array(depth_map)
        
        # Normalize depth map to 0-255 range
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        
        # Convert to RGB for ControlNet compatibility
        depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
        result = Image.fromarray(depth_rgb)
        
        # Resize to target resolution
        image_resolution = self.params.get('image_resolution', 512)
        result = result.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return result 