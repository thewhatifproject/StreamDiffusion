import numpy as np
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor


class PassthroughPreprocessor(BasePreprocessor):
    """
    Passthrough preprocessor for ControlNet
    
    Simply passes the input image through without any processing.
    Useful for ControlNets that expect the raw input image, such as:
    - Tile ControlNet
    - Reference ControlNet
    - Custom ControlNets that don't need preprocessing
    """
    
    def __init__(self, 
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize passthrough preprocessor
        
        Args:
            image_resolution: Output image resolution
            **kwargs: Additional parameters (ignored for passthrough)
        """
        super().__init__(
            image_resolution=image_resolution,
            **kwargs
        )
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Pass through the input image with minimal processing
        
        Args:
            image: Input image
            
        Returns:
            PIL Image (resized to target resolution if needed)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Resize to target resolution if specified
        image_resolution = self.params.get('image_resolution', 512)
        if image.size != (image_resolution, image_resolution):
            image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return image 