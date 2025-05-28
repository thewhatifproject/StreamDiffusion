import cv2
import numpy as np
from PIL import Image
from typing import Union
from .base import BasePreprocessor


class CannyPreprocessor(BasePreprocessor):
    """
    Canny edge detection preprocessor for ControlNet
    
    Detects edges in the input image using the Canny edge detection algorithm.
    """
    
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, **kwargs):
        """
        Initialize Canny preprocessor
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            **kwargs: Additional parameters
        """
        super().__init__(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            **kwargs
        )
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply Canny edge detection to the input image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected edges (black and white)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply Canny edge detection
        low_threshold = self.params.get('low_threshold', 100)
        high_threshold = self.params.get('high_threshold', 200)
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL Image (RGB format for consistency)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = Image.fromarray(edges_rgb)
        
        return result 