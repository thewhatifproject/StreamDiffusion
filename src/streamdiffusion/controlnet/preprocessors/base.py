from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import torch
import numpy as np
from PIL import Image


class BasePreprocessor(ABC):
    """
    Base class for ControlNet preprocessors
    
    All preprocessors should inherit from this class and implement the process method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the preprocessor
        
        Args:
            **kwargs: Preprocessor-specific parameters
        """
        self.params = kwargs
    
    @abstractmethod
    def process(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Process an image for ControlNet input
        
        Args:
            image: Input image in PIL, numpy array, or torch tensor format
            
        Returns:
            Processed PIL Image suitable for ControlNet conditioning
        """
        pass
    
    def validate_input(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Convert input to PIL Image for processing
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            if image.dim() == 4:  # Batch dimension
                image = image[0]  # Take first image from batch
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = image.permute(1, 2, 0)  # Convert to HWC
            
            # Convert to numpy and scale to 0-255
            image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        if isinstance(image, np.ndarray):
            # Ensure uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Convert to PIL Image
            if len(image.shape) == 3:
                image = Image.fromarray(image, 'RGB')
            else:
                image = Image.fromarray(image, 'L')
                
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        return image
    
    def resize_image(self, image: Image.Image, target_width: int = 512, target_height: int = 512) -> Image.Image:
        """
        Resize image to target dimensions
        
        Args:
            image: PIL Image to resize
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized PIL Image
        """
        return image.resize((target_width, target_height), Image.LANCZOS)
    
    def __call__(self, image: Union[Image.Image, np.ndarray, torch.Tensor], **kwargs) -> Image.Image:
        """
        Process an image (convenience method)
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Processed PIL Image
        """
        # Update parameters for this call
        params = {**self.params, **kwargs}
        
        # Store original params and update
        original_params = self.params
        self.params = params
        
        try:
            result = self.process(image)
        finally:
            # Restore original params
            self.params = original_params
            
        return result 