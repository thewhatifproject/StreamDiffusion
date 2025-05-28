import numpy as np
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from controlnet_aux import LineartDetector, LineartAnimeDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False


class LineartPreprocessor(BasePreprocessor):
    """
    Lineart detection preprocessor for ControlNet
    
    Extracts line art from input images using specialized line art detection models.
    Supports both realistic and anime-style line art extraction.
    """
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 coarse: bool = False,
                 anime_style: bool = True,
                 **kwargs):
        """
        Initialize Lineart preprocessor
        
        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            coarse: Whether to use coarse line art detection
            anime_style: Whether to use anime-style line art detection
            **kwargs: Additional parameters
        """
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            coarse=coarse,
            anime_style=anime_style,
            **kwargs
        )
        
        self._detector = None
    
    @property
    def detector(self):
        """Lazy loading of the line art detector"""
        if self._detector is None:
            anime_style = self.params.get('anime_style', True)
            
            if CONTROLNET_AUX_AVAILABLE:
                print(f"Loading {'Anime' if anime_style else 'Realistic'} Lineart detector from controlnet_aux")
                if anime_style:
                    self._detector = LineartAnimeDetector.from_pretrained('lllyasviel/Annotators')
                else:
                    self._detector = LineartDetector.from_pretrained('lllyasviel/Annotators')
            else:
                print("Warning: controlnet_aux not available, using fallback lineart implementation")
                self._detector = self._create_fallback_detector()
        return self._detector
    
    def _create_fallback_detector(self):
        """Create a simple fallback detector if controlnet_aux is not available"""
        class FallbackLineartDetector:
            def __call__(self, image, detect_resolution=512, image_resolution=512, coarse=False):
                if not OPENCV_AVAILABLE:
                    # Return a simple edge-based approximation
                    return self._simple_edge_detection(image)
                
                # Convert PIL to numpy
                if hasattr(image, 'size'):
                    image_np = np.array(image)
                else:
                    image_np = image
                
                # Convert to grayscale
                if len(image_np.shape) == 3:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_np
                
                # Apply adaptive threshold for line art effect
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10
                )
                
                # Invert so lines are black on white background
                lineart = 255 - adaptive_thresh
                
                # Convert back to RGB
                lineart_rgb = cv2.cvtColor(lineart, cv2.COLOR_GRAY2RGB)
                result = Image.fromarray(lineart_rgb)
                
                return result
        
        return FallbackLineartDetector()
    
    def _simple_edge_detection(self, image):
        """Simple edge detection fallback when OpenCV is not available"""
        # Convert to grayscale array
        image_np = np.array(image.convert('L'))
        
        # Simple Sobel-like edge detection
        h, w = image_np.shape
        edges = np.zeros_like(image_np)
        
        # Horizontal edges
        for i in range(1, h-1):
            for j in range(1, w-1):
                gx = (-1 * image_np[i-1, j-1] + 1 * image_np[i-1, j+1] +
                      -2 * image_np[i, j-1] + 2 * image_np[i, j+1] +
                      -1 * image_np[i+1, j-1] + 1 * image_np[i+1, j+1])
                
                gy = (-1 * image_np[i-1, j-1] - 2 * image_np[i-1, j] - 1 * image_np[i-1, j+1] +
                      1 * image_np[i+1, j-1] + 2 * image_np[i+1, j] + 1 * image_np[i+1, j+1])
                
                edges[i, j] = min(255, int(np.sqrt(gx*gx + gy*gy)))
        
        # Invert and convert to RGB
        edges = 255 - edges
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb.astype(np.uint8))
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply line art detection to the input image
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected line art (black lines on white background)
        """
        # Convert to PIL Image if needed
        image = self.validate_input(image)
        
        # Get parameters
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resolution = self.params.get('image_resolution', 512)
        coarse = self.params.get('coarse', False)
        
        # Resize for detection
        image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        
        # Detect line art
        if CONTROLNET_AUX_AVAILABLE and hasattr(self.detector, '__call__'):
            try:
                # Use controlnet_aux detector
                lineart_image = self.detector(
                    image_resized,
                    detect_resolution=detect_resolution,
                    image_resolution=image_resolution,
                    coarse=coarse
                )
            except Exception as e:
                print(f"Warning: Lineart detection failed, using fallback: {e}")
                lineart_image = self._create_fallback_detector()(
                    image_resized, detect_resolution, image_resolution, coarse
                )
        else:
            lineart_image = self.detector(
                image_resized, detect_resolution, image_resolution, coarse
            )
        
        # Resize to target resolution if needed
        if lineart_image.size != (image_resolution, image_resolution):
            lineart_image = lineart_image.resize((image_resolution, image_resolution), Image.LANCZOS)
        
        return lineart_image 