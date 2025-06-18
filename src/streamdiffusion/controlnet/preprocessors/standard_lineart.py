import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional
import time
from .base import BasePreprocessor


class StandardLineartPreprocessor(BasePreprocessor):
    """
    Real-time optimized Standard Lineart detection preprocessor for ControlNet
    
    Extracts line art from input images using traditional computer vision techniques.
    Uses Gaussian blur and intensity calculations to detect lines without requiring
    pre-trained models. Optimized for real-time performance.
    """
    
    def __init__(self, 
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 gaussian_sigma: float = 6.0,
                 intensity_threshold: int = 8,
                 **kwargs):
        """
        Initialize Standard Lineart preprocessor
        
        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            gaussian_sigma: Standard deviation for Gaussian blur
            intensity_threshold: Threshold for intensity calculation
            **kwargs: Additional parameters
        """
        print(f"StandardLineartPreprocessor.__init__: Initializing with detect_resolution={detect_resolution}, image_resolution={image_resolution}, gaussian_sigma={gaussian_sigma}, intensity_threshold={intensity_threshold}")
        
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            gaussian_sigma=gaussian_sigma,
            intensity_threshold=intensity_threshold,
            **kwargs
        )
        
        print("StandardLineartPreprocessor.__init__: Initialization complete")
    
    def _ensure_hwc3(self, x: np.ndarray) -> np.ndarray:
        """Ensure image has 3 channels (HWC3 format)"""
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y
    
    def _pad64(self, x: int) -> int:
        """Pad to nearest multiple of 64"""
        return int(np.ceil(float(x) / 64.0) * 64 - x)
    
    def _resize_image_with_pad(self, input_image: np.ndarray, resolution: int) -> tuple:
        """Resize image with padding to target resolution"""
        img = self._ensure_hwc3(input_image)
        H_raw, W_raw, _ = img.shape
        
        if resolution == 0:
            return img, lambda x: x
            
        k = float(resolution) / float(min(H_raw, W_raw))
        H_target = int(np.round(float(H_raw) * k))
        W_target = int(np.round(float(W_raw) * k))
        
        interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
        
        H_pad, W_pad = self._pad64(H_target), self._pad64(W_target)
        img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

        def remove_pad(x):
            return np.ascontiguousarray(x[:H_target, :W_target, ...].copy()).copy()

        return np.ascontiguousarray(img_padded.copy()).copy(), remove_pad
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply standard line art detection to the input image - real-time optimized
        
        Args:
            image: Input image
            
        Returns:
            PIL Image with detected line art (black lines on white background)
        """
        start_time = time.time()
        print("StandardLineartPreprocessor.process: Starting standard line art detection")
        
        # Convert to PIL Image if needed, then to numpy
        image = self.validate_input(image)
        if isinstance(image, Image.Image):
            input_image = np.array(image, dtype=np.uint8)
        else:
            input_image = image.astype(np.uint8)
            
        validation_time = time.time()
        print(f"StandardLineartPreprocessor.process: Input validation completed in {validation_time - start_time:.3f}s")
        
        # Get parameters
        detect_resolution = self.params.get('detect_resolution', 512)
        image_resolution = self.params.get('image_resolution', 512)
        gaussian_sigma = self.params.get('gaussian_sigma', 6.0)
        intensity_threshold = self.params.get('intensity_threshold', 8)
        
        print(f"StandardLineartPreprocessor.process: Using detect_resolution={detect_resolution}, image_resolution={image_resolution}, gaussian_sigma={gaussian_sigma}, intensity_threshold={intensity_threshold}")
        
        # Resize for detection with padding
        resize_start = time.time()
        input_image, remove_pad = self._resize_image_with_pad(input_image, detect_resolution)
        resize_time = time.time() - resize_start
        print(f"StandardLineartPreprocessor.process: Image resized and padded in {resize_time:.3f}s")
        
        # Apply standard lineart algorithm
        detection_start = time.time()
        print("StandardLineartPreprocessor.process: Starting standard lineart algorithm")
        
        # Convert to float32 for processing
        x = input_image.astype(np.float32)
        
        # Apply Gaussian blur
        g = cv2.GaussianBlur(x, (0, 0), gaussian_sigma)
        
        # Calculate intensity differences
        intensity = np.min(g - x, axis=2).clip(0, 255)
        
        # Normalize intensity
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        
        # Convert back to uint8
        detected_map = intensity.clip(0, 255).astype(np.uint8)
        
        detection_time = time.time() - detection_start
        print(f"StandardLineartPreprocessor.process: Standard lineart detection completed in {detection_time:.3f}s")
        
        # Remove padding and ensure HWC3 format
        postprocess_start = time.time()
        detected_map = self._ensure_hwc3(remove_pad(detected_map))
        
        # Resize to target resolution if needed
        if detected_map.shape[:2] != (image_resolution, image_resolution):
            detected_map = cv2.resize(detected_map, (image_resolution, image_resolution), interpolation=cv2.INTER_CUBIC)
        
        # Convert to PIL Image
        lineart_image = Image.fromarray(detected_map)
        
        postprocess_time = time.time() - postprocess_start
        print(f"StandardLineartPreprocessor.process: Post-processing completed in {postprocess_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"StandardLineartPreprocessor.process: Total processing time: {total_time:.3f}s")
        
        return lineart_image 