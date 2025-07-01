import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Optional
from .base import BasePreprocessor

try:
    from controlnet_aux import HEDdetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("HEDPreprocessor: controlnet_aux not available, using fallback implementation")


class HEDNetwork(nn.Module):
    """
    HED (Holistically-Nested Edge Detection) network implementation
    Based on VGG backbone with side outputs at multiple scales
    """
    
    def __init__(self):
        super(HEDNetwork, self).__init__()
        
        # VGG-16 backbone (first 13 conv layers)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, 1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        """Forward pass through HED network"""
        h, w = x.size()[2:]
        
        # Stage 1
        x1_1 = torch.relu(self.conv1_1(x))
        x1_2 = torch.relu(self.conv1_2(x1_1))
        x1_pool = self.pool(x1_2)
        
        # Stage 2
        x2_1 = torch.relu(self.conv2_1(x1_pool))
        x2_2 = torch.relu(self.conv2_2(x2_1))
        x2_pool = self.pool(x2_2)
        
        # Stage 3
        x3_1 = torch.relu(self.conv3_1(x2_pool))
        x3_2 = torch.relu(self.conv3_2(x3_1))
        x3_3 = torch.relu(self.conv3_3(x3_2))
        x3_pool = self.pool(x3_3)
        
        # Stage 4
        x4_1 = torch.relu(self.conv4_1(x3_pool))
        x4_2 = torch.relu(self.conv4_2(x4_1))
        x4_3 = torch.relu(self.conv4_3(x4_2))
        x4_pool = self.pool(x4_3)
        
        # Stage 5
        x5_1 = torch.relu(self.conv5_1(x4_pool))
        x5_2 = torch.relu(self.conv5_2(x5_1))
        x5_3 = torch.relu(self.conv5_3(x5_2))
        
        # Side outputs
        side1 = self.side1(x1_2)
        side2 = self.side2(x2_2)
        side3 = self.side3(x3_3)
        side4 = self.side4(x4_3)
        side5 = self.side5(x5_3)
        
        # Upsample to original size
        side1 = F.interpolate(side1, size=(h, w), mode='bilinear', align_corners=False)
        side2 = F.interpolate(side2, size=(h, w), mode='bilinear', align_corners=False)
        side3 = F.interpolate(side3, size=(h, w), mode='bilinear', align_corners=False)
        side4 = F.interpolate(side4, size=(h, w), mode='bilinear', align_corners=False)
        side5 = F.interpolate(side5, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fuse side outputs
        fuse = self.fuse(torch.cat([side1, side2, side3, side4, side5], dim=1))
        
        # Apply sigmoid
        fuse = torch.sigmoid(fuse)
        
        return fuse


class HEDPreprocessor(BasePreprocessor):
    """
    HED (Holistically-Nested Edge Detection) preprocessor
    
    Provides soft edge detection using neural networks.
    Supports both controlnet_aux (recommended) and custom fallback implementation.
    """
    
    _model_cache = {}
    
    def __init__(self, safe: bool = True, use_controlnet_aux: bool = True, **kwargs):
        """
        Initialize HED preprocessor
        
        Args:
            safe: Whether to use safe mode (if available)
            use_controlnet_aux: Whether to use controlnet_aux implementation
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.safe = safe
        self.use_controlnet_aux = use_controlnet_aux and CONTROLNET_AUX_AVAILABLE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load HED model with caching
        """
        if self.use_controlnet_aux:
            self._load_controlnet_aux_model()
        else:
            self._load_custom_model()
    
    def _load_controlnet_aux_model(self):
        """Load controlnet_aux HED model"""
        cache_key = f"hed_aux_{self.device}"
        
        if cache_key in self._model_cache:
            self.model = self._model_cache[cache_key]
            return
        
        print("HEDPreprocessor: Loading controlnet_aux HED model")
        try:
            # Initialize HED detector
            self.model = HEDdetector.from_pretrained("lllyasviel/Annotators")
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Cache the model
            self._model_cache[cache_key] = self.model
            print(f"HEDPreprocessor: Successfully loaded controlnet_aux model on {self.device}")
            
        except Exception as e:
            print(f"HEDPreprocessor: Failed to load controlnet_aux model: {e}")
            print("HEDPreprocessor: Falling back to custom implementation")
            self.use_controlnet_aux = False
            self._load_custom_model()
    
    def _load_custom_model(self):
        """Load custom HED model implementation"""
        cache_key = f"hed_custom_{self.device}_{self.dtype}"
        
        if cache_key in self._model_cache:
            self.model = self._model_cache[cache_key]
            return
        
        print("HEDPreprocessor: Loading custom HED model")
        self.model = HEDNetwork()
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # Cache the model
        self._model_cache[cache_key] = self.model
        print(f"HEDPreprocessor: Custom model loaded on {self.device}")
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply HED edge detection to the input image
        """
        if self.use_controlnet_aux:
            return self._process_with_controlnet_aux(image)
        else:
            return self._process_with_custom(image)
    
    def _process_with_controlnet_aux(self, image: Image.Image) -> Image.Image:
        """Process image using controlnet_aux HED"""
        try:
            # Get target dimensions
            target_width, target_height = self.get_target_dimensions()
            
            # Process with controlnet_aux
            result = self.model(image, output_type="pil")
            
            # Ensure result is PIL Image
            if not isinstance(result, Image.Image):
                if hasattr(result, 'save'):  # Likely PIL Image
                    result = result
                else:
                    # Convert from other formats
                    if isinstance(result, np.ndarray):
                        result = Image.fromarray(result)
                    else:
                        raise ValueError(f"Unexpected result type: {type(result)}")
            
            # Resize to target size if needed
            if result.size != (target_width, target_height):
                result = result.resize((target_width, target_height), Image.LANCZOS)
            
            return result
            
        except Exception as e:
            print(f"HEDPreprocessor: controlnet_aux processing failed: {e}")
            print("HEDPreprocessor: Falling back to custom implementation")
            self.use_controlnet_aux = False
            self._load_custom_model()
            return self._process_with_custom(image)
    
    def _process_with_custom(self, image: Image.Image) -> Image.Image:
        """Process image using custom HED implementation"""
        # Convert PIL to tensor for GPU processing
        image_tensor = self.pil_to_tensor(image).squeeze(0)  # Remove batch dim
        
        # Process with GPU-accelerated tensor method
        processed_tensor = self._process_tensor_core(image_tensor)
        
        # Convert back to PIL
        return self.tensor_to_pil(processed_tensor)
    
    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized HED processing using tensors (custom implementation)
        """
        with torch.no_grad():
            # Ensure correct input format and device
            image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
            
            # Normalize to [0, 1] if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Add batch dimension if needed
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Ensure RGB format
            if image_tensor.shape[1] == 1:
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
            
            # Run HED model
            edge_map = self.model(image_tensor)
            
            # Convert to 3-channel RGB format
            if edge_map.dim() == 4:
                edge_map = edge_map.repeat(1, 3, 1, 1).squeeze(0)
            elif edge_map.dim() == 3:
                edge_map = edge_map.repeat(3, 1, 1)
            
            # Ensure output is in [0, 1] range
            edge_map = torch.clamp(edge_map, 0.0, 1.0)
            
        return edge_map
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        """
        implementation = "controlnet_aux" if self.use_controlnet_aux else "custom"
        return {
            "preprocessor": "HED (Holistically-Nested Edge Detection)",
            "implementation": implementation,
            "safe_mode": self.safe,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "model_available": self.model is not None,
            "description": "Neural network-based soft edge detection"
        }
    
    @classmethod
    def create_optimized(cls, device: str = 'cuda', dtype: torch.dtype = torch.float16, **kwargs):
        """
        Create an optimized HED preprocessor
        
        Args:
            device: Target device ('cuda' or 'cpu')
            dtype: Data type for inference
            **kwargs: Additional parameters
            
        Returns:
            Optimized HEDPreprocessor instance
        """
        print(device)
        return cls(
            device=device,
            dtype=dtype,
            use_controlnet_aux=True,  # Prefer controlnet_aux for better quality
            safe=True,
            **kwargs
        ) 