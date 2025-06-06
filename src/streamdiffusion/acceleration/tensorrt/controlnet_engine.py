"""
ControlNet TensorRT Engine

This module provides TensorRT-accelerated ControlNet inference engines,
following the same patterns as the UNet engine but optimized for
ControlNet-specific inputs and outputs.
"""

import torch
from typing import List, Optional, Tuple, Dict, Any
from polygraphy import cuda

from .utilities import Engine


class ControlNetModelEngine:
    """TensorRT-accelerated ControlNet inference engine"""
    
    def __init__(self, engine_path: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        """
        Initialize ControlNet TensorRT engine
        
        Args:
            engine_path: Path to the compiled TensorRT engine
            stream: CUDA stream for inference
            use_cuda_graph: Whether to use CUDA graphs for optimization
        """
        self.engine = Engine(engine_path)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        
        # Load and activate the engine
        self.engine.load()
        self.engine.activate()
        
        # Cache for input/output names
        self._input_names = None
        self._output_names = None
    
    def __call__(self, 
                 sample: torch.Tensor,
                 timestep: torch.Tensor, 
                 encoder_hidden_states: torch.Tensor,
                 controlnet_cond: torch.Tensor,
                 conditioning_scale: float = 1.0,
                 text_embeds: Optional[torch.Tensor] = None,
                 time_ids: Optional[torch.Tensor] = None,
                 **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through TensorRT ControlNet engine
        
        Args:
            sample: Latent sample tensor (B, 4, H//8, W//8)
            timestep: Timestep tensor (B,)
            encoder_hidden_states: Text embeddings (B, 77, 768/1024/2048)
            controlnet_cond: Control conditioning image (B, 3, H, W)
            conditioning_scale: Scale factor for ControlNet conditioning
            text_embeds: Pooled text embeddings for SDXL (B, 1280)
            time_ids: Time/resolution conditioning for SDXL (B, 6)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (down_block_residuals, mid_block_residual)
        """
        # Ensure timestep is float32
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        
        # Prepare input dictionary
        input_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond
        }
        
        # Add SDXL-specific inputs if provided
        if text_embeds is not None:
            input_dict["text_embeds"] = text_embeds
        if time_ids is not None:
            input_dict["time_ids"] = time_ids
        
        # Prepare shape dictionary for buffer allocation
        shape_dict = {name: tensor.shape for name, tensor in input_dict.items()}
        
        # Add output shapes (this will be handled by the engine)
        # The engine will automatically determine output shapes
        
        # Allocate buffers and run inference
        self.engine.allocate_buffers(shape_dict=shape_dict, device=sample.device)
        
        outputs = self.engine.infer(
            input_dict,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        
        # Extract and organize outputs
        down_blocks, mid_block = self._extract_controlnet_outputs(outputs, conditioning_scale)
        
        return down_blocks, mid_block
    
    def _extract_controlnet_outputs(self, 
                                   outputs: Dict[str, torch.Tensor], 
                                   conditioning_scale: float) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract and organize ControlNet outputs from engine results
        
        Args:
            outputs: Raw outputs from TensorRT engine
            conditioning_scale: Scale factor to apply to all outputs
            
        Returns:
            Tuple of (down_block_residuals, mid_block_residual)
        """
        # Extract down block outputs (12 total)
        down_blocks = []
        for i in range(12):
            output_name = f"down_block_{i:02d}"
            if output_name in outputs:
                tensor = outputs[output_name]
                if conditioning_scale != 1.0:
                    tensor = tensor * conditioning_scale
                down_blocks.append(tensor)
        
        # Extract middle block output
        mid_block = None
        if "mid_block" in outputs:
            mid_block = outputs["mid_block"]
            if conditioning_scale != 1.0:
                mid_block = mid_block * conditioning_scale
        
        return down_blocks, mid_block
    
    def get_input_names(self) -> List[str]:
        """Get input names for this engine"""
        if self._input_names is None:
            # Extract from engine metadata if available
            # Fallback to standard names
            self._input_names = ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]
        return self._input_names
    
    def get_output_names(self) -> List[str]:
        """Get output names for this engine"""
        if self._output_names is None:
            # 12 down blocks + 1 middle block
            down_names = [f"down_block_{i:02d}" for i in range(12)]
            self._output_names = down_names + ["mid_block"]
        return self._output_names
    
    def to(self, *args, **kwargs):
        """Compatibility method - TensorRT engines don't need device movement"""
        pass

    def forward(self, *args, **kwargs):
        """Compatibility method - redirect to __call__"""
        return self.__call__(*args, **kwargs)


class HybridControlNet:
    """
    Wrapper that handles TensorRT/PyTorch fallback for ControlNet
    
    This wrapper attempts to use TensorRT when available and falls back
    to PyTorch ControlNet if TensorRT fails or is unavailable.
    """
    
    def __init__(self, 
                 model_id: str,
                 engine_path: Optional[str] = None,
                 pytorch_model: Optional[Any] = None,
                 stream: Optional[cuda.Stream] = None):
        """
        Initialize hybrid ControlNet wrapper
        
        Args:
            model_id: ControlNet model identifier
            engine_path: Path to TensorRT engine (if available)
            pytorch_model: Fallback PyTorch ControlNet model
            stream: CUDA stream for TensorRT engine
        """
        self.model_id = model_id
        self.engine_path = engine_path
        self.pytorch_model = pytorch_model
        self.stream = stream
        
        # Engine state
        self.trt_engine: Optional[ControlNetModelEngine] = None
        self.use_tensorrt = False
        self.fallback_reason = None
        
        # Try to load TensorRT engine if path provided
        if engine_path:
            self._try_load_tensorrt_engine()
    
    def _try_load_tensorrt_engine(self) -> bool:
        """
        Attempt to load TensorRT engine
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if self.engine_path and self.stream:
                self.trt_engine = ControlNetModelEngine(self.engine_path, self.stream)
                self.use_tensorrt = True
                return True
        except Exception as e:
            self.fallback_reason = f"TensorRT engine load failed: {e}"
            self.use_tensorrt = False
        
        return False
    
    def __call__(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with automatic TensorRT/PyTorch fallback
        
        Returns:
            Tuple of (down_block_residuals, mid_block_residual)
        """
        # Try TensorRT first if available
        if self.use_tensorrt and self.trt_engine:
            try:
                return self.trt_engine(*args, **kwargs)
            except Exception as e:
                print(f"Warning: TensorRT ControlNet failed ({e}), falling back to PyTorch")
                self.use_tensorrt = False
                self.fallback_reason = f"Runtime error: {e}"
        
        # Fallback to PyTorch
        if self.pytorch_model is None:
            raise RuntimeError(f"No PyTorch fallback available for ControlNet {self.model_id}")
        
        # Call PyTorch model - this should return the same format
        return self._call_pytorch_model(*args, **kwargs)
    
    def _call_pytorch_model(self, *args, **kwargs) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Call PyTorch ControlNet model with proper output formatting
        
        This method ensures the PyTorch model returns the same output format
        as the TensorRT engine.
        """
        # Call the PyTorch ControlNet
        result = self.pytorch_model(*args, **kwargs)
        
        # Handle different PyTorch ControlNet output formats
        if isinstance(result, tuple) and len(result) == 2:
            # Already in the expected format
            return result
        elif hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            # Diffusers ControlNet output format
            return result.down_block_res_samples, result.mid_block_res_sample
        else:
            # Try to extract from result
            if isinstance(result, (list, tuple)) and len(result) >= 13:
                # Assume first 12 are down blocks, last is middle
                return list(result[:12]), result[12]
            else:
                raise ValueError(f"Unexpected PyTorch ControlNet output format: {type(result)}")
    
    @property
    def is_using_tensorrt(self) -> bool:
        """Check if currently using TensorRT engine"""
        return self.trt_engine is not None
    
    def upgrade_to_tensorrt(self, engine_path: str) -> bool:
        """
        Upgrade to TensorRT engine after background compilation completes
        
        Args:
            engine_path: Path to the compiled TensorRT engine
            
        Returns:
            True if upgrade successful, False otherwise
        """
        try:
            print(f"Upgrading ControlNet {self.model_id} to TensorRT...")
            
            # Load TensorRT engine
            self.trt_engine = ControlNetModelEngine(engine_path, self.stream)
            self.engine_path = engine_path
            
            print(f"Successfully upgraded ControlNet {self.model_id} to TensorRT")
            return True
            
        except Exception as e:
            print(f"Failed to upgrade ControlNet {self.model_id} to TensorRT: {e}")
            self.trt_engine = None
            return False
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get current status information"""
        return {
            "model_id": self.model_id,
            "using_tensorrt": self.is_using_tensorrt,
            "engine_path": self.engine_path,
            "fallback_reason": self.fallback_reason,
            "has_pytorch_fallback": self.pytorch_model is not None
        } 