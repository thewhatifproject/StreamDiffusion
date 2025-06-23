"""ControlNet TensorRT engine pool management"""

import os
import time
import hashlib
from typing import Dict, Optional, Set, Union, Any, List
from pathlib import Path
import torch
from polygraphy import cuda

from .controlnet_engine import ControlNetModelEngine, HybridControlNet
from .controlnet_models import create_controlnet_model
from .builder import EngineBuilder, create_onnx_path
from .model_detection import detect_model_from_diffusers_unet


class ControlNetEnginePool:
    """Manages multiple ControlNet TensorRT engines"""
    
    def __init__(self, engine_dir: str, stream: Optional[cuda.Stream] = None):
        """Initialize ControlNet engine pool"""
        self.engine_dir = Path(engine_dir)
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        self.stream = stream if stream is not None else cuda.Stream()
        self.engines: Dict[str, HybridControlNet] = {}
        self.compiled_models: Set[str] = set()
        
        self._discover_existing_engines()
    
    def _discover_existing_engines(self) -> None:
        """Discover existing TensorRT engines in the engine directory"""
        for model_dir in self.engine_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("controlnet_"):
                engine_file = model_dir / "cnet.engine"
                if engine_file.exists():
                    dir_name = model_dir.name
                    if "--batch-" in dir_name:
                        model_part = dir_name.split("--batch-")[0]
                        model_id = model_part.replace("controlnet_", "").replace("_", "/")
                    else:
                        model_id = dir_name.replace("controlnet_", "").replace("_", "/")
                    
                    self.compiled_models.add(model_id)
                    print(f"Discovered existing ControlNet engine: {model_id}")
    
    
    
    
    def cleanup(self) -> None:
        """Clean up resources and stop background threads"""
        self.engines.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()
