"""
ControlNet Engine Pool Management

This module manages multiple ControlNet TensorRT engines, providing
on-demand compilation, loading, and hybrid TensorRT/PyTorch operation.
"""

import os
import threading
import hashlib
from typing import Dict, Optional, Set, Union, Any, List
from pathlib import Path
import torch
from polygraphy import cuda

from .controlnet_engine import ControlNetModelEngine, HybridControlNet
from .controlnet_models import create_controlnet_model
from .builder import EngineBuilder, create_onnx_path


class ControlNetEnginePool:
    """Manages multiple ControlNet TensorRT engines"""
    
    def __init__(self, engine_dir: str, stream: Optional[cuda.Stream] = None):
        """
        Initialize ControlNet engine pool
        
        Args:
            engine_dir: Directory to store TensorRT engines
            stream: CUDA stream for TensorRT engines (if None, creates new one)
        """
        self.engine_dir = Path(engine_dir)
        self.engine_dir.mkdir(parents=True, exist_ok=True)
        
        # CUDA stream management
        self.stream = stream if stream is not None else cuda.Stream()
        
        # Engine storage
        self.engines: Dict[str, HybridControlNet] = {}
        self.compiled_models: Set[str] = set()
        self.compiling_models: Set[str] = set()
        self.compilation_status: Dict[str, str] = {}
        
        # Compilation thread management
        self._compilation_threads: Dict[str, threading.Thread] = {}
        self._compilation_lock = threading.Lock()
        
        # Discover existing engines
        self._discover_existing_engines()
    
    def _discover_existing_engines(self) -> None:
        """Discover existing TensorRT engines in the engine directory"""
        for model_dir in self.engine_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("controlnet_"):
                engine_file = model_dir / "cnet.engine"
                if engine_file.exists():
                    # Extract model identifier from directory name
                    model_id = model_dir.name.replace("controlnet_", "").replace("_", "/")
                    self.compiled_models.add(model_id)
                    print(f"Discovered existing ControlNet engine: {model_id}")
    
    def get_or_load_engine(self, 
                          model_id: str,
                          pytorch_model: Any,
                          controlnet_type: str = "canny",
                          model_type: str = "sd15") -> HybridControlNet:
        """
        Get or load ControlNet engine (TensorRT if available, PyTorch fallback)
        
        Args:
            model_id: ControlNet model identifier
            pytorch_model: PyTorch ControlNet model for fallback
            controlnet_type: Type of ControlNet (canny, depth, etc.)
            model_type: Base model type (sd15, sdxl, etc.)
            
        Returns:
            HybridControlNet instance (TensorRT or PyTorch)
        """
        # Check if already loaded
        if model_id in self.engines:
            return self.engines[model_id]
        
        # Check for existing TensorRT engine using proper naming convention
        model_engine_dir = self._get_model_engine_dir(model_id)
        engine_path = model_engine_dir / "cnet.engine"
        
        # Create hybrid ControlNet wrapper
        hybrid_controlnet = HybridControlNet(
            model_id=model_id,
            engine_path=str(engine_path) if engine_path.exists() else None,
            pytorch_model=pytorch_model,
            stream=self.stream
        )
        
        # Store in pool
        self.engines[model_id] = hybrid_controlnet
        
        # Start background compilation if TensorRT not available
        if not hybrid_controlnet.is_using_tensorrt:
            self._start_background_compilation(
                model_id, pytorch_model, controlnet_type, model_type
            )
        
        return hybrid_controlnet
    
    def compile_controlnet_on_demand(self, 
                                   model_id: str,
                                   pytorch_model: Any,
                                   sample_inputs: Dict[str, torch.Tensor],
                                   controlnet_type: str = "canny",
                                   model_type: str = "sd15",
                                   **build_options) -> bool:
        """
        Compile ControlNet to TensorRT on-demand
        
        Args:
            model_id: ControlNet model identifier
            pytorch_model: PyTorch ControlNet model to compile
            sample_inputs: Sample inputs for shape inference
            controlnet_type: Type of ControlNet
            model_type: Base model type
            **build_options: Additional build options
            
        Returns:
            True if compilation started/completed, False if failed
        """
        engine_path = self._get_engine_path(model_id)
        
        # Check if already compiled
        if engine_path.exists():
            return True
        
        # Check if compilation is in progress
        if model_id in self._compilation_threads:
            thread = self._compilation_threads[model_id]
            if thread.is_alive():
                return False  # Compilation in progress
        
        try:
            return self._compile_controlnet_sync(
                model_id, pytorch_model, sample_inputs, 
                controlnet_type, model_type, **build_options
            )
        except Exception as e:
            print(f"ControlNet compilation failed for {model_id}: {e}")
            self.compilation_status[model_id] = f"Failed: {e}"
            return False
    
    def _start_background_compilation(self, 
                                    model_id: str,
                                    pytorch_model: Any, 
                                    controlnet_type: str,
                                    model_type: str):
        """
        Start background compilation of ControlNet to TensorRT
        
        Args:
            model_id: ControlNet model identifier
            pytorch_model: PyTorch ControlNet model
            controlnet_type: Type of ControlNet
            model_type: Base model type
        """
        if model_id in self.compiling_models:
            return  # Already compiling
            
        self.compiling_models.add(model_id)
        
        def compile_worker():
            try:
                model_engine_dir = self._get_model_engine_dir(model_id)
                engine_path = model_engine_dir / "cnet.engine"
                
                print(f"Starting background compilation of {model_id} to TensorRT...")
                print(f"Engine will be saved to: {engine_path}")
                
                success = self._compile_controlnet(
                    pytorch_model, controlnet_type, model_type, str(engine_path)
                )
                
                if success:
                    print(f"Successfully compiled {model_id} to TensorRT")
                    # Update the existing hybrid model to use TensorRT
                    if model_id in self.engines:
                        self.engines[model_id].upgrade_to_tensorrt(str(engine_path))
                else:
                    print(f"Failed to compile {model_id} to TensorRT, will continue using PyTorch")
                    
            except Exception as e:
                print(f"Error compiling {model_id}: {e}")
            finally:
                self.compiling_models.discard(model_id)
        
        threading.Thread(target=compile_worker, daemon=True).start()
    
    def _compile_controlnet(self, 
                           pytorch_model: Any,
                           controlnet_type: str, 
                           model_type: str,
                           engine_path: str) -> bool:
        """
        Compile ControlNet to TensorRT
        
        Args:
            pytorch_model: PyTorch ControlNet model
            controlnet_type: Type of ControlNet
            model_type: Base model type
            engine_path: Path to save the compiled TensorRT engine
            
        Returns:
            True if compilation succeeded, False if failed
        """
        try:
            print(f"Starting ControlNet compilation: {controlnet_type} ({model_type})")
            
            # Create ControlNet TensorRT model definition
            batch_size = 1
            height = 512
            width = 512
            
            # Determine embedding dimension and other parameters
            if model_type.lower() in ["sdxl", "sdxl-turbo"]:
                embedding_dim = 2048
            else:
                embedding_dim = 768
            
            controlnet_model = create_controlnet_model(
                model_type=model_type,
                controlnet_type=controlnet_type,
                max_batch_size=batch_size,
                min_batch_size=1,
                embedding_dim=embedding_dim,
                image_height=height,
                image_width=width,
                static_batch=True,
                static_shape=True
            )
            
            # Move model to GPU
            pytorch_model = pytorch_model.to(torch.device("cuda"), dtype=torch.float16)
            
            # Create engine directory
            engine_dir = Path(engine_path).parent
            onnx_dir = engine_dir / "onnx"
            onnx_dir.mkdir(exist_ok=True)
            
            # Create ONNX paths
            base_name = "controlnet"
            onnx_path = create_onnx_path(base_name, onnx_dir, opt=False)
            onnx_opt_path = create_onnx_path(base_name, onnx_dir, opt=True)
            
            print(f"ONNX path: {onnx_path}")
            print(f"ONNX optimized path: {onnx_opt_path}")
            print(f"Engine path: {engine_path}")
            
            # Build TensorRT engine
            builder = EngineBuilder(controlnet_model, pytorch_model, device=torch.device("cuda"))
            builder.build(
                str(onnx_path),
                str(onnx_opt_path), 
                str(engine_path),
                opt_batch_size=batch_size,
                opt_image_height=height,
                opt_image_width=width
            )
            
            print(f"Successfully compiled ControlNet engine: {engine_path}")
            return True
            
        except Exception as e:
            print(f"ControlNet compilation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_sample_inputs(self, model_type: str) -> Dict[str, torch.Tensor]:
        """Generate sample inputs for ControlNet compilation"""
        # Standard dimensions
        batch_size = 1
        height, width = 512, 512
        
        if model_type.lower() in ["sdxl", "sdxl-turbo"]:
            embedding_dim = 2048
            text_embed_dim = 1280
        else:
            embedding_dim = 768
            text_embed_dim = None
        
        inputs = {
            "sample": torch.randn(batch_size, 4, height // 8, width // 8, dtype=torch.float16),
            "timestep": torch.tensor([0.5], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(batch_size, 77, embedding_dim, dtype=torch.float16),
            "controlnet_cond": torch.randn(batch_size, 3, height, width, dtype=torch.float16)
        }
        
        # Add SDXL-specific inputs
        if text_embed_dim:
            inputs["text_embeds"] = torch.randn(batch_size, text_embed_dim, dtype=torch.float16)
            inputs["time_ids"] = torch.randn(batch_size, 6, dtype=torch.float16)
        
        return inputs
    
    def _get_engine_path(self, model_id: str) -> Path:
        """Get the engine path for a model ID"""
        # Create safe filename from model ID
        safe_name = self._safe_filename(model_id)
        return self.engine_dir / f"controlnet_{safe_name}.engine"
    
    def _safe_filename(self, model_id: str) -> str:
        """Convert model ID to safe filename"""
        # Replace problematic characters and hash long names
        safe_id = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        if len(safe_id) > 100:  # Hash very long names
            hash_suffix = hashlib.md5(model_id.encode()).hexdigest()[:8]
            safe_id = safe_id[:90] + "_" + hash_suffix
        return safe_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all ControlNets in the pool"""
        status = {
            "total_engines": len(self.engines),
            "compiled_models": len(self.compiled_models),
            "active_compilations": len([t for t in self._compilation_threads.values() if t.is_alive()]),
            "engines": {}
        }
        
        for model_id, engine in self.engines.items():
            engine_status = engine.status.copy()
            engine_status["compilation_status"] = self.compilation_status.get(model_id, "Unknown")
            status["engines"][model_id] = engine_status
        
        return status
    
    def cleanup(self) -> None:
        """Clean up resources and stop background threads"""
        # Wait for compilation threads to finish (with timeout)
        for model_id, thread in list(self._compilation_threads.items()):
            if thread.is_alive():
                print(f"Waiting for ControlNet compilation to finish: {model_id}")
                thread.join(timeout=10)  # 10 second timeout
        
        self._compilation_threads.clear()
        self.engines.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()

    def _get_model_engine_dir(self, model_id: str) -> Path:
        """
        Get the engine directory for a specific ControlNet model
        
        Args:
            model_id: ControlNet model identifier
            
        Returns:
            Path to model's engine directory
        """
        # Convert model_id to a safe directory name
        # Replace problematic characters for directory names
        safe_name = model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_name = "controlnet_" + safe_name
        
        model_dir = Path(self.engine_dir) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir 