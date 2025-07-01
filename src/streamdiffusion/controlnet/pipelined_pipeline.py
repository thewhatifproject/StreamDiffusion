import torch
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import numpy as np
import concurrent.futures
import logging
from .base_controlnet_pipeline import BaseControlNetPipeline

logger = logging.getLogger(__name__)


class PipelinedControlNetPipeline(BaseControlNetPipeline):
    """ControlNet pipeline with inter-frame parallelism using thread-safe PIL copies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_frame_future = None
        self._next_frame_result = None
        
    def update_control_image_efficient(self, control_image: Union[str, Image.Image, np.ndarray, torch.Tensor], index: Optional[int] = None) -> None:
        """Enhanced with inter-frame pipeline parallelism"""
        
        # Single ControlNet case - use existing sync logic
        if index is not None:
            if not (0 <= index < len(self.controlnets)):
                raise IndexError(f"{self.model_type} ControlNet index {index} out of range")
            if self.controlnet_scales[index] == 0:
                return
            preprocessor = self.preprocessors[index]
            processed_image = self._prepare_control_image(control_image, preprocessor)
            self.controlnet_images[index] = processed_image
            return
        
        # Multi-ControlNet pipeline processing
        self._wait_for_previous_preprocessing()
        self._start_next_frame_preprocessing(control_image)
        self._apply_current_frame_preprocessing()

    def _start_next_frame_preprocessing(self, control_image):
        """Start preprocessing for next frame in background thread"""
        
        if not any(scale > 0 for scale in self.controlnet_scales):
            self._next_frame_future = None
            return
        
        if (self._last_input_frame is not None and 
            isinstance(control_image, (torch.Tensor, np.ndarray, Image.Image)) and 
            control_image is self._last_input_frame):
            self._next_frame_future = None
            return
        
        self._last_input_frame = control_image
        
        # Prepare preprocessor groups first (avoid duplication)
        preprocessor_groups = {}
        active_indices = []
        for i, scale in enumerate(self.controlnet_scales):
            if scale > 0:
                active_indices.append(i)
                preprocessor = self.preprocessors[i]
                preprocessor_key = id(preprocessor) if preprocessor is not None else 'passthrough'
                
                if preprocessor_key not in preprocessor_groups:
                    preprocessor_groups[preprocessor_key] = {
                        'preprocessor': preprocessor,
                        'indices': []
                    }
                preprocessor_groups[preprocessor_key]['indices'].append(i)
        
        if not active_indices:
            self._next_frame_future = None
            return
        
        # Optimize input preparation based on type
        if isinstance(control_image, torch.Tensor):
            # Fast path: tensor already available, no cloning needed (read-only)
            control_image_safe = None
            control_tensor = control_image  # Direct reference, no clone!
        elif isinstance(control_image, Image.Image):
            # PIL path: single copy, derive tensor from copy
            control_image_safe = control_image.copy()
            control_tensor = self._to_tensor_safe(control_image_safe)
        elif isinstance(control_image, str):
            # Load path: load once, derive tensor
            from diffusers.utils import load_image
            control_image_safe = load_image(control_image)
            control_tensor = self._to_tensor_safe(control_image_safe)
        else:
            # Numpy/other: use as-is
            control_image_safe = control_image
            control_tensor = None
        
        # Submit optimized background processing
        self._next_frame_future = self._preprocessor_executor.submit(
            self._process_frame_preprocessing_optimized,
            preprocessor_groups,
            control_image_safe,
            control_tensor,
            active_indices
        )

    def _process_frame_preprocessing_optimized(self, preprocessor_groups, control_image_safe, control_tensor, active_indices):
        """Optimized preprocessing in background thread"""
        
        try:
            processed_cache = {}
            
            if len(preprocessor_groups) > 1:
                # Parallel processing for multiple preprocessors
                futures = []
                for prep_key, group in preprocessor_groups.items():
                    # NO double copying - use references
                    future = self._preprocessor_executor.submit(
                        self._process_single_preprocessor_optimized,
                        prep_key, 
                        group, 
                        control_image_safe,  # No extra copy!
                        control_tensor       # No clone!
                    )
                    futures.append((future, prep_key, group))
                
                # Collect results
                for future, prep_key, group in futures:
                    result = future.result()
                    if result[2] is not None:
                        cache_key = f"prep_{prep_key}"
                        processed_cache[cache_key] = result[2]
                        
            else:
                # Single preprocessor - direct processing
                prep_key, group = next(iter(preprocessor_groups.items()))
                result = self._process_single_preprocessor_optimized(
                    prep_key, group, control_image_safe, control_tensor
                )
                if result[2] is not None:
                    cache_key = f"prep_{prep_key}"
                    processed_cache[cache_key] = result[2]
            
            return {
                'processed_cache': processed_cache,
                'preprocessor_groups': preprocessor_groups,
                'active_indices': active_indices,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

    def _process_single_preprocessor_optimized(self, preprocessor_key, group, control_image_safe, control_tensor_safe):
        """Optimized thread-safe preprocessor logic"""
        try:
            preprocessor = group['preprocessor']
            
            # Prioritize tensor path for speed (no PIL conversion overhead)
            if (preprocessor is not None and 
                hasattr(preprocessor, 'process_tensor') and 
                control_tensor_safe is not None):
                try:
                    processed_image = self._prepare_control_image(control_tensor_safe, preprocessor)
                    return preprocessor_key, group['indices'], processed_image
                except Exception:
                    pass  # Fall through to PIL processing
            
            # PIL processing fallback (only if tensor path failed)
            if control_image_safe is not None:
                processed_image = self._prepare_control_image(control_image_safe, preprocessor)
                return preprocessor_key, group['indices'], processed_image
            
            # Last resort: create tensor from scratch (should be rare)
            if control_tensor_safe is not None:
                processed_image = self._prepare_control_image(control_tensor_safe, preprocessor)
                return preprocessor_key, group['indices'], processed_image
            
            return preprocessor_key, group['indices'], None
            
        except Exception as e:
            logger.error(f"_process_single_preprocessor_optimized: Preprocessor {preprocessor_key} failed: {e}")
            return preprocessor_key, group['indices'], None

    def _wait_for_previous_preprocessing(self):
        """Wait for previous frame preprocessing with optimized timeout"""
        if hasattr(self, '_next_frame_future') and self._next_frame_future is not None:
            try:
                # Reduced timeout: 50ms instead of 100ms for 45ms preprocessing
                self._next_frame_result = self._next_frame_future.result(timeout=0.05)
            except concurrent.futures.TimeoutError:
                logger.warning("PipelinedControlNet: Preprocessing timeout - using previous results")
                self._next_frame_result = None
            except Exception as e:
                logger.error(f"PipelinedControlNet: Preprocessing error: {e}")
                self._next_frame_result = None
        else:
            self._next_frame_result = None

    def _apply_current_frame_preprocessing(self):
        """Apply preprocessing results from previous iteration"""
        if not hasattr(self, '_next_frame_result') or self._next_frame_result is None:
            return
        
        result = self._next_frame_result
        if result['status'] != 'success':
            return
        
        # Apply results to pipeline state
        processed_cache = result['processed_cache']
        preprocessor_groups = result['preprocessor_groups']
        
        # Update controlnet_images with processed results
        for prep_key, group in preprocessor_groups.items():
            cache_key = f"prep_{prep_key}"
            if cache_key in processed_cache:
                processed_image = processed_cache[cache_key]
                for index in group['indices']:
                    self.controlnet_images[index] = processed_image
        
        # Update cache
        self._preprocessed_cache.clear()
        self._preprocessed_cache.update(processed_cache)

    def _to_tensor_safe(self, image):
        """Thread-safe tensor conversion"""
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()
        return to_tensor(image).unsqueeze(0).to(device=self.device, dtype=self.dtype) 