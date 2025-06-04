# TensorRT ControlNet Files for Reference

## Overview

We are working on enabling TensorRT with controlnets in stream diffusion. We have a reference project, ComfyUI_TensorRT, that has successfully enabled building tensorrt engines with controlnet support. This reference project is the only on earth that has ever done this, so we must pay VERY close attention to it. We are not reinventing the wheel, we are adapting a PROVEN solution to our own purposes in StreamDiffusion. 

What I expect, is to be able to run any StreamDiffusion script with the acceleration parameter set to "tensorrt", and it will build the engine with controlnet support. We should be focused on SDTurbo at first. Please review the necessary files to get yourself up to speed.

This document provides a comprehensive analysis of all files that must be investigated to translate the ComfyUI_TensorRT ControlNet functionality to StreamDiffusion. ComfyUI_TensorRT is the only project on earth that has successfully enabled TensorRT engines with ControlNet support, making this translation critical for achieving our goal.

## Reference Project Files (ComfyUI_TensorRT) - Critical to Study

### Core ControlNet Implementation Files

#### 1. `custom_nodes/ComfyUI_TensorRT/models/baseline.py`
**Why Important**: Contains the `TRTModelUtil` base class with `use_control` flag and fundamental ControlNet infrastructure. This is the foundation that enables ControlNet support across all model types.

Key components:
- `use_control` parameter in constructor
- `get_control()` method (abstract, implemented by subclasses)
- Input/output configuration dictionaries
- Dynamic shape evaluation methods

#### 2. `custom_nodes/ComfyUI_TensorRT/models/sd_unet.py`
**Why Important**: Contains the `UNetTRT` class with the critical `get_control()` method that dynamically generates ControlNet input tensor configurations based on UNet architecture (channel multipliers, downsampling factors). This is the ONLY implementation on earth that has solved this problem.

Key components:
- `UNetTRT.get_control()` method - the core innovation
- Channel multiplier and resolution block calculations
- Input/output control tensor shape generation
- Model-specific implementations (SD15_TRT, SDXL_TRT, etc.)

#### 3. `custom_nodes/ComfyUI_TensorRT/models/supported_models.py`
**Why Important**: Contains model detection logic (`detect_version_from_model()`) and model helper creation (`get_helper_from_model()`). This is how ComfyUI_TensorRT dynamically determines model characteristics and creates the appropriate TRT helper with ControlNet support.

Key components:
- `detect_version_from_model()` - extracts model type from ComfyUI model
- `get_helper_from_model()` - creates TRT helper with ControlNet support
- Model mapping dictionary
- Version-specific helper instantiation

#### 4. `custom_nodes/ComfyUI_TensorRT/tensorrt_diffusion_model.py`
**Why Important**: Contains the `TRTDiffusionBackbone` class with the critical `__call__` method that handles ControlNet inputs at runtime. Shows exactly how control tensors are processed and fed to the TensorRT engine.

Key components:
- `TRTDiffusionBackbone.__call__()` method
- ControlNet input processing and mapping
- TensorRT engine execution with control inputs
- Dynamic input tensor allocation

### Supporting Infrastructure Files

#### 5. `custom_nodes/ComfyUI_TensorRT/tensorrt_nodes.py`
**Why Important**: Shows the complete build process for TensorRT engines with ControlNet support, including how model helpers are used during compilation.

Key components:
- TensorRT engine building workflow
- Model helper integration during compilation
- Dynamic/static build configurations
- ONNX export integration

#### 6. `custom_nodes/ComfyUI_TensorRT/onnx_utils/export.py`
**Why Important**: Contains ONNX export logic that handles ControlNet inputs during model compilation. The `get_backbone()` function shows how ControlNet inputs are mapped to ONNX model inputs.

Key components:
- `get_backbone()` function - critical for ControlNet ONNX export
- ControlNet input mapping to ONNX inputs
- Dynamic input name generation
- Model wrapping for export

#### 7. `custom_nodes/ComfyUI_TensorRT/__init__.py`
**Why Important**: Shows the node registration and integration pattern.

## ComfyUI Source Code Files - Critical Dependencies

### Model Loading and Detection Infrastructure

#### 8. `comfy/supported_models.py`
**Why Important**: Contains all the model configuration classes (SD15, SDXL, etc.) that ComfyUI_TensorRT relies on to detect model characteristics. Each class has a `unet_config` dict that contains model architecture details needed for ControlNet input generation.

Key components:
- Model configuration classes (SD15, SDXL, etc.)
- `unet_config` dictionaries with architecture details
- Channel multipliers, context dimensions, model channels
- Model-specific configurations

#### 9. `comfy/supported_models_base.py`
**Why Important**: Contains the `BASE` class that all model configurations inherit from. Has the `matches()` method used by `model_config_from_unet_config()` to detect model types.

Key components:
- `BASE` class with common model infrastructure
- `matches()` method for model type detection
- `unet_config` and `unet_extra_config` patterns
- Model instantiation methods

#### 10. `comfy/model_detection.py`
**Why Important**: Contains the core detection functions that ComfyUI_TensorRT depends on for model identification and configuration extraction.

Key components:
- `detect_unet_config()`: Extracts UNet architecture from state dict
- `model_config_from_unet_config()`: Maps UNet config to model class
- `model_config_from_unet()`: Complete pipeline from state_dict to model config
- Channel multiplier and transformer depth detection

#### 11. `comfy/model_base.py`
**Why Important**: Contains the `BaseModel` class and model type definitions. Shows how ComfyUI creates model instances with their configurations, which ComfyUI_TensorRT needs to extract UNet architecture details.

Key components:
- `BaseModel` class implementation
- Model type enumeration
- UNet configuration processing
- Model instantiation and loading

#### 12. `comfy/sd.py`
**Why Important**: Contains model loading functions like `load_state_dict_guess_config()` and `load_diffusion_model_state_dict()` that ComfyUI_TensorRT depends on for model loading and configuration extraction.

Key components:
- `load_state_dict_guess_config()` - main model loading function
- Model detection and configuration pipeline
- State dict processing and model instantiation

## StreamDiffusion Files - Need Modification

### Core TensorRT Acceleration Files

#### 13. `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/models.py`
**Why Important**: Current UNet model definition that needs to be extended with ControlNet support using the ComfyUI_TensorRT pattern. This is where we'll add the `use_control` flag and `get_control()` method.

Required changes:
- Add `use_control` parameter to UNet class
- Implement `get_control()` method for ControlNet input generation
- Add model-specific ControlNet configurations
- Update input/output shape calculations

#### 14. `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/engine.py`
**Why Important**: Runtime engine that needs to be modified to handle ControlNet inputs similar to `TRTDiffusionBackbone.__call__()` in ComfyUI_TensorRT.

Required changes:
- Update `UNet2DConditionModelEngine.__call__()` to handle ControlNet inputs
- Add ControlNet input processing and mapping
- Update buffer allocation for ControlNet tensors
- Integrate with ControlNet pipeline

#### 15. `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/builder.py`
**Why Important**: Engine building utilities that may need updates to support ControlNet-aware model compilation.

Required changes:
- Update build process to include ControlNet inputs
- Add ControlNet-aware ONNX export
- Update optimization profiles for ControlNet tensors

#### 16. `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/utilities.py`
**Why Important**: ONNX export and TensorRT utilities that may need ControlNet input handling during compilation.

Required changes:
- Update ONNX export to include ControlNet inputs
- Add ControlNet input name generation
- Update dynamic axes for ControlNet tensors

#### 17. `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/__init__.py`
**Why Important**: Main acceleration function that orchestrates the entire TensorRT compilation process and needs to be updated for ControlNet support.

Required changes:
- Update `accelerate_with_tensorrt()` to detect ControlNet usage
- Add ControlNet-aware model compilation
- Integrate ControlNet pipeline with TensorRT engines

### ControlNet Integration Files

#### 18. `StreamDiffusion/src/streamdiffusion/controlnet/base_controlnet_pipeline.py`
**Why Important**: Current ControlNet implementation that patches StreamDiffusion's UNet step. This patching mechanism needs to be adapted to work with TensorRT engines instead of PyTorch models.

Required changes:
- Detect when TensorRT acceleration is used
- Adapt ControlNet input preparation for TensorRT engines
- Update patching mechanism for TensorRT compatibility
- Modify control tensor processing

#### 19. `StreamDiffusion/src/streamdiffusion/controlnet/controlnet_pipeline.py`
**Why Important**: Entry point for ControlNet functionality that may need updates for TensorRT compatibility.

Required changes:
- Add TensorRT compatibility checks
- Update ControlNet initialization for TensorRT

#### 20. `StreamDiffusion/utils/wrapper.py`
**Why Important**: High-level wrapper that integrates ControlNet with StreamDiffusion. The `_apply_controlnet_patch()` method needs to understand when TensorRT is being used and adapt accordingly.

Required changes:
- Update `_apply_controlnet_patch()` to handle TensorRT acceleration
- Add TensorRT detection logic
- Modify ControlNet integration for TensorRT engines

### Example and Configuration Files

#### 21. `StreamDiffusion/examples/controlnet/controlnet_webcam_demo.py`
**Why Important**: Current ControlNet demo that needs to be updated to support TensorRT acceleration parameter.

Required changes:
- Add TensorRT acceleration option
- Update configuration for TensorRT + ControlNet
- Add error handling for TensorRT compilation

#### 22. `StreamDiffusion/configs/controlnet_examples/sdturbo_depth_trt_example.yaml`
**Why Important**: Configuration example that shows how TensorRT + ControlNet should be configured.

Required changes:
- Complete configuration example for TensorRT + ControlNet
- Add all necessary parameters and settings

## Key Translation Challenges

### 1. Model Detection Adaptation
**Challenge**: ComfyUI_TensorRT uses `model.model.model_config.__class__.__name__` which depends on ComfyUI's model loading infrastructure.

**Solution Required**: Create equivalent detection logic for diffusers models in StreamDiffusion. Need to extract model type, architecture details, and UNet configuration from diffusers models.

### 2. UNet Architecture Extraction
**Challenge**: The `get_control()` method in ComfyUI_TensorRT extracts `channel_mult`, `num_res_blocks`, and `input_block_chans` from ComfyUI's model configs.

**Solution Required**: Extract equivalent architecture details from diffusers UNet models. Map diffusers model configurations to ComfyUI-style architecture descriptions.

### 3. ControlNet Input Configuration
**Challenge**: ComfyUI_TensorRT dynamically creates input tensors for each ControlNet layer based on UNet architecture. This is the core innovation that must be precisely translated.

**Solution Required**: Recreate the `get_control()` logic for diffusers models. Calculate correct tensor shapes for each ControlNet input based on UNet downsampling and channel configurations.

### 4. Runtime Integration
**Challenge**: ComfyUI_TensorRT patches the diffusion model directly, while StreamDiffusion uses pipeline patching.

**Solution Required**: Bridge the two patching approaches. Ensure ControlNet inputs are correctly processed and fed to TensorRT engines during inference.

### 5. ONNX Export Integration
**Challenge**: ComfyUI_TensorRT includes ControlNet inputs during ONNX export via the `get_backbone()` function.

**Solution Required**: Integrate ControlNet input handling into StreamDiffusion's ONNX export process. Ensure all ControlNet inputs are included in the exported model.

## Critical Insights from ComfyUI Analysis

### Model Configuration Pattern
ComfyUI uses a hierarchical model detection system where `detect_unet_config()` extracts architecture details from the state_dict, then `model_config_from_unet_config()` matches this to a model class. This pattern must be replicated for diffusers models.

### Architecture Details Storage
ComfyUI stores critical UNet details like `channel_mult`, `num_res_blocks`, and `transformer_depth` in the model configs, which ComfyUI_TensorRT uses to generate ControlNet input shapes. We need equivalent extraction from diffusers models.

### Dynamic Model Instantiation
ComfyUI's `get_model()` method in each model class creates the actual PyTorch model, which ComfyUI_TensorRT introspects to extract ControlNet input requirements. We need similar introspection capabilities for diffusers models.

## Implementation Strategy

1. **Phase 1**: Study and understand ComfyUI_TensorRT's ControlNet implementation patterns
2. **Phase 2**: Create diffusers-equivalent model detection and architecture extraction
3. **Phase 3**: Implement ControlNet-aware TensorRT model classes for StreamDiffusion
4. **Phase 4**: Update StreamDiffusion's TensorRT acceleration pipeline for ControlNet support
5. **Phase 5**: Integrate with existing ControlNet pipeline and test with SDTurbo

## Success Criteria

The translation will be successful when:
- Any StreamDiffusion script can use `acceleration="tensorrt"` with ControlNet enabled
- TensorRT engines are built with correct ControlNet input tensors
- Runtime performance matches or exceeds PyTorch ControlNet inference
- All ControlNet preprocessors and models work correctly with TensorRT acceleration
- SDTurbo + ControlNet + TensorRT works as the primary target configuration 