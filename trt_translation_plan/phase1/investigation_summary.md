# TensorRT ControlNet Translation Investigation Summary

## 🔍 Investigation Overview

I have conducted a comprehensive analysis of both ComfyUI_TensorRT (the reference implementation) and StreamDiffusion (the target) to understand how to enable TensorRT acceleration with ControlNet support. The investigation examined 22 critical files across both codebases to identify the exact translation requirements.

## 🎯 Key Findings

### ✅ What ComfyUI_TensorRT Does Right (The Reference Implementation)

1. **Dynamic ControlNet Input Generation** (`custom_nodes/ComfyUI_TensorRT/models/sd_unet.py`, lines 80-101)
   - The `get_control()` method is the **core innovation** - it dynamically calculates ControlNet input tensor shapes based on UNet architecture
   - Uses `channel_mult`, `num_res_blocks`, and `input_block_chans` to determine correct tensor dimensions
   - Generates `input_control_*`, `output_control_*`, and `middle_control_0` tensors with precise shapes

2. **Model Detection Infrastructure** (`custom_nodes/ComfyUI_TensorRT/models/supported_models.py`)
   - `detect_version_from_model()` extracts model type: `model.model.model_config.__class__.__name__`
   - Maps to TRT helper classes with ControlNet support enabled via `from_model()` class methods

3. **Runtime ControlNet Processing** (`custom_nodes/ComfyUI_TensorRT/tensorrt_diffusion_model.py`, lines 225-229)
   ```python
   if self.model.use_control and control is not None:
       for control_layer, control_tensors in control.items():
           for i, tensor in enumerate(control_tensors):
               model_inputs[f"{control_layer}_control_{i}"] = tensor
   ```

4. **ONNX Export Integration** (`custom_nodes/ComfyUI_TensorRT/onnx_utils/export.py`, lines 58-70)
   - `get_backbone()` creates wrapper that organizes ControlNet inputs during export
   - Maps control arguments to input/output/middle control dictionaries

### ❌ What StreamDiffusion Currently Lacks

1. **No Diffusers→ComfyUI Model Mapping**
   - StreamDiffusion uses diffusers models, ComfyUI_TensorRT expects ComfyUI model configs
   - Need to extract `channel_mult`, `num_res_blocks` from diffusers UNet configurations

2. **No ControlNet-Aware TensorRT Models**
   - Current `UNet` class in `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/models.py` has no `use_control` flag
   - Missing `get_control()` method for ControlNet input generation

3. **No TensorRT Engine ControlNet Support**
   - `UNet2DConditionModelEngine.__call__()` only handles basic inputs: sample, timestep, encoder_hidden_states
   - No processing of ControlNet conditioning tensors

4. **No Compilation Integration**
   - `accelerate_with_tensorrt()` doesn't detect ControlNet usage or compile with ControlNet support

## 🔧 Critical Translation Challenges Identified

### 1. **Architecture Parameter Extraction** ⚠️ HIGH RISK
**Problem**: ComfyUI_TensorRT relies on ComfyUI's model configs with explicit `channel_mult`, `num_res_blocks` parameters. Diffusers stores this differently.

**Solution Required**: 
```python
# ComfyUI style (what TensorRT expects)
channel_mult = (1, 2, 4, 4)  # From model config
num_res_blocks = (2, 2, 2, 2)  # From model config

# Diffusers style (what we have)
block_out_channels = [320, 640, 1280, 1280]  # From unet.config
layers_per_block = 2  # Single value, not per-layer
```

### 2. **Input Block Channel Calculation** ⚠️ CRITICAL
**Problem**: The `set_block_chans()` method in ComfyUI_TensorRT (lines 43-55) calculates the exact channel dimensions and downsampling factors for each UNet block. This is **essential** for ControlNet input shape calculation.

**Must Replicate Exactly**:
```python
def set_block_chans(self):
    ch = self.hidden_size
    ds = 1
    input_block_chans = [(self.hidden_size, ds)]
    for level, mult in enumerate(self.channel_mult):
        for nr in range(self.num_res_blocks[level]):
            ch = mult * self.hidden_size
            input_block_chans.append((ch, ds))
        if level != len(self.channel_mult) - 1:
            out_ch = ch
            ch = out_ch
            ds *= 2
            input_block_chans.append((ch, ds))
    return input_block_chans
```

### 3. **Runtime Integration Bridge** ⚠️ MEDIUM RISK
**Problem**: StreamDiffusion uses pipeline-level ControlNet patching (`_patch_stream_diffusion()`) while ComfyUI_TensorRT uses model-level patching.

**Solution**: Detect TensorRT mode in ControlNet pipeline and adapt conditioning preparation.

## 🚀 Immediate Action Items

### Phase 1: Foundation (Week 1)
1. **Create Model Detection Module**
   - Map diffusers UNet configs to ComfyUI-style parameters
   - Support SD15, SDXL, SDTurbo initially

2. **Extend TensorRT UNet Model**
   - Add `use_control` flag and `get_control()` method
   - Implement `set_block_chans()` equivalent for diffusers

### Phase 2: Integration (Week 2)  
1. **Update TensorRT Engine**
   - Modify `UNet2DConditionModelEngine.__call__()` to handle ControlNet inputs
   - Add ControlNet input processing to buffer allocation

2. **Create ONNX Export Wrapper**
   - Implement `ControlNetUNetWrapper` similar to ComfyUI_TensorRT's `get_backbone()`

### Phase 3: Pipeline Integration (Week 3)
1. **Update Acceleration Function**
   - Detect ControlNet usage in `accelerate_with_tensorrt()`
   - Use ControlNet-aware compilation when needed

2. **Patch ControlNet Pipeline**
   - Detect TensorRT mode in `_patch_stream_diffusion()`
   - Adapt conditioning preparation for TensorRT engines

## 🧪 Validation Strategy

### Reference Validation
- Compare outputs with ComfyUI_TensorRT using same models/ControlNets
- Validate ControlNet input tensor shapes match exactly
- Ensure runtime performance meets expectations

### Progressive Testing
1. **SDTurbo + Depth ControlNet** (primary target)
2. **SD15 + Canny ControlNet** 
3. **SDXL + Multiple ControlNets**

## 💡 Key Implementation Insights

### 1. **The `get_control()` Method is Everything**
This single method contains the core innovation that makes ControlNet + TensorRT possible. It must be translated with pixel-perfect accuracy.

### 2. **Architecture Detection is Critical**
Without correct `channel_mult` and `num_res_blocks` extraction from diffusers models, ControlNet input shapes will be wrong and compilation will fail.

### 3. **Leverage Existing Patterns**
StreamDiffusion already has excellent ControlNet support for PyTorch. We need to bridge this to TensorRT engines, not rebuild from scratch.

### 4. **ComfyUI_TensorRT Solved the Hard Problems**
The reference implementation has working solutions for:
- ControlNet input calculation
- ONNX export with control inputs  
- Runtime tensor processing
- Model architecture introspection

Our job is translation, not innovation.

## ⚡ Success Criteria

**Primary Goal**: Any StreamDiffusion script with `acceleration="tensorrt"` + ControlNet works identically to PyTorch version but faster.

**Target Configuration**: 
```yaml
acceleration: "tensorrt"
controlnets:
  - model_id: "thibaud/controlnet-sd21-depth-diffusers"
    conditioning_scale: 0.5
    preprocessor: "depth_tensorrt"
```

**Success Metrics**:
- ✅ TensorRT engines compile successfully with ControlNet inputs
- ✅ Runtime inference produces correct results  
- ✅ Performance improvement over PyTorch ControlNet
- ✅ SDTurbo + ControlNet + TensorRT working end-to-end

## 🎯 Next Steps

1. **Start with Model Detection** - Create the diffusers→ComfyUI parameter mapping
2. **Implement Core `get_control()`** - This is the critical path blocker
3. **Test with Simple Case** - SDTurbo + single ControlNet first
4. **Validate Against Reference** - Compare with ComfyUI_TensorRT outputs
5. **Scale to Full Implementation** - Add all model types and features

The investigation has provided a clear roadmap. ComfyUI_TensorRT has proven this is possible and shown us exactly how to do it. The implementation is now a matter of careful translation rather than research. 