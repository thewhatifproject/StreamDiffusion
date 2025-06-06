# StreamDiffusion TensorRT ControlNet Implementation Plan

## Executive Summary

After thorough investigation of both ComfyUI_TensorRT (reference) and StreamDiffusion (target), I have identified the exact steps needed to enable TensorRT acceleration with ControlNet support in StreamDiffusion. This plan outlines a systematic approach to translate the proven ComfyUI_TensorRT implementation to StreamDiffusion's architecture.

## Current State Analysis

### ✅ What Works (ComfyUI_TensorRT)
- **Model Architecture Detection**: `detect_version_from_model()` extracts model type from ComfyUI models
- **ControlNet Input Generation**: `get_control()` method dynamically creates ControlNet input tensors based on UNet architecture
- **Runtime Processing**: `TRTDiffusionBackbone.__call__()` handles ControlNet inputs during inference
- **ONNX Export Integration**: `get_backbone()` includes ControlNet inputs in model export

### ❌ What's Missing (StreamDiffusion)
- **No Model Detection**: StreamDiffusion lacks diffusers→ComfyUI model mapping
- **No ControlNet-Aware TRT Models**: UNet class has no `use_control` flag or `get_control()` method
- **No Runtime ControlNet Support**: TensorRT engine doesn't handle ControlNet inputs
- **No Compilation Integration**: TensorRT compilation doesn't detect/handle ControlNet usage

## Implementation Strategy

### Phase 1: Model Detection and Architecture Extraction 

#### 1.1 Create Diffusers Model Detector
**File**: `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/model_detection.py`

```python
def detect_model_from_diffusers_unet(unet: UNet2DConditionModel) -> str:
    """
    Detect model type from diffusers UNet configuration
    Maps diffusers models to ComfyUI model types for compatibility
    """
    # Extract key architecture parameters
    in_channels = unet.config.in_channels
    block_out_channels = unet.config.block_out_channels
    cross_attention_dim = unet.config.cross_attention_dim
    
    # Model detection logic based on architecture
    if cross_attention_dim == 768 and block_out_channels == [320, 640, 1280, 1280]:
        return "SD15"
    elif cross_attention_dim == 2048 and block_out_channels == [320, 640, 1280]:
        return "SDXL" 
    # Add more model patterns...

def extract_unet_architecture(unet: UNet2DConditionModel) -> dict:
    """
    Extract UNet architecture details needed for ControlNet input generation
    """
    return {
        "model_channels": unet.config.block_out_channels[0],
        "channel_mult": tuple(c // unet.config.block_out_channels[0] for c in unet.config.block_out_channels),
        "num_res_blocks": tuple(unet.config.layers_per_block),
        "context_dim": unet.config.cross_attention_dim,
        "in_channels": unet.config.in_channels,
    }
```

#### 1.2 Update TensorRT Models with ControlNet Support
**File**: `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/models.py`

Add ControlNet-aware base class:
```python
class UNetTRT(UNet):  # Inherit from existing UNet class
    def __init__(self, use_control: bool = False, unet_arch: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.use_control = use_control
        self.unet_arch = unet_arch or {}
        
        if self.use_control and self.unet_arch:
            self.control_inputs = self.get_control()
            self._add_control_inputs()
    
    def get_control(self) -> dict:
        """
        Generate ControlNet input configurations based on UNet architecture
        Translates ComfyUI_TensorRT's get_control() method for diffusers models
        """
        if not self.unet_arch:
            return {}
            
        # Calculate input block channels (mimics ComfyUI_TensorRT logic)
        input_block_chans = self._calculate_input_block_channels()
        
        control_inputs = {}
        
        # Generate input control tensors
        for i, (ch, ds) in enumerate(reversed(input_block_chans)):
            control_inputs[f"input_control_{i}"] = {
                "batch": self.min_batch,  # Will be dynamic
                "channels": ch,
                "height": self.min_latent_shape * ds,  # Will be dynamic  
                "width": self.min_latent_shape * ds,   # Will be dynamic
            }
        
        # Generate output control tensors
        for i, (ch, ds) in enumerate(input_block_chans):
            control_inputs[f"output_control_{i}"] = {
                "batch": self.min_batch,
                "channels": ch, 
                "height": self.min_latent_shape * ds,
                "width": self.min_latent_shape * ds,
            }
        
        # Middle control tensor
        ch, ds = input_block_chans[-1]
        control_inputs["middle_control_0"] = {
            "batch": self.min_batch,
            "channels": ch,
            "height": self.min_latent_shape * ds, 
            "width": self.min_latent_shape * ds,
        }
        
        return control_inputs
```

### Phase 2: Runtime Engine Updates

#### 2.1 Update UNet Engine for ControlNet Support  
**File**: `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/engine.py`

Modify `UNet2DConditionModelEngine.__call__()`:
```python
def __call__(
    self,
    latent_model_input: torch.Tensor,
    timestep: torch.Tensor, 
    encoder_hidden_states: torch.Tensor,
    controlnet_conditioning: Optional[Dict[str, List[torch.Tensor]]] = None,
    **kwargs,
) -> Any:
    # Existing input processing...
    
    # Add ControlNet inputs to shape_dict and input_dict
    if controlnet_conditioning is not None:
        for control_layer, control_tensors in controlnet_conditioning.items():
            for i, tensor in enumerate(control_tensors):
                input_name = f"{control_layer}_control_{i}"
                shape_dict[input_name] = tensor.shape
                input_dict[input_name] = tensor
    
    self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)
    
    noise_pred = self.engine.infer(input_dict, self.stream, use_cuda_graph=self.use_cuda_graph)["latent"]
    return UNet2DConditionOutput(sample=noise_pred)
```

#### 2.2 Create ControlNet-Aware UNet Wrapper
**File**: `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/controlnet_wrapper.py`

```python
class ControlNetUNetWrapper(torch.nn.Module):
    """
    Wrapper that combines UNet with ControlNet inputs for ONNX export
    Mimics ComfyUI_TensorRT's get_backbone() functionality
    """
    def __init__(self, unet: UNet2DConditionModel, control_input_names: List[str]):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
    
    def forward(self, sample, timestep, encoder_hidden_states, *control_args):
        # Organize control inputs
        control_dict = {"input": [], "output": [], "middle": []}
        
        for i, control_input in enumerate(control_args):
            input_name = self.control_input_names[i + 3]  # Skip sample, timestep, encoder_hidden_states
            if "input" in input_name:
                control_dict["input"].append(control_input)
            elif "output" in input_name:
                control_dict["output"].append(control_input)  
            elif "middle" in input_name:
                control_dict["middle"].append(control_input)
        
        # Call UNet with ControlNet conditioning
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=control_dict["input"] if control_dict["input"] else None,
            mid_block_additional_residual=control_dict["middle"][0] if control_dict["middle"] else None,
        )
```

### Phase 3: Compilation Pipeline Integration

#### 3.1 Update TensorRT Acceleration Function
**File**: `StreamDiffusion/src/streamdiffusion/acceleration/tensorrt/__init__.py`

```python
def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    engine_build_options: dict = {},
):
    # Detect if ControlNet is being used
    use_controlnet = hasattr(stream, 'controlnets') and len(stream.controlnets) > 0
    
    if use_controlnet:
        print("🎛️ ControlNet detected - enabling TensorRT ControlNet support")
        
        # Detect model architecture
        model_type = detect_model_from_diffusers_unet(stream.unet)
        unet_arch = extract_unet_architecture(stream.unet)
        
        print(f"📋 Detected model: {model_type}")
        print(f"🏗️ Architecture: {unet_arch}")
        
        # Create ControlNet-aware UNet model
        unet_model = UNetTRT(
            fp16=True,
            device=stream.device, 
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=stream.text_encoder.config.hidden_size,
            unet_dim=stream.unet.config.in_channels,
            use_control=True,
            unet_arch=unet_arch,
        )
        
        # Wrap UNet for ControlNet-aware ONNX export
        control_input_names = unet_model.get_input_names()
        wrapped_unet = ControlNetUNetWrapper(stream.unet, control_input_names)
        
        # Compile with ControlNet support
        compile_unet(
            wrapped_unet,  # Use wrapped UNet
            unet_model,
            create_onnx_path("unet", onnx_dir, opt=False),
            create_onnx_path("unet", onnx_dir, opt=True), 
            unet_engine_path,
            **engine_build_options,
        )
    else:
        # Existing non-ControlNet compilation path
        pass
```

#### 3.2 Update ControlNet Pipeline Integration  
**File**: `StreamDiffusion/src/streamdiffusion/controlnet/base_controlnet_pipeline.py`

Modify `_patch_stream_diffusion()`:
```python
def _patch_stream_diffusion(self) -> None:
    """Patch StreamDiffusion to support ControlNet with TensorRT"""
    # Detect if TensorRT acceleration is being used
    is_tensorrt = hasattr(self.stream.unet, 'engine')
    
    if is_tensorrt:
        print("🚀 TensorRT ControlNet mode enabled")
        self._original_unet_step = self.stream.unet_step
        
        def patched_unet_step_tensorrt(x_t_latent, t_list, idx=None):
            # Process ControlNet conditioning
            if self.controlnets:
                controlnet_conditioning = self._prepare_tensorrt_conditioning(x_t_latent, t_list)
            else:
                controlnet_conditioning = None
            
            # Call TensorRT engine with ControlNet inputs
            return self._original_unet_step(
                x_t_latent, 
                t_list, 
                idx=idx,
                controlnet_conditioning=controlnet_conditioning
            )
        
        self.stream.unet_step = patched_unet_step_tensorrt
    else:
        # Existing PyTorch ControlNet path
        pass

def _prepare_tensorrt_conditioning(self, x_t_latent, t_list) -> Dict[str, List[torch.Tensor]]:
    """
    Prepare ControlNet conditioning in TensorRT format
    Organizes control tensors to match TensorRT engine input expectations
    """
    # Process each ControlNet and organize outputs
    # This mimics ComfyUI_TensorRT's conditioning preparation
    # Returns dict with "input", "output", "middle" keys containing tensor lists
    pass
```

### Phase 4: Testing and Validation

#### 4.1 Unit Tests
Create comprehensive tests for:
- Model detection accuracy
- ControlNet input generation 
- ONNX export with ControlNet inputs
- TensorRT engine compilation
- Runtime inference with ControlNet

#### 4.2 Integration Tests  
- SDTurbo + Depth ControlNet + TensorRT
- SDXL + Canny ControlNet + TensorRT
- Multiple ControlNets + TensorRT
- Performance benchmarking vs PyTorch

#### 4.3 Example Update
**File**: `StreamDiffusion/examples/controlnet/controlnet_webcam_demo.py`

Add TensorRT acceleration option:
```python
# Add to argument parser
parser.add_argument("--acceleration", choices=["none", "tensorrt"], default="none")

# Update stream initialization 
if args.acceleration == "tensorrt":
    print("🚀 Enabling TensorRT acceleration...")
    stream = accelerate_with_tensorrt(
        stream, 
        engine_dir="./engines",
        max_batch_size=args.batch_size,
    )
```

## Key Technical Insights

### 1. Model Architecture Mapping
The core challenge is mapping diffusers UNet configurations to ComfyUI-style architecture descriptions. ComfyUI_TensorRT relies on specific parameters like `channel_mult` and `num_res_blocks` that need to be derived from diffusers configs.

### 2. ControlNet Input Generation  
ComfyUI_TensorRT's `get_control()` method is the critical innovation. It calculates exact tensor shapes for each ControlNet layer based on UNet downsampling patterns. This must be precisely replicated for diffusers models.

### 3. Runtime Integration Bridge
StreamDiffusion uses pipeline-level ControlNet patching while ComfyUI_TensorRT uses model-level patching. The solution is to detect TensorRT usage in the ControlNet pipeline and adapt the conditioning preparation accordingly.

### 4. ONNX Export Adaptation
ComfyUI_TensorRT's `get_backbone()` wrapper approach needs to be adapted for diffusers models, ensuring all ControlNet inputs are included in the exported ONNX model.

## Success Metrics

### ✅ Functional Requirements
- [ ] Any StreamDiffusion script works with `acceleration="tensorrt"` + ControlNet
- [ ] Any StreamDiffusion script works with any acceleration setting, with or WITHOUT Controlnet
- [ ] TensorRT engines build successfully with ControlNet inputs
- [ ] Runtime inference produces correct results
- [ ] All ControlNet preprocessors work with TensorRT
- [ ] SDTurbo + ControlNet + TensorRT as primary target

### ⚡ Performance Requirements  
- [ ] TensorRT ControlNet performance ≥ PyTorch ControlNet performance
- [ ] Memory usage within acceptable bounds
- [ ] Engine compilation time < 10 minutes for common models
- [ ] First inference latency < 2x subsequent inferences

## Implementation Priority

### 🔥 Critical Path (Week 1-2)
1. Model detection and architecture extraction
2. ControlNet-aware UNet model classes
3. Basic ONNX export with ControlNet inputs
4. Simple TensorRT engine compilation

### 🎯 Core Features (Week 3-4)  
1. Runtime engine ControlNet support
2. ControlNet pipeline TensorRT integration
3. SDTurbo + Depth ControlNet working end-to-end
4. Basic testing and validation

### 🚀 Polish and Optimization (Week 5-6)
1. Multiple ControlNet support
2. Performance optimization
3. Comprehensive testing
4. Documentation and examples

## Risk Mitigation

### 🚨 High Risk Areas
- **Model Detection Accuracy**: Ensure all supported models are correctly identified
- **Tensor Shape Calculation**: ControlNet input shapes must be pixel-perfect
- **Memory Management**: TensorRT engines with ControlNet inputs may use significant memory
- **ONNX Export Compatibility**: Complex control flow may challenge ONNX export

### 🛡️ Mitigation Strategies
- Extensive testing against ComfyUI_TensorRT reference outputs
- Gradual rollout starting with SDTurbo only
- Memory profiling and optimization
- Fallback to PyTorch ControlNet if TensorRT compilation fails

## Conclusion

This implementation plan provides a systematic approach to translate ComfyUI_TensorRT's proven ControlNet functionality to StreamDiffusion. The strategy leverages the existing working patterns from ComfyUI_TensorRT while adapting them to StreamDiffusion's diffusers-based architecture.

The key insight is that ComfyUI_TensorRT has already solved the hard problems (ControlNet input calculation, runtime processing, ONNX export). Our task is to carefully translate these solutions to work with diffusers models instead of ComfyUI models.

Success depends on precise implementation of the `get_control()` method for diffusers models and careful integration with StreamDiffusion's existing ControlNet pipeline architecture. 