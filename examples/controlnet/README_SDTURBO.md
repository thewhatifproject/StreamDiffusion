# Turbo ControlNet for StreamDiffusion

This document explains how to use the Turbo ControlNet functionality in StreamDiffusion for real-time image-to-image generation with minimal latency, supporting both SD Turbo and SD-XL Turbo.

## Overview

Turbo ControlNet provides:
- **Ultra-fast inference**: Single/few-step generation with SD Turbo and SD-XL Turbo
- **Real-time img2img**: Image-to-image generation with ControlNet conditioning
- **Minimal latency**: Optimized for webcam and real-time applications
- **Multiple ControlNet support**: Use multiple ControlNets simultaneously
- **Advanced control parameters**: Control guidance start/end, strength adjustment
- **Unified demo**: Single script that automatically supports SD 1.5, SD Turbo, and SD-XL Turbo

## Key Features

### Performance Optimizations
- **Few-step inference**: SD Turbo (1 step) and SD-XL Turbo (2-4 steps) for high-quality results
- **Tiny AutoEncoder**: TAESD for SD Turbo, TAESDXL for SD-XL Turbo
- **GPU-optimized preprocessing**: Efficient control image processing
- **Memory-efficient**: Reduced memory usage compared to standard pipelines

### Advanced Parameters
- **Strength**: Controls how much the input image is modified (0.1-1.0)
- **Control Guidance Start/End**: Fine-tune when ControlNet guidance is applied
- **No CFG**: Turbo models work best with guidance_scale=0.0
- **Multiple ControlNets**: Combine different types of control (pose + depth, etc.)

## Quick Start

### 1. Unified ControlNet Demo

The main `controlnet_webcam_demo.py` now automatically supports SD 1.5, SD Turbo, and SD-XL Turbo based on your configuration file:

```bash
# Run with SD Turbo configuration (512x512, 1 step)
python controlnet_webcam_demo.py --config ../configs/controlnet_examples/sdturbo_canny_example.yaml

# Run with SD-XL Turbo configuration (1024x1024, 2 steps)
python controlnet_webcam_demo.py --config ../configs/controlnet_examples/sdxlturbo_canny_example.yaml

# Run with SD 1.5 configuration (512x512, 50 steps)
python controlnet_webcam_demo.py --config ../configs/controlnet_examples/canny_sd15_example.yaml

# Use custom prompt with SD-XL Turbo
python controlnet_webcam_demo.py --config ../configs/controlnet_examples/sdxlturbo_canny_example.yaml --prompt "a cyberpunk robot, neon lights, futuristic"

# Adjust Turbo-specific parameters
python controlnet_webcam_demo.py --config sdxlturbo_config.yaml --strength 0.6 --controlnet-scale 0.7
```

### 2. The script automatically detects the pipeline type

The demo reads the `pipeline_type` field from your configuration and automatically:
- Uses the appropriate pipeline (SD 1.5 StreamDiffusion, SD Turbo img2img, or SD-XL Turbo img2img)
- Sets appropriate resolution (512x512 for SD 1.5/SD Turbo, 1024x1024 for SD-XL Turbo)
- Enables relevant controls (strength adjustment for Turbo variants)
- Shows appropriate performance metrics and UI information

## Configuration Examples

### SD Turbo Canny ControlNet
```yaml
model_id: "stabilityai/sd-turbo"
pipeline_type: "sdturbo"  # Single-step SD Turbo
prompt: "a beautiful artwork, highly detailed"
guidance_scale: 0.0  # No CFG for SD Turbo
num_inference_steps: 1
strength: 0.8
width: 512
height: 512
use_taesd: true

controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    conditioning_scale: 0.8
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 50
      high_threshold: 100
    control_guidance_start: 0.0
    control_guidance_end: 1.0
```

### SD-XL Turbo Canny ControlNet
```yaml
model_id: "stabilityai/sdxl-turbo"
pipeline_type: "sdxlturbo"  # Multi-step SD-XL Turbo
prompt: "a beautiful artwork, highly detailed, masterpiece, cinematic lighting"
negative_prompt: "blurry, low quality, distorted, render, 3D, oversaturated"
guidance_scale: 0.0  # No CFG for SD-XL Turbo
num_inference_steps: 2
strength: 0.5  # Lower default strength for SD-XL Turbo
width: 1024
height: 1024
use_taesd: true

controlnets:
  - model_id: "diffusers/controlnet-canny-sdxl-1.0"  # SD-XL specific ControlNet
    conditioning_scale: 0.5
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
    control_guidance_start: 0.0
    control_guidance_end: 1.0
```

### SD 1.5 ControlNet (existing)
```yaml
model_id: "runwayml/stable-diffusion-v1-5"
pipeline_type: "sd1.5"  # Uses StreamDiffusion pipeline
prompt: "a beautiful landscape, highly detailed"
guidance_scale: 1.2
num_inference_steps: 50
width: 512
height: 512

controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    conditioning_scale: 1.0
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
```

## API Usage

### Basic Usage
```python
from streamdiffusion.controlnet import (
    create_controlnet_pipeline_auto,
    StreamDiffusionControlNetConfig,
    ControlNetConfig
)

# Create SD-XL Turbo configuration
config = StreamDiffusionControlNetConfig(
    model_id="stabilityai/sdxl-turbo",
    pipeline_type="sdxlturbo",  # Auto-selects SD-XL Turbo pipeline
    prompt="a beautiful landscape",
    strength=0.5,
    width=1024,
    height=1024,
    controlnets=[
        ControlNetConfig(
            model_id="diffusers/controlnet-canny-sdxl-1.0",
            conditioning_scale=0.5,
            preprocessor="canny"
        )
    ]
)

# Create pipeline (automatically selects SD-XL Turbo based on config)
pipeline = create_controlnet_pipeline_auto(config)

# Generate image
output = pipeline(
    image=input_image,  # PIL Image for img2img
    strength=0.5,
    num_inference_steps=2
)
```

### Advanced Usage with Multiple ControlNets
```python
# Multi-ControlNet SD-XL Turbo configuration
config = StreamDiffusionControlNetConfig(
    model_id="stabilityai/sdxl-turbo",
    pipeline_type="sdxlturbo",
    prompt="a person in a detailed environment",
    width=1024,
    height=1024,
    controlnets=[
        ControlNetConfig(
            model_id="diffusers/controlnet-openpose-sdxl-1.0",
            conditioning_scale=0.6,
            preprocessor="openpose"
        ),
        ControlNetConfig(
            model_id="diffusers/controlnet-depth-sdxl-1.0",
            conditioning_scale: 0.4,
            preprocessor="depth"
        )
    ]
)

pipeline = create_controlnet_pipeline_auto(config)
```

## Performance Comparison

| Pipeline Type | Resolution | Steps | Typical Speed | Use Case |
|---------------|------------|-------|---------------|----------|
| SD 1.5 + ControlNet | 512x512 | 20-50 | ~500-1000ms | High quality, detailed generation |
| SD Turbo + ControlNet | 512x512 | 1 | ~50-150ms | Real-time, interactive applications |
| SD-XL Turbo + ControlNet | 1024x1024 | 2-4 | ~200-400ms | High resolution, fast generation |

## Interactive Controls

The unified demo supports different controls based on the pipeline type:

### Common Controls
- **'q'**: Quit the application
- **'s'**: Save current output image
- **'c'**: Toggle control image preview
- **'+'**: Increase ControlNet conditioning scale
- **'-'**: Decrease ControlNet conditioning scale  
- **'p'**: Change prompt interactively

### Turbo Specific Controls (SD Turbo & SD-XL Turbo)
- **'>'**: Increase img2img strength
- **'<'**: Decrease img2img strength

## Parameters Guide

### SD Turbo Specific Parameters

- **`strength`** (0.1-1.0): How much to modify the input image
  - `0.6-0.9`: Recommended range for SD Turbo
  - Higher values work well due to single-step inference

- **`guidance_scale`**: Should be `0.0` for SD Turbo (no CFG)
- **`num_inference_steps`**: Should be `1` for SD Turbo

### SD-XL Turbo Specific Parameters

- **`strength`** (0.1-1.0): How much to modify the input image
  - `0.3-0.7`: Recommended range for SD-XL Turbo
  - Lower values often work better due to higher quality base

- **`guidance_scale`**: Should be `0.0` for SD-XL Turbo (no CFG)
- **`num_inference_steps`**: Should be `2-4` for SD-XL Turbo
- **`width/height`**: Should be `1024` for optimal SD-XL performance

### ControlNet Parameters

- **`conditioning_scale`** (0.0-2.0): Strength of ControlNet influence
  - SD Turbo: `0.6-1.0` typically works well
  - SD-XL Turbo: `0.3-0.8` often sufficient due to higher base quality
  - SD 1.5: `0.8-1.2` for strong control

- **`control_guidance_start`** (0.0-1.0): When to start ControlNet guidance
- **`control_guidance_end`** (0.0-1.0): When to end ControlNet guidance

## Switching Between Pipelines

To switch between different architectures, simply change the configuration:

```bash
# SD 1.5 with Canny (512x512, 50 steps)
python controlnet_webcam_demo.py --config canny_sd15_example.yaml

# SD Turbo with Canny (512x512, 1 step)
python controlnet_webcam_demo.py --config sdturbo_canny_example.yaml

# SD-XL Turbo with Canny (1024x1024, 2 steps)
python controlnet_webcam_demo.py --config sdxlturbo_canny_example.yaml
```

The script automatically detects the `pipeline_type` field and adapts accordingly.

## Model Compatibility

### Base Models
- **SD Turbo**: Use `stabilityai/sd-turbo`
- **SD-XL Turbo**: Use `stabilityai/sdxl-turbo`
- **SD 1.5**: Any SD 1.5 compatible model

### ControlNets
- **SD Turbo**: Use SD 1.5 ControlNets (e.g., `lllyasviel/control_v11p_sd15_canny`)
- **SD-XL Turbo**: Use SD-XL ControlNets (e.g., `diffusers/controlnet-canny-sdxl-1.0`)
- **SD 1.5**: Use SD 1.5 ControlNets

### Preprocessors
All standard preprocessors are supported across all pipeline types.

## Troubleshooting

### Common Issues

1. **Slow performance with Turbo models**: 
   - Ensure `use_taesd: true` in config
   - Use `guidance_scale: 0.0`
   - Set appropriate steps (1 for SD Turbo, 2-4 for SD-XL Turbo)

2. **Poor quality outputs**:
   - Adjust `conditioning_scale` (lower for SD-XL Turbo)
   - Adjust `strength` parameter
   - Check preprocessor parameters
   - For SD-XL Turbo: ensure using SD-XL specific ControlNets

3. **Memory issues**:
   - SD-XL Turbo requires more VRAM due to 1024x1024 resolution
   - Reduce resolution if needed
   - Use `dtype: "float16"`
   - Enable memory optimizations

4. **Resolution mismatches**:
   - SD-XL Turbo works best at 1024x1024
   - SD Turbo works best at 512x512
   - The demo auto-detects but can be overridden with `--resolution`

## Future Extensions

The unified implementation is designed to be extensible for:
- Additional Turbo variants (LCM, Lightning, etc.)
- Custom ControlNet architectures
- Additional optimization techniques
- Multi-model ensembles

## Performance Tips

### For SD Turbo
1. **Use TAESD**: Always enable `use_taesd: true`
2. **Single step**: Keep `num_inference_steps: 1`
3. **No CFG**: Set `guidance_scale: 0.0`
4. **Optimize resolution**: Use 512x512

### For SD-XL Turbo
1. **Use TAESDXL**: Always enable `use_taesd: true`
2. **Few steps**: Use `num_inference_steps: 2-4`
3. **No CFG**: Set `guidance_scale: 0.0`
4. **Native resolution**: Use 1024x1024 for best quality
5. **Lower ControlNet scales**: Often 0.3-0.6 is sufficient

### For SD 1.5
1. **Use StreamDiffusion optimizations**: Enable TensorRT acceleration
2. **Optimal step count**: Use appropriate `t_index_list`
3. **Batch processing**: Use denoising batches for efficiency 