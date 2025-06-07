# WIP: ControlNet Support for StreamDiffusion

This implementation adds comprehensive ControlNet support to StreamDiffusion, enabling real-time image generation conditioned on various types of control inputs like edges, depth maps, pose information, and more.

## Features

- Chain multiple ControlNets for complex conditioning
- Use YAML/JSON configs to define your ControlNet pipelines
- Extensible preprocessing system for different ControlNet types
- Drop-in replacement for standard StreamDiffusion with ControlNet capabilities

### Basic Setup

1. Clone or navigate to your StreamDiffusion directory
2. The ControlNet implementation is already included in `src/streamdiffusion/controlnet/`
3. Import and use as shown in the examples below

## Quick Start

### Method 1: Configuration-Based (Recommended)

```python
from streamdiffusion.controlnet import load_controlnet_config, create_controlnet_pipeline

# Load configuration
config = load_controlnet_config("configs/controlnet_examples/canny_example.yaml")

# Create pipeline
pipeline = create_controlnet_pipeline(config)

# Use with images
from PIL import Image
input_image = Image.open("your_image.jpg")
output = pipeline(input_image)
```

### Method 2: Programmatic Setup

```python
from streamdiffusion import StreamDiffusion
from streamdiffusion.controlnet import ControlNetPipeline, ControlNetConfig
from diffusers import StableDiffusionPipeline

# Create base pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
stream = StreamDiffusion(pipe, t_index_list=[32, 45])

# Add ControlNet
controlnet_pipeline = ControlNetPipeline(stream)
controlnet_config = ControlNetConfig(
    model_id="lllyasviel/control_v11p_sd15_canny",
    conditioning_scale=1.0,
    preprocessor="canny"
)
controlnet_pipeline.add_controlnet(controlnet_config)

# Prepare and use
controlnet_pipeline.prepare("a beautiful landscape")
output = controlnet_pipeline(input_image)
```

## Configuration Files

### Example: Canny Edge ControlNet

```yaml
# configs/controlnet_examples/canny_example.yaml
model_id: "stabilityai/sd-turbo"
prompt: "a beautiful landscape, highly detailed"
width: 512
height: 512

controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    conditioning_scale: 1.0
    preprocessor: "canny"
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
    enabled: true
```

### Example: Multiple ControlNets

```yaml
# configs/controlnet_examples/multi_controlnet_example.yaml
model_id: "stabilityai/sd-turbo"
prompt: "a person in a detailed environment"

controlnets:
  # Human pose detection
  - model_id: "lllyasviel/control_v11p_sd15_openpose"
    conditioning_scale: 0.8
    preprocessor: "openpose"
    enabled: true
    
  # Depth information
  - model_id: "lllyasviel/control_v11f1p_sd15_depth"
    conditioning_scale: 0.6
    preprocessor: "depth"
    enabled: true
```

## 🎮 Example Scripts

### Webcam Demo

```bash

# the depth trt examples assume you have a tensorrt engine for depthanything as seen in this repo
# https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt
# additionally, for all of the demo configs, they must be updated to suite your environment. 

# test with simple gui
python ./examples/controlnet/controlnet_webcam_gui_demo.py --config ./configs/controlnet_examples/sdturbo_depth_trt_example.yaml

# Basic webcam test with ControlNet (SD1.5 config)
python ./examples/controlnet/controlnet_webcam_demo.py --config ./configs/controlnet_examples/depth_trt_example.yaml

# Basic webcam test with ControlNet (SDTurbo config)
python ./examples/controlnet/controlnet_webcam_demo.py --config ./configs/controlnet_examples/sdturbo_depth_trt_example.yaml

# Basic webcam test with ControlNet (SDXL config)
python ./examples/controlnet/controlnet_webcam_demo.py --config ./configs/controlnet_examples/sdxlturbo_depth_trt_example.yaml

# Basic optimized webcam demo, lacking keyboard controls but slightly more performant 
python ./examples/controlnet/controlnet_webcam_demo.py --config ./configs/controlnet_examples/depth_trt_example.yaml

# Basic webcam test with ControlNet (SD1.5 config)
python ./examples/controlnet/controlnet_webcam_demo.py --config ./configs/controlnet_examples/lineart_example.yaml


### High Performance Single Controlnet Simulation Demo
#this demo generates images of a square moving around the screen and uses a single canny controlnet to show the performance loss in comparison to the single.py high performance demo
python ./examples/optimal-performance/controlnet_single.py

### Configuration Demo

```bash
# Test with a single image
python ./examples/controlnet_config_demo.py --config ./configs/controlnet_examples/depth_example.yaml --input test_image.jpg

# Batch processing
python ./examples/controlnet_config_demo.py --config ./configs/controlnet_examples/canny_example.yaml --input image_folder/

# Interactive mode
python ./examples/controlnet_config_demo.py --config ./configs/controlnet_examples/multi_controlnet_example.yaml --interactive

# Create example configurations
python ./examples/controlnet_config_demo.py --create-examples
```

## 🔧 Supported ControlNets

### Preprocessors

 - canny
 - depth (WIP)
 - depth tensorrt (WIP)
 - lineart
 - openpose 


## Reference

### ControlNetPipeline

```python
class ControlNetPipeline:
    def add_controlnet(self, controlnet_config: ControlNetConfig, control_image=None) -> int
    def remove_controlnet(self, index: int) -> None
    def clear_controlnets(self) -> None
    def update_control_image(self, index: int, control_image) -> None
    def update_controlnet_scale(self, index: int, scale: float) -> None
```

### ControlNetConfig

```python
@dataclass
class ControlNetConfig:
    model_id: str                    # ControlNet model ID or path
    conditioning_scale: float = 1.0  # Strength (0.0 to 5.0)
    preprocessor: Optional[str] = None
    preprocessor_params: Dict[str, Any] = field(default_factory=dict)
    control_image_path: Optional[str] = None
    enabled: bool = True
```

## Adding Preprocessors

1. Create a new file in `src/streamdiffusion/controlnet/preprocessors/`
2. Inherit from `BasePreprocessor`
3. Implement the `process` method
4. Add to `__init__.py` and register

```python
from streamdiffusion.controlnet.preprocessors import BasePreprocessor, register_preprocessor

class MyCustomPreprocessor(BasePreprocessor):
    def process(self, image):
        # Your custom preprocessing logic
        processed_image = your_processing_function(image)
        return processed_image

# Register your preprocessor
register_preprocessor("my_custom", MyCustomPreprocessor)

# Use in configuration
controlnet_config = ControlNetConfig(
    model_id="your/controlnet",
    preprocessor="my_custom"
)
```


## Some Models 

```python 
"lllyasviel/control_v11p_sd15_canny",         
        "lllyasviel/control_v11f1p_sd15_depth",       
        "lllyasviel/control_v11p_sd15_openpose",      
        "lllyasviel/control_v11p_sd15_scribble",      
        "lllyasviel/sd-controlnet-hed",               
        "lllyasviel/control_v11p_sd15_mlsd",          
        "lllyasviel/control_v11p_sd15_normalbae",     
        "lllyasviel/control_v11p_sd15_seg",           
        "lllyasviel/control_v11p_sd15_lineart",       
        "lllyasviel/control_v11p_sd15s2_lineart_anime", 
        "monster-labs/control_v1p_sd15_qrcode_monster", 
        "monster-labs/control_v1p_sd15_qrcode_monster/v2",  # QR code model v2 (in v2 subfolder)
        "lllyasviel/control_v11p_sd15_inpaint",       
        "lllyasviel/control_v11e_sd15_shuffle",       
        "lllyasviel/control_v11e_sd15_ip2p",          
        "lllyasviel/control_v11f1e_sd15_tile"         
        ```