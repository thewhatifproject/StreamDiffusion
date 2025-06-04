#!/usr/bin/env python3
"""
Debug script for ControlNet TensorRT dimension issues

This script helps diagnose dimension mismatches by:
1. Loading the actual UNet and ControlNet models
2. Examining their internal structure
3. Testing tensor shapes during forward pass
4. Comparing with expected TensorRT input shapes
"""

import torch
import sys
from pathlib import Path

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent / "src"))

from diffusers import StableDiffusionPipeline, ControlNetModel
from streamdiffusion.acceleration.tensorrt.model_detection import (
    detect_model_from_diffusers_unet, 
    extract_unet_architecture, 
    validate_architecture
)
from streamdiffusion.acceleration.tensorrt.models import UNet
from streamdiffusion.acceleration.tensorrt.controlnet_wrapper import create_controlnet_wrapper


def debug_controlnet_dimensions():
    """Debug ControlNet dimension calculation issues"""
    
    print("🔍 ControlNet Dimension Debug Script")
    print("=" * 50)
    
    # Load the pipeline that's causing issues
    model_path = "C:/_dev/comfy/ComfyUI/models/checkpoints/sd_turbo.safetensors"
    controlnet_id = "thibaud/controlnet-sd21-depth-diffusers"
    
    print(f"📥 Loading pipeline from: {model_path}")
    try:
        pipe = StableDiffusionPipeline.from_single_file(model_path).to("cuda", torch.float16)
        print(f"✅ Loaded pipeline successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return
    
    print(f"📥 Loading ControlNet: {controlnet_id}")
    try:
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to("cuda")
        print(f"✅ Loaded ControlNet successfully")
    except Exception as e:
        print(f"❌ Failed to load ControlNet: {e}")
        return
    
    # Analyze UNet architecture
    print("\n🏗️  UNet Architecture Analysis")
    print("-" * 30)
    
    unet = pipe.unet
    config = unet.config
    
    print(f"UNet Config:")
    print(f"  in_channels: {config.in_channels}")
    print(f"  block_out_channels: {config.block_out_channels}")
    print(f"  cross_attention_dim: {config.cross_attention_dim}")
    print(f"  layers_per_block: {getattr(config, 'layers_per_block', 'Not set')}")
    print(f"  down_block_types: {getattr(config, 'down_block_types', 'Not set')}")
    print(f"  up_block_types: {getattr(config, 'up_block_types', 'Not set')}")
    
    # Model detection
    print(f"\n🔍 Model Detection")
    print("-" * 20)
    
    try:
        model_type = detect_model_from_diffusers_unet(unet)
        print(f"Detected model type: {model_type}")
        
        unet_arch = extract_unet_architecture(unet)
        unet_arch = validate_architecture(unet_arch, model_type)
        
        print(f"Architecture extracted:")
        for key, value in unet_arch.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Model detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test TensorRT UNet model creation
    print(f"\n🎛️  TensorRT UNet Model Creation")
    print("-" * 35)
    
    try:
        # CRITICAL FIX: Use the actual test resolution, not the minimum configured resolution
        test_latent_height, test_latent_width = 512 // 8, 512 // 8
        print(f"🔧 Using actual test latent resolution: {test_latent_height}x{test_latent_width}")
        
        # Temporarily override the min_latent_shape to match our test conditions
        unet_arch_corrected = unet_arch.copy()
        
        trt_unet_model = UNet(
            fp16=True,
            device="cuda",
            max_batch_size=1,
            min_batch_size=1,
            embedding_dim=pipe.text_encoder.config.hidden_size,
            unet_dim=unet.config.in_channels,
            use_control=True,
            unet_arch=unet_arch,
        )
        
        # CRITICAL FIX: Override the min_latent_shape to match test conditions
        trt_unet_model.min_latent_shape = test_latent_height
        
        # Regenerate control inputs with correct resolution
        control_inputs = trt_unet_model.get_control()
        print(f"✅ TensorRT UNet model created with {len(control_inputs)} control inputs")
        
        for name, shape_spec in control_inputs.items():
            print(f"  {name}: {shape_spec}")
        
    except Exception as e:
        print(f"❌ TensorRT UNet creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test ControlNet wrapper creation
    print(f"\n🔗 ControlNet Wrapper Test")
    print("-" * 25)
    
    try:
        control_input_names = trt_unet_model.get_input_names()
        print(f"Input names: {control_input_names}")
        
        wrapped_unet = create_controlnet_wrapper(unet, control_input_names)
        print(f"✅ ControlNet wrapper created successfully")
        
    except Exception as e:
        print(f"❌ ControlNet wrapper creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test tensor creation and forward pass
    print(f"\n🧪 Tensor Dimension Test")
    print("-" * 25)
    
    batch_size = 1
    height, width = 512, 512
    latent_height, latent_width = height // 8, width // 8
    
    print(f"Testing with batch_size={batch_size}, latent_size={latent_height}x{latent_width}")
    
    try:
        # Create base inputs
        sample = torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float16, device="cuda")
        timestep = torch.tensor([1], dtype=torch.float16, device="cuda")
        encoder_hidden_states = torch.randn(batch_size, 77, pipe.text_encoder.config.hidden_size, dtype=torch.float16, device="cuda")
        
        print(f"Base inputs created:")
        print(f"  sample: {sample.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
        
        # Create control inputs
        control_args = []
        for name, shape_spec in control_inputs.items():
            if name not in ["sample", "timestep", "encoder_hidden_states"]:
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                
                control_tensor = torch.randn(batch_size, channels, control_height, control_width, dtype=torch.float16, device="cuda")
                control_args.append(control_tensor)
                print(f"  {name}: {control_tensor.shape}")
        
        print(f"Created {len(control_args)} control tensors")
        
        # Test if the wrapper can handle these dimensions
        print(f"\n🧪 Testing forward pass...")
        try:
            with torch.no_grad():
                output = wrapped_unet(sample, timestep, encoder_hidden_states, *control_args)
                print(f"✅ Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            print(f"This is likely the same error we see during ONNX export")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_controlnet_dimensions() 