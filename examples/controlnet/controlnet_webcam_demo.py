#!/usr/bin/env python3
"""
ControlNet Webcam Demo for StreamDiffusion

This script demonstrates real-time image generation using webcam input with ControlNet.
It uses the StreamDiffusionWrapper with ControlNet support for proper separation of concerns.
Supports SD 1.5, SD Turbo, and SD-XL Turbo pipelines based on configuration.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import time
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.controlnet import load_controlnet_config
from streamdiffusion.image_utils import postprocess_image
# Add StreamDiffusion to path
# sys.path.append(str(Path(__file__).parent.parent))


def debug_unet_architecture(wrapper):
    """Debug UNet architecture to understand channel expectations"""
    print("\n🔍 DEBUGGING UNET ARCHITECTURE:")
    
    # Access the underlying UNet
    unet = wrapper.stream.unet
    print(f"UNet config: {unet.config}")
    print(f"UNet class: {type(unet).__name__}")
    
    # Print down block configuration
    print(f"Down block types: {unet.config.down_block_types}")
    print(f"Block out channels: {unet.config.block_out_channels}")
    
    # Inspect down blocks
    print("\n📋 DOWN BLOCKS ANALYSIS:")
    for i, block in enumerate(unet.down_blocks):
        print(f"Down block {i}: {type(block).__name__}")
        print(f"  Block out channels: {block.out_channels if hasattr(block, 'out_channels') else 'N/A'}")
        
        # Look for ResNet blocks
        if hasattr(block, 'resnets'):
            for j, resnet in enumerate(block.resnets):
                print(f"    ResNet {j}: in={resnet.in_channels}, out={resnet.out_channels}")
    
    print(f"\n🎯 CRITICAL INSIGHT:")
    print(f"Standard SD UNet has 4 down blocks with channels: {unet.config.block_out_channels}")
    print(f"But ControlNet tensors should match the ResNet OUTPUT channels, not block_out_channels")


def debug_controlnet_tensors(wrapper):
    """Debug ControlNet tensor processing"""
    print("\n🔍 DEBUGGING CONTROLNET TENSORS:")
    
    # Check if TensorRT mode is active
    print(f"TensorRT active: {hasattr(wrapper.stream, 'trt_unet')}")
    
    if hasattr(wrapper.stream, 'trt_unet'):
        print("🚀 TensorRT MODE:")
        try:
            # Inspect TensorRT inputs
            trt_model = wrapper.stream.trt_unet
            print(f"TensorRT inputs: {list(trt_model.input_names)}")
            for i, input_name in enumerate(trt_model.input_names):
                if 'control' in input_name.lower():
                    print(f"  Control input {i}: {input_name}")
        except Exception as e:
            print(f"TensorRT inspection failed: {e}")


def debug_tensor_matching_deep(wrapper, sample_tensor, timestep_tensor, encoder_hidden_states):
    """Deep debug of tensor matching during UNet forward pass"""
    print("\n🔬 DEEP TENSOR MATCHING DEBUG:")
    
    # Create a small test to understand the processing
    try:
        # Temporarily monkey patch the UNet forward to capture internal processing
        original_forward = wrapper.stream.unet.forward
        
        def debug_forward(*args, **kwargs):
            print(f"\n📍 UNet Forward Called:")
            print(f"Args: {len(args)}")
            print(f"Kwargs keys: {list(kwargs.keys())}")
            
            # Check down_block_additional_residuals processing
            if 'down_block_additional_residuals' in kwargs:
                residuals = kwargs['down_block_additional_residuals']
                print(f"\n🎯 DOWN_BLOCK_ADDITIONAL_RESIDUALS:")
                print(f"Type: {type(residuals)}")
                print(f"Length: {len(residuals) if residuals else 'None'}")
                if residuals:
                    for i, res in enumerate(residuals):
                        if res is not None:
                            print(f"  Residual {i}: {res.shape}")
                        else:
                            print(f"  Residual {i}: None")
            
            # Call original with error catching
            try:
                result = original_forward(*args, **kwargs)
                print("✅ UNet forward succeeded")
                return result
            except Exception as e:
                print(f"❌ UNet forward failed: {e}")
                print(f"Error type: {type(e).__name__}")
                
                # Try to extract more info about the failing operation
                import traceback
                print("📍 Full traceback:")
                traceback.print_exc()
                raise
        
        # Apply the monkey patch temporarily
        wrapper.stream.unet.forward = debug_forward
        
        # Try a simple forward pass
        print("\n🧪 TESTING SIMPLE FORWARD PASS:")
        try:
            # Create minimal test inputs
            test_sample = torch.randn(1, 4, 64, 64, device=sample_tensor.device, dtype=sample_tensor.dtype)
            test_timestep = torch.tensor([500], device=timestep_tensor.device, dtype=timestep_tensor.dtype)
            test_encoder = torch.randn(1, 77, 768, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            
            # Test without ControlNet first
            print("Testing WITHOUT ControlNet...")
            result = wrapper.stream.unet(test_sample, test_timestep, test_encoder)
            print("✅ Base UNet works")
            
            # Test with fake ControlNet residuals
            print("\nTesting WITH ControlNet (fake residuals)...")
            fake_residuals = [
                torch.randn(1, 320, 64, 64, device=test_sample.device, dtype=test_sample.dtype),
                torch.randn(1, 640, 64, 64, device=test_sample.device, dtype=test_sample.dtype),
                torch.randn(1, 1280, 64, 64, device=test_sample.device, dtype=test_sample.dtype),
                None  # 4th position is None for DownBlock2D
            ]
            result = wrapper.stream.unet(
                test_sample, 
                test_timestep, 
                test_encoder,
                down_block_additional_residuals=fake_residuals
            )
            print("✅ UNet with fake ControlNet works!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        finally:
            # Restore original forward
            wrapper.stream.unet.forward = original_forward
            
    except Exception as e:
        print(f"❌ Deep debug failed: {e}")


def debug_actual_tensor_shapes(sample_tensor, control_tensors):
    """Debug actual tensor shapes during forward pass"""
    print(f"\n🔍 TENSOR SHAPE DEBUGGING:")
    print(f"📏 Sample tensor shape: {sample_tensor.shape}")
    
    if control_tensors:
        if isinstance(control_tensors, dict):
            for key, tensors in control_tensors.items():
                if isinstance(tensors, list):
                    for i, tensor in enumerate(tensors):
                        print(f"📏 Control {key}[{i}] shape: {tensor.shape}")
                else:
                    print(f"📏 Control {key} shape: {tensors.shape}")
        elif isinstance(control_tensors, list):
            for i, tensor in enumerate(control_tensors):
                print(f"📏 Control tensor[{i}] shape: {tensor.shape}")
        else:
            print(f"📏 Control tensors type: {type(control_tensors)}")


def get_current_controlnet_scale(wrapper, index=0):
    """Safely get current ControlNet scale"""
    if not wrapper.use_controlnet:
        return 0.0
    if hasattr(wrapper.stream, 'controlnet_scales') and len(wrapper.stream.controlnet_scales) > index:
        return wrapper.stream.controlnet_scales[index]
    return 0.0

def has_controlnets(wrapper):
    """Check if wrapper has active ControlNets"""
    if not wrapper.use_controlnet:
        return False
    return hasattr(wrapper.stream, 'controlnets') and len(getattr(wrapper.stream, 'controlnets', [])) > 0

def main():
    parser = argparse.ArgumentParser(description="ControlNet Webcam Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent.parent / "configs" / "controlnet_examples" / "sdturbo_depth_trt_example.yaml"
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--controlnet-scale", type=float,
                       help="Override ControlNet conditioning scale from config")
    parser.add_argument("--show-preprocessed", action="store_true",
                       help="Show the preprocessed control image in a separate window")
    parser.add_argument("--resolution", type=int, default=None,
                       help="Camera and output resolution (auto-detects from pipeline type if not specified)")
    
    args = parser.parse_args()
    
    print("🎨 Starting ControlNet Webcam Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Detect pipeline type
    pipeline_type = getattr(config, 'pipeline_type', 'sd1.5')
    print(f"🔧 Pipeline type: {pipeline_type}")
    
    # Set default resolution based on pipeline type if not specified
    if args.resolution is None:
        if pipeline_type == 'sdxlturbo':
            args.resolution = 1024  # SD-XL Turbo default
        else:
            args.resolution = 512   # SD 1.5 and SD Turbo default
    
    # Override parameters if provided
    model_id = args.model if args.model else config.model_id
    prompt = args.prompt if args.prompt else config.prompt
    if args.controlnet_scale is not None:
        config.controlnets[0].conditioning_scale = args.controlnet_scale
    
    # Update resolution in config
    config.width = args.resolution
    config.height = args.resolution
    
    # Determine t_index_list and other parameters based on pipeline type
    if pipeline_type == 'sdturbo':
        t_index_list = [0]  # Single step for SD Turbo
        cfg_type = "none"
        use_lcm_lora = False
        use_tiny_vae = True
    elif pipeline_type == 'sdxlturbo':
        t_index_list = [0, 16]  # Two steps for SD-XL Turbo  
        cfg_type = "none"
        use_lcm_lora = False
        use_tiny_vae = False
    else:  # sd1.5
        t_index_list = getattr(config, 't_index_list', [32, 45])
        cfg_type = getattr(config, 'cfg_type', 'self')
        use_lcm_lora = getattr(config, 'use_lcm_lora', True)
        use_tiny_vae = getattr(config, 'use_tiny_vae', True)
    
    # Create ControlNet configuration for wrapper
    controlnet_config = {
        'model_id': config.controlnets[0].model_id,
        'preprocessor': config.controlnets[0].preprocessor,
        'conditioning_scale': config.controlnets[0].conditioning_scale,
        'enabled': config.controlnets[0].enabled,
        'preprocessor_params': getattr(config.controlnets[0], 'preprocessor_params', None),
        'pipeline_type': pipeline_type,  # Add pipeline_type for patching
    }
    
    print("🔄 Creating StreamDiffusion pipeline with ControlNet...")
    print(f"📝 Using ControlNet: {controlnet_config['model_id']}")
    print(f"🔧 Preprocessor: {controlnet_config['preprocessor']}")
    
    # Create StreamDiffusionWrapper with ControlNet support
    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=args.resolution,
        height=args.resolution,
        warmup=10,
        acceleration=getattr(config, 'acceleration', 'none'),  # RESTORED: Test TensorRT with fixed format
        do_add_noise=True,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        use_denoising_batch=True,
        cfg_type=cfg_type,
        seed=getattr(config, 'seed', 2),
        use_safety_checker=False,
        # ControlNet options
        use_controlnet=True,
        controlnet_config=controlnet_config,
    )
    
    print("✓ Pipeline created successfully")
    
    # Debug UNet architecture (COMMENTED OUT FOR CLEAN OUTPUT)
    # debug_unet_architecture(wrapper)
    # debug_controlnet_tensors(wrapper)
    
    # # Prepare some test tensors for deep debugging
    # test_sample = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float16)
    # test_timestep = torch.tensor([500, 600], device="cuda", dtype=torch.float16)
    # test_encoder = torch.randn(2, 77, 768, device="cuda", dtype=torch.float16)
    
    # # Deep debug tensor matching
    # debug_tensor_matching_deep(wrapper, test_sample, test_timestep, test_encoder)
    
    # Prepare the model with prompt
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=getattr(config, 'negative_prompt', ''),
        num_inference_steps=getattr(config, 'num_inference_steps', 50),
        guidance_scale=getattr(config, 'guidance_scale', 1.2 if cfg_type != "none" else 1.0),
        delta=getattr(config, 'delta', 1.0),
    )
    
    print("🔍 DEBUGGING: Checking if TensorRT compilation succeeded...")
    if hasattr(wrapper.stream, 'unet') and hasattr(wrapper.stream.unet, 'engine'):
        print("✅ TensorRT engine detected - compilation succeeded!")
    else:
        print("❌ No TensorRT engine - running in PyTorch mode")
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution)
    
    print("✓ Camera opened successfully")
    print(f"📝 Prompt: {prompt}")
    print(f"🎛️  ControlNet Scale: {controlnet_config['conditioning_scale']}")
    print(f"⚡ Steps: {len(t_index_list)}")
    print(f"📏 Resolution: {args.resolution}x{args.resolution}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    print("  - Press 'c' to toggle control image preview")
    print("  - Press '+' to increase ControlNet scale")
    print("  - Press '-' to decrease ControlNet scale")
    print("  - Press 'p' to change prompt interactively")
    print("  - Press 'd' to debug tensor shapes during inference")
    
    frame_count = 0
    show_preprocessed = args.show_preprocessed
    fps_counter = []
    debug_tensor_shapes = False
    
    # Get preprocessor name for display
    preprocessor_name = controlnet_config['preprocessor'].replace("_", " ").title()
    
    # Profiling variables
    profile_times = {
        'preprocessing': [],
        'generation': [],
        'display': [],
        'total': []
    }
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Profile preprocessing
            prep_start = time.time()
            wrapper.update_control_image_efficient(frame_pil)
            prep_time = time.time() - prep_start
            profile_times['preprocessing'].append(prep_time)
            
            # Debug tensor shapes if requested
            if debug_tensor_shapes and frame_count > 0:
                print(f"\n🔍 FRAME {frame_count} TENSOR DEBUG:")
                # Try to access sample tensor (this is a bit hacky but for debugging)
                try:
                    sample_tensor = torch.randn(1, 4, args.resolution // 8, args.resolution // 8, device='cuda', dtype=torch.float16)
                    print(f"📏 Sample latent tensor shape: {sample_tensor.shape}")
                    
                    # Get ControlNet tensors if available
                    if hasattr(wrapper.stream, 'controlnets'):
                        print(f"📋 Number of ControlNets: {len(wrapper.stream.controlnets)}")
                except Exception as e:
                    print(f"❌ Debug tensor access failed: {e}")
            
            # Profile generation
            gen_start = time.time()
            
            # Generate image using wrapper
            output_image = wrapper(frame_pil)
            
            gen_time = time.time() - gen_start
            profile_times['generation'].append(gen_time)
            
            # Profile display processing
            display_start = time.time()
            # Convert back to BGR for display
            output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Get preprocessed control image from cache (avoid reprocessing)
            control_cv = None
            control_pil = None
            if show_preprocessed:
                control_pil = wrapper.get_last_processed_image(0)
                if control_pil is not None:
                    control_cv = cv2.cvtColor(np.array(control_pil), cv2.COLOR_RGB2BGR)
            
            # Create display layout
            display_frame = cv2.resize(frame, (args.resolution, args.resolution))
            output_display = cv2.resize(output_cv, (args.resolution, args.resolution))
            
            if show_preprocessed and control_cv is not None:
                control_display = cv2.resize(control_cv, (args.resolution//2, args.resolution//2))
                # Create 3-panel layout: Input | Control | Output
                top_row = np.hstack([
                    cv2.resize(display_frame, (args.resolution//2, args.resolution//2)), 
                    control_display
                ])
                bottom_row = cv2.resize(output_display, (args.resolution, args.resolution//2))
                combined = np.vstack([top_row, bottom_row])
            else:
                # Simple side-by-side layout
                combined = np.hstack([display_frame, output_display])
            
            display_time = time.time() - display_start
            profile_times['display'].append(display_time)
            
            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            fps_counter.append(frame_time)
            profile_times['total'].append(frame_time)
            
            if len(fps_counter) > 30:  # Keep last 30 frames
                fps_counter.pop(0)
            avg_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            
            # Add info overlay
            current_scale = get_current_controlnet_scale(wrapper)
            info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Scale: {current_scale:.2f}"
            if debug_tensor_shapes:
                info_text += " | DEBUG ON"
            
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add pipeline type info
            pipeline_display_name = {
                'sd1.5': 'SD 1.5',
                'sdturbo': 'SD TURBO',
                'sdxlturbo': 'SD-XL TURBO'
            }.get(pipeline_type, pipeline_type.upper())
            
            pipeline_text = f"{pipeline_display_name} | Preprocessor: {preprocessor_name}"
            if pipeline_type in ['sdturbo', 'sdxlturbo']:
                steps = len(t_index_list)
                pipeline_text += f" | Steps: {steps}"
            cv2.putText(combined, pipeline_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add timing info
            if frame_count > 0:
                timing_text = f"Prep: {prep_time*1000:.1f}ms | Gen: {gen_time*1000:.1f}ms"
                cv2.putText(combined, timing_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Add labels
            if show_preprocessed and control_cv is not None:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, f"{preprocessor_name}", (args.resolution//2 + 10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                label_text = f"{pipeline_display_name} Generated"
                cv2.putText(combined, label_text, (10, combined.shape[0] - args.resolution//2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                label_text = f"{pipeline_display_name} Generated"
                cv2.putText(combined, label_text, (args.resolution + 10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'ControlNet StreamDiffusion - {preprocessor_name}', combined)
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current output
                try:
                    timestamp = int(time.time())
                    preprocessor_safe = controlnet_config['preprocessor'].replace("/", "_").replace("\\", "_")
                    output_path = f"controlnet_{pipeline_type}_{preprocessor_safe}_output_{timestamp}.png"
                    output_image.save(output_path)
                    print(f"💾 Saved output to {output_path}")
                    
                    if show_preprocessed and control_cv is not None:
                        control_path = f"controlnet_{pipeline_type}_{preprocessor_safe}_control_{timestamp}.png"
                        control_pil.save(control_path)
                        print(f"💾 Saved control image to {control_path}")
                        
                except Exception as save_error:
                    print(f"❌ Failed to save: {save_error}")
            elif key == ord('c'):
                # Toggle control image preview
                show_preprocessed = not show_preprocessed
                print(f"🖼️  Control image preview: {'ON' if show_preprocessed else 'OFF'}")
            elif key == ord('d'):
                # Toggle tensor shape debugging
                debug_tensor_shapes = not debug_tensor_shapes
                print(f"🔍 Tensor shape debugging: {'ON' if debug_tensor_shapes else 'OFF'}")
            elif key == ord('+'):
                # Increase ControlNet scale
                if has_controlnets(wrapper):
                    current_scale = get_current_controlnet_scale(wrapper)
                    new_scale = min(2.0, current_scale + 0.1)
                    wrapper.update_controlnet_scale(0, new_scale)
                    print(f"📈 ControlNet scale: {new_scale:.2f}")
            elif key == ord('-'):
                # Decrease ControlNet scale
                if has_controlnets(wrapper):
                    current_scale = get_current_controlnet_scale(wrapper)
                    new_scale = max(0.0, current_scale - 0.1)
                    wrapper.update_controlnet_scale(0, new_scale)
                    print(f"📉 ControlNet scale: {new_scale:.2f}")
            elif key == ord('p'):
                # Interactive prompt change
                print(f"\n🎨 Enter new prompt (or press Enter to keep current):")
                try:
                    new_prompt = input(f"Current: {prompt}\nNew: ").strip()
                    if new_prompt:
                        # Update prompt via wrapper
                        wrapper.stream.update_prompt(new_prompt)
                        prompt = new_prompt
                        print(f"✓ Updated prompt: {new_prompt}")
                except:
                    print("❌ Failed to update prompt")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 ControlNet demo finished")
        if fps_counter:
            avg_fps = len(fps_counter) / sum(fps_counter)
            print(f"📊 Average FPS: {avg_fps:.2f}")
            if pipeline_type == 'sdturbo':
                print(f"⚡ SD Turbo achieved real-time performance with single-step inference")
            elif pipeline_type == 'sdxlturbo':
                print(f"⚡ SD-XL Turbo achieved high-quality performance with multi-step inference")


if __name__ == "__main__":
    main() 