#!/usr/bin/env python3
"""
ControlNet Webcam Demo for StreamDiffusion

Legacy demo. Use the GUI demo instead.
"""

#TODO: remove this file

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import time
import os
from collections import deque  # OPTIMIZATION: For efficient sliding window FPS tracking

# OPTIMIZATION: Remove heavy imports from module level - import when needed

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
    # OPTIMIZATION: Import heavy modules only when actually needed
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from utils.wrapper import StreamDiffusionWrapper
    from streamdiffusion.controlnet import load_controlnet_config
    from streamdiffusion.image_utils import postprocess_image
    
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
    parser.add_argument("--simple", action="store_true",
                       help="Simple mode: disable all text overlays and interactive features for maximum performance")
    
    args = parser.parse_args()
    
    print("🎨 Starting ControlNet Webcam Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Detect pipeline type
    pipeline_type = config.get('pipeline_type', 'sd1.5')
    print(f"🔧 Pipeline type: {pipeline_type}")
    
    # Set default resolution based on pipeline type if not specified
    if args.resolution is None:
        if pipeline_type == 'sdxlturbo':
            args.resolution = 1024  # SD-XL Turbo default
        else:
            args.resolution = 512   # SD 1.5 and SD Turbo default
    
    # Override parameters if provided
    model_id = args.model if args.model else config['model_id']
    prompt = args.prompt if args.prompt else config['prompt']
    if args.controlnet_scale is not None:
        config['controlnets'][0]['conditioning_scale'] = args.controlnet_scale
    
    # Update resolution in config
    config['width'] = args.resolution
    config['height'] = args.resolution
    
    # Determine t_index_list and other parameters based on pipeline type
    t_index_list = config.get('t_index_list', [0,16])
    if pipeline_type == 'sdturbo':
        cfg_type = config.get('cfg_type', "none")
        use_lcm_lora = config.get('use_lcm_lora', False)
        use_tiny_vae = config.get('use_tiny_vae', True)
    elif pipeline_type == 'sdxlturbo':
        cfg_type = config.get('cfg_type', "none")
        use_lcm_lora = config.get('use_lcm_lora', False)
        use_tiny_vae = config.get('use_tiny_vae', False)
    else:  # sd1.5
        cfg_type = config.get('cfg_type', 'self')
        use_lcm_lora = config.get('use_lcm_lora', True)
        use_tiny_vae = config.get('use_tiny_vae', True)
    
    # Create ControlNet configuration for wrapper
    controlnet_config = {
        'model_id': config['controlnets'][0]['model_id'],
        'preprocessor': config['controlnets'][0]['preprocessor'],
        'conditioning_scale': config['controlnets'][0]['conditioning_scale'],
        'enabled': config['controlnets'][0]['enabled'],
        'preprocessor_params': config['controlnets'][0].get('preprocessor_params', None),
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
        acceleration=config.get('acceleration', 'none'),  # RESTORED: Test TensorRT with fixed format
        do_add_noise=True,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        use_denoising_batch=True,
        cfg_type=cfg_type,
        seed=config.get('seed', 2),
        use_safety_checker=False,
        # ControlNet options
        use_controlnet=True,
        controlnet_config=controlnet_config,
    )
    
    print("✓ Pipeline created successfully")
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=config.get('negative_prompt', ''),
        num_inference_steps=config.get('num_inference_steps', 50),
        guidance_scale=config.get('guidance_scale', 1.1 if cfg_type != "none" else 1.0),
        delta=config.get('delta', 1.0),
    )
    
    print("DEBUGGING: Checking if TensorRT compilation succeeded...")
    if hasattr(wrapper.stream, 'unet') and hasattr(wrapper.stream.unet, 'engine'):
        print("TensorRT engine detected - compilation succeeded!")
    else:
        print("No TensorRT engine - running in PyTorch mode")
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution)
    
    print("✓ Camera opened successfully")
    print(f"Prompt: {prompt}")
    print(f"ControlNet Scale: {controlnet_config['conditioning_scale']}")
    print(f"Steps: {len(t_index_list)}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    
    if not args.simple:
        print("\n Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current output")
        print("  - Press 'c' to toggle control image preview")
        print("  - Press '+' to increase ControlNet scale")
        print("  - Press '-' to decrease ControlNet scale")
        print("  - Press 'p' to change prompt interactively")
        print("  - Press 'd' to debug tensor shapes during inference")
    else:
        print("\nSimple mode enabled - press 'q' to quit")
    
    frame_count = 0
    show_preprocessed = args.show_preprocessed and not args.simple  # Disable in simple mode
    # OPTIMIZATION: Replace list with deque for O(1) operations instead of O(n) pop(0)
    fps_counter = deque(maxlen=30) if not args.simple else None
    debug_tensor_shapes = False
    
    # OPTIMIZATION: Pre-compute display strings and cache them
    if not args.simple:
        preprocessor_name = controlnet_config['preprocessor'].replace("_", " ").title()
        pipeline_display_name = {
            'sd1.5': 'SD 1.5',
            'sdturbo': 'SD TURBO', 
            'sdxlturbo': 'SD-XL TURBO'
        }.get(pipeline_type, pipeline_type.upper())
        
        # Pre-compute static text elements
        steps_text = f" | Steps: {len(t_index_list)}" if pipeline_type in ['sdturbo', 'sdxlturbo'] else ""
        pipeline_text_base = f"{pipeline_display_name} | Preprocessor: {preprocessor_name}{steps_text}"
        generated_label = f"{pipeline_display_name} Generated"
        window_title = f'ControlNet StreamDiffusion - {preprocessor_name}'
        
        # Pre-compute font constants to avoid repeated lookups
        FONT_MAIN = cv2.FONT_HERSHEY_SIMPLEX
        COLOR_GREEN = (0, 255, 0)
        COLOR_CYAN = (0, 255, 255) 
        COLOR_YELLOW = (255, 255, 0)
        COLOR_WHITE = (255, 255, 255)
    else:
        preprocessor_name = ""
        window_title = 'ControlNet StreamDiffusion'
    
    # Profiling variables (disabled in simple mode)
    profile_times = None if args.simple else {
        'preprocessing': [],
        'generation': [],
        'display': [],
        'total': []
    }
    
    # OPTIMIZATION: Use deque for simple mode FPS tracking too
    simple_fps_times = deque(maxlen=30) if args.simple else None
    
    # OPTIMIZATION: Pre-allocate reusable buffers for image processing
    target_size = (args.resolution, args.resolution)
    half_size = (args.resolution//2, args.resolution//2)
    
    # Pre-allocate numpy arrays to avoid repeated allocations
    frame_rgb_buffer = None
    output_bgr_buffer = None
    control_bgr_buffer = None
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # OPTIMIZATION: Resize camera frame first, then convert to RGB (fewer pixels to process)
            frame_resized = cv2.resize(frame, target_size)
            
            # OPTIMIZATION: Reuse buffer for BGR->RGB conversion
            if frame_rgb_buffer is None:
                frame_rgb_buffer = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            else:
                cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB, dst=frame_rgb_buffer)
            
            frame_pil = Image.fromarray(frame_rgb_buffer)
            
            # Profile preprocessing (disabled in simple mode)
            if not args.simple:
                prep_start = time.time()
            wrapper.update_control_image_efficient(frame_pil)
            if not args.simple:
                prep_time = time.time() - prep_start
                profile_times['preprocessing'].append(prep_time)
            
            # Debug tensor shapes if requested (disabled in simple mode)
            if not args.simple and debug_tensor_shapes and frame_count > 0:
                print(f"\n🔍 FRAME {frame_count} TENSOR DEBUG:")
                # Try to access sample tensor (this is a bit hacky but for debugging)
                try:
                    sample_tensor = torch.randn(1, 4, args.resolution // 8, args.resolution // 8, device='cuda', dtype=torch.float16)
                    print(f"📏 Sample latent tensor shape: {sample_tensor.shape}")
                    
                    # Get ControlNet tensors if available
                    if hasattr(wrapper.stream, 'controlnets'):
                        print(f"Number of ControlNets: {len(wrapper.stream.controlnets)}")
                except Exception as e:
                    print(f"Debug tensor access failed: {e}")
            
            # Profile generation (disabled in simple mode)
            if not args.simple:
                gen_start = time.time()
            
            # Generate image using wrapper
            output_image = wrapper(frame_pil)
            
            if not args.simple:
                gen_time = time.time() - gen_start
                profile_times['generation'].append(gen_time)
            
            # Profile display processing (disabled in simple mode)
            if not args.simple:
                display_start = time.time()
            
            # OPTIMIZATION: Reuse buffer for output RGB->BGR conversion
            output_array = np.array(output_image)
            if output_bgr_buffer is None or output_bgr_buffer.shape != output_array.shape:
                output_bgr_buffer = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            else:
                cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR, dst=output_bgr_buffer)
            
            # Get preprocessed control image from cache (avoid reprocessing)
            control_bgr_buffer = None
            control_pil = None
            if show_preprocessed:
                control_pil = wrapper.get_last_processed_image(0)
                if control_pil is not None:
                    # OPTIMIZATION: Reuse buffer for control image conversion
                    control_array = np.array(control_pil)
                    if control_bgr_buffer is None or control_bgr_buffer.shape != control_array.shape:
                        control_bgr_buffer = cv2.cvtColor(control_array, cv2.COLOR_RGB2BGR)
                    else:
                        cv2.cvtColor(control_array, cv2.COLOR_RGB2BGR, dst=control_bgr_buffer)
            
            # OPTIMIZATION: Use already resized frame and output (eliminate redundant resizes)
            display_frame = frame_resized  # Already resized above
            output_display = cv2.resize(output_bgr_buffer, target_size) if output_bgr_buffer.shape[:2] != target_size else output_bgr_buffer
            
            if show_preprocessed and control_bgr_buffer is not None:
                control_display = cv2.resize(control_bgr_buffer, half_size)
                # Create 3-panel layout: Input | Control | Output
                # OPTIMIZATION: Resize display_frame only once for the layout
                input_half = cv2.resize(display_frame, half_size)
                top_row = np.hstack([input_half, control_display])
                bottom_row = cv2.resize(output_display, (args.resolution, args.resolution//2))
                combined = np.vstack([top_row, bottom_row])
            else:
                # Simple side-by-side layout
                combined = np.hstack([display_frame, output_display])
            
            # Profiling and FPS tracking (disabled in simple mode)
            if not args.simple:
                display_time = time.time() - display_start
                profile_times['display'].append(display_time)
                
                # Calculate FPS
                end_time = time.time()
                frame_time = end_time - start_time
                # OPTIMIZATION: deque automatically maintains maxlen, no need for manual pop
                fps_counter.append(frame_time)
                profile_times['total'].append(frame_time)
                
                # OPTIMIZATION: More efficient FPS calculation with deque
                avg_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            else:
                # In simple mode, lightweight FPS tracking
                end_time = time.time()
                frame_time = end_time - start_time
                # OPTIMIZATION: deque automatically maintains maxlen
                simple_fps_times.append(frame_time)
                
                # Print FPS to console every 30 frames (much faster than cv2.putText)
                if frame_count % 30 == 0 and frame_count > 0:
                    avg_fps = len(simple_fps_times) / sum(simple_fps_times) if simple_fps_times else 0
                    print(f"Simple Mode FPS: {avg_fps:.1f}")
                
                avg_fps = 0  # Don't calculate for overlay
            
            # OPTIMIZATION: Reduce putText calls and use pre-computed constants
            if not args.simple:
                current_scale = get_current_controlnet_scale(wrapper)
                info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Scale: {current_scale:.2f}"
                if debug_tensor_shapes:
                    info_text += " | DEBUG ON"
                
                cv2.putText(combined, info_text, (10, 30), 
                           FONT_MAIN, 0.6, COLOR_GREEN, 2)
                
                # Add pipeline type info (use pre-computed base)
                cv2.putText(combined, pipeline_text_base, (10, 60), 
                           FONT_MAIN, 0.5, COLOR_CYAN, 1)
                
                # Add timing info
                if frame_count > 0:
                    timing_text = f"Prep: {prep_time*1000:.1f}ms | Gen: {gen_time*1000:.1f}ms"
                    cv2.putText(combined, timing_text, (10, 90), 
                               FONT_MAIN, 0.4, COLOR_YELLOW, 1)
                
                # OPTIMIZATION: Consolidate label rendering
                if show_preprocessed and control_bgr_buffer is not None:
                    # 3-panel layout labels
                    cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                               FONT_MAIN, 0.5, COLOR_WHITE, 1)
                    cv2.putText(combined, preprocessor_name, (args.resolution//2 + 10, combined.shape[0] - 10), 
                               FONT_MAIN, 0.5, COLOR_WHITE, 1)
                    cv2.putText(combined, generated_label, (10, combined.shape[0] - args.resolution//2 - 10), 
                               FONT_MAIN, 0.5, COLOR_WHITE, 1)
                else:
                    # 2-panel layout labels  
                    cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                               FONT_MAIN, 0.5, COLOR_WHITE, 1)
                    cv2.putText(combined, generated_label, (args.resolution + 10, combined.shape[0] - 10), 
                               FONT_MAIN, 0.5, COLOR_WHITE, 1)
            
            cv2.imshow(window_title, combined)
            frame_count += 1
            
            # Handle key presses (simplified in simple mode)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif not args.simple:
                # Full interactive features only in non-simple mode
                if key == ord('s'):
                    # Save current output
                    try:
                        timestamp = int(time.time())
                        preprocessor_safe = controlnet_config['preprocessor'].replace("/", "_").replace("\\", "_")
                        output_path = f"controlnet_{pipeline_type}_{preprocessor_safe}_output_{timestamp}.png"
                        output_image.save(output_path)
                        print(f"Saved output to {output_path}")
                        
                        if show_preprocessed and control_bgr_buffer is not None:
                            control_path = f"controlnet_{pipeline_type}_{preprocessor_safe}_control_{timestamp}.png"
                            control_pil.save(control_path)
                            print(f"Saved control image to {control_path}")
                            
                    except Exception as save_error:
                        print(f"Failed to save: {save_error}")
                elif key == ord('c'):
                    # Toggle control image preview
                    show_preprocessed = not show_preprocessed
                    print(f"Control image preview: {'ON' if show_preprocessed else 'OFF'}")
                elif key == ord('d'):
                    # Toggle tensor shape debugging
                    debug_tensor_shapes = not debug_tensor_shapes
                    print(f"Tensor shape debugging: {'ON' if debug_tensor_shapes else 'OFF'}")
                elif key == ord('+'):
                    # Increase ControlNet scale
                    if has_controlnets(wrapper):
                        current_scale = get_current_controlnet_scale(wrapper)
                        new_scale = min(2.0, current_scale + 0.1)
                        wrapper.update_controlnet_scale(0, new_scale)
                        print(f"ControlNet scale: {new_scale:.2f}")
                elif key == ord('-'):
                    # Decrease ControlNet scale
                    if has_controlnets(wrapper):
                        current_scale = get_current_controlnet_scale(wrapper)
                        new_scale = max(0.0, current_scale - 0.1)
                        wrapper.update_controlnet_scale(0, new_scale)
                        print(f"ControlNet scale: {new_scale:.2f}")
                elif key == ord('p'):
                    # Interactive prompt change
                    print(f"\nEnter new prompt (or press Enter to keep current):")
                    try:
                        new_prompt = input(f"Current: {prompt}\nNew: ").strip()
                        if new_prompt:
                            # Update prompt via wrapper
                            wrapper.stream.update_prompt(new_prompt)
                            prompt = new_prompt
                            print(f"✓ Updated prompt: {new_prompt}")
                    except:
                        print("Failed to update prompt")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("ControlNet demo finished")
        if not args.simple and fps_counter:
            avg_fps = len(fps_counter) / sum(fps_counter)
            print(f"Average FPS: {avg_fps:.2f}")
            if pipeline_type == 'sdturbo':
                print(f"SD Turbo achieved real-time performance with single-step inference")
            elif pipeline_type == 'sdxlturbo':
                print(f"SD-XL Turbo achieved high-quality performance with multi-step inference")
        elif args.simple and simple_fps_times:
            avg_fps = len(simple_fps_times) / sum(simple_fps_times)
            print(f"Simple Mode Final FPS: {avg_fps:.2f}")
            print(f"Performance optimizations were active - this is your maximum speed!")


if __name__ == "__main__":
    main() 