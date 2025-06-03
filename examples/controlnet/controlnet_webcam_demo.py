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
        acceleration=getattr(config, 'acceleration', 'none'),
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
    
    # Prepare the model with prompt
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=getattr(config, 'negative_prompt', ''),
        num_inference_steps=getattr(config, 'num_inference_steps', 50),
        guidance_scale=getattr(config, 'guidance_scale', 1.2 if cfg_type != "none" else 1.0),
        delta=getattr(config, 'delta', 1.0),
    )
    
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
    
    frame_count = 0
    show_preprocessed = args.show_preprocessed
    fps_counter = []
    
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
            
            # Show profiling info every 30 frames
            if frame_count % 30 == 0 and frame_count > 0:
                recent_prep = profile_times['preprocessing'][-30:]
                recent_gen = profile_times['generation'][-30:]
                recent_display = profile_times['display'][-30:]
                recent_total = profile_times['total'][-30:]
                
                print(f"\n📊 Performance Profile (last 30 frames):")
                print(f"  Preprocessing: {sum(recent_prep)/len(recent_prep)*1000:.1f}ms avg")
                print(f"  Generation:    {sum(recent_gen)/len(recent_gen)*1000:.1f}ms avg")
                print(f"  Display:       {sum(recent_display)/len(recent_display)*1000:.1f}ms avg")
                print(f"  Total:         {sum(recent_total)/len(recent_total)*1000:.1f}ms avg")
                print(f"  FPS:           {avg_fps:.1f}")
                if pipeline_type == 'sdturbo':
                    print(f"  ⚡ SD Turbo single-step inference: {gen_time*1000:.1f}ms")
                elif pipeline_type == 'sdxlturbo':
                    print(f"  ⚡ SD-XL Turbo multi-step inference: {gen_time*1000:.1f}ms")
            
            # Add info overlay
            current_scale = get_current_controlnet_scale(wrapper)
            info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Scale: {current_scale:.2f}"
            
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