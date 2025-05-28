#!/usr/bin/env python3
"""
Lineart ControlNet Webcam Demo for StreamDiffusion

This script demonstrates real-time anime-style line art generation using webcam input.
It extracts line art from the webcam feed and uses it to condition the generation with anime-style ControlNet.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys
import time

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from streamdiffusion.controlnet import (
    load_controlnet_config, 
    create_controlnet_pipeline,
    ControlNetConfig
)
from streamdiffusion.image_utils import postprocess_image


def main():
    parser = argparse.ArgumentParser(description="Lineart ControlNet Webcam Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent / "configs" / "controlnet_examples" / "lineart_example.yaml"
    default_model = r"C:\_dev\comfy\ComfyUI\models\checkpoints\kohaku-v2.1.safetensors"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to lineart ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str, 
                       default=str(default_model),
                       help="Path to base model file")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--controlnet-scale", type=float, default=1.0,
                       help="ControlNet conditioning scale")
    parser.add_argument("--show-lineart", action="store_true",
                       help="Show the extracted line art in a separate window")
    
    args = parser.parse_args()
    
    print("🎨 Starting Lineart ControlNet Webcam Demo")
    
    # Load configuration
    config = load_controlnet_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")
    
    # Override parameters if provided
    if args.model:
        config.model_id = args.model
    if args.prompt:
        config.prompt = args.prompt
    if args.controlnet_scale != 1.0:
        config.controlnets[0].conditioning_scale = args.controlnet_scale
    
    # Create ControlNet pipeline
    print("🔄 Creating Lineart ControlNet pipeline...")
    print(f"📝 Using ControlNet: {config.controlnets[0].model_id}")
    pipeline = create_controlnet_pipeline(config)
    print("✓ Pipeline created successfully")
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    
    print("✓ Camera opened successfully")
    print(f"📝 Prompt: {config.prompt}")
    print(f"🎛️  ControlNet Scale: {config.controlnets[0].conditioning_scale}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    print("  - Press 'l' to toggle line art preview")
    print("  - Press '+' to increase ControlNet scale")
    print("  - Press '-' to decrease ControlNet scale")
    print("  - Press 'p' to change prompt interactively")
    
    frame_count = 0
    show_lineart = args.show_lineart
    fps_counter = []
    
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
            
            # Update ControlNet with current frame
            pipeline.update_control_image(0, frame_pil)
            
            # Generate image
            x_output = pipeline(frame_pil)
            output_image = postprocess_image(x_output, output_type="pil")[0]
            
            # Convert back to BGR for display
            output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Get line art if showing
            lineart_cv = None
            if show_lineart and len(pipeline.preprocessors) > 0:
                preprocessor = pipeline.preprocessors[0]
                if preprocessor is not None:
                    lineart_pil = preprocessor.process(frame_pil)
                    lineart_cv = cv2.cvtColor(np.array(lineart_pil), cv2.COLOR_RGB2BGR)
            
            # Create display layout
            display_frame = cv2.resize(frame, (512, 512))
            output_display = cv2.resize(output_cv, (512, 512))
            
            if show_lineart and lineart_cv is not None:
                lineart_display = cv2.resize(lineart_cv, (256, 256))
                # Create 3-panel layout: Input | Lineart | Output
                top_row = np.hstack([
                    cv2.resize(display_frame, (256, 256)), 
                    lineart_display
                ])
                bottom_row = cv2.resize(output_display, (512, 256))
                combined = np.vstack([top_row, bottom_row])
            else:
                # Simple side-by-side layout
                combined = np.hstack([display_frame, output_display])
            
            # Calculate FPS
            end_time = time.time()
            frame_time = end_time - start_time
            fps_counter.append(frame_time)
            if len(fps_counter) > 30:  # Keep last 30 frames
                fps_counter.pop(0)
            avg_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            
            # Add info overlay
            current_scale = pipeline.controlnet_scales[0]
            info_text = f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Scale: {current_scale:.2f}"
            
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add labels
            if show_lineart and lineart_cv is not None:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Line Art", (266, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Generated", (10, combined.shape[0] - 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(combined, "Input", (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Generated", (522, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Lineart ControlNet StreamDiffusion', combined)
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current output
                try:
                    timestamp = int(time.time())
                    output_path = f"lineart_output_{timestamp}.png"
                    output_image.save(output_path)
                    print(f"💾 Saved output to {output_path}")
                    
                    if show_lineart and lineart_cv is not None:
                        lineart_path = f"lineart_control_{timestamp}.png"
                        lineart_pil.save(lineart_path)
                        print(f"💾 Saved line art to {lineart_path}")
                        
                except Exception as save_error:
                    print(f"❌ Failed to save: {save_error}")
            elif key == ord('l'):
                # Toggle line art preview
                show_lineart = not show_lineart
                print(f"🖼️  Line art preview: {'ON' if show_lineart else 'OFF'}")
            elif key == ord('+'):
                new_scale = min(2.0, pipeline.controlnet_scales[0] + 0.1)
                pipeline.update_controlnet_scale(0, new_scale)
                print(f"📈 ControlNet scale: {new_scale:.2f}")
            elif key == ord('-'):
                new_scale = max(0.0, pipeline.controlnet_scales[0] - 0.1)
                pipeline.update_controlnet_scale(0, new_scale)
                print(f"📉 ControlNet scale: {new_scale:.2f}")
            elif key == ord('p'):
                # Interactive prompt change
                print("\n🎨 Enter new prompt (or press Enter to keep current):")
                try:
                    new_prompt = input(f"Current: {config.prompt}\nNew: ").strip()
                    if new_prompt:
                        pipeline.stream.update_prompt(new_prompt)
                        config.prompt = new_prompt
                        print(f"✓ Updated prompt: {new_prompt}")
                except:
                    print("❌ Failed to update prompt")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Lineart demo finished")
        if fps_counter:
            avg_fps = len(fps_counter) / sum(fps_counter)
            print(f"📊 Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main() 