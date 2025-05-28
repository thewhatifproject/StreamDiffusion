#!/usr/bin/env python3
"""
ControlNet Webcam Demo for StreamDiffusion

This script demonstrates real-time image generation using ControlNet with webcam input.
It applies edge detection to the webcam feed and uses it to condition the generation.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from streamdiffusion.controlnet import (
    load_controlnet_config, 
    create_controlnet_pipeline,
    ControlNetConfig
)
from streamdiffusion.image_utils import postprocess_image


def main():
    parser = argparse.ArgumentParser(description="ControlNet Webcam Demo")
    parser.add_argument("--config", type=str, 
                       default="../configs/controlnet_examples/canny_example.yaml",
                       help="Path to ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str, 
                       default="../models/checkpoints/kohaku-v2.1.safetensors",
                       help="Path to base model file")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    
    args = parser.parse_args()
    
    print("🚀 Starting ControlNet Webcam Demo")
    
    # Load configuration
    try:
        config = load_controlnet_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Override model path if provided
    if args.model:
        config.model_id = args.model
    
    # Override prompt if provided
    if args.prompt:
        config.prompt = args.prompt
    
    # Create ControlNet pipeline
    try:
        print("🔄 Creating ControlNet pipeline...")
        pipeline = create_controlnet_pipeline(config)
        print("✓ Pipeline created successfully")
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return
    
    print("✓ Camera opened successfully")
    print(f"📝 Prompt: {config.prompt}")
    print("🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    print("  - Press 'r' to reset")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Update ControlNet with current frame
            if len(pipeline.controlnets) > 0:
                pipeline.update_control_image(0, frame_pil)
            
            # Generate image
            try:
                x_output = pipeline(frame_pil)
                output_image = postprocess_image(x_output, output_type="pil")[0]
                
                # Convert back to BGR for display
                output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
                
                # Resize for display if needed
                display_width = 1024
                aspect_ratio = output_cv.shape[0] / output_cv.shape[1]
                display_height = int(display_width * aspect_ratio)
                output_display = cv2.resize(output_cv, (display_width, display_height))
                
                # Create side-by-side display
                input_display = cv2.resize(frame, (display_width // 2, display_height))
                output_display_half = cv2.resize(output_cv, (display_width // 2, display_height))
                
                combined = np.hstack([input_display, output_display_half])
                
                # Add text overlay
                cv2.putText(combined, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, "Input", (10, display_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "ControlNet Output", (display_width // 2 + 10, display_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('ControlNet StreamDiffusion', combined)
                
                frame_count += 1
                
            except Exception as e:
                print(f"⚠️  Generation failed: {e}")
                cv2.imshow('ControlNet StreamDiffusion', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current output
                try:
                    output_path = f"controlnet_output_{frame_count}.png"
                    output_image.save(output_path)
                    print(f"💾 Saved output to {output_path}")
                except:
                    print("❌ Failed to save output")
            elif key == ord('r'):
                # Reset frame count
                frame_count = 0
                print("🔄 Reset frame counter")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Demo finished")


if __name__ == "__main__":
    main() 