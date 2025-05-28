#!/usr/bin/env python3
"""
Enhanced Webcam Test with ControlNet Support

This is an enhanced version of the original webcam_test.py that demonstrates
how to integrate ControlNet functionality with StreamDiffusion.
"""

import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.controlnet import ControlNetPipeline, ControlNetConfig
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="StreamDiffusion Webcam Test with ControlNet")
    parser.add_argument("--model", type=str, 
                       default="../models/checkpoints/kohaku-v2.1.safetensors",
                       help="Path to the model file")
    parser.add_argument("--controlnet", type=str,
                       default="lllyasviel/control_v11p_sd15_canny",
                       help="ControlNet model to use")
    parser.add_argument("--prompt", type=str,
                       default="1girl with dog hair, thick frame glasses, masterpiece",
                       help="Text prompt for generation")
    parser.add_argument("--use-controlnet", action="store_true",
                       help="Enable ControlNet conditioning")
    parser.add_argument("--controlnet-scale", type=float, default=1.0,
                       help="ControlNet conditioning scale")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    
    args = parser.parse_args()
    
    print("🚀 Loading StreamDiffusion with ControlNet support...")
    
    # Load the local model
    pipe = StableDiffusionPipeline.from_single_file(args.model).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    
    # Initialize StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=[32, 45],
        torch_dtype=torch.float16,
    )
    
    # Load and fuse LCM
    stream.load_lcm_lora()
    stream.fuse_lora()
    
    # Enable acceleration
    pipe.enable_xformers_memory_efficient_attention()
    
    # Create ControlNet pipeline if requested
    if args.use_controlnet:
        print(f"🎛️  Setting up ControlNet: {args.controlnet}")
        controlnet_pipeline = ControlNetPipeline(stream)
        
        # Create ControlNet configuration
        controlnet_config = ControlNetConfig(
            model_id=args.controlnet,
            conditioning_scale=args.controlnet_scale,
            preprocessor="canny",  # Use Canny edge detection
            preprocessor_params={
                "low_threshold": 100,
                "high_threshold": 200
            }
        )
        
        # Add ControlNet to pipeline
        controlnet_pipeline.add_controlnet(controlnet_config)
        
        # Use ControlNet pipeline
        generation_pipeline = controlnet_pipeline
        print("✓ ControlNet enabled")
    else:
        generation_pipeline = stream
        print("✓ Using standard StreamDiffusion")
    
    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}")
    
    print("✓ Webcam opened")
    
    # Prepare generation
    generation_pipeline.prepare(args.prompt)
    print(f"📝 Prompt: {args.prompt}")
    
    print("\n🎮 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current output")
    if args.use_controlnet:
        print("  - Press '+' to increase ControlNet scale")
        print("  - Press '-' to decrease ControlNet scale")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
        
        # Update ControlNet image if using ControlNet
        if args.use_controlnet and hasattr(generation_pipeline, 'update_control_image'):
            generation_pipeline.update_control_image(0, frame_pil)
        
        # Generate image
        try:
            x_output = generation_pipeline(frame_pil)
            output_image = postprocess_image(x_output, output_type="pil")[0]
            
            # Convert back to BGR for display
            output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            
            # Create side-by-side display
            display_frame = cv2.resize(frame, (512, 512))
            combined = np.hstack([display_frame, output_cv])
            
            # Add info overlay
            info_text = f"Frame: {frame_count}"
            if args.use_controlnet:
                current_scale = generation_pipeline.controlnet_scales[0]
                info_text += f" | ControlNet: {current_scale:.2f}"
            
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Input", (10, 500), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Generated", (522, 500), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('StreamDiffusion + ControlNet', combined)
            frame_count += 1
            
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            cv2.imshow('StreamDiffusion + ControlNet', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            try:
                output_path = f"controlnet_webcam_output_{frame_count}.png"
                output_image.save(output_path)
                print(f"💾 Saved: {output_path}")
            except:
                print("❌ Failed to save")
        elif key == ord('+') and args.use_controlnet:
            new_scale = min(2.0, generation_pipeline.controlnet_scales[0] + 0.1)
            generation_pipeline.update_controlnet_scale(0, new_scale)
            print(f"📈 ControlNet scale: {new_scale:.2f}")
        elif key == ord('-') and args.use_controlnet:
            new_scale = max(0.0, generation_pipeline.controlnet_scales[0] - 0.1)
            generation_pipeline.update_controlnet_scale(0, new_scale)
            print(f"📉 ControlNet scale: {new_scale:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Demo finished")


if __name__ == "__main__":
    main() 