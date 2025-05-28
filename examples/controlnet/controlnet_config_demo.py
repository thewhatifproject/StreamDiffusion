#!/usr/bin/env python3
"""
ControlNet Configuration Demo for StreamDiffusion

This script demonstrates how to use different ControlNet configurations
with static images for testing and experimentation.
"""

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
    ControlNetConfig,
    create_example_configs
)
from streamdiffusion.image_utils import postprocess_image


def test_single_image(pipeline, input_image_path, output_dir):
    """Test ControlNet pipeline with a single input image"""
    print(f"🖼️  Processing {input_image_path}")
    
    # Load input image
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Update ControlNet with input image
    for i in range(len(pipeline.controlnets)):
        pipeline.update_control_image(i, input_image)
    
    # Generate output
    start_time = time.time()
    x_output = pipeline(input_image)
    generation_time = time.time() - start_time
    
    output_image = postprocess_image(x_output, output_type="pil")[0]
    
    # Save results
    input_name = Path(input_image_path).stem
    output_path = output_dir / f"{input_name}_controlnet_output.png"
    output_image.save(output_path)
    
    # Save control images from preprocessors
    for i, preprocessor in enumerate(pipeline.preprocessors):
        if preprocessor is not None:
            control_image = preprocessor.process(input_image)
            control_path = output_dir / f"{input_name}_control_{i}.png"
            control_image.save(control_path)
    
    print(f"✓ Generated in {generation_time:.2f}s, saved to {output_path}")
    return output_image


def batch_test(pipeline, input_dir, output_dir):
    """Test ControlNet pipeline with a directory of images"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Processing {len(image_files)} images from {input_dir}")
    
    total_time = 0
    for i, image_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] ", end="")
        try:
            start_time = time.time()
            test_single_image(pipeline, image_file, output_dir)
            total_time += time.time() - start_time
        except Exception as e:
            print(f"❌ Failed to process {image_file}: {e}")
    
    avg_time = total_time / len(image_files)
    print(f"📊 Batch complete. Average generation time: {avg_time:.2f}s")


def interactive_demo(pipeline):
    """Interactive demo for testing different prompts and scales"""
    print("\n🎮 Interactive ControlNet Demo")
    print("Commands:")
    print("  prompt <text> - Change the prompt")
    print("  scale <index> <value> - Change ControlNet scale")
    print("  list - List current ControlNets")
    print("  generate <image_path> - Generate with input image")
    print("  quit - Exit interactive mode")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
                
            if command[0] == "quit":
                break
            elif command[0] == "prompt":
                new_prompt = " ".join(command[1:])
                pipeline.stream.update_prompt(new_prompt)
                print(f"✓ Updated prompt: {new_prompt}")
            elif command[0] == "scale":
                if len(command) >= 3:
                    index = int(command[1])
                    scale = float(command[2])
                    pipeline.update_controlnet_scale(index, scale)
                    print(f"✓ Updated ControlNet {index} scale to {scale}")
                else:
                    print("Usage: scale <index> <value>")
            elif command[0] == "list":
                print("ControlNets:")
                for i, (cn, scale) in enumerate(zip(pipeline.controlnets, pipeline.controlnet_scales)):
                    print(f"  {i}: scale={scale}")
            elif command[0] == "generate":
                if len(command) >= 2:
                    image_path = command[1]
                    if Path(image_path).exists():
                        output_dir = Path("interactive_outputs")
                        output_dir.mkdir(exist_ok=True)
                        test_single_image(pipeline, image_path, output_dir)
                    else:
                        print(f"❌ Image not found: {image_path}")
                else:
                    print("Usage: generate <image_path>")
            else:
                print(f"❌ Unknown command: {command[0]}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="ControlNet Configuration Demo")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to ControlNet configuration file")
    parser.add_argument("--input", type=str,
                       help="Input image or directory")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--create-examples", action="store_true",
                       help="Create example configuration files")
    
    args = parser.parse_args()
    
    if args.create_examples:
        output_dir = Path("example_configs")
        create_example_configs(output_dir)
        print(f"✓ Created example configurations in {output_dir}")
        return
    
    print("🚀 Starting ControlNet Configuration Demo")
    
    # Load configuration
    try:
        config = load_controlnet_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
        print(f"📝 Base model: {config.model_id}")
        print(f"📝 Prompt: {config.prompt}")
        print(f"📝 ControlNets: {len(config.controlnets)}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Create ControlNet pipeline
    try:
        print("🔄 Creating ControlNet pipeline...")
        pipeline = create_controlnet_pipeline(config)
        print("✓ Pipeline created successfully")
        
        # Print ControlNet info
        for i, (cn_config, scale) in enumerate(zip(config.controlnets, pipeline.controlnet_scales)):
            print(f"  ControlNet {i}: {cn_config.model_id} (scale: {scale})")
        
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return
    
    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.interactive:
        interactive_demo(pipeline)
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            test_single_image(pipeline, input_path, output_dir)
        elif input_path.is_dir():
            batch_test(pipeline, input_path, output_dir)
        else:
            print(f"❌ Input path not found: {args.input}")
    else:
        print("❌ No input specified. Use --input or --interactive")
        print("💡 Try: python controlnet_config_demo.py --create-examples")


if __name__ == "__main__":
    main() 