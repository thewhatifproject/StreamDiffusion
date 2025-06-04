#!/usr/bin/env python3
"""
TensorRT ControlNet Demo for StreamDiffusion

This example demonstrates how to use TensorRT acceleration with ControlNet support
in StreamDiffusion. It shows the key features of the implementation:

1. Automatic model detection and architecture extraction
2. ControlNet-aware TensorRT engine compilation
3. Runtime ControlNet conditioning with TensorRT engines
4. Performance comparison between PyTorch and TensorRT modes

Usage:
    python examples/tensorrt_controlnet_demo.py --prompt "a robot walking in a park" \
                                                --controlnet_model "thibaud/controlnet-sd21-depth-diffusers" \
                                                --control_image examples/depth_map.png \
                                                --output_dir ./outputs
"""

import argparse
import time
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# StreamDiffusion imports
from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from streamdiffusion.controlnet import BaseControlNetPipeline
from streamdiffusion.controlnet.config import ControlNetConfig


def create_sample_depth_map(width=512, height=512):
    """Create a sample depth map for testing"""
    print("🎨 Creating sample depth map...")
    
    # Create a simple gradient depth map
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a depth map with a circular gradient
    center_x, center_y = width // 2, height // 2
    distance = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    depth_map = 1.0 - np.clip(distance * 2, 0, 1)  # Invert so center is closer
    
    # Convert to PIL Image
    depth_image = Image.fromarray((depth_map * 255).astype(np.uint8), mode='L')
    
    return depth_image


def setup_base_stream(model_id="stabilityai/sd-turbo", **kwargs):
    """Setup base StreamDiffusion pipeline"""
    print(f"🚀 Setting up StreamDiffusion with {model_id}...")
    
    stream = StreamDiffusion(
        model_id_or_path=model_id,
        t_index_list=[35, 45],  # Good for SDTurbo
        torch_dtype=torch.float16,
        device="cuda",
        **kwargs
    )
    
    return stream


def benchmark_inference(pipeline, num_iterations=10):
    """Benchmark inference performance"""
    print(f"⏱️ Benchmarking inference ({num_iterations} iterations)...")
    
    # Warmup
    for _ in range(3):
        try:
            _ = pipeline.stream()
        except:
            pass  # Ignore warmup errors
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    successful_iterations = 0
    for i in range(num_iterations):
        try:
            result = pipeline.stream()
            successful_iterations += 1
        except Exception as e:
            print(f"❌ Iteration {i} failed: {e}")
            continue
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    if successful_iterations > 0:
        avg_time = (end_time - start_time) / successful_iterations
        print(f"✅ Average inference time: {avg_time:.3f}s ({successful_iterations}/{num_iterations} successful)")
        return avg_time
    else:
        print("❌ All iterations failed!")
        return None


def main():
    parser = argparse.ArgumentParser(description="TensorRT ControlNet Demo")
    parser.add_argument("--model_id", type=str, default="stabilityai/sd-turbo",
                        help="Base model ID")
    parser.add_argument("--controlnet_model", type=str, 
                        default="thibaud/controlnet-sd21-depth-diffusers",
                        help="ControlNet model ID")
    parser.add_argument("--prompt", type=str, 
                        default="a robot walking in a park, highly detailed",
                        help="Text prompt")
    parser.add_argument("--control_image", type=str, default=None,
                        help="Path to control image (depth map)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--engine_dir", type=str, default="./engines",
                        help="TensorRT engine directory")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--guidance_scale", type=float, default=1.2,
                        help="Guidance scale")
    parser.add_argument("--controlnet_scale", type=float, default=0.8,
                        help="ControlNet conditioning scale")
    parser.add_argument("--skip_pytorch", action="store_true",
                        help="Skip PyTorch benchmark")
    parser.add_argument("--skip_tensorrt", action="store_true",
                        help="Skip TensorRT compilation and benchmark")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Only run benchmarks, don't save images")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create control image
    if args.control_image and Path(args.control_image).exists():
        print(f"📁 Loading control image: {args.control_image}")
        control_image = Image.open(args.control_image).convert('L')
        control_image = control_image.resize((args.width, args.height))
    else:
        print("🎨 No control image provided, creating sample depth map")
        control_image = create_sample_depth_map(args.width, args.height)
        
        # Save sample depth map
        depth_path = os.path.join(args.output_dir, "sample_depth_map.png")
        control_image.save(depth_path)
        print(f"💾 Saved sample depth map: {depth_path}")
    
    # Setup ControlNet configuration
    controlnet_config = ControlNetConfig(
        model_id=args.controlnet_model,
        conditioning_scale=args.controlnet_scale,
        enabled=True,
        preprocessor=None,  # Use raw depth map
    )
    
    print("\n" + "="*60)
    print("🧪 PYTORCH CONTROLNET PIPELINE")
    print("="*60)
    
    pytorch_time = None
    if not args.skip_pytorch:
        try:
            # Setup PyTorch pipeline
            stream = setup_base_stream(
                model_id=args.model_id,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
            )
            stream.prepare(prompt=args.prompt)
            
            # Add ControlNet
            controlnet_pipeline = BaseControlNetPipeline(stream)
            controlnet_pipeline.add_controlnet(controlnet_config, control_image)
            
            print("🐍 PyTorch ControlNet pipeline ready!")
            
            # Benchmark PyTorch
            pytorch_time = benchmark_inference(controlnet_pipeline)
            
            # Generate sample image
            if not args.benchmark_only:
                print("🖼️ Generating PyTorch sample...")
                result = controlnet_pipeline.stream()
                if result is not None:
                    pytorch_path = os.path.join(args.output_dir, "pytorch_result.png")
                    result.save(pytorch_path)
                    print(f"💾 Saved PyTorch result: {pytorch_path}")
            
            # Cleanup
            del controlnet_pipeline, stream
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ PyTorch pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🚀 TENSORRT CONTROLNET PIPELINE")
    print("="*60)
    
    tensorrt_time = None
    if not args.skip_tensorrt:
        try:
            # Setup TensorRT pipeline
            stream = setup_base_stream(
                model_id=args.model_id,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
            )
            
            # Add ControlNet BEFORE TensorRT acceleration
            controlnet_pipeline = BaseControlNetPipeline(stream)
            controlnet_pipeline.add_controlnet(controlnet_config, control_image)
            
            print("🏗️ Compiling TensorRT engines with ControlNet support...")
            
            # Apply TensorRT acceleration (should detect ControlNet automatically)
            stream = accelerate_with_tensorrt(
                stream=stream,
                engine_dir=args.engine_dir,
                max_batch_size=2,
                min_batch_size=1,
                use_cuda_graph=True,
                engine_build_options={
                    "opt_batch_size": 1,
                    "max_workspace_size": 4 << 30,  # 4GB
                }
            )
            
            # Prepare prompt after TensorRT acceleration
            stream.prepare(prompt=args.prompt)
            
            print("🚀 TensorRT ControlNet pipeline ready!")
            
            # Verify ControlNet integration
            if hasattr(stream.unet, 'use_control') and stream.unet.use_control:
                print("✅ TensorRT UNet has ControlNet support enabled")
                print(f"🏗️ Architecture: {getattr(stream.unet, 'unet_arch', {}).get('model_channels', 'unknown')}")
            else:
                print("⚠️ TensorRT UNet does not have ControlNet support")
            
            # Benchmark TensorRT
            tensorrt_time = benchmark_inference(controlnet_pipeline)
            
            # Generate sample image
            if not args.benchmark_only:
                print("🖼️ Generating TensorRT sample...")
                result = controlnet_pipeline.stream()
                if result is not None:
                    tensorrt_path = os.path.join(args.output_dir, "tensorrt_result.png")
                    result.save(tensorrt_path)
                    print(f"💾 Saved TensorRT result: {tensorrt_path}")
            
            # Cleanup
            del controlnet_pipeline, stream
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ TensorRT pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Performance summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    
    if pytorch_time:
        print(f"🐍 PyTorch:  {pytorch_time:.3f}s per inference")
    if tensorrt_time:
        print(f"🚀 TensorRT: {tensorrt_time:.3f}s per inference")
    
    if pytorch_time and tensorrt_time:
        speedup = pytorch_time / tensorrt_time
        print(f"⚡ Speedup:  {speedup:.2f}x")
        
        if speedup > 1.1:
            print("✅ TensorRT acceleration successful!")
        else:
            print("⚠️ TensorRT speedup lower than expected")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main() 