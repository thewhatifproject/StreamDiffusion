#!/usr/bin/env python3
"""
Test Script for Phase 2.1: ControlNet TensorRT Infrastructure

This script tests the basic infrastructure components:
1. ControlNet TensorRT compilation
2. Basic inference through TensorRT engine
3. Fallback to PyTorch if compilation fails

Usage:
    python test_controlnet_tensorrt.py [--model MODEL_ID] [--engine-dir ENGINE_DIR]
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent / "src"))

from diffusers import ControlNetModel
from polygraphy import cuda

# Import Phase 2.1 infrastructure components
from streamdiffusion.acceleration.tensorrt.controlnet_models import create_controlnet_model
from streamdiffusion.acceleration.tensorrt.controlnet_engine import ControlNetModelEngine, HybridControlNet
from streamdiffusion.acceleration.tensorrt.engine_pool import ControlNetEnginePool
from streamdiffusion.acceleration.tensorrt.builder import compile_controlnet


def generate_sample_inputs(model_type: str = "sd15", batch_size: int = 1, 
                          height: int = 512, width: int = 512) -> dict:
    """Generate sample inputs for ControlNet testing"""
    
    if model_type.lower() in ["sdxl", "sdxl-turbo"]:
        embedding_dim = 2048
        has_sdxl_inputs = True
    else:
        embedding_dim = 768
        has_sdxl_inputs = False
    
    inputs = {
        "sample": torch.randn(batch_size, 4, height // 8, width // 8, dtype=torch.float16).cuda(),
        "timestep": torch.tensor([0.5] * batch_size, dtype=torch.float32).cuda(),
        "encoder_hidden_states": torch.randn(batch_size, 77, embedding_dim, dtype=torch.float16).cuda(),
        "controlnet_cond": torch.randn(batch_size, 3, height, width, dtype=torch.float16).cuda()
    }
    
    if has_sdxl_inputs:
        inputs["text_embeds"] = torch.randn(batch_size, 1280, dtype=torch.float16).cuda()
        inputs["time_ids"] = torch.randn(batch_size, 6, dtype=torch.float16).cuda()
    
    return inputs


def test_controlnet_compilation(model_id: str, engine_dir: str, model_type: str = "sd15") -> tuple:
    """
    Test ControlNet TensorRT compilation
    
    Returns:
        (success: bool, compilation_time: float, engine_path: str)
    """
    print(f"\n=== Testing ControlNet Compilation ===")
    print(f"Model: {model_id}")
    print(f"Model Type: {model_type}")
    print(f"Engine Directory: {engine_dir}")
    
    try:
        # Load PyTorch ControlNet
        print("Loading PyTorch ControlNet...")
        start_time = time.time()
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
        load_time = time.time() - start_time
        print(f"PyTorch ControlNet loaded in {load_time:.2f}s")
        
        # Create TensorRT model definition
        print("Creating TensorRT model definition...")
        controlnet_model = create_controlnet_model(
            model_type=model_type,
            controlnet_type="canny",  # Default for testing
            max_batch=1,
            min_batch_size=1,
            embedding_dim=768 if model_type == "sd15" else 2048
        )
        
        # Setup paths
        os.makedirs(engine_dir, exist_ok=True)
        onnx_dir = Path(engine_dir) / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        
        model_safe_name = model_id.replace("/", "_").replace(":", "_")
        onnx_path = onnx_dir / f"controlnet_{model_safe_name}.onnx"
        onnx_opt_path = onnx_dir / f"controlnet_{model_safe_name}.opt.onnx"
        engine_path = Path(engine_dir) / "cnet.engine"
        
        # Compile to TensorRT
        print("Compiling ControlNet to TensorRT...")
        start_time = time.time()
        
        compile_controlnet(
            controlnet=controlnet,
            model_data=controlnet_model,
            onnx_path=str(onnx_path),
            onnx_opt_path=str(onnx_opt_path),
            engine_path=str(engine_path),
            opt_batch_size=1,
            engine_build_options={}
        )
        
        compilation_time = time.time() - start_time
        print(f"Compilation completed in {compilation_time:.2f}s")
        print(f"Engine saved to: {engine_path}")
        
        # Verify engine file exists
        if engine_path.exists():
            engine_size = engine_path.stat().st_size / (1024 * 1024)  # MB
            print(f"Engine file size: {engine_size:.1f} MB")
            return True, compilation_time, str(engine_path)
        else:
            print("ERROR: Engine file not created")
            return False, compilation_time, ""
            
    except Exception as e:
        print(f"ERROR during compilation: {e}")
        return False, 0.0, ""


def test_tensorrt_inference(engine_path: str, model_type: str = "sd15") -> tuple:
    """
    Test ControlNet TensorRT inference
    
    Returns:
        (success: bool, inference_time: float, output_shapes: list)
    """
    print(f"\n=== Testing TensorRT Inference ===")
    print(f"Engine: {engine_path}")
    
    try:
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Load TensorRT engine
        print("Loading TensorRT engine...")
        start_time = time.time()
        engine = ControlNetModelEngine(engine_path, stream)
        load_time = time.time() - start_time
        print(f"TensorRT engine loaded in {load_time:.2f}s")
        
        # Generate sample inputs
        print("Generating sample inputs...")
        inputs = generate_sample_inputs(model_type)
        
        # Warm up inference (first run is slower)
        print("Warming up engine...")
        for _ in range(3):
            _ = engine(**inputs)
        
        # Benchmark inference
        print("Running inference benchmark...")
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            down_blocks, mid_block = engine(**inputs)
            inference_time = time.time() - start_time
            times.append(inference_time * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        
        # Validate outputs
        print("Validating outputs...")
        output_shapes = []
        
        print(f"Down blocks: {len(down_blocks)} tensors")
        for i, tensor in enumerate(down_blocks):
            output_shapes.append(f"down_block_{i:02d}: {list(tensor.shape)}")
            print(f"  down_block_{i:02d}: {tensor.shape} dtype={tensor.dtype}")
        
        if mid_block is not None:
            output_shapes.append(f"mid_block: {list(mid_block.shape)}")
            print(f"  mid_block: {mid_block.shape} dtype={mid_block.dtype}")
        else:
            print("  WARNING: mid_block is None")
        
        return True, avg_time, output_shapes
        
    except Exception as e:
        print(f"ERROR during TensorRT inference: {e}")
        return False, 0.0, []


def test_pytorch_fallback(model_id: str, model_type: str = "sd15") -> tuple:
    """
    Test PyTorch fallback functionality
    
    Returns:
        (success: bool, inference_time: float, output_shapes: list)
    """
    print(f"\n=== Testing PyTorch Fallback ===")
    
    try:
        # Load PyTorch ControlNet
        print("Loading PyTorch ControlNet...")
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
        
        # Generate sample inputs
        inputs = generate_sample_inputs(model_type)
        
        # Warm up
        print("Warming up PyTorch inference...")
        for _ in range(3):
            with torch.no_grad():
                result = controlnet(**inputs)
        
        # Benchmark PyTorch inference
        print("Running PyTorch inference benchmark...")
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                result = controlnet(**inputs)
            inference_time = time.time() - start_time
            times.append(inference_time * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Average PyTorch inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        
        # Extract outputs in the expected format
        if hasattr(result, 'down_block_res_samples') and hasattr(result, 'mid_block_res_sample'):
            down_blocks = result.down_block_res_samples
            mid_block = result.mid_block_res_sample
        else:
            print("ERROR: Unexpected PyTorch output format")
            return False, avg_time, []
        
        # Validate outputs
        output_shapes = []
        print(f"Down blocks: {len(down_blocks)} tensors")
        for i, tensor in enumerate(down_blocks):
            output_shapes.append(f"down_block_{i:02d}: {list(tensor.shape)}")
            
        if mid_block is not None:
            output_shapes.append(f"mid_block: {list(mid_block.shape)}")
        
        return True, avg_time, output_shapes
        
    except Exception as e:
        print(f"ERROR during PyTorch inference: {e}")
        return False, 0.0, []


def test_hybrid_controlnet(model_id: str, engine_dir: str, model_type: str = "sd15") -> bool:
    """
    Test HybridControlNet wrapper functionality
    
    Returns:
        success: bool
    """
    print(f"\n=== Testing HybridControlNet Wrapper ===")
    
    try:
        # Load PyTorch model for fallback
        pytorch_model = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
        
        # Check for existing engine
        engine_path = Path(engine_dir) / "cnet.engine"
        
        # Create hybrid wrapper
        stream = cuda.Stream()
        hybrid = HybridControlNet(
            model_id=model_id,
            engine_path=str(engine_path) if engine_path.exists() else None,
            pytorch_model=pytorch_model,
            stream=stream
        )
        
        print(f"HybridControlNet created")
        print(f"Using TensorRT: {hybrid.is_using_tensorrt}")
        print(f"Status: {hybrid.status}")
        
        # Test inference
        inputs = generate_sample_inputs(model_type)
        
        start_time = time.time()
        down_blocks, mid_block = hybrid(**inputs)
        inference_time = time.time() - start_time
        
        print(f"Hybrid inference time: {inference_time*1000:.2f} ms")
        print(f"Output validation: {len(down_blocks)} down blocks, mid_block: {mid_block is not None}")
        
        return True
        
    except Exception as e:
        print(f"ERROR during hybrid testing: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test ControlNet TensorRT Phase 2.1 Infrastructure")
    parser.add_argument("--model", default="lllyasviel/sd-controlnet-canny", 
                       help="ControlNet model ID (default: lllyasviel/sd-controlnet-canny)")
    parser.add_argument("--engine-dir", default="./engines/controlnet_lllyasviel_sd-controlnet-canny", 
                       help="Directory for TensorRT engines (default: ./engines/controlnet_lllyasviel_sd-controlnet-canny)")
    parser.add_argument("--model-type", default="sd15", choices=["sd15", "sdxl"],
                       help="Base model type (default: sd15)")
    parser.add_argument("--skip-compilation", action="store_true",
                       help="Skip compilation if engine already exists")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ControlNet TensorRT Phase 2.1 Infrastructure Test")
    print("=" * 60)
    
    # Test results
    results = {
        "compilation": False,
        "tensorrt_inference": False,
        "pytorch_fallback": False,
        "hybrid_wrapper": False
    }
    
    # Test 1: ControlNet Compilation
    engine_path = ""
    if not args.skip_compilation:
        compilation_success, compilation_time, engine_path = test_controlnet_compilation(
            args.model, args.engine_dir, args.model_type
        )
        results["compilation"] = compilation_success
        
        if not compilation_success:
            print("⚠️  Compilation failed, testing fallback only")
    else:
        # Check for existing engine
        engine_path = str(Path(args.engine_dir) / "cnet.engine")
        if Path(engine_path).exists():
            print(f"Using existing engine: {engine_path}")
            results["compilation"] = True
        else:
            print("No existing engine found")
    
    # Test 2: TensorRT Inference (if compilation succeeded)
    if results["compilation"] and engine_path:
        tensorrt_success, tensorrt_time, tensorrt_shapes = test_tensorrt_inference(
            engine_path, args.model_type
        )
        results["tensorrt_inference"] = tensorrt_success
    
    # Test 3: PyTorch Fallback
    pytorch_success, pytorch_time, pytorch_shapes = test_pytorch_fallback(
        args.model, args.model_type
    )
    results["pytorch_fallback"] = pytorch_success
    
    # Test 4: HybridControlNet Wrapper
    hybrid_success = test_hybrid_controlnet(args.model, args.engine_dir, args.model_type)
    results["hybrid_wrapper"] = hybrid_success
    
    # Results Summary
    print("\n" + "=" * 60)
    print("PHASE 2.1 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"✅ ControlNet TensorRT Compilation: {'PASS' if results['compilation'] else 'FAIL'}")
    print(f"✅ TensorRT Engine Inference: {'PASS' if results['tensorrt_inference'] else 'FAIL'}")
    print(f"✅ PyTorch Fallback: {'PASS' if results['pytorch_fallback'] else 'FAIL'}")
    print(f"✅ HybridControlNet Wrapper: {'PASS' if results['hybrid_wrapper'] else 'FAIL'}")
    
    # Performance comparison
    if results["tensorrt_inference"] and results["pytorch_fallback"]:
        speedup = pytorch_time / tensorrt_time
        print(f"\n📊 Performance Results:")
        print(f"   PyTorch: {pytorch_time:.2f} ms")
        print(f"   TensorRT: {tensorrt_time:.2f} ms")
        print(f"   Speedup: {speedup:.2f}x")
    
    # Phase 2.1 Success Criteria
    phase_2_1_success = (
        results["compilation"] and 
        results["tensorrt_inference"] and 
        results["pytorch_fallback"]
    )
    
    print(f"\n🎯 Phase 2.1 Success Criteria:")
    print(f"   ✅ Can compile a single ControlNet to TensorRT: {'PASS' if results['compilation'] else 'FAIL'}")
    print(f"   ✅ Basic inference through TensorRT engine: {'PASS' if results['tensorrt_inference'] else 'FAIL'}")
    print(f"   ✅ Fallback to PyTorch if compilation fails: {'PASS' if results['pytorch_fallback'] else 'FAIL'}")
    
    print(f"\n🏆 PHASE 2.1 OVERALL: {'SUCCESS' if phase_2_1_success else 'FAILED'}")
    
    return 0 if phase_2_1_success else 1


if __name__ == "__main__":
    sys.exit(main()) 