#!/usr/bin/env python3
"""
Test ControlNet TensorRT Integration

This script tests the full integration of ControlNet TensorRT acceleration
with the StreamDiffusion pipeline, verifying automatic compilation on model load.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add StreamDiffusion to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def test_controlnet_integration():
    """Test full ControlNet TensorRT integration"""
    
    print("=" * 60)
    print("CONTROLNET TENSORRT INTEGRATION TEST")
    print("=" * 60)
    
    # Setup test environment
    device = torch.device("cuda")
    model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id = "lllyasviel/sd-controlnet-canny"  # Use known working model
    engine_dir = Path("./engines")
    
    # Expected engine directory structure
    expected_engine_dir = engine_dir / "controlnet" / "controlnet_lllyasviel_sd-controlnet-canny"
    expected_engine_path = expected_engine_dir / "cnet.engine"
    
    print(f"Model ID: {model_id}")
    print(f"ControlNet Model ID: {controlnet_model_id}")
    print(f"Engine directory: {engine_dir}")
    print(f"Expected ControlNet engine: {expected_engine_path}")
    
    # Clean up any existing engines for clean test
    if expected_engine_path.exists():
        print(f"Removing existing engine: {expected_engine_path}")
        expected_engine_path.unlink()
    
    # Step 1: Initialize StreamDiffusion pipeline
    print("\n1. Initializing StreamDiffusion pipeline...")
    try:
        stream = StreamDiffusion.from_pretrained(
            model_id_or_path=model_id,
            t_index_list=[0, 16, 32, 45],
            warmup=10
        )
        print("✅ StreamDiffusion pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize StreamDiffusion: {e}")
        return False
    
    # Step 2: Enable TensorRT acceleration
    print("\n2. Enabling TensorRT acceleration...")
    try:
        stream = accelerate_with_tensorrt(
            stream=stream,
            engine_dir=str(engine_dir),
            max_batch_size=1,
            engine_build_options={
                "opt_image_height": 512,
                "opt_image_width": 512,
            }
        )
        print("✅ TensorRT acceleration enabled")
        
        # Verify ControlNet engine pool was initialized
        if hasattr(stream, 'controlnet_engine_pool'):
            print("✅ ControlNet engine pool initialized")
        else:
            print("❌ ControlNet engine pool not found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to enable TensorRT acceleration: {e}")
        return False
    
    # Step 3: Load ControlNet (should trigger synchronous compilation)
    print("\n3. Loading ControlNet (should trigger synchronous compilation)...")
    initial_time = time.time()
    
    try:
        stream.load_controlnet(
            controlnet_type="canny",
            controlnet_model_id=controlnet_model_id
        )
        load_time = time.time() - initial_time
        print(f"✅ ControlNet loaded in {load_time:.2f}s")
        
        # Check if engine pool has the model
        if controlnet_model_id in stream.controlnet_engine_pool.engines:
            hybrid_controlnet = stream.controlnet_engine_pool.engines[controlnet_model_id]
            print(f"✅ ControlNet found in engine pool")
            
            if hybrid_controlnet.is_using_tensorrt:
                print("✅ ControlNet is using TensorRT engine")
            else:
                print("⚠️  ControlNet is using PyTorch fallback (compilation may have failed)")
        else:
            print("❌ ControlNet not found in engine pool")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load ControlNet: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Verify engine file creation
    print("\n4. Verifying engine file creation...")
    
    if expected_engine_path.exists():
        engine_size_mb = expected_engine_path.stat().st_size / (1024 * 1024)
        print(f"✅ Engine file created: {expected_engine_path}")
        print(f"   Engine size: {engine_size_mb:.1f} MB")
    else:
        print(f"❌ Engine file not found: {expected_engine_path}")
        
        # List what's actually in the controlnet directory
        controlnet_dir = engine_dir / "controlnet"
        if controlnet_dir.exists():
            print(f"   Contents of {controlnet_dir}:")
            for item in controlnet_dir.iterdir():
                print(f"     {item}")
        return False
    
    # Step 5: Test engine pool status
    print("\n5. Checking engine pool status...")
    try:
        status = stream.controlnet_engine_pool.get_status()
        print(f"✅ Engine pool status:")
        print(f"   Total engines: {status['total_engines']}")
        print(f"   Compiled models: {status['compiled_models']}")
        
        for model_id, engine_status in status['engines'].items():
            print(f"   {model_id}: {engine_status}")
            
    except Exception as e:
        print(f"❌ Failed to get engine pool status: {e}")
        return False
    
    # Step 6: Test ControlNet inference 
    print("\n6. Testing ControlNet inference...")
    try:
        # Create test control image
        import numpy as np
        from PIL import Image
        
        control_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        control_image = Image.fromarray(control_image)
        
        # Test inference
        inference_start = time.time()
        result = stream(
            prompt="A beautiful landscape",
            control_image=control_image
        )
        inference_time = time.time() - inference_start
        
        print(f"✅ ControlNet inference successful in {inference_time:.3f}s")
        
        if result is not None:
            print("✅ Generated result image")
        else:
            print("❌ No result image generated")
            return False
            
    except Exception as e:
        print(f"❌ ControlNet inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("✅ ControlNet TensorRT integration is working correctly")
    print("=" * 60)
    
    return True


def main():
    """Main test function"""
    success = test_controlnet_integration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 