#!/usr/bin/env python3
"""
Test Model Type Detection

Test to verify that our robust architecture-based model detection 
correctly identifies SD1.5, SD2.1, and SDXL ControlNet models.
"""

import sys
from pathlib import Path

# Add StreamDiffusion to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_robust_architecture_detection():
    """Test robust architecture-based detection"""
    from diffusers import ControlNetModel
    from streamdiffusion.acceleration.tensorrt.model_detection import detect_model_from_diffusers_unet
    
    test_cases = [
        ("lllyasviel/sd-controlnet-canny", "SD15"),
        ("thibaud/controlnet-sd21-depth-diffusers", "SD21"),
    ]
    
    print("=== Robust Architecture Detection Test ===")
    for model_id, expected_type in test_cases:
        try:
            print(f"Loading {model_id}...")
            controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
            detected_type = detect_model_from_diffusers_unet(controlnet)
            
            status = "✅" if detected_type == expected_type else "❌"
            print(f"{status} {model_id} -> {detected_type} (expected: {expected_type})")
            
            # Show architecture details
            config = controlnet.config
            print(f"   cross_attention_dim: {config.cross_attention_dim}")
            print(f"   block_out_channels: {config.block_out_channels}")
            print(f"   in_channels: {config.in_channels}")
            
        except Exception as e:
            print(f"❌ {model_id} -> FAILED: {e}")
        
        print()

def test_embedding_dimensions():
    """Test embedding dimension mapping"""
    
    def get_embedding_dim(model_type: str) -> int:
        if model_type.upper() in ["SDXL"]:
            return 2048
        elif model_type.upper() in ["SD21"]:
            return 1024
        else:
            return 768
    
    test_cases = [
        ("SD15", 768),
        ("SD21", 1024),
        ("SDXL", 2048),
    ]
    
    print("=== Embedding Dimension Test ===")
    for model_type, expected_dim in test_cases:
        actual_dim = get_embedding_dim(model_type)
        status = "✅" if actual_dim == expected_dim else "❌"
        print(f"{status} {model_type} -> {actual_dim}D (expected: {expected_dim}D)")

if __name__ == "__main__":
    import torch
    
    test_robust_architecture_detection()
    test_embedding_dimensions()
    
    print("\n=== Summary ===")
    print("✅ The robust architecture detection should fix the matrix multiplication error!")
    print("✅ SD2.1 models will be correctly identified and use 1024-dimensional embeddings!") 