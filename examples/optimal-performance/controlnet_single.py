import os
import sys
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, get_context
from typing import Literal, Optional, Union
from PIL import Image

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from src.streamdiffusion.controlnet import SDTurboControlNetPipeline, ControlNetConfig, StreamDiffusionControlNetConfig, create_sdturbo_controlnet_pipeline


def dummy_control_image_generator():
    """Generate a simple dummy control image (optimized for speed)"""
    frame_count = 0
    # Pre-generate a set of control images to cycle through (avoids real-time generation)
    control_images = []
    
    print("Pre-generating control images for maximum speed...")
    for i in range(60):  # 60 frames to cycle through
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255  # Smaller for speed
        
        # Simple moving pattern
        x = int((i % 40) * 256 / 40)
        y = int(((i // 2) % 40) * 256 / 40)
        
        # Simple rectangle (faster than complex shapes)
        cv2.rectangle(img, (x, y), (min(x + 50, 256), min(y + 50, 256)), (0, 0, 0), -1)
        control_images.append(img)
    
    print("Control images pre-generated")
    
    while True:
        yield control_images[frame_count % len(control_images)]
        frame_count += 1


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt: str,
    model_id_or_path: str,
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_canny",
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    use_dummy_input: bool = True,
    camera_id: int = 0,
    control_strength: float = 0.5,  # Reduced from 1.0 for speed
    control_skip_frames: int = 3,   # Only update control every N frames
    use_tiny_control: bool = True,  # Use smaller control images
) -> None:
    """
    OPTIMIZED process for generating images using ControlNet with maximum performance.
    """
    
    # Setup control image source
    if use_dummy_input:
        control_generator = dummy_control_image_generator()
        cap = None
        print("Using optimized dummy control image generator")
    else:
        cap = cv2.VideoCapture(camera_id)
        if use_tiny_control:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)  # Smaller for speed
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        cap.set(cv2.CAP_PROP_FPS, 60)
        control_generator = None
        print(f"Using webcam {camera_id} for control input")
    
    # Create ControlNet configuration OPTIMIZED FOR SPEED
    controlnet_config = ControlNetConfig(
        model_id=controlnet_model_id,
        conditioning_scale=control_strength,  # Reduced strength = less computation
        preprocessor="canny",
        preprocessor_params={
            "low_threshold": 100,    # Higher thresholds = less edges = faster
            "high_threshold": 200,
        },
        enabled=True
    )
    
    # Pipeline configuration MAXIMALLY optimized for FPS
    config = StreamDiffusionControlNetConfig(
        model_id=model_id_or_path,
        controlnets=[controlnet_config],
        prompt=prompt,
        negative_prompt="",  # Empty for speed
        width=512,
        height=512,
        acceleration=acceleration,
        dtype="float16",
        device="cuda",
        t_index_list=[0],  # Single step
        cfg_type="none",   # No CFG 
        guidance_scale=1.0,
        num_inference_steps=1,
        seed=2,
        use_lcm_lora=False,  # SD Turbo doesn't need this
    )
    
    print("Creating OPTIMIZED ControlNet pipeline...")
    start_time = time.time()
    stream = create_sdturbo_controlnet_pipeline(config)
    print(f"Pipeline created in {time.time() - start_time:.2f}s")
    
    # Extended warmup for TensorRT optimization
    print("Performing extended warmup for maximum optimization...")
    warmup_image = np.random.randint(0, 255, (256 if use_tiny_control else 512, 256 if use_tiny_control else 512, 3), dtype=np.uint8)
    
    # First update control image once
    stream.update_control_image_efficient(warmup_image)
    
    # Then do pure generation warmup (like the original demo)
    for i in range(20):  # More warmup iterations
        try:
            _ = stream.stream.txt2img_sd_turbo(1).cpu().detach()
            if i == 0:
                print("First warmup iteration completed")
            elif i % 5 == 0:
                print(f"Warmup iteration {i}")
        except Exception as e:
            print(f"Warmup iteration {i} failed: {e}")
    
    print("Warmup completed, starting OPTIMIZED generation...")
    
    # Performance tracking
    frame_count = 0
    total_time = 0
    control_update_count = 0
    last_control_image = None
    
    while True:
        try:
            start_time = time.time()
            
            # OPTIMIZATION: Only update control every N frames
            should_update_control = (frame_count % control_skip_frames == 0)
            
            if should_update_control:
                # Get control image
                if use_dummy_input:
                    control_img = next(control_generator)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read from camera")
                        continue
                    
                    # OPTIMIZATION: Smaller control images
                    if use_tiny_control:
                        control_img = cv2.resize(frame, (256, 256))
                    else:
                        control_img = cv2.resize(frame, (512, 512))
                    control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
                
                # OPTIMIZATION: Only update if image changed significantly
                if last_control_image is not None and use_dummy_input is False:
                    diff = np.mean(np.abs(control_img.astype(float) - last_control_image.astype(float)))
                    if diff < 10:  # Threshold for "significant change"
                        should_update_control = False
                
                if should_update_control:
                    # Update control image efficiently
                    stream.update_control_image_efficient(control_img)
                    last_control_image = control_img.copy() if not use_dummy_input else None
                    control_update_count += 1
            
            # Generate image using SD Turbo optimized path (same as original demo)
            x_outputs = stream.stream.txt2img_sd_turbo(1).cpu().detach()
            queue.put(x_outputs, block=False)
            
            # Calculate and report FPS
            frame_time = time.time() - start_time
            fps = 1 / frame_time
            fps_queue.put(fps)
            
            # Print performance stats
            frame_count += 1
            total_time += frame_time
            if frame_count % 100 == 0:
                avg_fps = frame_count / total_time
                control_rate = control_update_count / frame_count * 100
                print(f"Frame {frame_count}: Current FPS: {fps:.1f}, Average FPS: {avg_fps:.1f}, Control Updates: {control_rate:.1f}%")
            
        except KeyboardInterrupt:
            avg_fps = frame_count / total_time if total_time > 0 else 0
            control_rate = control_update_count / frame_count * 100 if frame_count > 0 else 0
            print(f"Final stats - Frames: {frame_count}, Average FPS: {avg_fps:.1f}, Control Updates: {control_rate:.1f}%")
            if cap:
                cap.release()
            return
        except Exception as e:
            print(f"Generation error: {e}")
            continue


def main(
    prompt: str = "beautiful landscape, high quality, photorealistic",
    model_id_or_path: str = "stabilityai/sd-turbo", 
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_canny",
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_dummy_input: bool = True,
    camera_id: int = 0,
    control_strength: float = 0.3,  # Lower default for speed
    control_skip_frames: int = 5,   # Update control less frequently  
    use_tiny_control: bool = True,  # Use smaller control images for speed
) -> None:
    """
    MAXIMUM PERFORMANCE ControlNet demo - optimized to get as close as possible to 111 FPS.
    
    Parameters
    ----------
    prompt : str
        The prompt for image generation.
    model_id_or_path : str
        The base model to use (SD Turbo recommended for max FPS).
    controlnet_model_id : str
        The ControlNet model ID to use.
    acceleration : Literal["none", "xformers", "tensorrt"]
        Acceleration method (tensorrt strongly recommended for max FPS).
    use_dummy_input : bool
        Whether to use procedural dummy control images or webcam input.
    camera_id : int
        Camera ID if using webcam input.
    control_strength : float
        ControlNet conditioning strength (lower = faster).
    control_skip_frames : int
        Only update control image every N frames (higher = faster).
    use_tiny_control : bool
        Use 256x256 control images instead of 512x512 for speed.
    """
    print("🚀 MAXIMUM PERFORMANCE ControlNet Demo")
    print("=" * 50)
    print(f"Model: {model_id_or_path}")
    print(f"ControlNet: {controlnet_model_id}")
    print(f"Acceleration: {acceleration}")
    print(f"Input source: {'Dummy generator' if use_dummy_input else f'Camera {camera_id}'}")
    print(f"Control strength: {control_strength}")
    print(f"Control skip frames: {control_skip_frames}")
    print(f"Tiny control images: {use_tiny_control}")
    print(f"Prompt: {prompt}")
    print("=" * 50)
    
    if acceleration != "tensorrt":
        print("⚠️  WARNING: TensorRT acceleration strongly recommended for maximum FPS!")
    
    if control_strength > 0.5:
        print("⚠️  WARNING: High control strength may reduce FPS. Try --control_strength=0.3")
    
    ctx = get_context('spawn')
    queue = ctx.Queue(maxsize=1)  # Minimal buffer for maximum speed
    fps_queue = ctx.Queue(maxsize=5)
    
    # Start image generation process
    process1 = ctx.Process(
        target=image_generation_process,
        args=(queue, fps_queue, prompt, model_id_or_path, controlnet_model_id, acceleration, use_dummy_input, camera_id, control_strength, control_skip_frames, use_tiny_control),
    )
    process1.start()

    # Start image viewer process
    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    try:
        process1.join()
        process2.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        process1.terminate()
        process2.terminate()
        process1.join()
        process2.join()


if __name__ == "__main__":
    fire.Fire(main) 