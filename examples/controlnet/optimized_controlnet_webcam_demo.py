#!/usr/bin/env python3
"""
Optimized ControlNet Webcam Demo for StreamDiffusion

This script demonstrates optimized real-time image generation using webcam input with ControlNet.
Uses multiprocessing pattern from optimal performance examples.
"""

import os
import sys
import threading
import time
import tkinter as tk
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Optional
from pathlib import Path

import cv2
import fire
import numpy as np
import torch
from PIL import Image, ImageTk

# Add StreamDiffusion to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from streamdiffusion.controlnet import (
    load_controlnet_config, 
    create_controlnet_pipeline,
)
from streamdiffusion.image_utils import postprocess_image


def webcam_capture_process(
    frame_queue: Queue,
    camera_index: int = 0,
    resolution: int = 512,
) -> None:
    """
    Process for capturing frames from webcam and putting them in a queue.
    
    Parameters
    ----------
    frame_queue : Queue
        Queue to put captured frames in
    camera_index : int
        Camera device index
    resolution : int
        Camera resolution
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    print(f"✓ Camera {camera_index} opened successfully")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((resolution, resolution))
            
            # Non-blocking put - skip frame if queue is full to maintain real-time performance
            try:
                frame_queue.put(frame_pil, block=False)
            except:
                # Queue full, skip this frame to maintain real-time performance
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("📷 Camera capture stopped")


def controlnet_generation_process(
    frame_queue: Queue,
    output_queue: Queue,
    fps_queue: Queue,
    config_path: str,
    model_override: Optional[str] = None,
    prompt_override: Optional[str] = None,
    controlnet_scale_override: Optional[float] = None,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Process for generating images using ControlNet based on input frames.
    Similar to image_generation_process in optimal performance examples.
    
    Parameters
    ----------
    frame_queue : Queue
        Queue to get input frames from
    output_queue : Queue
        Queue to put generated images in
    fps_queue : Queue
        Queue to put FPS measurements in
    config_path : str
        Path to ControlNet configuration file
    model_override : Optional[str]
        Override base model path from config
    prompt_override : Optional[str]
        Override prompt from config
    controlnet_scale_override : Optional[float]
        Override ControlNet conditioning scale
    acceleration : Literal["none", "xformers", "tensorrt"]
        Acceleration method
    """
    # Load configuration
    config = load_controlnet_config(config_path)
    
    # Apply overrides
    if model_override:
        config.model_id = model_override
    if prompt_override:
        config.prompt = prompt_override
    if controlnet_scale_override is not None:
        config.controlnets[0].conditioning_scale = controlnet_scale_override
    
    # Override acceleration from CLI
    print(f"🚀 Using acceleration: {acceleration}")
    config.acceleration = acceleration
    
    print(f"🔄 Creating ControlNet pipeline...")
    print(f"📝 Using ControlNet: {config.controlnets[0].model_id}")
    print(f"🔧 Preprocessor: {config.controlnets[0].preprocessor}")
    print(f"🎛️  Scale: {config.controlnets[0].conditioning_scale}")
    print(f"🚀 Acceleration: {config.acceleration}")
    
    # Create ControlNet pipeline
    pipeline = create_controlnet_pipeline(config)
    print("✓ ControlNet pipeline created successfully")
    
    generation_count = 0
    
    try:
        while True:
            # Get latest frame (non-blocking, similar to optimal examples)
            current_frame = None
            try:
                # Get the most recent frame, discard older ones
                while not frame_queue.empty():
                    current_frame = frame_queue.get(block=False)
            except:
                pass
            
            if current_frame is None:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            start_time = time.time()
            
            try:
                # Update control image efficiently
                pipeline.update_control_image_efficient(current_frame)
                
                # Generate (similar to stream.stream.txt2img_sd_turbo in optimal examples)
                x_output = pipeline(current_frame)
                output_image = postprocess_image(x_output, output_type="pil")[0]
                
                # Get control image for display
                control_image = pipeline.get_last_processed_image(0)
                
                # Put result in output queue (non-blocking)
                try:
                    output_queue.put({
                        'generated': output_image,
                        'input': current_frame,
                        'control': control_image,
                        'generation_count': generation_count
                    }, block=False)
                    generation_count += 1
                except:
                    pass  # Queue full, skip
                
                # Calculate and report FPS (like optimal examples)
                generation_time = time.time() - start_time
                fps = 1.0 / generation_time
                try:
                    fps_queue.put(fps, block=False)
                except:
                    pass  # Queue full, skip
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
                
    except KeyboardInterrupt:
        print("🛑 Generation process interrupted")


def update_image_display(
    image_data: Image.Image, 
    label: tk.Label, 
    target_size: int = 320
) -> None:
    """
    Update the image displayed on a Tkinter label.
    
    Parameters
    ----------
    image_data : Image.Image
        The image to be displayed
    label : tk.Label
        The label to update
    target_size : int
        Target display size
    """
    tk_image = ImageTk.PhotoImage(image_data.resize((target_size, target_size)))
    label.configure(image=tk_image, width=target_size, height=target_size)
    label.image = tk_image  # Keep reference to prevent garbage collection


def display_update_thread(
    output_queue: Queue,
    fps_queue: Queue,
    labels: List[tk.Label],
    fps_label: tk.Label,
    show_control: bool = False
) -> None:
    """
    Thread for updating the display with generated images.
    Similar to _receive_images in optimal performance examples.
    
    Parameters
    ----------
    output_queue : Queue
        Queue to receive output data from
    fps_queue : Queue
        Queue to receive FPS data from
    labels : List[tk.Label]
        Labels for displaying images [input, control, generated]
    fps_label : tk.Label
        Label for FPS display
    show_control : bool
        Whether to show control images
    """
    generation_count = 0
    fps_history = []
    
    while True:
        try:
            # Update images
            if not output_queue.empty():
                output_data = output_queue.get(block=False)
                
                # Update input image
                labels[0].after(0, update_image_display, output_data['input'], labels[0])
                
                # Update control image if available and requested
                if show_control and output_data['control'] is not None and len(labels) > 1:
                    labels[1].after(0, update_image_display, output_data['control'], labels[1])
                
                # Update generated image
                target_label = labels[2] if len(labels) > 2 else labels[-1]
                target_label.after(0, update_image_display, output_data['generated'], target_label)
                
                generation_count = output_data['generation_count']
            
            # Update FPS
            if not fps_queue.empty():
                fps = fps_queue.get(block=False)
                fps_history.append(fps)
                if len(fps_history) > 30:  # Keep last 30 measurements
                    fps_history.pop(0)
                
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                fps_label.config(text=f"FPS: {avg_fps:.1f} | Generated: {generation_count}")
            
            time.sleep(0.0005)  # Small sleep like in optimal examples
            
        except KeyboardInterrupt:
            break


def display_process(
    output_queue: Queue,
    fps_queue: Queue,
    show_control: bool = False,
    preprocessor_name: str = "ControlNet"
) -> None:
    """
    Process for displaying images using Tkinter.
    Similar to receive_images in optimal performance examples.
    
    Parameters
    ----------
    output_queue : Queue
        Queue to receive images from
    fps_queue : Queue
        Queue to receive FPS data from
    show_control : bool
        Whether to show control images
    preprocessor_name : str
        Name of the preprocessor for display
    """
    root = tk.Tk()
    root.title(f"Optimized ControlNet StreamDiffusion - {preprocessor_name}")
    
    # Create labels for images
    if show_control:
        # 3-panel layout: Input | Control | Generated
        input_label = tk.Label(root, text="Input")
        control_label = tk.Label(root, text=f"Control ({preprocessor_name})")
        generated_label = tk.Label(root, text="Generated")
        
        input_label.grid(row=0, column=0, padx=5, pady=5)
        control_label.grid(row=0, column=1, padx=5, pady=5)
        generated_label.grid(row=0, column=2, padx=5, pady=5)
        
        labels = [input_label, control_label, generated_label]
    else:
        # 2-panel layout: Input | Generated
        input_label = tk.Label(root, text="Input")
        generated_label = tk.Label(root, text="Generated")
        
        input_label.grid(row=0, column=0, padx=5, pady=5)
        generated_label.grid(row=0, column=1, padx=5, pady=5)
        
        labels = [input_label, generated_label]
    
    # FPS label
    fps_label = tk.Label(root, text="FPS: 0", font=("Arial", 12))
    fps_label.grid(row=1, column=0, columnspan=len(labels), pady=5)
    
    # Info label
    info_label = tk.Label(root, text="Press Ctrl+C in terminal to stop", font=("Arial", 10))
    info_label.grid(row=2, column=0, columnspan=len(labels), pady=5)
    
    # Start display update thread
    thread = threading.Thread(
        target=display_update_thread,
        args=(output_queue, fps_queue, labels, fps_label, show_control),
        daemon=True
    )
    thread.start()
    
    def on_closing():
        print("🖼️  Display window closed")
        root.quit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


def main(
    config: str = None,
    camera: int = 0,
    model: str = None,
    prompt: str = None,
    controlnet_scale: float = None,
    show_control: bool = False,
    resolution: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
) -> None:
    """
    Main function to start the optimized ControlNet webcam demo.
    Similar to main() in optimal performance examples.
    
    Parameters
    ----------
    config : str
        Path to ControlNet configuration file
    camera : int
        Camera device index
    model : str
        Override base model path from config
    prompt : str
        Override prompt from config
    controlnet_scale : float
        Override ControlNet conditioning scale from config
    show_control : bool
        Show the preprocessed control image
    resolution : int
        Camera and output resolution
    acceleration : Literal["none", "xformers", "tensorrt"]
        Acceleration method
    """
    # Default config path
    if config is None:
        script_dir = Path(__file__).parent
        config = str(script_dir.parent / "configs" / "controlnet_examples" / "depth_trt_example.yaml")
    
    print("🚀 Starting Optimized ControlNet Webcam Demo")
    print(f"📁 Config: {config}")
    print(f"📷 Camera: {camera}")
    print(f"📏 Resolution: {resolution}x{resolution}")
    print(f"🚀 Acceleration: {acceleration}")
    print(f"🖼️  Show control: {show_control}")
    
    # Load config to get preprocessor name for display
    temp_config = load_controlnet_config(config)
    preprocessor_name = temp_config.controlnets[0].preprocessor.replace("_", " ").title()
    
    # Create multiprocessing context (like optimal examples)
    ctx = get_context('spawn')
    
    # Create queues with appropriate sizes
    frame_queue = ctx.Queue(maxsize=2)  # Small buffer to maintain real-time performance
    output_queue = ctx.Queue(maxsize=5)  # Buffer for generated images
    fps_queue = ctx.Queue(maxsize=10)
    
    # Start webcam capture process
    capture_process = ctx.Process(
        target=webcam_capture_process,
        args=(frame_queue, camera, resolution)
    )
    capture_process.start()
    
    # Start ControlNet generation process (like image_generation_process in optimal examples)
    generation_process = ctx.Process(
        target=controlnet_generation_process,
        args=(
            frame_queue, output_queue, fps_queue, config,
            model, prompt, controlnet_scale, acceleration
        )
    )
    generation_process.start()
    
    # Start display process (like receive_images in optimal examples)
    display_proc = ctx.Process(
        target=display_process,
        args=(output_queue, fps_queue, show_control, preprocessor_name)
    )
    display_proc.start()
    
    print("\n🎮 Demo started! Close the display window or press Ctrl+C to stop.")
    
    try:
        # Wait for processes to complete (like optimal examples)
        capture_process.join()
        generation_process.join()
        display_proc.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping demo...")
        
        # Terminate processes gracefully
        capture_process.terminate()
        generation_process.terminate()
        display_proc.terminate()
        
        # Wait for clean shutdown
        capture_process.join(timeout=2)
        generation_process.join(timeout=2)
        display_proc.join(timeout=2)
        
    print("🏁 Demo finished")


if __name__ == "__main__":
    fire.Fire(main) 