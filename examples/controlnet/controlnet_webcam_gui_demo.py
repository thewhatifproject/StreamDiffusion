#!/usr/bin/env python3
"""
ControlNet Webcam GUI Demo for StreamDiffusion

This script demonstrates real-time image generation using webcam input with ControlNet
and a GUI control panel for adjusting parameters in real-time.
Supports multiple ControlNets with independent strength controls.
"""

import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import argparse
from pathlib import Path
import sys
import time
import os
import threading
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class ControlNetGUI:
    def __init__(self, wrapper, config, args):
        self.wrapper = wrapper
        self.config = config
        self.args = args
        
        # Emergency shutdown flag
        self.emergency_shutdown = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("ControlNet StreamDiffusion - GUI Demo")
        self.root.geometry("400x800")
        
        # Video processing state
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # GUI variables
        self.prompt_var = tk.StringVar(value=config['prompt'] if 'prompt' in config else "")
        self.negative_prompt_var = tk.StringVar(value=config.get('negative_prompt', ''))
        self.guidance_scale_var = tk.DoubleVar(value=config.get('guidance_scale', 1.1))
        self.num_steps_var = tk.IntVar(value=config.get('num_inference_steps', 50))
        self.seed_var = tk.IntVar(value=config.get('seed', 2))
        
        # Temporal consistency variables
        self.frame_buffer_size_var = tk.IntVar(value=config.get('frame_buffer_size', 1))
        self.delta_var = tk.DoubleVar(value=config.get('delta', 1.0))
        self.cfg_type_var = tk.StringVar(value=config.get('cfg_type', 'self'))
        
        # ControlNet variables - support multiple ControlNets
        self.controlnet_vars = []
        self.controlnet_enabled_vars = []
        
        # Initialize ControlNet variables based on config
        if 'controlnets' in config and config['controlnets']:
            for i, cn_config in enumerate(config['controlnets']):
                scale_var = tk.DoubleVar(value=cn_config['conditioning_scale'])
                enabled_var = tk.BooleanVar(value=cn_config.get('enabled', True))
                self.controlnet_vars.append(scale_var)
                self.controlnet_enabled_vars.append(enabled_var)
        else:
            # Default single ControlNet
            self.controlnet_vars.append(tk.DoubleVar(value=1.0))
            self.controlnet_enabled_vars.append(tk.BooleanVar(value=True))
        
        # Status variables
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.status_var = tk.StringVar(value="Ready")
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main controls tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main Controls")
        
        # ControlNet tab
        controlnet_frame = ttk.Frame(notebook)
        notebook.add(controlnet_frame, text="ControlNets")
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        
        self.setup_main_controls(main_frame)
        self.setup_controlnet_controls(controlnet_frame)
        self.setup_advanced_controls(advanced_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=2)
        
    def setup_main_controls(self, parent):
        """Setup main parameter controls"""
        # Prompt controls
        prompt_frame = ttk.LabelFrame(parent, text="Prompts", padding=10)
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(prompt_frame, text="Prompt:").pack(anchor=tk.W)
        prompt_entry = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        prompt_entry.pack(fill=tk.X, pady=2)
        prompt_entry.insert(tk.END, self.prompt_var.get())
        
        def update_prompt(*args):
            self.prompt_var.set(prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        prompt_entry.bind('<KeyRelease>', update_prompt)
        
        ttk.Label(prompt_frame, text="Negative Prompt:").pack(anchor=tk.W, pady=(10,0))
        neg_prompt_entry = tk.Text(prompt_frame, height=2, wrap=tk.WORD)
        neg_prompt_entry.pack(fill=tk.X, pady=2)
        neg_prompt_entry.insert(tk.END, self.negative_prompt_var.get())
        
        def update_neg_prompt(*args):
            self.negative_prompt_var.set(neg_prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        neg_prompt_entry.bind('<KeyRelease>', update_neg_prompt)
        
        # Generation parameters
        gen_frame = ttk.LabelFrame(parent, text="Generation Parameters", padding=10)
        gen_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Guidance Scale
        ttk.Label(gen_frame, text="Guidance Scale:").pack(anchor=tk.W)
        guidance_frame = ttk.Frame(gen_frame)
        guidance_frame.pack(fill=tk.X, pady=2)
        
        guidance_scale = ttk.Scale(guidance_frame, from_=0.1, to=10.0, 
                                 variable=self.guidance_scale_var, orient=tk.HORIZONTAL,
                                 command=self.update_pipeline_params)
        guidance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        guidance_label = ttk.Label(guidance_frame, text=f"{self.guidance_scale_var.get():.1f}")
        guidance_label.pack(side=tk.RIGHT, padx=(5,0))
        self.guidance_scale_var.trace('w', lambda *args: guidance_label.config(text=f"{self.guidance_scale_var.get():.1f}"))
        
        # Number of steps
        ttk.Label(gen_frame, text="Inference Steps:").pack(anchor=tk.W, pady=(10,0))
        steps_frame = ttk.Frame(gen_frame)
        steps_frame.pack(fill=tk.X, pady=2)
        
        steps_scale = ttk.Scale(steps_frame, from_=1, to=100, 
                              variable=self.num_steps_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        steps_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        steps_label = ttk.Label(steps_frame, text=f"{self.num_steps_var.get()}")
        steps_label.pack(side=tk.RIGHT, padx=(5,0))
        self.num_steps_var.trace('w', lambda *args: steps_label.config(text=f"{self.num_steps_var.get()}"))
        
        # Seed
        ttk.Label(gen_frame, text="Seed:").pack(anchor=tk.W, pady=(10,0))
        seed_frame = ttk.Frame(gen_frame)
        seed_frame.pack(fill=tk.X, pady=2)
        
        seed_scale = ttk.Scale(seed_frame, from_=0, to=999999, 
                              variable=self.seed_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        seed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        seed_label = ttk.Label(seed_frame, text=f"{self.seed_var.get()}")
        seed_label.pack(side=tk.RIGHT, padx=(5,0))
        self.seed_var.trace('w', lambda *args: seed_label.config(text=f"{self.seed_var.get()}"))
        
    def setup_controlnet_controls(self, parent):
        """Setup ControlNet-specific controls"""
        info_frame = ttk.LabelFrame(parent, text="ControlNet Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        if 'controlnets' in self.config and self.config['controlnets']:
            for i, cn_config in enumerate(self.config['controlnets']):
                cn_info_text = f"ControlNet {i+1}: {cn_config['model_id'].split('/')[-1]}\nPreprocessor: {cn_config['preprocessor']}"
                ttk.Label(info_frame, text=cn_info_text, font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=2)
        
        # ControlNet strength controls
        controls_frame = ttk.LabelFrame(parent, text="ControlNet Strengths", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.controlnet_frames = []
        
        for i, (scale_var, enabled_var) in enumerate(zip(self.controlnet_vars, self.controlnet_enabled_vars)):
            cn_frame = ttk.Frame(controls_frame)
            cn_frame.pack(fill=tk.X, pady=5)
            
            # ControlNet name and enable checkbox
            header_frame = ttk.Frame(cn_frame)
            header_frame.pack(fill=tk.X, pady=2)
            
            cn_name = f"ControlNet {i+1}"
            if 'controlnets' in self.config and i < len(self.config['controlnets']):
                preprocessor = self.config['controlnets'][i]['preprocessor']
                cn_name = f"ControlNet {i+1} ({preprocessor})"
            
            enabled_cb = ttk.Checkbutton(header_frame, text=cn_name, variable=enabled_var,
                                       command=lambda idx=i: self.toggle_controlnet(idx))
            enabled_cb.pack(side=tk.LEFT)
            
            # Strength slider
            strength_frame = ttk.Frame(cn_frame)
            strength_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(strength_frame, text="Strength:").pack(side=tk.LEFT)
            
            strength_scale = ttk.Scale(strength_frame, from_=0.0, to=2.0,
                                     variable=scale_var, orient=tk.HORIZONTAL,
                                     command=lambda val, idx=i: self.update_controlnet_strength(idx, val))
            strength_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            strength_label = ttk.Label(strength_frame, text=f"{scale_var.get():.2f}")
            strength_label.pack(side=tk.RIGHT)
            
            # Update label when scale changes
            scale_var.trace('w', lambda *args, lbl=strength_label, var=scale_var: 
                          lbl.config(text=f"{var.get():.2f}"))
            
            self.controlnet_frames.append(cn_frame)
        
        # Global ControlNet controls
        global_frame = ttk.LabelFrame(parent, text="Global ControlNet Controls", padding=10)
        global_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame = ttk.Frame(global_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Enable All", 
                  command=self.enable_all_controlnets).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Disable All", 
                  command=self.disable_all_controlnets).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset Strengths", 
                  command=self.reset_controlnet_strengths).pack(side=tk.LEFT, padx=2)
        
    def setup_advanced_controls(self, parent):
        """Setup advanced controls"""
        # Camera settings
        camera_frame = ttk.LabelFrame(parent, text="Camera Settings", padding=10)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_frame, text=f"Camera Device: {self.args.camera}").pack(anchor=tk.W)
        ttk.Label(camera_frame, text=f"Resolution: {self.args.resolution}x{self.args.resolution}").pack(anchor=tk.W)
        
        # Pipeline info
        pipeline_frame = ttk.LabelFrame(parent, text="Pipeline Information", padding=10)
        pipeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        pipeline_type = self.config.get('pipeline_type', 'sd1.5')
        model_id = self.config['model_id'] if 'model_id' in self.config else "Unknown"
        acceleration = self.config.get('acceleration', 'none')
        
        ttk.Label(pipeline_frame, text=f"Pipeline Type: {pipeline_type}").pack(anchor=tk.W)
        ttk.Label(pipeline_frame, text=f"Model: {model_id.split('/')[-1]}").pack(anchor=tk.W)
        ttk.Label(pipeline_frame, text=f"Acceleration: {acceleration}").pack(anchor=tk.W)
        
        if acceleration.lower() == 'tensorrt':
            ttk.Label(pipeline_frame, text="ℹ️ TensorRT: Frame buffer size locked at config value", 
                     font=('TkDefaultFont', 8), foreground='blue').pack(anchor=tk.W, pady=(5,0))
        
        # Performance settings
        perf_frame = ttk.LabelFrame(parent, text="Performance", padding=10)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Show FPS", variable=self.show_fps_var).pack(anchor=tk.W)
        
        # Temporal consistency settings
        temporal_frame = ttk.LabelFrame(parent, text="Temporal Consistency", padding=10)
        temporal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Check if TensorRT is enabled
        tensorrt_enabled = acceleration.lower() == 'tensorrt'
        
        # Frame Buffer Size
        ttk.Label(temporal_frame, text="Frame Buffer Size:").pack(anchor=tk.W)
        buffer_frame = ttk.Frame(temporal_frame)
        buffer_frame.pack(fill=tk.X, pady=2)
        
        buffer_scale = ttk.Scale(buffer_frame, from_=1, to=10, 
                               variable=self.frame_buffer_size_var, orient=tk.HORIZONTAL,
                               command=self.update_pipeline_params,
                               state="disabled" if tensorrt_enabled else "normal")
        buffer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        buffer_label = ttk.Label(buffer_frame, text=f"{self.frame_buffer_size_var.get()}")
        buffer_label.pack(side=tk.RIGHT, padx=(5,0))
        self.frame_buffer_size_var.trace('w', lambda *args: buffer_label.config(text=f"{self.frame_buffer_size_var.get()}"))
        
        if tensorrt_enabled:
            ttk.Label(temporal_frame, text="⚠️ Frame Buffer Size locked (TensorRT enabled)", 
                     font=('TkDefaultFont', 8), foreground='orange').pack(anchor=tk.W)
        
        # Delta (temporal stability)
        ttk.Label(temporal_frame, text="Delta (Temporal Stability):").pack(anchor=tk.W, pady=(10,0))
        delta_frame = ttk.Frame(temporal_frame)
        delta_frame.pack(fill=tk.X, pady=2)
        
        delta_scale = ttk.Scale(delta_frame, from_=0.1, to=2.0, 
                              variable=self.delta_var, orient=tk.HORIZONTAL,
                              command=self.update_pipeline_params)
        delta_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        delta_label = ttk.Label(delta_frame, text=f"{self.delta_var.get():.2f}")
        delta_label.pack(side=tk.RIGHT, padx=(5,0))
        self.delta_var.trace('w', lambda *args: delta_label.config(text=f"{self.delta_var.get():.2f}"))
        
        # CFG Type dropdown
        ttk.Label(temporal_frame, text="CFG Type:").pack(anchor=tk.W, pady=(10,0))
        cfg_combobox = ttk.Combobox(temporal_frame, textvariable=self.cfg_type_var, 
                                   values=["none", "full", "self", "initialize"], 
                                   state="readonly", width=15)
        cfg_combobox.pack(anchor=tk.W, pady=2)
        cfg_combobox.bind('<<ComboboxSelected>>', self.update_pipeline_params)
        
        # Add help text
        if tensorrt_enabled:
            help_text = ("TensorRT Mode: Frame buffer size fixed at config value\n"
                        "Delta: Lower = more stable, higher = more responsive\n" 
                        "CFG Type: none/initialize = fastest, self/full = higher quality")
        else:
            help_text = ("Frame Buffer: Higher = more temporal consistency, more VRAM\n"
                        "Delta: Lower = more stable, higher = more responsive\n" 
                        "CFG Type: none/initialize = fastest, self/full = higher quality")
        ttk.Label(temporal_frame, text=help_text, font=('TkDefaultFont', 8), 
                 foreground='gray').pack(anchor=tk.W, pady=(5,0))
        
        # Export settings
        export_frame = ttk.LabelFrame(parent, text="Export Options", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_control_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(export_frame, text="Save Control Images", 
                       variable=self.save_control_images_var).pack(anchor=tk.W)
        
    def update_pipeline_params(self, *args):
        """Update pipeline parameters"""
        if hasattr(self.wrapper, 'stream'):
            try:
                # Update prompt - this can be done dynamically
                if hasattr(self.wrapper.stream, 'update_prompt'):
                    self.wrapper.stream.update_prompt(self.prompt_var.get())
                
                # For other parameters, we need to re-prepare the pipeline
                if self.running:
                    # Check if TensorRT is enabled
                    acceleration = self.config.get('acceleration', 'none')
                    tensorrt_enabled = acceleration.lower() == 'tensorrt'
                    
                    # Check if we need to recreate the entire pipeline (expensive operations)
                    needs_pipeline_recreation = False
                    
                    if (hasattr(self.wrapper, 'frame_buffer_size') and 
                        self.wrapper.frame_buffer_size != self.frame_buffer_size_var.get()):
                        if tensorrt_enabled:
                            print("update_pipeline_params: Cannot change frame buffer size with TensorRT acceleration!")
                            print("update_pipeline_params: TensorRT engines are compiled with fixed batch sizes.")
                            print("update_pipeline_params: Please update the config file and restart the application.")
                            # Reset the GUI value to current wrapper value
                            self.frame_buffer_size_var.set(self.wrapper.frame_buffer_size)
                            return
                        else:
                            needs_pipeline_recreation = True
                    
                    if (hasattr(self.wrapper.stream, 'cfg_type') and 
                        self.wrapper.stream.cfg_type != self.cfg_type_var.get()):
                        needs_pipeline_recreation = True
                    
                    if needs_pipeline_recreation:
                        print("update_pipeline_params: Warning - Frame buffer size or CFG type change requires pipeline recreation (expensive)")
                        print("update_pipeline_params: Consider restarting the application with new settings for optimal performance")
                    
                    # Re-prepare with new parameters (works for most changes)
                    self.wrapper.prepare(
                        prompt=self.prompt_var.get(),
                        negative_prompt=self.negative_prompt_var.get(),
                        num_inference_steps=self.num_steps_var.get(),
                        guidance_scale=self.guidance_scale_var.get(),
                        delta=self.delta_var.get(),
                    )
                    
                    # Update seed in the stream
                    if hasattr(self.wrapper.stream, 'generator'):
                        self.wrapper.stream.generator.manual_seed(int(self.seed_var.get()))
                    elif hasattr(self.wrapper.stream, 'set_seed'):
                        self.wrapper.stream.set_seed(int(self.seed_var.get()))
                
            except Exception as e:
                print(f"update_pipeline_params: Failed to update pipeline params: {e}")
    
    def update_controlnet_strength(self, index, value):
        """Update ControlNet strength"""
        try:
            new_value = float(value)
            if hasattr(self.wrapper, 'update_controlnet_scale'):
                self.wrapper.update_controlnet_scale(index, new_value)
        except Exception:
            pass
    
    def toggle_controlnet(self, index):
        """Toggle ControlNet on/off"""
        try:
            enabled = self.controlnet_enabled_vars[index].get()
            strength = self.controlnet_vars[index].get() if enabled else 0.0
            if hasattr(self.wrapper, 'update_controlnet_scale'):
                self.wrapper.update_controlnet_scale(index, strength)
        except Exception:
            pass
    
    def enable_all_controlnets(self):
        """Enable all ControlNets"""
        for var in self.controlnet_enabled_vars:
            var.set(True)
        for i in range(len(self.controlnet_vars)):
            self.toggle_controlnet(i)
    
    def disable_all_controlnets(self):
        """Disable all ControlNets"""
        for var in self.controlnet_enabled_vars:
            var.set(False)
        for i in range(len(self.controlnet_vars)):
            self.toggle_controlnet(i)
    
    def reset_controlnet_strengths(self):
        """Reset all ControlNet strengths to 1.0"""
        for var in self.controlnet_vars:
            var.set(1.0)
        for i in range(len(self.controlnet_vars)):
            self.update_controlnet_strength(i, 1.0)
    
    def reset_defaults(self):
        """Reset all parameters to defaults"""
        # Reset prompts
        self.prompt_var.set(self.config.get('prompt', ''))
        self.negative_prompt_var.set(self.config.get('negative_prompt', ''))
        
        # Reset generation params
        self.guidance_scale_var.set(self.config.get('guidance_scale', 1.1))
        self.num_steps_var.set(self.config.get('num_inference_steps', 50))
        self.seed_var.set(self.config.get('seed', 2))
        
        # Reset temporal consistency variables
        self.frame_buffer_size_var.set(self.config.get('frame_buffer_size', 1))
        self.delta_var.set(self.config.get('delta', 1.0))
        self.cfg_type_var.set(self.config.get('cfg_type', 'self'))
        
        # Reset ControlNet strengths
        self.reset_controlnet_strengths()
        self.enable_all_controlnets()
        
        # Update pipeline
        self.update_pipeline_params()
    
    def toggle_camera(self):
        """Start/stop camera capture"""
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture and processing"""
        try:
            self.cap = cv2.VideoCapture(self.args.camera)
            if not self.cap.isOpened():
                error_msg = f"Could not open camera {self.args.camera}"
                messagebox.showerror("Error", error_msg)
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.resolution)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.resolution)
            
            self.running = True
            self.start_button.config(text="Stop Camera")
            self.status_var.set("Running")
            
            # Start processing threads
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
            self.display_thread = threading.Thread(target=self.display_results, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            self.display_thread.start()
            
        except Exception as e:
            error_msg = f"Failed to start camera: {str(e)}"
            messagebox.showerror("Error", error_msg)
    
    def stop_camera(self):
        """Stop camera capture"""
        if not self.running:
            return
            
        self.running = False
        
        # Release camera first to stop capture thread quickly
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None
        
        # Clear queues immediately to unblock processing threads
        self._clear_queues()
        
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process any pending events
        except Exception:
            pass
        
        # Update GUI
        self.start_button.config(text="Start Camera")
        self.status_var.set("Stopped")
        
        # Brief pause to allow threads to see the running=False state
        import time
        time.sleep(0.05)
    
    def _clear_queues(self):
        """Clear all queues safely"""
        # Clear frame queue
        cleared_frames = 0
        try:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                    cleared_frames += 1
                    if cleared_frames > 100:  # Safety limit
                        break
                except queue.Empty:
                    break
        except Exception:
            pass
        
        # Clear result queue
        cleared_results = 0
        try:
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                    cleared_results += 1
                    if cleared_results > 100:  # Safety limit
                        break
                except queue.Empty:
                    break
        except Exception:
            pass
    
    def capture_frames(self):
        """Capture frames from camera"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame
                frame_resized = cv2.resize(frame, (self.args.resolution, self.args.resolution))
                
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame_resized)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
            else:
                break
    
    def process_frames(self):
        """Process frames with StreamDiffusion"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Convert frame to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Update control image
                self.wrapper.update_control_image_efficient(frame_pil)
                
                # Generate image
                output_image = self.wrapper(frame_pil)
                
                # Calculate processing time
                process_time = time.time() - start_time
                
                # Put result in queue
                try:
                    self.result_queue.put_nowait((frame, output_image, process_time))
                except queue.Full:
                    # Skip if queue is full
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"process_frames: Processing error: {e}")
                if not self.running:
                    break
    
    def display_results(self):
        """Display results and update GUI"""
        window_name = "ControlNet StreamDiffusion - GUI Demo"
        window_created = False
        
        while self.running:
            try:
                # Use short timeout and check running state frequently
                try:
                    frame, output_image, process_time = self.result_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                
                if not self.running:
                    break
                
                # Update FPS counter
                self.fps_counter.append(process_time)
                self.frame_count += 1
                
                if self.show_fps_var.get() and len(self.fps_counter) > 0:
                    avg_fps = len(self.fps_counter) / sum(self.fps_counter)
                    try:
                        self.fps_var.set(f"FPS: {avg_fps:.1f}")
                    except:
                        pass  # GUI might be destroyed
                
                if not self.running:
                    break
                
                # Convert output to display format
                output_array = np.array(output_image)
                output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                
                # Create side-by-side display
                combined = np.hstack([frame, output_bgr])
                
                # Add labels
                cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, "Generated", (self.args.resolution + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if not self.running:
                    break
                
                # Show result
                try:
                    cv2.imshow(window_name, combined)
                    window_created = True
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if not self.running:
                        break
                    
                    if key == ord('q'):
                        self.root.after(0, self.stop_camera)
                        break
                    
                    # Check if window was closed
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            self.root.after(0, self.stop_camera)
                            break
                    except:
                        self.root.after(0, self.stop_camera)
                        break
                        
                except Exception:
                    break
                    
            except Exception:
                break
        
        # Cleanup
        if window_created:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except:
                pass
    
    def save_output(self):
        """Save current output"""
        if not self.running:
            messagebox.showwarning("Warning", "Camera is not running")
            return
        
        try:
            # Get current frame from queue if available
            if not self.result_queue.empty():
                frame, output_image, _ = self.result_queue.queue[-1]  # Get latest
                
                timestamp = int(time.time())
                output_path = f"controlnet_gui_output_{timestamp}.png"
                output_image.save(output_path)
                
                if self.save_control_images_var.get():
                    # Save control images if available
                    for i in range(len(self.controlnet_vars)):
                        try:
                            control_image = self.wrapper.get_last_processed_image(i)
                            if control_image:
                                control_path = f"controlnet_gui_control_{i}_{timestamp}.png"
                                control_image.save(control_path)
                        except:
                            pass
                
                messagebox.showinfo("Success", f"Saved output to {output_path}")
            else:
                messagebox.showwarning("Warning", "No output to save")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"run: Unexpected error in GUI: {e}")
        finally:
            self.cleanup()
    
    def on_closing(self):
        """Handle window closing"""
        # Start emergency shutdown timer as absolute fallback
        self.emergency_shutdown = True
        emergency_thread = threading.Thread(target=self._emergency_exit, daemon=True)
        emergency_thread.start()
        
        # Stop camera if running
        if self.running:
            self.stop_camera()
            
            # Very short wait
            import time
            time.sleep(0.3)
            
            # Nuclear option - destroy all OpenCV windows immediately
            try:
                cv2.destroyAllWindows()
                for _ in range(3):  # Try multiple times quickly
                    cv2.waitKey(1)
            except:
                pass
        
        # Force destroying GUI
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        
        import os
        os._exit(0)  # Nuclear option - immediate exit
    
    def _emergency_exit(self):
        """Emergency exit thread - kills application after 2 seconds no matter what"""
        import time
        import os
        time.sleep(2.0)  # Wait 2 seconds maximum
        if self.emergency_shutdown:
            os._exit(1)  # Force kill with error code
    
    def cleanup(self):
        """Cleanup resources"""
        # Ensure camera is stopped
        if self.running:
            self.running = False
            
        # Release camera if not already done
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception:
                self.cap = None  # Set to None anyway
        
        # Close all OpenCV windows aggressively
        try:
            cv2.destroyAllWindows()
            # Process any remaining events
            for _ in range(5):  # Try a few times
                cv2.waitKey(1)
        except Exception:
            pass
        
        # Clear queues to prevent hanging
        try:
            self._clear_queues()
        except Exception:
            pass


def main():
    # Import heavy modules only when needed
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from utils.wrapper import StreamDiffusionWrapper
    from streamdiffusion.controlnet import load_controlnet_config
    
    parser = argparse.ArgumentParser(description="ControlNet Webcam GUI Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent.parent / "configs" / "controlnet_examples" / "multi_controlnet_example.yaml"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config),
                       help="Path to ControlNet configuration file")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--resolution", type=int, default=None,
                       help="Camera and output resolution (auto-detects from pipeline type if not specified)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_controlnet_config(args.config)
    
    # Detect pipeline type
    pipeline_type = config.get('pipeline_type', 'sd1.5')
    
    # Display ControlNet information
    if 'controlnets' in config and config['controlnets']:
        pass  # ControlNets found
    else:
        print("Warning: No ControlNets found in configuration")
    
    # Set default resolution based on pipeline type if not specified
    if args.resolution is None:
        if pipeline_type == 'sdxlturbo':
            args.resolution = 1024  # SD-XL Turbo default
        else:
            args.resolution = 512   # SD 1.5 and SD Turbo default
    
    # Override parameters if provided
    model_id = args.model if args.model else config['model_id']
    prompt = args.prompt if args.prompt else config['prompt']
    
    # Update resolution in config for GUI display
    config['width'] = args.resolution
    config['height'] = args.resolution
    
    # Determine t_index_list and other parameters based on pipeline type
    t_index_list = config.get('t_index_list', [0,16])
    if pipeline_type == 'sdturbo':
        cfg_type = config.get('cfg_type', "none")
        use_lcm_lora = config.get('use_lcm_lora', False)
        use_tiny_vae = config.get('use_tiny_vae', True)
    elif pipeline_type == 'sdxlturbo':
        cfg_type = config.get('cfg_type', "none")
        use_lcm_lora = config.get('use_lcm_lora', False)
        use_tiny_vae = config.get('use_tiny_vae', False)
    else:  # sd1.5
        cfg_type = config.get('cfg_type', 'self')
        use_lcm_lora = config.get('use_lcm_lora', True)
        use_tiny_vae = config.get('use_tiny_vae', True)
    
    # Create ControlNet configurations for wrapper - support multiple ControlNets
    controlnet_configs = []
    if 'controlnets' in config and config['controlnets']:
        for cn_config in config['controlnets']:
            controlnet_config = {
                'model_id': cn_config['model_id'],
                'preprocessor': cn_config['preprocessor'],
                'conditioning_scale': cn_config['conditioning_scale'],
                'enabled': cn_config.get('enabled', True),
                'preprocessor_params': cn_config.get('preprocessor_params', None),
                'pipeline_type': pipeline_type,
                'control_guidance_start': cn_config.get('control_guidance_start', 0.0),
                'control_guidance_end': cn_config.get('control_guidance_end', 1.0),
            }
            controlnet_configs.append(controlnet_config)
    else:
        # Fallback single ControlNet for compatibility
        controlnet_configs = [{
            'model_id': 'lllyasviel/sd-controlnet-depth',
            'preprocessor': 'depth_midas',
            'conditioning_scale': 1.0,
            'enabled': True,
            'preprocessor_params': None,
            'pipeline_type': pipeline_type,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }]
    
    # Create StreamDiffusionWrapper with ControlNet support
    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=config.get('frame_buffer_size', 1),
        width=args.resolution,
        height=args.resolution,
        warmup=10,
        acceleration=config.get('acceleration', 'none'),
        do_add_noise=True,
        use_lcm_lora=use_lcm_lora,
        use_tiny_vae=use_tiny_vae,
        use_denoising_batch=True,
        cfg_type=config.get('cfg_type', cfg_type),
        seed=config.get('seed', 2),
        use_safety_checker=False,
        # ControlNet options
        use_controlnet=True,
        controlnet_config=controlnet_configs,
    )
    
    # Prepare pipeline
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=config.get('negative_prompt', ''),
        num_inference_steps=config.get('num_inference_steps', 50),
        guidance_scale=config.get('guidance_scale', 1.1 if cfg_type != "none" else 1.0),
        delta=config.get('delta', 1.0),
    )
    
    # Create and run GUI
    gui = ControlNetGUI(wrapper, config, args)
    gui.run()


if __name__ == "__main__":
    main() 