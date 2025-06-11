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
    def __init__(self, args):
        self.args = args
        self.wrapper = None
        self.config = None
        
        # Emergency shutdown flag
        self.emergency_shutdown = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("ControlNet StreamDiffusion - GUI Demo (Config Selectable)")
        self.root.geometry("400x800")
        
        # Video processing state
        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # GUI variables - initialize with defaults
        self.prompt_var = tk.StringVar(value="")
        self.negative_prompt_var = tk.StringVar(value="")
        self.guidance_scale_var = tk.DoubleVar(value=1.1)
        self.num_steps_var = tk.IntVar(value=50)
        self.seed_var = tk.IntVar(value=2)
        
        # Temporal consistency variables
        self.frame_buffer_size_var = tk.IntVar(value=1)
        self.delta_var = tk.DoubleVar(value=1.0)
        self.cfg_type_var = tk.StringVar(value="self")
        
        # ControlNet variables - support multiple ControlNets
        self.controlnet_vars = []
        self.controlnet_enabled_vars = []
        
        # Status variables
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.status_var = tk.StringVar(value="No config loaded")
        self.config_status_var = tk.StringVar(value="No config loaded")
        
        # Initialize GUI
        self.setup_gui()
        
        # Load initial config if provided
        if hasattr(args, 'config') and args.config:
            self.load_config(args.config)
        
    def load_config(self, config_path):
        """Load configuration and populate GUI controls"""
        try:
            # Import here to avoid loading at module level
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
            from streamdiffusion.controlnet import load_controlnet_config
            
            # Stop camera if running
            if self.running:
                self.stop_camera()
            
            # Clear existing wrapper to force recreation on next start
            self.wrapper = None
            
            # Load configuration
            self.config = load_controlnet_config(config_path)
            
            # Override parameters if provided in args
            if hasattr(self.args, 'model') and self.args.model:
                self.config['model_id'] = self.args.model
            if hasattr(self.args, 'prompt') and self.args.prompt:
                self.config['prompt'] = self.args.prompt
            
            # Set default resolution based on pipeline type if not specified in args
            if not hasattr(self.args, 'resolution') or self.args.resolution is None:
                pipeline_type = self.config.get('pipeline_type', 'sd1.5')
                if pipeline_type == 'sdxlturbo':
                    self.args.resolution = 1024
                else:
                    self.args.resolution = 512
            
            # Update resolution in config for GUI display
            self.config['width'] = self.args.resolution
            self.config['height'] = self.args.resolution
            
            # Update GUI with new config values (but don't create wrapper yet)
            self._update_gui_from_config()
            
            # Update status
            config_name = Path(config_path).name
            self.config_status_var.set(f"Config: {config_name}")
            self.status_var.set("Config loaded - Ready to start camera")
            
            # Enable start button
            if hasattr(self, 'start_button'):
                self.start_button.config(state="normal")
            
            print(f"load_config: Successfully loaded config: {config_path}")
            print("load_config: Pipeline will be created when camera is started")
            
        except Exception as e:
            error_msg = f"Failed to load config: {str(e)}"
            print(f"load_config: {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.config_status_var.set("Config load failed")
            self.status_var.set("Error")
            if hasattr(self, 'start_button'):
                self.start_button.config(state="disabled")
    
    def _create_wrapper(self):
        """Create the StreamDiffusionWrapper from current config"""
        if not self.config:
            raise ValueError("No config loaded")
        
        print("_create_wrapper: Starting pipeline creation...")
        
        # Import here to avoid loading at module level
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        from utils.wrapper import StreamDiffusionWrapper
        import torch
        
        # Determine parameters based on pipeline type
        pipeline_type = self.config.get('pipeline_type', 'sd1.5')
        t_index_list = self.config.get('t_index_list', [0,16])
        
        print(f"_create_wrapper: Pipeline type: {pipeline_type}")
        print(f"_create_wrapper: Model: {self.config['model_id']}")
        
        if pipeline_type == 'sdturbo':
            cfg_type = self.config.get('cfg_type', "none")
            use_lcm_lora = self.config.get('use_lcm_lora', False)
            use_tiny_vae = self.config.get('use_tiny_vae', True)
        elif pipeline_type == 'sdxlturbo':
            cfg_type = self.config.get('cfg_type', "none")
            use_lcm_lora = self.config.get('use_lcm_lora', False)
            use_tiny_vae = self.config.get('use_tiny_vae', False)
        else:  # sd1.5
            cfg_type = self.config.get('cfg_type', 'self')
            use_lcm_lora = self.config.get('use_lcm_lora', True)
            use_tiny_vae = self.config.get('use_tiny_vae', True)
        
        # Create ControlNet configurations for wrapper
        controlnet_configs = []
        if 'controlnets' in self.config and self.config['controlnets']:
            print(f"_create_wrapper: Loading {len(self.config['controlnets'])} ControlNet(s)...")
            for cn_config in self.config['controlnets']:
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
                print(f"_create_wrapper: - {cn_config['model_id']} ({cn_config['preprocessor']})")
        else:
            # Fallback single ControlNet for compatibility
            print("_create_wrapper: Using fallback depth ControlNet...")
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
        
        print("_create_wrapper: Creating StreamDiffusionWrapper...")
        
        # Create StreamDiffusionWrapper
        self.wrapper = StreamDiffusionWrapper(
            model_id_or_path=self.config['model_id'],
            t_index_list=t_index_list,
            mode="img2img",
            output_type="pil",
            device="cuda",
            dtype=torch.float16,
            frame_buffer_size=self.config.get('frame_buffer_size', 1),
            width=self.args.resolution,
            height=self.args.resolution,
            warmup=10,
            acceleration=self.config.get('acceleration', 'none'),
            do_add_noise=True,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            use_denoising_batch=True,
            cfg_type=cfg_type,
            seed=self.config.get('seed', 2),
            use_safety_checker=False,
            # ControlNet options
            use_controlnet=True,
            controlnet_config=controlnet_configs,
        )
        
        print("_create_wrapper: Preparing pipeline...")
        
        # Prepare pipeline
        self.wrapper.prepare(
            prompt=self.config.get('prompt', ''),
            negative_prompt=self.config.get('negative_prompt', ''),
            num_inference_steps=self.config.get('num_inference_steps', 50),
            guidance_scale=self.config.get('guidance_scale', 1.1 if cfg_type != "none" else 1.0),
            delta=self.config.get('delta', 1.0),
        )
        
        print("_create_wrapper: Pipeline creation completed!")
    
    def _update_gui_from_config(self):
        """Update GUI variables from loaded config"""
        if not self.config:
            return
        
        # Update prompt variables
        self.prompt_var.set(self.config.get('prompt', ''))
        self.negative_prompt_var.set(self.config.get('negative_prompt', ''))
        
        # Manually update the Text widgets (they don't auto-bind to StringVars)
        if hasattr(self, 'prompt_entry'):
            self.prompt_entry.delete(1.0, tk.END)
            self.prompt_entry.insert(1.0, self.config.get('prompt', ''))
            
        if hasattr(self, 'neg_prompt_entry'):
            self.neg_prompt_entry.delete(1.0, tk.END)
            self.neg_prompt_entry.insert(1.0, self.config.get('negative_prompt', ''))
        
        # Update generation parameters
        self.guidance_scale_var.set(self.config.get('guidance_scale', 1.1))
        self.num_steps_var.set(self.config.get('num_inference_steps', 50))
        self.seed_var.set(self.config.get('seed', 2))
        
        # Update temporal consistency variables
        self.frame_buffer_size_var.set(self.config.get('frame_buffer_size', 1))
        self.delta_var.set(self.config.get('delta', 1.0))
        self.cfg_type_var.set(self.config.get('cfg_type', 'self'))
        
        # Clear and recreate ControlNet variables
        self.controlnet_vars.clear()
        self.controlnet_enabled_vars.clear()
        
        if 'controlnets' in self.config and self.config['controlnets']:
            for cn_config in self.config['controlnets']:
                scale_var = tk.DoubleVar(value=cn_config['conditioning_scale'])
                enabled_var = tk.BooleanVar(value=cn_config.get('enabled', True))
                self.controlnet_vars.append(scale_var)
                self.controlnet_enabled_vars.append(enabled_var)
        else:
            # Default single ControlNet
            self.controlnet_vars.append(tk.DoubleVar(value=1.0))
            self.controlnet_enabled_vars.append(tk.BooleanVar(value=True))
        
        # Refresh GUI layout
        self._refresh_gui_layout()
    
    def _refresh_gui_layout(self):
        """Refresh GUI layout after config change"""
        # Clear existing ControlNet info
        for widget in self.controlnet_info_frame.winfo_children():
            widget.destroy()
        
        # Clear existing ControlNet controls  
        for widget in self.controlnet_controls_frame.winfo_children():
            widget.destroy()
        
        # Repopulate with new config
        if self.config:
            self._populate_controlnet_info(self.controlnet_info_frame)
            self._populate_controlnet_controls(self.controlnet_controls_frame)
        else:
            ttk.Label(self.controlnet_info_frame, text="No configuration loaded. Please browse and select a config file.", 
                     font=('TkDefaultFont', 9), foreground='gray').pack(anchor=tk.W, pady=2)
    
    def browse_config(self):
        """Open file dialog to select config file"""
        script_dir = Path(__file__).parent
        default_dir = script_dir.parent.parent / "configs" / "controlnet_examples"
        
        config_path = filedialog.askopenfilename(
            title="Select ControlNet Configuration",
            initialdir=str(default_dir) if default_dir.exists() else None,
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ]
        )
        
        if config_path:
            self.load_config(config_path)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Config selection section at the top
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Config status and browse button
        config_controls_frame = ttk.Frame(config_frame)
        config_controls_frame.pack(fill=tk.X)
        
        ttk.Label(config_controls_frame, textvariable=self.config_status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(config_controls_frame, text="Browse Config...", command=self.browse_config).pack(side=tk.RIGHT, padx=(5,0))
        
        # Add a separator
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
        
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
        
        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.toggle_camera, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=2)
    
    def setup_main_controls(self, parent):
        """Setup main parameter controls"""
        # Prompt controls
        prompt_frame = ttk.LabelFrame(parent, text="Prompts", padding=10)
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(prompt_frame, text="Prompt:").pack(anchor=tk.W)
        self.prompt_entry = tk.Text(prompt_frame, height=3, wrap=tk.WORD)
        self.prompt_entry.pack(fill=tk.X, pady=2)
        self.prompt_entry.insert(tk.END, self.prompt_var.get())
        
        def update_prompt(*args):
            self.prompt_var.set(self.prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        self.prompt_entry.bind('<KeyRelease>', update_prompt)
        
        ttk.Label(prompt_frame, text="Negative Prompt:").pack(anchor=tk.W, pady=(10,0))
        self.neg_prompt_entry = tk.Text(prompt_frame, height=2, wrap=tk.WORD)
        self.neg_prompt_entry.pack(fill=tk.X, pady=2)
        self.neg_prompt_entry.insert(tk.END, self.negative_prompt_var.get())
        
        def update_neg_prompt(*args):
            self.negative_prompt_var.set(self.neg_prompt_entry.get(1.0, tk.END).strip())
            self.update_pipeline_params()
        
        self.neg_prompt_entry.bind('<KeyRelease>', update_neg_prompt)
        
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
        
        # Store reference to info frame for dynamic updates
        self.controlnet_info_frame = info_frame
        
        # Initial message when no config is loaded
        if not self.config:
            ttk.Label(info_frame, text="No configuration loaded. Please browse and select a config file.", 
                     font=('TkDefaultFont', 9), foreground='gray').pack(anchor=tk.W, pady=2)
        else:
            self._populate_controlnet_info(info_frame)
        
        # ControlNet strength controls
        controls_frame = ttk.LabelFrame(parent, text="ControlNet Strengths", padding=10)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Store reference for dynamic updates
        self.controlnet_controls_frame = controls_frame
        self.controlnet_frames = []
        
        # Populate controls if config exists
        if self.config:
            self._populate_controlnet_controls(controls_frame)
        
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
    
    def _populate_controlnet_info(self, parent):
        """Populate ControlNet information in the given frame"""
        if 'controlnets' in self.config and self.config['controlnets']:
            for i, cn_config in enumerate(self.config['controlnets']):
                cn_info_text = f"ControlNet {i+1}: {cn_config['model_id'].split('/')[-1]}\nPreprocessor: {cn_config['preprocessor']}"
                ttk.Label(parent, text=cn_info_text, font=('TkDefaultFont', 8)).pack(anchor=tk.W, pady=2)
    
    def _populate_controlnet_controls(self, parent):
        """Populate ControlNet strength controls in the given frame"""
        for i, (scale_var, enabled_var) in enumerate(zip(self.controlnet_vars, self.controlnet_enabled_vars)):
            cn_frame = ttk.Frame(parent)
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
    
    def setup_advanced_controls(self, parent):
        """Setup advanced controls"""
        # Camera settings
        camera_frame = ttk.LabelFrame(parent, text="Camera Settings", padding=10)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_frame, text=f"Camera Device: {self.args.camera}").pack(anchor=tk.W)
        resolution_text = f"{self.args.resolution}x{self.args.resolution}" if hasattr(self.args, 'resolution') and self.args.resolution else "Auto-detect"
        ttk.Label(camera_frame, text=f"Resolution: {resolution_text}").pack(anchor=tk.W)
        
        # Pipeline info
        pipeline_frame = ttk.LabelFrame(parent, text="Pipeline Information", padding=10)
        pipeline_frame.pack(fill=tk.X, padx=5, pady=5)
        
        if self.config:
            pipeline_type = self.config.get('pipeline_type', 'sd1.5')
            model_id = self.config['model_id'] if 'model_id' in self.config else "Unknown"
            acceleration = self.config.get('acceleration', 'none')
            
            ttk.Label(pipeline_frame, text=f"Pipeline Type: {pipeline_type}").pack(anchor=tk.W)
            ttk.Label(pipeline_frame, text=f"Model: {model_id.split('/')[-1]}").pack(anchor=tk.W)
            ttk.Label(pipeline_frame, text=f"Acceleration: {acceleration}").pack(anchor=tk.W)
            
            if acceleration.lower() == 'tensorrt':
                ttk.Label(pipeline_frame, text="TensorRT: Frame buffer size locked at config value", 
                         font=('TkDefaultFont', 8), foreground='blue').pack(anchor=tk.W, pady=(5,0))
        else:
            ttk.Label(pipeline_frame, text="No configuration loaded", 
                     font=('TkDefaultFont', 9), foreground='gray').pack(anchor=tk.W, pady=2)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(parent, text="Performance", padding=10)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_fps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Show FPS", variable=self.show_fps_var).pack(anchor=tk.W)
        
        # Temporal consistency settings
        temporal_frame = ttk.LabelFrame(parent, text="Temporal Consistency", padding=10)
        temporal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Check if TensorRT is enabled
        acceleration = self.config.get('acceleration', 'none') if self.config else 'none'
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
            ttk.Label(temporal_frame, text="Frame Buffer Size locked (TensorRT enabled)", 
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
        if not self.wrapper or not hasattr(self.wrapper, 'stream'):
            # No wrapper loaded yet, just update the GUI variables
            return
            
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
            if self.wrapper and hasattr(self.wrapper, 'update_controlnet_scale'):
                self.wrapper.update_controlnet_scale(index, new_value)
        except Exception:
            pass
    
    def toggle_controlnet(self, index):
        """Toggle ControlNet on/off"""
        try:
            enabled = self.controlnet_enabled_vars[index].get()
            strength = self.controlnet_vars[index].get() if enabled else 0.0
            if self.wrapper and hasattr(self.wrapper, 'update_controlnet_scale'):
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
        if not self.config:
            messagebox.showwarning("Warning", "No configuration loaded. Please load a config file first.")
            return
            
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
        # Check if config is loaded
        if not self.config:
            messagebox.showerror("Error", "Please load a configuration file first")
            return
        
        try:
            # Create wrapper if it doesn't exist yet
            if not self.wrapper:
                print("start_camera: Creating StreamDiffusion pipeline...")
                self.status_var.set("Loading pipeline...")
                self.root.update()  # Update GUI to show status change
                
                self._create_wrapper()
                print("start_camera: Pipeline created successfully")
            
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
            print(f"start_camera: {error_msg}")
            messagebox.showerror("Error", error_msg)
            # Reset status on error
            if self.config:
                self.status_var.set("Config loaded - Ready to start camera")
            else:
                self.status_var.set("No config loaded")
    
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
        if self.config:
            self.status_var.set("Config loaded - Ready to start camera")
        else:
            self.status_var.set("No config loaded")
        
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
        while self.running and self.wrapper:
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
        if not self.running or not self.wrapper:
            messagebox.showwarning("Warning", "Camera is not running or pipeline not loaded")
            return
        
        try:
            # Get current frame from queue if available
            if not self.result_queue.empty():
                # Get the most recent result by draining the queue
                latest_result = None
                try:
                    while True:
                        latest_result = self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                if latest_result:
                    frame, output_image, _ = latest_result
                    
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
    parser = argparse.ArgumentParser(description="ControlNet Webcam GUI Demo")
    
    # Get the script directory to make paths relative to it
    script_dir = Path(__file__).parent
    default_config = script_dir.parent.parent / "configs" / "controlnet_examples" / "multi_controlnet_example.yaml"
    
    parser.add_argument("--config", type=str, 
                       default=str(default_config) if default_config.exists() else None,
                       help="Path to ControlNet configuration file (optional)")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device index")
    parser.add_argument("--model", type=str,
                       help="Override base model path from config")
    parser.add_argument("--prompt", type=str,
                       help="Override prompt from config")
    parser.add_argument("--resolution", type=int, default=None,
                       help="Camera and output resolution (auto-detects from pipeline type if not specified)")
    
    args = parser.parse_args()
    
    print("ControlNet StreamDiffusion GUI Demo")
    print("Configuration can be loaded from the GUI or specified via --config argument")
    print("Pipeline creation is deferred until 'Start Camera' is clicked for faster startup")
    
    # Create and run GUI
    gui = ControlNetGUI(args)
    gui.run()


if __name__ == "__main__":
    main() 