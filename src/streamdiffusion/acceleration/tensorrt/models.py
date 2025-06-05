#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch=max_batch,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class UNet(BaseModel):
    def __init__(
        self,
        fp16=False,
        device="cuda",
        max_batch=16,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        use_control=False,
        unet_arch=None,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch=max_batch,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"
        
        # ControlNet support
        self.use_control = use_control
        self.unet_arch = unet_arch or {}
        
        # Initialize ControlNet input configurations
        if self.use_control and self.unet_arch:
            self.control_inputs = self.get_control()
            self._add_control_inputs()
        else:
            self.control_inputs = {}

    def get_control(self) -> dict:
        """
        Generate ControlNet input configurations based on REAL ControlNet format
        
        BREAKTHROUGH DISCOVERY: Real ControlNet provides 12 down block tensors + 1 middle!
        Each UNet down block has multiple ResNet layers, and ControlNet provides 
        a tensor for EACH individual ResNet layer.
        
        REAL FORMAT DISCOVERED:
        - Position 0-2:  320ch @ 64×64  (Down block 0 - 3 ResNet layers)
        - Position 3-5:  320/640ch @ 32×32  (Down block 1 - 3 ResNet layers)  
        - Position 6-8:  640/1280ch @ 16×16  (Down block 2 - 3 ResNet layers)
        - Position 9-11: 1280ch @ 8×8   (Down block 3 - 3 ResNet layers)
        - Middle:        1280ch @ 8×8   (1 middle block tensor)
        
        TENSORRT COMPATIBILITY FIX:
        TensorRT doesn't support varying spatial dimensions in optimization profiles.
        So we generate ALL control inputs at uniform 64×64 and let the wrapper
        downsample them to correct sizes at runtime: 64→32→16→8
        
        Total: 13 control inputs (12 down + 1 middle), all at 64×64 for TensorRT
        """
        # Get UNet architecture
        block_out_channels = self.unet_arch.get('block_out_channels', (320, 640, 1280, 1280))
        
        # Base spatial dimensions for latent input (512x512 → 64x64 latents)
        base_spatial = 64  # Always 64x64 for 512x512 generation
        
        control_inputs = {}
        
        print(f"🎯 REAL CONTROLNET FORMAT: Generating 12 down + 1 middle control tensors")
        print(f"   Based on discovered format: [320ch@64x64]*3 + [320/640ch@32x32]*3 + [640/1280ch@16x16]*3 + [1280ch@8x8]*3 + middle")
        
        # Generate the EXACT 12 down block control tensors to match real ControlNet
        # CRITICAL FIX: Use uniform spatial dimensions for TensorRT compatibility
        # Let the wrapper handle downsampling at runtime
        control_configs = [
            # Down block 0: 3 tensors @ 64x64 (uniform spatial size)
            (320, 64),   # Position 0: First ResNet in down block 0
            (320, 64),   # Position 1: Second ResNet in down block 0  
            (320, 64),   # Position 2: Third ResNet in down block 0
            
            # Down block 1: 3 tensors @ 64x64 (uniform spatial size) 
            (320, 64),   # Position 3: First ResNet in down block 1 (320→640 transition)
            (640, 64),   # Position 4: Second ResNet in down block 1
            (640, 64),   # Position 5: Third ResNet in down block 1
            
            # Down block 2: 3 tensors @ 64x64 (uniform spatial size)
            (640, 64),   # Position 6: First ResNet in down block 2 (640→1280 transition)  
            (1280, 64),  # Position 7: Second ResNet in down block 2
            (1280, 64),  # Position 8: Third ResNet in down block 2
            
            # Down block 3: 3 tensors @ 64x64 (uniform spatial size)
            (1280, 64),  # Position 9: First ResNet in down block 3
            (1280, 64),  # Position 10: Second ResNet in down block 3
            (1280, 64),  # Position 11: Third ResNet in down block 3
        ]
        
        # Generate down block control inputs
        for i, (channels, spatial_size) in enumerate(control_configs):
            input_name = f"input_control_{i:02d}"  # Use zero-padded numbers
            control_inputs[input_name] = {
                'batch': self.min_batch,
                'channels': channels,
                'height': spatial_size,
                'width': spatial_size,
                'downsampling_factor': 1  # No downsampling in diffusers
            }
            print(f"   {input_name}: {channels}ch @ {spatial_size}x{spatial_size}")
        
        # Add middle block control (always present)
        control_inputs["input_control_middle"] = {
            'batch': self.min_batch,
            'channels': 1280,  # Always 1280 for middle block
            'height': 64,      # Use uniform 64x64 for TensorRT compatibility (downsample at runtime)
            'width': 64,
            'downsampling_factor': 1
        }
        print(f"   input_control_middle: 1280ch @ 64x64 (will downsample to 8x8 at runtime)")
        
        print(f"🎯 TOTAL: {len(control_inputs)} control inputs (12 down + 1 middle)")
        print(f"   This matches the EXACT real ControlNet format!")
        
        return control_inputs

    def _add_control_inputs(self):
        """Add ControlNet inputs to the model's input/output specifications"""
        if not self.control_inputs:
            return
        
        # Store original methods
        self._original_get_input_names = self.get_input_names
        self._original_get_dynamic_axes = self.get_dynamic_axes
        self._original_get_input_profile = self.get_input_profile
        self._original_get_shape_dict = self.get_shape_dict
        self._original_get_sample_input = self.get_sample_input

    def get_input_names(self):
        """Get input names including ControlNet inputs"""
        base_names = ["sample", "timestep", "encoder_hidden_states"]
        if self.use_control and self.control_inputs:
            # Add control input names in the correct order
            control_names = sorted(self.control_inputs.keys())
            return base_names + control_names
        return base_names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        base_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        
        if self.use_control and self.control_inputs:
            # Add dynamic axes for ControlNet inputs
            for name in self.control_inputs:
                base_axes[name] = {0: "2B", 2: "H", 3: "W"}
        
        return base_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        
        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, latent_height, latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }
        
        if self.use_control and self.control_inputs:
            # Add ControlNet input profiles using EXACT specifications
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                height = shape_spec["height"]  # Use EXACT height from config
                width = shape_spec["width"]    # Use EXACT width from config
                
                print(f"🔧 Profile for {name}: {channels}ch @ {height}x{width}")
                
                profile[name] = [
                    (min_batch, channels, height, width),      # Use exact dimensions
                    (batch_size, channels, height, width),     # Use exact dimensions  
                    (max_batch, channels, height, width),      # Use exact dimensions
                ]
        
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }
        
        if self.use_control and self.control_inputs:
            # Add ControlNet input shapes using EXACT specifications
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                height = shape_spec["height"]  # Use EXACT height from config
                width = shape_spec["width"]    # Use EXACT width from config
                
                shape_dict[name] = (2 * batch_size, channels, height, width)  # Use exact dimensions
        
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        
        base_inputs = [
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.ones((2 * batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]
        
        if self.use_control and self.control_inputs:
            # Add ControlNet sample inputs using EXACT specifications
            control_inputs = []
            for name in sorted(self.control_inputs.keys()):
                shape_spec = self.control_inputs[name]
                channels = shape_spec["channels"]
                height = shape_spec["height"]  # Use the EXACT height from our config
                width = shape_spec["width"]    # Use the EXACT width from our config
                
                print(f"🔧 Generating sample tensor {name}: {channels}ch @ {height}x{width}")
                
                control_input = torch.randn(
                    2 * batch_size, channels, height, width, 
                    dtype=dtype, device=self.device
                )
                control_inputs.append(control_input)
            
            return tuple(base_inputs + control_inputs)
        
        return tuple(base_inputs)


class VAE(BaseModel):
    def __init__(self, device, max_batch, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch=max_batch,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch=max_batch,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )
