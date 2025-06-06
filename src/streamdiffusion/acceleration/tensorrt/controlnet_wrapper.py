"""
ControlNet-Aware UNet Wrapper for ONNX Export

This module provides a wrapper that combines UNet with ControlNet inputs for ONNX export.
It mimics ComfyUI_TensorRT's get_backbone() functionality, organizing control arguments
into the proper format for diffusers UNet models.
"""

import torch
from typing import List, Optional, Dict, Any
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ControlNetUNetWrapper(torch.nn.Module):
    """
    Wrapper that combines UNet with ControlNet inputs for ONNX export
    
    This wrapper organizes ControlNet inputs during ONNX export, ensuring that
    all control tensors are properly mapped to the UNet's ControlNet conditioning
    inputs (down_block_additional_residuals, mid_block_additional_residual).
    """
    
    def __init__(self, unet: UNet2DConditionModel, control_input_names: List[str]):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        
        # Pre-compute control input organization for efficiency
        self.input_control_indices = []
        self.output_control_indices = []
        self.middle_control_indices = []
        
        for i, name in enumerate(control_input_names):
            if name in ["sample", "timestep", "encoder_hidden_states"]:
                continue  # Skip base UNet inputs
                
            if "input_control" in name:
                self.input_control_indices.append(i)
            elif "output_control" in name:
                self.output_control_indices.append(i)
            elif "middle_control" in name:
                self.middle_control_indices.append(i)
        
        print(f"🔗 ControlNet wrapper initialized with {len(self.input_control_indices)} input controls, "
              f"{len(self.output_control_indices)} output controls, {len(self.middle_control_indices)} middle controls")
    
    def forward(self, sample, timestep, encoder_hidden_states, *control_args):
        """
        Forward pass that organizes control inputs and calls UNet
        
        Args:
            sample: Latent sample tensor
            timestep: Timestep tensor
            encoder_hidden_states: Text embeddings
            *control_args: Variable number of ControlNet input tensors
            
        Returns:
            UNet output (noise prediction)
        """
        print(f"🔗 ControlNet wrapper forward: sample.shape={sample.shape}")
        print(f"🔗 Received {len(control_args)} control tensors")
        
        # tensors at proper resolutions:
        # [320ch@64x64]*3 + [320/640ch@32x32]*3 + [640/1280ch@16x16]*3 + [1280ch@8x8]*3 + middle@8x8
        
        down_block_controls = []
        mid_block_control = None
        
        # Extract the 12 down block control tensors + 1 middle control tensor
        input_control_count = len(self.input_control_indices)
        print(f"🔗 Processing {input_control_count} input controls - NO INTERPOLATION NEEDED!")
        
        if input_control_count > 0:
            # Get all control tensors at their CORRECT native sizes
            all_control_tensors = []
            middle_tensor = None
            
            for i, idx in enumerate(self.input_control_indices):
                control_arg_idx = idx - 3  # Adjust for sample, timestep, encoder_hidden_states
                if control_arg_idx < len(control_args):
                    tensor = control_args[control_arg_idx]
                    
                    # Check if this is the middle control tensor (last one)
                    if i == input_control_count - 1:  # Last tensor is middle
                        middle_tensor = tensor
                        print(f"🔗 Middle control: {tensor.shape} (already correct size!)")
                    else:
                        # No interpolation needed - tensor is already at correct size!
                        all_control_tensors.append(tensor)
                        print(f"🔗 Down control {i}: {tensor.shape} (already correct size!)")
            
            # Validate we have exactly 12 down block tensors
            if len(all_control_tensors) == 12:
                down_block_controls = all_control_tensors
                mid_block_control = middle_tensor
            else:
                print(f"ERROR: Expected 12 down block tensors, got {len(all_control_tensors)}")
                # Fallback: empty controls
                down_block_controls = None
                mid_block_control = None
        
        # Prepare UNet arguments
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        
        # Pass control tensors directly - no processing needed!
        if down_block_controls:
            # print(f"Adding down_block_additional_residuals:")
            # print(f"   Type: list with {len(down_block_controls)} tensors")
            # for i, ctrl in enumerate(down_block_controls):
            #     print(f"   Position {i}: {ctrl.shape}")
            unet_kwargs['down_block_additional_residuals'] = down_block_controls
        
        if mid_block_control is not None:
            # print(f"Adding mid_block_additional_residual: {mid_block_control.shape}")
            unet_kwargs['mid_block_additional_residual'] = mid_block_control
        
        # print(f"Calling UNet with {len(unet_kwargs)} arguments")
        
        # Call UNet with ControlNet conditioning
        return self.unet(**unet_kwargs)


class MultiControlNetUNetWrapper(torch.nn.Module):
    """
    Advanced wrapper for multiple ControlNets with different scales
    
    This wrapper can handle multiple ControlNet inputs with different conditioning scales,
    combining their outputs before passing to the UNet.
    """
    
    def __init__(self, 
                 unet: UNet2DConditionModel, 
                 control_input_names: List[str],
                 num_controlnets: int = 1,
                 conditioning_scales: Optional[List[float]] = None):
        super().__init__()
        self.unet = unet
        self.control_input_names = control_input_names
        self.num_controlnets = num_controlnets
        self.conditioning_scales = conditioning_scales or [1.0] * num_controlnets
        
        # Organize control indices per ControlNet
        self.controlnet_indices = []
        controls_per_net = (len(control_input_names) - 3) // num_controlnets  # -3 for base inputs
        
        for cn_idx in range(num_controlnets):
            start_idx = 3 + cn_idx * controls_per_net  # Skip sample, timestep, encoder_hidden_states
            end_idx = start_idx + controls_per_net
            self.controlnet_indices.append(list(range(start_idx, end_idx)))
        
        print(f"Multi-ControlNet wrapper initialized for {num_controlnets} ControlNets")
    
    def forward(self, sample, timestep, encoder_hidden_states, *control_args):
        """
        Forward pass for multiple ControlNets
        
        Args:
            sample: Latent sample tensor
            timestep: Timestep tensor
            encoder_hidden_states: Text embeddings
            *control_args: Control tensors for all ControlNets
            
        Returns:
            UNet output with combined ControlNet conditioning
        """
        combined_down_controls = None
        combined_mid_control = None
        
        # Process each ControlNet
        for cn_idx, indices in enumerate(self.controlnet_indices):
            scale = self.conditioning_scales[cn_idx]
            if scale == 0:
                continue  # Skip disabled ControlNets
            
            # Extract controls for this ControlNet
            cn_controls = [control_args[i - 3] for i in indices if i - 3 < len(control_args)]
            
            if not cn_controls:
                continue
            
            # Assume first half are down controls, last is middle control
            # This is a simplified assumption - real implementation would need
            # more sophisticated control tensor organization
            num_down = len(cn_controls) - 1
            down_controls = cn_controls[:num_down]
            mid_control = cn_controls[num_down] if num_down < len(cn_controls) else None
            
            # Apply conditioning scale
            scaled_down = [ctrl * scale for ctrl in down_controls]
            scaled_mid = mid_control * scale if mid_control is not None else None
            
            # Combine with previous ControlNets
            if combined_down_controls is None:
                combined_down_controls = scaled_down
                combined_mid_control = scaled_mid
            else:
                # Add to existing controls
                for i in range(min(len(combined_down_controls), len(scaled_down))):
                    combined_down_controls[i] += scaled_down[i]
                if scaled_mid is not None and combined_mid_control is not None:
                    combined_mid_control += scaled_mid
        
        # Prepare UNet arguments
        unet_kwargs = {
            'sample': sample,
            'timestep': timestep,
            'encoder_hidden_states': encoder_hidden_states,
            'return_dict': False,
        }
        
        # Add combined ControlNet conditioning
        if combined_down_controls:
            unet_kwargs['down_block_additional_residuals'] = list(reversed(combined_down_controls))
        if combined_mid_control is not None:
            unet_kwargs['mid_block_additional_residual'] = combined_mid_control
        
        return self.unet(**unet_kwargs)


def create_controlnet_wrapper(unet: UNet2DConditionModel, 
                            control_input_names: List[str],
                            num_controlnets: int = 1,
                            conditioning_scales: Optional[List[float]] = None) -> torch.nn.Module:
    """
    Factory function to create appropriate ControlNet wrapper
    
    Args:
        unet: Diffusers UNet model
        control_input_names: List of input names including ControlNet inputs
        num_controlnets: Number of ControlNets (default: 1)
        conditioning_scales: Conditioning scales for each ControlNet
        
    Returns:
        Appropriate wrapper module for ONNX export
    """
    if num_controlnets == 1:
        return ControlNetUNetWrapper(unet, control_input_names)
    else:
        return MultiControlNetUNetWrapper(
            unet, control_input_names, num_controlnets, conditioning_scales
        )


def organize_control_tensors(control_tensors: List[torch.Tensor], 
                           control_input_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """
    Organize control tensors by type (input, output, middle)
    
    This function helps organize ControlNet tensors for runtime processing,
    similar to how the wrapper organizes them for ONNX export.
    
    Args:
        control_tensors: List of ControlNet tensors
        control_input_names: Names corresponding to each tensor
        
    Returns:
        Dictionary with 'input', 'output', 'middle' keys containing organized tensors
    """
    organized = {'input': [], 'output': [], 'middle': []}
    
    for tensor, name in zip(control_tensors, control_input_names):
        if "input_control" in name:
            organized['input'].append(tensor)
        elif "output_control" in name:
            organized['output'].append(tensor)
        elif "middle_control" in name:
            organized['middle'].append(tensor)
    
    return organized 