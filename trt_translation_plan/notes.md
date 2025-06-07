# TensorRT Batch Size Mismatch Issue

## Problem
StreamDiffusion + TensorRT fails when `frame_buffer_size > 1` due to batch size mismatch between engine compilation and runtime.

**Error**: `Expected dimensions are [6,4,64,64]. Set dimensions are [4,4,64,64]`

## Root Cause
- **Compilation**: UNet engine built with `trt_unet_batch_size = 6`
- **Runtime**: `x_t_latent` tensor has batch size 4
- **Result**: TensorRT engine rejects mismatched input

## Technical Details

### Batch Size Calculation (compilation time)
```python
# StreamDiffusion.__init__()
if self.cfg_type == "full":
    self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
# Example: 2 * 2 * 3 = 12 or (2+1) * 3 = 9, but debug shows 6
```

### Runtime Tensor Creation (inference time)
```python
# StreamDiffusion.predict_x0_batch()
def predict_x0_batch(self, x_t_latent):  # Input: batch_size=1
    if self.denoising_steps_num > 1:
        x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)  # Results in batch_size=4
```

## Disconnect
Engine compiled for batch size 6, runtime produces batch size 4.

## Working Fix
```python
# In predict_x0_batch() before concatenation:
if x_t_latent.shape[0] == 1 and self.frame_bff_size > 1:
    x_t_latent = x_t_latent.repeat(self.frame_bff_size, 1, 1, 1)
```

## Investigation Needed
1. Is `frame_buffer_size > 1` + TensorRT supported?
2. Is batch size calculation formula correct?
3. Should runtime tensor creation match compilation parameters?

**Files**: `pipeline.py`, `acceleration/tensorrt/__init__.py`, `wrapper.py`

**Note**: This is a core StreamDiffusion TensorRT integration issue, not ControlNet-specific.
