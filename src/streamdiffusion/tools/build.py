import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
import ast
from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def accelerate_pipeline(is_sdxl, model_id, height, width, num_timesteps, controlnet_dicts, export_dir):

    pipe_type = StableDiffusionPipeline
    vae_model = "madebyollin/taesd"
    if is_sdxl:
        vae_model = "madebyollin/taesdxl"
        pipe_type = StableDiffusionXLPipeline

    # load vae
    vae = AutoencoderTiny.from_pretrained(vae_model)

    pipe = pipe_type.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        vae=vae
    ).to('cuda')

    # StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=list(range(num_timesteps)),
        torch_dtype=torch.float16,
        height=height,
        width=width
    )
    
    if controlnet_dicts is not None:
        stream.load_controlnet(controlnet_dicts)
        print(f"Use controlnet: {controlnet_dicts}")

    # Set batch sizes
    vae_batch_size = 1
    unet_batch_size = num_timesteps

    # build models
    accelerate_with_tensorrt(
        stream=stream,
        engine_dir=str(export_dir),
        unet_batch_size=(unet_batch_size, unet_batch_size),
        vae_batch_size=(vae_batch_size, vae_batch_size),
        model_id=model_id,
        controlnet_dicts=controlnet_dicts,
        unet_engine_build_options={
            'opt_image_height': height,
            'opt_image_width': width,
            'min_image_resolution': min(height, width),
            'max_image_resolution': max(height, width),
            'opt_batch_size': unet_batch_size,
            'build_static_batch': True,
            'build_dynamic_shape': False
        },
        vae_engine_build_options={
            'opt_image_height': height,
            'opt_image_width': width,
            'min_image_resolution': min(height, width),
            'max_image_resolution': max(height, width),
            'opt_batch_size': vae_batch_size,
            'build_static_batch': True,
            'build_dynamic_shape': False
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--sdxl',
                        type=bool, default=False)
    parser.add_argument('--model_id',
                        type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--export_dir',
                        type=Path, required=True, help='Directory for generated models')
    parser.add_argument('--height',
                        type=int, required=True, help='image height')
    parser.add_argument('--width',
                        type=int, required=True, help='image width')
    parser.add_argument('--num_timesteps',
                        type=int, default=1, help='number of timesteps')
    parser.add_argument('--controlnet_dicts',
                    type=ast.literal_eval,  
                    default=None)

    args = parser.parse_args()

    accelerate_pipeline(
        args.sdxl,
        args.model_id,
        args.height,
        args.width,
        args.num_timesteps,
        args.controlnet_dicts,
        args.export_dir
    )

# Usage:
# docker run -it --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/oylo/models:/root/app/engines builder
#python -m src/streamdiffusion/acceleration/tensorrt/build.py --model_id stabilityai/sd-turbo --height 768 --width 768 --num_timesteps 2 --export_dir /engines --sdxl True --controlnet_dicts "[{'xinsir/controlnet-canny-sdxl-1.0': 0.8}]"