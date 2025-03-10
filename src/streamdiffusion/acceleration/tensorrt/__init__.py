import gc
import os
import torch
from polygraphy import cuda
from ...pipeline import StreamDiffusion
from .engine import (
    AutoencoderKLEngine,
    UNet2DConditionModelEngine,
    UNet2DConditionControlNetModelEngine,
    UNet2DConditionXLControlNetModelEngine
)

from tools.convert_stable_diffusion_controlnet_to_onnx import (UNet2DConditionControlNetModel, UNet2DConditionXLControlNetModel)

def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    is_controlnet_enabled: bool = False,
    use_cuda_graph: bool = False,
):
    # Salva la configurazione originale del VAE
    vae_config = stream.vae.config
    vae_dtype = stream.vae.dtype

    # Rimuove i riferimenti ai modelli originari
    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    # Percorsi dei file engine già generati
    unet_engine_path = os.path.join(engine_dir, "unet", "model.engine")
    vae_encoder_engine_path = os.path.join(engine_dir, "vae_encoder", "model.engine")
    vae_decoder_engine_path = os.path.join(engine_dir, "vae_decoder", "model.engine")

    # Crea un CUDA stream
    cuda_stream = cuda.Stream()

    # Carica l'engine UNet in base all'abilitazione del ControlNet
    if is_controlnet_enabled:
        
        if stream.sdxl:
            stream.unet = UNet2DConditionXLControlNetModelEngine(
            unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph
        )
        else:
            stream.unet = UNet2DConditionControlNetModelEngine(
            unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph
        )

    else:
        stream.unet = UNet2DConditionModelEngine(
            unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph
        )

    # Carica l'engine VAE (encoder e decoder)
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )

    # Ripristina la configurazione e il dtype del VAE
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    return stream