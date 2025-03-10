import gc
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from ...unet_with_control import UNet2DConditionControlNetModel
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine, UNet2DConditionControlNetModelEngine
from .models import VAE, BaseModel, UNet, VAEEncoder, UNetXLTurbo, UNetXL, UNetWithControlNet, UNetXLWithControlNet


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    opt_batch_size = engine_build_options.pop('opt_batch_size', opt_batch_size)
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options=None,
):
    if engine_build_options is None:
        engine_build_options = {}
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options=None,
):
    if engine_build_options is None:
        engine_build_options = {}
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    opt_batch_size = engine_build_options.pop('opt_batch_size', opt_batch_size)
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )

def compile_control_unet(
    unet: UNet2DConditionControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options=None,
):
    if engine_build_options is None:
        engine_build_options = {}
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    unet.unet = unet.unet.to(torch.device("cuda"), dtype=torch.float16)
    unet.controlnets = [controlnet.to(torch.device("cuda"), dtype=torch.float16) for controlnet in unet.controlnets]
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    opt_batch_size = engine_build_options.pop('opt_batch_size', opt_batch_size)
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )

def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    unet_batch_size: tuple = (1, 2),
    vae_batch_size: tuple = (1, 1),
    is_controlnet_enabled: bool = False,
    num_controlnets: int = 0,
    unet_engine_build_options=None,
    vae_engine_build_options=None,
    use_cuda_graph: bool = False,
):
    # argument default values should not be mutable
    if vae_engine_build_options is None:
        vae_engine_build_options = {}
    if unet_engine_build_options is None:
        unet_engine_build_options = {}

    # fix opt_batch_size
    if unet_engine_build_options.get("opt_batch_size", None) is None:
        unet_engine_build_options["opt_batch_size"] = unet_batch_size[1]
    if vae_engine_build_options.get("opt_batch_size", None) is None:
        vae_engine_build_options["opt_batch_size"] = vae_batch_size[1]
        
    # take refs of models from pipeline
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"
    
    if stream.sdxl and is_controlnet_enabled:
        unet_model = UNetXLWithControlNet(
        fp16=True,
        device=stream.device,
        num_controlnets=num_controlnets,
        min_batch_size=unet_batch_size[0],
        max_batch_size=unet_batch_size[1],
        encoder_hidden_states_dim=text_encoder.config.hidden_size,
        unet_dim=unet.unet.config.in_channels,
    )
    
    elif stream.sdxl and not is_controlnet_enabled:
        unet_model = UNetXL(
        fp16=True,
        device=stream.device,
        min_batch_size=unet_batch_size[0],
        max_batch_size=unet_batch_size[1],
        encoder_hidden_states_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
        
    elif not stream.sdxl and is_controlnet_enabled:
        
        unet_model = UNetWithControlNet(
        fp16=True,
        device=stream.device,
        num_controlnets=num_controlnets,
        min_batch_size=unet_batch_size[0],
        max_batch_size=unet_batch_size[1],
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
        
    elif not stream.sdxl and not is_controlnet_enabled:
    
        unet_model = UNet(
        fp16=True,
        device=stream.device,
        min_batch_size=unet_batch_size[0],
        max_batch_size=unet_batch_size[1],
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )

    vae_decoder_model = VAE(
        device=stream.device,
        min_batch_size=vae_batch_size[0],
        max_batch_size=vae_batch_size[1]
    )

    vae_encoder_model = VAEEncoder(
        device=stream.device,
        min_batch_size=vae_batch_size[0],
        max_batch_size=vae_batch_size[1]
    )

    if not os.path.exists(unet_engine_path):
        if not is_controlnet_enabled:
            compile_unet(
                unet,
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                engine_build_options=unet_engine_build_options)
        else:
            print("In compile unet control")
            compile_control_unet(
                unet,
                unet_model,
                create_onnx_path("unet", onnx_dir, opt=False),
                create_onnx_path("unet", onnx_dir, opt=True),
                unet_engine_path,
                engine_build_options=unet_engine_build_options)
    else:
        del unet

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            engine_build_options=vae_engine_build_options)
        print("In compile vae decoder")

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            engine_build_options=vae_engine_build_options)
        print("In compile vae encoder")

    del vae

    cuda_stream = cuda.Stream()
    
    if is_controlnet_enabled:
        stream.unet = UNet2DConditionControlNetModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=False)
    else:
        stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=False)

    print("In stream unet")

    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    
    print("In autoencoder kl")

    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream
