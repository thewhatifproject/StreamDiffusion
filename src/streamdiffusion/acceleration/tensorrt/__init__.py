import gc
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from pathlib import Path
from ...pipeline import StreamDiffusion
from ...unet_with_control import UNet2DConditionControlNetModel
from .builder import EngineBuilder
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine, UNet2DConditionControlNetModelEngine
from .models import VAE, BaseModel, UNet, UNetWithControlNet, UNetXLTurboWithControlNet, VAEEncoder, UNetXLTurbo

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
    opt_batch_size = engine_build_options.pop('opt_batch_size', opt_batch_size)
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
    engine_build_options: dict = {},
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
    engine_build_options: dict = {},
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

def create_prefix(
    model_id_or_path: str,
    unet_batch_size: tuple = (1, 2),
    controlnet_prefix: str = "",
    vae_batch_size: tuple = (1, 1)
):
    maybe_path = Path(model_id_or_path)
    if maybe_path.exists():
        return f"{maybe_path.stem}--unet_batch_size-{unet_batch_size}--vae_batch_size-{vae_batch_size}--controlnet-{controlnet_prefix}"
    else:
        return f"{model_id_or_path}--unet_batch_size-{unet_batch_size}--vae_batch_size-{vae_batch_size}--controlnet-{controlnet_prefix}"

def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    unet_batch_size: tuple = (1, 2),
    vae_batch_size: tuple = (1, 1),
    model_id: str = "stabilityai/sdxl-turbo",
    controlnet_dicts: bool = False,
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
        
    controlnet_prefix = "disabled"
    if controlnet_dicts:    
        controlnet_prefix = "_".join([f"{list(d.keys())[0]}-{list(d.values())[0]}" for d in controlnet_dicts])
        
    engine_dir = Path(engine_dir)
    
    unet_engine_path = os.path.join(
        engine_dir,
        create_prefix(
            model_id_or_path=model_id,
            controlnet_prefix=controlnet_prefix,
            unet_batch_size=unet_batch_size,
            vae_batch_size=vae_batch_size
        ),
        "unet.engine",
    )
    vae_encoder_engine_path = os.path.join(
        engine_dir,
        create_prefix(
            model_id_or_path=model_id,
            controlnet_prefix=controlnet_prefix,
            unet_batch_size=unet_batch_size,
            vae_batch_size=vae_batch_size
        ),
        "vae_encoder.engine",
    )
    vae_decoder_engine_path = os.path.join(
        engine_dir,
        create_prefix(
            model_id_or_path=model_id,
            controlnet_prefix=controlnet_prefix,
            unet_batch_size=unet_batch_size,
            vae_batch_size=vae_batch_size
        ),
        "vae_decoder.engine",
    )
    
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae
    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()
    
    if not os.path.exists(unet_engine_path):
        os.makedirs(os.path.dirname(unet_engine_path), exist_ok=True)
        
        if controlnet_dicts is not None and not stream.sdxl:
            
            unet_model = UNetWithControlNet(
            fp16=True,
            device=stream.device,
            min_batch_size=unet_batch_size[0],
            max_batch_size=unet_batch_size[1],
            num_controlnets=len(controlnet_dicts),
            embedding_dim=stream.text_encoder.config.hidden_size,
            unet_dim=unet.unet.config.in_channels,
        )
            
            compile_control_unet(
                unet,
                unet_model,
                unet_engine_path + ".onnx",
                unet_engine_path + ".opt.onnx",
                unet_engine_path,
                engine_build_options=unet_engine_build_options  
                )
            
        elif controlnet_dicts is not None and stream.sdxl:
            
            unet_model = UNetXLTurboWithControlNet(
            fp16=True,
            device=stream.device,
            min_batch_size=unet_batch_size[0],
            max_batch_size=unet_batch_size[1],
            num_controlnets=len(controlnet_dicts),
            embedding_dim=stream.text_encoder.config.hidden_size,
            unet_dim=unet.unet.config.in_channels,
        )
            
            compile_control_unet(
                unet,
                unet_model,
                unet_engine_path + ".onnx",
                unet_engine_path + ".opt.onnx",
                unet_engine_path,
                engine_build_options=unet_engine_build_options  
                )
            
        
        elif stream.sdxl:
            
            unet_model = UNetXLTurbo(
                fp16=True,
                device=stream.device,
                min_batch_size=unet_batch_size[0],
                max_batch_size=unet_batch_size[1],
                embedding_dim=text_encoder.config.hidden_size,
                unet_dim=unet.config.in_channels,
            )
            
            compile_unet(
            unet,
            unet_model,
            unet_engine_path + ".onnx",
            unet_engine_path + ".opt.onnx",
            unet_engine_path,
            engine_build_options=unet_engine_build_options        
            )
        
        else:
            unet_model = UNet(
            fp16=True,
            device=stream.device,
            min_batch_size=unet_batch_size[0],
            max_batch_size=unet_batch_size[1],
            embedding_dim=text_encoder.config.hidden_size,
            unet_dim=unet.config.in_channels,
            )
            
            compile_unet(
            unet,
            unet_model,
            unet_engine_path + ".onnx",
            unet_engine_path + ".opt.onnx",
            unet_engine_path,
            engine_build_options=unet_engine_build_options        
            )
    
    else:
        del unet
    
    if not os.path.exists(vae_encoder_engine_path):
        os.makedirs(os.path.dirname(vae_encoder_engine_path), exist_ok=True)
        
        vae_encoder_model = VAEEncoder(
        device=stream.device,
        min_batch_size=vae_batch_size[0],
        max_batch_size=vae_batch_size[1]
        )
        
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            vae_encoder_engine_path + ".onnx",
            vae_encoder_engine_path + ".opt.onnx",
            vae_encoder_engine_path,
            engine_build_options=vae_engine_build_options
        )
        
        
    if not os.path.exists(vae_decoder_engine_path):
        os.makedirs(os.path.dirname(vae_decoder_engine_path), exist_ok=True)
        
        vae_decoder_model = VAE(
        device=stream.device,
        min_batch_size=vae_batch_size[0],
        max_batch_size=vae_batch_size[1]
        )
        
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            vae_decoder_engine_path + ".onnx",
            vae_decoder_engine_path + ".opt.onnx",
            vae_decoder_engine_path,
            engine_build_options=vae_engine_build_options        
            )
    
    del vae
    
    #cuda_stream = cuda.Stream()
    
    #if controlnet_dicts is not None:
    #    stream.unet = UNet2DConditionControlNetModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph)
    #else:
    #    stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph)
    
    #stream.vae = AutoencoderKLEngine(
    #    vae_encoder_engine_path,
    #    vae_decoder_engine_path,
    #    cuda_stream,
    #    stream.pipe.vae_scale_factor,
    #    use_cuda_graph=use_cuda_graph,
    #)
    #setattr(stream.vae, "config", vae_config)
    #setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    #return stream