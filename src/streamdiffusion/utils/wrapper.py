import traceback
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from diffusers import  AutoencoderTiny, ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion

torch.set_float32_matmul_precision('high') #test
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        controlnet_dicts: Optional[List[Dict[str, float]]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        HyperSD_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda", "mps"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        acceleration: bool = False,
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "none",
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False
    ):
        self.sd_turbo = "turbo" in model_id_or_path
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = len(t_index_list) * frame_buffer_size
        self.use_safety_checker = use_safety_checker
        self.is_controlnet_enabled = controlnet_dicts is not None
        self.default_tiny_vae = "madebyollin/taesdxl"

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            controlnet_dicts=controlnet_dicts,
            lcm_lora_id=lcm_lora_id,
            HyperSD_lora_id=HyperSD_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            CM_lora_type=CM_lora_type,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed
        )
        
        #Eliminare se gira cosi
        #if hasattr(self.stream.unet, 'config'):
        #    self.stream.unet.config.addition_embed_type = None
            
        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        controlnet_images: Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        assert (self.is_controlnet_enabled and controlnet_images is not None) or (
            not self.is_controlnet_enabled and controlnet_images is None
        ), "If ControlNet is disabled, please do not provide controlnet_images, vice versa."
        return self.img2img(image, prompt, controlnet_images)

    def img2img(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        controlnet_images: Optional[Union[str, Image.Image, list[str], list[Image.Image], torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        if isinstance(controlnet_images, str) or isinstance(controlnet_images, Image.Image):
            controlnet_images = self.preprocess_image(controlnet_images, is_controlnet_image=True)

        if isinstance(controlnet_images, list):
            controlnet_images = [self.preprocess_image(img, is_controlnet_image=True) for img in controlnet_images]
            controlnet_images = torch.stack(controlnet_images)

        image_tensor = self.stream(image, controlnet_images=controlnet_images)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image], is_controlnet_image: bool = False) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return (
            self.stream.image_processor.preprocess(image, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if not is_controlnet_image
            else self.stream.controlnet_image_processor.preprocess(image, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
        )

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if self.frame_buffer_size > 1:
            return self.stream.image_processor.postprocess(image_tensor.cpu(), output_type=output_type)
        else:
            return self.stream.image_processor.postprocess(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        controlnet_dicts: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        HyperSD_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: bool = False,
        do_add_noise: bool = True,
        CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "none",
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2
    ) -> StreamDiffusion:
        
        if acceleration:
            print ("Init acceleration inductor...")
            self.dtype = torch.bfloat16
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
            #torch._inductor.config.force_fuse_int_mm_with_mul = True
            #torch._inductor.config.use_mixed_mm = True

        if self.is_controlnet_enabled:
            controlnets = [
                ControlNetModel.from_pretrained(list(controlnet_dict.keys())[0]).to(self.device, self.dtype)
                for controlnet_dict in controlnet_dicts
            ]
            try:
                pipe: StableDiffusionXLControlNetPipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    model_id_or_path, controlnet=controlnets,
                ).to(device=self.device, dtype=self.dtype)
                pipe.controlnet_conditioning_scales = [list(d.values())[0] for d in controlnet_dicts]
            except ValueError:
                pipe: StableDiffusionXLControlNetPipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                    model_id_or_path, controlnet=controlnets,
                ).to(device=self.device, dtype=self.dtype)
                pipe.controlnet_conditioning_scales = [list(d.values())[0] for d in controlnet_dicts]
        else:
            try:  # Load from local directory
                pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)
            except ValueError:  # Load from huggingface
                pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
                    model_id_or_path,
                ).to(device=self.device, dtype=self.dtype)

            except Exception:  # No model found
                traceback.print_exc()
                print("Model load has failed. Doesn't exist.")
                exit()
        
        #if acceleration:
        #    print ("Fuse QKV Projections...")
        #    pipe.fuse_qkv_projections()

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=True,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            if CM_lora_type == "lcm":
                print("-----------------Using lcm-----------------")
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            elif CM_lora_type == "Hyper_SD":
                print(f"-----------------Using Hyper_SD {HyperSD_lora_id}-----------------")
                if HyperSD_lora_id is not None:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD", model_name=HyperSD_lora_id
                    )
                elif HyperSD_lora_id is None and controlnet_dicts is not None:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD",
                        model_name="Hyper-SD15-4step-lora.safetensors",
                    )
                    print("To generate better results with ControlNet, using 4-steps Hyper-SD instead of 1-step.")
                else:
                    stream.load_HyperSD_lora(
                        pretrained_model_name_or_path_or_dict="ByteDance/Hyper-SD",
                        model_name="Hyper-SD15-1step-lora.safetensors",
                    )
                    print("Using 1-step Hyper-SD.")
                stream.fuse_lora()
            else:  # CM_lora_type == "none"
                pass

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=pipe.device, dtype=self.dtype)
            else:
                stream.vae = AutoencoderTiny.from_pretrained(self.default_tiny_vae).to(
                    device=pipe.device, dtype=self.dtype
                )

        #if acceleration and pipe is not None:

            print ("Memory format conversion...")
            stream.unet.to(memory_format=torch.channels_last)
            stream.vae.to(memory_format=torch.channels_last)
            if self.is_controlnet_enabled:
                stream.controlnet.to(memory_format=torch.channels_last)
            
            print ("Apply dynamic quantization...")
            from torchao import swap_conv2d_1x1_to_linear, apply_dynamic_quant
            swap_conv2d_1x1_to_linear(stream.unet, self.conv_filter_fn)
            swap_conv2d_1x1_to_linear(stream.vae, self.conv_filter_fn)
            swap_conv2d_1x1_to_linear(stream.controlnet, self.conv_filter_fn)
            apply_dynamic_quant(stream.unet, self.dynamic_quant_filter_fn)
            apply_dynamic_quant(stream.vae, self.dynamic_quant_filter_fn)
            apply_dynamic_quant(stream.controlnet, self.dynamic_quant_filter_fn)

            print ("Apply torch compile optimization...")
            stream.unet = torch.compile(stream.unet, mode="reduce-overhead", fullgraph=True)
            stream.vae.decode = torch.compile(stream.vae.decode, mode="reduce-overhead", fullgraph=True)
            #stream.vae.encode = torch.compile(stream.vae.encode, mode="reduce-overhead", fullgraph=True)
            if self.is_controlnet_enabled:
                stream.controlnet = torch.compile(stream.controlnet, mode="reduce-overhead", fullgraph=True)

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.Generator(),
            seed=seed,
        )

        if self.use_safety_checker:
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )
            from transformers import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
    
    def dynamic_quant_filter_fn(mod, *args):
        return (
            isinstance(mod, torch.nn.Linear)
            and mod.in_features > 16
            and (mod.in_features, mod.out_features)
            not in [
                (1280, 640),
                (1920, 1280),
                (1920, 640),
                (2048, 1280),
                (2048, 2560),
                (2560, 1280),
                (256, 128),
                (2816, 1280),
                (320, 640),
                (512, 1536),
                (512, 256),
                (512, 512),
                (640, 1280),
                (640, 1920),
                (640, 320),
                (640, 5120),
                (640, 640),
                (960, 320),
                (960, 640),
            ]
        )


    def conv_filter_fn(mod, *args):
        return (
            isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
        )