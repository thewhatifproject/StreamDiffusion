import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from diffusers import ControlNetModel, LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from huggingface_hub import hf_hub_download

from streamdiffusion.image_filter import SimilarImageFilter

class StreamDiffusion:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        t_index_list: Optional[List[int]] = None,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        generator: Optional[torch.Generator] = torch.Generator(),
        denoising_steps_num: Optional[int] = None,
        CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "none",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = generator

        if self.device.type == "mps":
            self.timer_event = getattr(torch, str(self.device).split(":", 1)[0])

        self.height = height
        self.width = width
        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.cfg_type = cfg_type
        self.CM_lora_type = CM_lora_type
        self.frame_bff_size = frame_buffer_size

        # Imposta t_index_list e il numero di passi di denoising
        if t_index_list is None and denoising_steps_num is None:
            raise ValueError("Please provide either t_index_list or num_denosing_steps")
        if t_index_list is not None:
            self.denoising_steps_num = len(t_index_list)
            self.t_list = t_index_list
        else:
            self.denoising_steps_num = denoising_steps_num

        # Calcola il batch size per il modello unet
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        # Imposta i componenti della pipeline
        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        self.controlnet_image_processor = VaeImageProcessor(
            pipe.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet  # In modalità integrazione diretta, self.unet include il modulo ControlNet
        self.vae = pipe.vae

        self.inference_time_ema = 0
        # Aggiorna il flag sdxl per riconoscere sia la versione standard sia quella ControlNet
        self.sdxl = True
        # Flag per indicare se il ControlNet è integrato
        self.controlnet_enabled = hasattr(pipe, "controlnet") and pipe.controlnet is not None

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.CM_lora_type = "lcm"
        self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name, **kwargs)

    def load_HyperSD_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]] = "ByteDance/Hyper-SD",
        adapter_name: Optional[Any] = None,
        model_name: Optional[str] = "Hyper-SD15-1step-lora.safetensors",
        **kwargs,
    ) -> None:
        self.CM_lora_type = "Hyper_SD"
        self.pipe.load_lora_weights(hf_hub_download(pretrained_model_name_or_path_or_dict, model_name), adapter_name, **kwargs)

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs)

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(fuse_unet=fuse_unet, fuse_text_encoder=fuse_text_encoder, lora_scale=lora_scale, safe_fusing=safe_fusing)

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    def init_generator(self, seed: Optional[int] = None) -> None:
        if seed is None or seed < 0:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def update_scheduler(self, t_index_list: Optional[List[int]] = None, num_inference_steps: int = 50) -> None:
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        if t_index_list is not None:
            if len(t_index_list) != self.denoising_steps_num:
                raise ValueError(f"Length of provided t_index_list ({len(t_index_list)}) does not match denoising_steps_num ({self.denoising_steps_num}).")
            self.t_list = t_index_list
        if self.t_list is None:
            raise ValueError("Please provide t_index_list")
        if max(self.t_list) >= len(self.timesteps):
            raise ValueError(f"Maximum value in t_index_list is out of range. Current timesteps: {self.timesteps}")
        self.sub_timesteps = [self.timesteps[t] for t in self.t_list]
        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(sub_timesteps_tensor, repeats=self.frame_bff_size if self.use_denoising_batch else 1, dim=0)
        c_skip_list, c_out_list = [], []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        alpha_prod_t_sqrt_list = [self.scheduler.alphas_cumprod[t].sqrt() for t in self.sub_timesteps]
        beta_prod_t_sqrt_list = [(1 - self.scheduler.alphas_cumprod[t]).sqrt() for t in self.sub_timesteps]
        alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=self.frame_bff_size if self.use_denoising_batch else 1, dim=0)
        self.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=self.frame_bff_size if self.use_denoising_batch else 1, dim=0)

    @torch.no_grad()
    def update_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = None) -> None:
        if negative_prompt is None:
            encoder_output = self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        else:
            encoder_output = self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            self.negative_prompt_embeds = encoder_output[1]
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)
        if self.sdxl:
            self.add_text_embeds = encoder_output[2]
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            self.add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=encoder_output[0].dtype, text_encoder_projection_dim=text_encoder_projection_dim)
        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = self.negative_prompt_embeds.repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = self.negative_prompt_embeds.repeat(self.frame_bff_size, 1, 1)
        if self.cfg_type in ["initialize", "full"]:
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)

    @torch.no_grad()
    def update_noise(self, noise: Optional[torch.Tensor] = None) -> None:
        if noise is None:
            self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width), generator=self.generator).to(device=self.device, dtype=self.dtype)
        else:
            if noise.size() != self.init_noise.size():
                raise ValueError(f"Noise size {noise.size()} does not match expected {self.init_noise.size()}")
            self.init_noise = noise.to(device=self.device, dtype=self.dtype)
        self.stock_noise = torch.zeros_like(self.init_noise)

    @torch.no_grad()
    def update_cfg_setting(self, guidance_scale: float = 1.0, delta: float = 1.0) -> None:
        self.guidance_scale = guidance_scale
        self.delta = delta

    def init_stream_buffer(self, x_t_latent_buffer: Optional[torch.Tensor] = None) -> None:
        if self.denoising_steps_num > 1:
            B, C, H, W = (self.denoising_steps_num - 1) * self.frame_bff_size, 4, self.latent_height, self.latent_width
            if x_t_latent_buffer is None:
                self.x_t_latent_buffer = torch.zeros((B, C, H, W), device=self.device, dtype=self.dtype)
            else:
                if x_t_latent_buffer.size() != (B, C, H, W):
                    raise ValueError(f"x_t_latent_buffer size {x_t_latent_buffer.size()} does not match expected {(B, C, H, W)}")
                self.x_t_latent_buffer = x_t_latent_buffer
        else:
            self.x_t_latent_buffer = None

    def init_control_stream_buffer(self, controlnet_images_buffer: Optional[torch.Tensor] = None) -> None:
        if self.denoising_steps_num > 1:
            B, C, H, W = (self.denoising_steps_num - 1) * self.frame_bff_size, 3, self.height, self.width
            if controlnet_images_buffer is None:
                self.controlnet_images_buffer = torch.zeros((B, C, H, W), device=self.device, dtype=self.dtype)
            else:
                if controlnet_images_buffer.size() != (B, C, H, W):
                    raise ValueError(f"controlnet_images_buffer size {controlnet_images_buffer.size()} does not match expected {(B, C, H, W)}")
                self.controlnet_images_buffer = controlnet_images_buffer
        else:
            self.controlnet_images_buffer = None

    @torch.inference_mode()
    def prepare(self, prompt: str, negative_prompt: Optional[str] = None, num_inference_steps: int = 50, guidance_scale: float = 1.2, delta: float = 1.0, generator: Optional[torch.Generator] = None, seed: Optional[int] = None, t_index_list: Optional[List[int]] = None) -> None:
        if generator is not None:
            self.generator = generator
        self.init_generator(seed)
        self.init_stream_buffer()
        self.init_control_stream_buffer()
        self.update_cfg_setting(guidance_scale, delta)
        if (self.cfg_type in ["initialize", "full"]) and negative_prompt is None:
            negative_prompt = ""
        self.update_prompt(prompt, negative_prompt)
        self.update_scheduler(t_index_list, num_inference_steps)
        self.update_noise()

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, t_index: int) -> torch.Tensor:
        return self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise

    def scheduler_step_batch(self, model_pred_batch: torch.Tensor, x_t_latent_batch: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor:
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch) / self.alpha_prod_t_sqrt
            return self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            return self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch

    @torch.inference_mode()
    def unet_step(self, x_t_latent: torch.Tensor, t_list: Union[torch.Tensor, List[int]],
                added_cond_kwargs, idx: Optional[int] = None,
                controlnet_images: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg_type == "initialize":
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.cfg_type == "full":
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Assicurati che added_cond_kwargs contenga "text_embeds"
        if "text_embeds" not in added_cond_kwargs:
            if hasattr(self, "add_text_embeds") and self.add_text_embeds is not None:
                added_cond_kwargs["text_embeds"] = self.add_text_embeds.to(self.device)
            else:
                raise ValueError("Missing 'text_embeds' in added_cond_kwargs. Ensure update_prompt() sia stato chiamato correttamente.")

        if controlnet_images is not None and self.controlnet_enabled:
            cond_scale = self.controlnet_conditioning_scales if hasattr(self, "controlnet_conditioning_scales") else [1.0]
            print("QUIII")
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                controlnet_cond=controlnet_images,
                added_cond_kwargs=added_cond_kwargs,
                conditioning_scale=cond_scale,
                guess_mode=False,
                return_dict=False,
            )
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                self.prompt_embeds,  # ora è posizionale
                added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
        else:
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        # Resto del metodo invariato: applica guidance, gestione R-CFG, ecc.
        if self.cfg_type == "initialize":
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat([model_pred[0:1], self.stock_noise[1:]], dim=0)
        elif self.cfg_type == "full":
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.cfg_type in ["self", "initialize"]:
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type in ["self", "initialize"]:
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat([self.alpha_prod_t_sqrt[1:], torch.ones_like(self.alpha_prod_t_sqrt[0:1])], dim=0)
                delta_x = alpha_next * delta_x
                beta_next = torch.concat([self.beta_prod_t_sqrt[1:], torch.ones_like(self.beta_prod_t_sqrt[0:1])], dim=0)
                delta_x = delta_x / beta_next
                init_noise = torch.concat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
                self.stock_noise = init_noise + delta_x
        else:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
        return denoised_batch, model_pred
 
    @torch.inference_mode()
    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(device=self.device, dtype=self.vae.dtype)
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    @torch.inference_mode()
    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(x_0_pred_out / self.vae.config.scaling_factor, return_dict=False)[0]

    def predict_x0_batch(self, x_t_latent: torch.Tensor, controlnet_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        added_cond_kwargs = {}
        if self.sdxl:
        # Assicurati che self.add_text_embeds sia stato settato in update_prompt
            if not hasattr(self, "add_text_embeds") or self.add_text_embeds is None:
                # Fallback: usa prompt_embeds come text_embeds se non sono disponibili add_text_embeds
                added_cond_kwargs = {"text_embeds": self.prompt_embeds.to(self.device)}
            else:
                added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), 
                                    "time_ids": self.add_time_ids.to(self.device)}
        else:
                added_cond_kwargs = {}
        if controlnet_images is not None:
            prev_controlnet_images = self.controlnet_images_buffer
        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, self.x_t_latent_buffer), dim=0)
                self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)
            if self.sdxl:
                added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
            if controlnet_images is not None:
                controlnet_images = torch.cat((controlnet_images, prev_controlnet_images), dim=0)
            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs, controlnet_images=controlnet_images)
            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.CM_lora_type == "Hyper_SD":
                    self.x_t_latent_buffer = (self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[1:] * model_pred[:-1])
                elif self.CM_lora_type in ["lcm", "none"]:
                    if self.do_add_noise:
                        self.x_t_latent_buffer = (self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + self.beta_prod_t_sqrt[1:] * self.init_noise[1:])
                    else:
                        self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(1).repeat(self.frame_bff_size)
                if self.sdxl:
                    added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx, controlnet_images=controlnet_images, added_cond_kwargs=added_cond_kwargs)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.CM_lora_type == "Hyper_SD":
                        x_t_latent = (self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[idx + 1] * model_pred)
                    elif self.CM_lora_type in ["lcm", "none"]:
                        if self.do_add_noise:
                            x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred + self.beta_prod_t_sqrt[idx + 1] * torch.randn_like(x_0_pred, device=self.device, dtype=self.dtype)
                        else:
                            x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred
        return x_0_pred_out

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None):

        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        return torch.tensor([add_time_ids], dtype=dtype)

    @torch.inference_mode()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None, x_t_latent: Optional[torch.Tensor] = None, controlnet_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.device.type == "mps":
            start = self.timer_event.Event(enable_timing=True)
            end = self.timer_event.Event(enable_timing=True)
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        start.record()
        if x_t_latent is not None:
            x_t_latent = x_t_latent.to(device=self.device, dtype=self.dtype)
        else:
            if x is not None:
                if self.similar_image_filter:
                    x = self.similar_filter(x)
                    if x is None:
                        time.sleep(self.inference_time_ema)
                        return self.prev_image_result
                x_t_latent = self.encode_image(x)
            else:
                x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(device=self.device, dtype=self.dtype)
        x_0_pred_out = self.predict_x0_batch(x_t_latent, controlnet_images=controlnet_images)
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        self.prev_image_result = x_output
        end.record()
        if self.device.type == "mps":
            self.timer_event.synchronize()
        else:
            torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output