import torch
from diffusers import ControlNetModel, UNet2DConditionModel

class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, controlnets: list[ControlNetModel], controlnet_scales: list[float]):
        super().__init__()
        self.unet = unet.to(unet.device, dtype=unet.dtype)
        self.controlnets = [controlnet.to(unet.device, dtype=unet.dtype) for controlnet in controlnets]
        self.controlnet_scales = controlnet_scales

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Estrai i parametri dalla lista degli argomenti o, se non presenti, da kwargs.
        sample = args[0]
        timestep = args[1]
        # Usa pop per rimuovere 'encoder_hidden_states' da kwargs se presente
        if len(args) > 2:
            encoder_hidden_states = args[2]
        else:
            encoder_hidden_states = kwargs.pop("encoder_hidden_states", None)
        # Stessa cosa per 'controlnet_images'
        if len(args) > 3:
            controlnet_images = args[3]
        else:
            controlnet_images = kwargs.pop("controlnet_images", None)

        # Passa esplicitamente gli argomenti e **kwargs senza duplicati.
        for i in range(len(self.controlnets)):
            down_samples, mid_sample = self.controlnets[i](
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_images[i],
                guess_mode=False,
                return_dict=False,
                **kwargs
            )
            down_samples = [down_sample * self.controlnet_scales[i] for down_sample in down_samples]
            mid_sample *= self.controlnet_scales[i]
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    prev + curr for prev, curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
            **kwargs
        )
        return noise_pred