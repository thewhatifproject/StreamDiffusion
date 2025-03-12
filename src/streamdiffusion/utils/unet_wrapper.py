import torch

class WrappedUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, *args, **kwargs):
        # Assicuriamoci che 'encoder_hidden_states' venga sempre passato,
        # anche se è presente in kwargs.
        return self.unet(*args, **kwargs)