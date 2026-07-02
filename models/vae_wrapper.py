# /home/ihjung/2026/ADLip/models/vae_wrapper.py

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    def __init__(self, model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(model_name)
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)
        return mean  # [B, 4, H/8, W/8]

    def decode(self, z):
        # no_grad 제거 → gradient가 z_out까지 흘러야 함
        z = self.post_quant_conv(z)
        return self.decoder(z)