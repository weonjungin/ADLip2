"""
models/generator.py

구조:
    z_ref  [B, 16, 4, 12, 12]
    audio  [B, 16, 5, 384]

    1. Spatial CrossAttention (프레임별 독립)
       Q = z spatial tokens [B*16, 144, dim]
       K = V = audio tokens [B*16, 5, dim]

    2. Temporal SelfAttention (spatial position별)
       Q = K = V = [B*144, 16, dim]

    3. FFN

    4. delta = z - z0
       z_out = z_ref + alpha * delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_dim_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_dim_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_dim_ratio, dim),
        )

    def forward(self, z, audio):
        z_norm = self.norm1(z)
        attn_out, _ = self.attn(query=z_norm, key=audio, value=audio)
        z = z + attn_out
        z = z + self.ffn(self.norm2(z))
        return z


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_dim_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_dim_ratio),
            nn.GELU(),
            nn.Linear(dim * ffn_dim_ratio, dim),
        )

    def forward(self, z):
        B, T, S, D = z.shape
        z_t = z.permute(0, 2, 1, 3).reshape(B*S, T, D)
        z_norm = self.norm1(z_t)
        attn_out, _ = self.attn(query=z_norm, key=z_norm, value=z_norm)
        z_t = z_t + attn_out
        z_t = z_t + self.ffn(self.norm2(z_t))
        z = z_t.reshape(B, S, T, D).permute(0, 2, 1, 3)
        return z


class TemporalLipGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=4,
        spatial_len=144,
        audio_in=384,
        audio_len=5,
        n_frames=16,
        dim=512,
        num_heads=8,
        spatial_layers=2,
        temporal_layers=2,
        ffn_dim_ratio=4,
        dropout=0.1,
        alpha=0.3,
    ):
        super().__init__()
        self.n_frames    = n_frames
        self.spatial_len = spatial_len
        self.alpha       = alpha

        self.z_proj    = nn.Linear(latent_dim, dim)
        self.z_unproj  = nn.Linear(dim, latent_dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_in, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.spatial_pos  = nn.Parameter(torch.randn(1, 1, spatial_len, dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, dim) * 0.02)
        self.audio_pos    = nn.Parameter(torch.randn(1, 1, audio_len, dim) * 0.02)

        self.spatial_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, ffn_dim_ratio, dropout)
            for _ in range(spatial_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalSelfAttentionBlock(dim, num_heads, ffn_dim_ratio, dropout)
            for _ in range(temporal_layers)
        ])

    def forward(self, z_ref, audio):
        B, T, C, H, W = z_ref.shape
        S = H * W

        z_flat = z_ref.reshape(B*T, C, H, W).permute(0,2,3,1).reshape(B*T, S, C)
        z0 = self.z_proj(z_flat).reshape(B, T, S, -1)
        z0 = z0 + self.spatial_pos + self.temporal_pos

        a = self.audio_proj(audio) + self.audio_pos

        z = z0.reshape(B*T, S, -1)
        a_flat = a.reshape(B*T, audio.shape[2], -1)
        for layer in self.spatial_layers:
            z = layer(z, a_flat)
        z = z.reshape(B, T, S, -1)

        for layer in self.temporal_layers:
            z = layer(z)

        delta_tokens = z - z0
        delta_flat   = self.z_unproj(delta_tokens.reshape(B*T, S, -1))
        delta = delta_flat.reshape(B, T, S, C)
        delta = delta.reshape(B, T, H, W, C).permute(0,1,4,2,3)

        z_out = z_ref + self.alpha * delta
        return z_out, delta
