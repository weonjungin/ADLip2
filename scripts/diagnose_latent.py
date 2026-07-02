import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, "/home/jiweon/projects/ADLip2")
sys.path.insert(0, "/home/jiweon/projects/lip-sync-score/src")

from omegaconf import OmegaConf
from models.vae_wrapper import VAEWrapper
from models.generator import TemporalLipGenerator
from data.dataset import LipSyncDataset
from torch.utils.data import DataLoader

# 설정
cfg = OmegaConf.load("/home/jiweon/projects/ADLip2/configs/expC3.yaml")
device = torch.device("cuda:0")

# VAE
vae = VAEWrapper(cfg.vae.model_name).to(device)
vae.eval()

# Generator
with torch.no_grad():
    z_dummy = vae.encode(torch.zeros(1,3,96,96).to(device))
    _, C, H, W = z_dummy.shape

model = TemporalLipGenerator(
    latent_dim=C, spatial_len=H*W,
    audio_in=384, audio_len=cfg.model.audio_len,
    n_frames=16, dim=cfg.model.dim,
    num_heads=cfg.model.num_heads,
    spatial_layers=cfg.model.spatial_layers,
    temporal_layers=cfg.model.temporal_layers,
    ffn_dim_ratio=cfg.model.ffn_dim_ratio,
    dropout=cfg.model.dropout,
    alpha=cfg.model.alpha,
).to(device)

# 체크포인트 로드
ckpt = torch.load(
    "/home/jiweon/projects/ADLip2/results/batch2_alpha0.35_sync0.15_20260502_214513/best.pt",
    map_location=device, weights_only=False
)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# 데이터 몇 개만
dataset = LipSyncDataset(
    cfg.data.root, split="val",
    samples_per_video=5,
    splits_dir=cfg.data.splits_dir,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 진단
all_cos = []
all_z_out_mean = []
all_z_gt_mean = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        ref_lips_A = batch["ref_lips_A"].to(device)
        target_lips = batch["target_lips"].to(device)
        audios = batch["audios"].to(device)

        B, T, c, h, w = ref_lips_A.shape
        z_A = vae.encode(ref_lips_A.reshape(B*T, c, h, w))
        z_A = z_A.reshape(B, T, *z_A.shape[1:])
        z_gt = vae.encode(target_lips.reshape(B*T, c, h, w))
        z_gt = z_gt.reshape(B, T, *z_gt.shape[1:])

        z_A_in = z_A * 0.5
        z_out_A, delta = model(z_A_in, audios)

        # 코사인 유사도
        cos = F.cosine_similarity(
            z_out_A.reshape(B, -1),
            z_gt.reshape(B, -1)
        ).mean().item()

        all_cos.append(cos)
        all_z_out_mean.append(z_out_A.mean().item())
        all_z_gt_mean.append(z_gt.mean().item())

        print(f"batch {i}: cos={cos:.4f} "
              f"z_out mean={z_out_A.mean():.4f} std={z_out_A.std():.4f} | "
              f"z_gt mean={z_gt.mean():.4f} std={z_gt.std():.4f} | "
              f"delta norm={delta.abs().mean():.4f}")

print(f"\n평균 cosine similarity: {np.mean(all_cos):.4f}")
print(f"z_out 평균: {np.mean(all_z_out_mean):.4f}")
print(f"z_gt 평균:  {np.mean(all_z_gt_mean):.4f}")

# 프레임별 코사인 유사도 (첫 번째 배치만)
print("\n=== 프레임별 cosine similarity (batch 0) ===")
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i > 0:
            break
        ref_lips_A = batch["ref_lips_A"].to(device)
        target_lips = batch["target_lips"].to(device)
        audios = batch["audios"].to(device)

        B, T, c, h, w = ref_lips_A.shape
        z_A = vae.encode(ref_lips_A.reshape(B*T, c, h, w))
        z_A = z_A.reshape(B, T, *z_A.shape[1:])
        z_gt = vae.encode(target_lips.reshape(B*T, c, h, w))
        z_gt = z_gt.reshape(B, T, *z_gt.shape[1:])

        z_A_in = z_A * 0.5
        z_out_A, delta = model(z_A_in, audios)

        for f in range(16):
            cos_f = F.cosine_similarity(
                z_out_A[:, f].reshape(B, -1),
                z_gt[:, f].reshape(B, -1)
            ).mean().item()
            print(f"  frame {f:2d}: cos={cos_f:.4f}")