import torch
import torch.nn.functional as F
import numpy as np
import sys
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0, "/home/jiweon/projects/ADLip2")
from omegaconf import OmegaConf
from models.vae_wrapper import VAEWrapper

device = torch.device("cuda:0")
cfg = OmegaConf.load("/home/jiweon/projects/ADLip2/configs/expC3.yaml")

vae = VAEWrapper(cfg.vae.model_name).to(device)
vae.eval()

# face_frames에서 샘플 몇 장 로드
import os
from glob import glob

data_root = "/media/HDD/jiweon/processed/HDTF/HDTF_processed/WDA_HillaryClinton_000/face_frames"
frames = sorted(glob(f"{data_root}/*.png"))[:50]

transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in frames]).to(device)

with torch.no_grad():
    z = vae.encode(imgs)
    recon = vae.decode(z)

# 픽셀 복원 품질
mse = F.mse_loss(recon, imgs).item()
psnr = 10 * np.log10(4.0 / mse)  # 픽셀 범위 [-1,1] → max=2, max^2=4

# lip 영역만 (하단 절반) MSE
lip_recon = recon[:, :, 48:, :]
lip_orig  = imgs[:, :, 48:, :]
lip_mse = F.mse_loss(lip_recon, lip_orig).item()
lip_psnr = 10 * np.log10(4.0 / lip_mse)

print(f"[전체] MSE={mse:.6f}  PSNR={psnr:.2f} dB")
print(f"[lip]  MSE={lip_mse:.6f}  PSNR={lip_psnr:.2f} dB")
print(f"z shape: {z.shape}  mean={z.mean():.4f}  std={z.std():.4f}")
