"""
save_generator_output.py

C2 best.pt로 HDTF 데이터 전체를 forward해서 z_out.npy 저장

사용법:
    CUDA_VISIBLE_DEVICES=0 python scripts/save_generator_output.py \
        --ckpt results/C2_20260508_174715/best.pt \
        --cfg configs/expC2.yaml \
        --out_root /media/HDD/jiweon/hdtf_zout \
        --device cuda:0
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/jiweon/projects/ADLip2")
from omegaconf import OmegaConf
from models.vae_wrapper import VAEWrapper
from models.generator import TemporalLipGenerator

N_FRAMES = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="results/C2_20260508_174715/best.pt")
    parser.add_argument("--cfg",      default="configs/expC2.yaml")
    parser.add_argument("--out_root", default="/media/HDD/jiweon/hdtf_zout")
    parser.add_argument("--data_root",default="/home/ihjung/HDTF_ssd/processed")
    parser.add_argument("--latent_root", default="/media/HDD/jiweon/hdtf_latents")
    parser.add_argument("--device",   default="cuda:0")
    parser.add_argument("--splits_dir", default="/home/jiweon/projects/ADLip2/data/splits")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg    = OmegaConf.load(args.cfg)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # VAE
    vae = VAEWrapper(cfg.vae.model_name).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        z_dummy = vae.encode(torch.zeros(1, 3, 96, 96).to(device))
        _, C, H, W = z_dummy.shape
        spatial_len = H * W
    print(f"Latent shape: {z_dummy.shape}", flush=True)

    # Generator
    model = TemporalLipGenerator(
        latent_dim      = C,
        spatial_len     = spatial_len,
        audio_in        = 384,
        audio_len       = cfg.model.audio_len,
        n_frames        = N_FRAMES,
        dim             = cfg.model.dim,
        num_heads       = cfg.model.num_heads,
        spatial_layers  = cfg.model.spatial_layers,
        temporal_layers = cfg.model.temporal_layers,
        ffn_dim_ratio   = cfg.model.ffn_dim_ratio,
        dropout         = cfg.model.dropout,
        alpha           = cfg.model.alpha,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f"Loaded: {args.ckpt}", flush=True)

    # 비디오 목록 (train + val)
    video_names = []
    for split in ["train", "val"]:
        split_file = Path(args.splits_dir) / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                video_names += [l.strip() for l in f if l.strip()]

    # 유효 비디오 필터링
    valid = []
    for name in video_names:
        lp = Path(args.latent_root) / name / "latent.npy"
        ap = Path(args.data_root)   / name / "audio_feature.npy"
        mp = Path(args.data_root)   / name / "mel_latentsync.npy"
        if lp.exists() and ap.exists() and mp.exists():
            valid.append(name)

    print(f"Total videos: {len(valid)}", flush=True)

    skip = 0
    done = 0

    for name in tqdm(valid, desc="Saving z_out"):
        out_path = out_root / name / "z_out.npy"

        if out_path.exists():
            skip += 1
            continue

        (out_root / name).mkdir(parents=True, exist_ok=True)

        try:
            # latent 로드
            latent = np.load(
                str(Path(args.latent_root) / name / "latent.npy"),
                mmap_mode='r'
            )  # [T, 4, 12, 12]
            audio = np.load(
                str(Path(args.data_root) / name / "audio_feature.npy"),
                mmap_mode='r'
            )  # [T, 384]

            T = latent.shape[0]
            if T < N_FRAMES:
                skip += 1
                continue

            all_zout = []

            with torch.no_grad():
                # 16프레임 슬라이딩 윈도우
                for t in range(0, T - N_FRAMES + 1, N_FRAMES):
                    # lip latent
                    z = torch.tensor(
                        np.array(latent[t:t + N_FRAMES]),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)  # [1, 16, 4, 12, 12]

                    # audio window [1, 16, 5, 384]
                    windows = []
                    for i in range(N_FRAMES):
                        idx = min(t + i, T - 1)
                        idxs = np.arange(idx - 2, idx + 3)
                        idxs = np.clip(idxs, 0, T - 1)
                        windows.append(audio[idxs])
                    aud = torch.tensor(
                        np.stack(windows),
                        dtype=torch.float32
                    ).unsqueeze(0).to(device)  # [1, 16, 5, 384]

                    z_in = z * 0.5
                    z_out, _ = model(z_in, aud)  # [1, 16, 4, 12, 12]
                    all_zout.append(z_out.squeeze(0).cpu().numpy())

            if len(all_zout) == 0:
                skip += 1
                continue

            # [N_windows, 16, 4, 12, 12] → [T', 4, 12, 12]
            z_out_all = np.concatenate(all_zout, axis=0)
            np.save(str(out_path), z_out_all)
            done += 1

        except Exception as e:
            print(f"ERROR {name}: {e}", flush=True)
            skip += 1

    print(f"\nDone: {done}  Skip: {skip}", flush=True)
    print(f"Saved to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
