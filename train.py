"""
train.py - 16프레임 temporal 구조

Stage 1:
    latent + same + diff + silence + dreg + smooth

Stage 2:
    + sync loss

Stage 3:
    + our loss
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import argparse
import sys
import logging

sys.path.insert(0, "/home/jiweon/projects/ADLip2")
sys.path.insert(0, "/home/jiweon/projects/lip-sync-score/src")

from omegaconf import OmegaConf
from models.vae_wrapper import VAEWrapper
from torchvision.models import vgg16
from models.generator import TemporalLipGenerator
from models.latentsync import SyncNet as LatentSyncNet
from models.discriminator import MouthDiscriminator
from data.dataset import LipSyncDataset

from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
from lipsyncscore.loss.contrastive import SyncNetContrastiveLoss

FACE_SIZE = 256
N_FRAMES  = 16
CENTER    = N_FRAMES // 2  # 8


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ──────────────────────────────────────────
# SyncNet
# ──────────────────────────────────────────
SYNCNET_CONFIG = {
    "audio_encoder": {
        "in_channels": 1,
        "block_out_channels": [32,64,128,256,512,1024,2048],
        "downsample_factors": [[2,1],2,2,1,2,2,[2,3]],
        "attn_blocks": [0,0,0,0,0,0,0],
        "dropout": 0.0,
    },
    "visual_encoder": {
        "in_channels": 48,
        "block_out_channels": [64,128,256,256,512,1024,2048,2048],
        "downsample_factors": [[1,2],2,2,2,2,2,2,2],
        "attn_blocks": [0,0,0,0,0,0,0,0],
        "dropout": 0.0,
    },
}

def load_latentsync(ckpt_path, device):
    model = LatentSyncNet(SYNCNET_CONFIG).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"LatentSync loaded & frozen: {ckpt_path}", flush=True)
    return model

def load_our_syncnet(cfg, device):
    model = SyncNetTemporal(
        in_frames=cfg.our.get("in_frames", 16),
        emb_dim=cfg.our.get("emb_dim", 256),
        pooling=cfg.our.get("pooling", "mean"),
        lip_in_channels = cfg.our.get("lip_in_channels", 4),
    ).to(device)

    ckpt = torch.load(cfg.our.ckpt_path, map_location=device, weights_only=False)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.train()

    for p in model.parameters():
        p.requires_grad = True

    print(f"Our SyncNetTemporal loaded & trainable: {cfg.our.ckpt_path}", flush=True)
    return model


# ──────────────────────────────────────────
# paste pred_lips → face_window
# ──────────────────────────────────────────
def paste_lips_to_face(face_window, pred_lips, lip_bbox):
    """
    face_window: [B, 48, 128, 256]  # 하단 절반
    pred_lips:   [B, 16, 3, 96, 96]
    lip_bbox:    [B, 16, 6] = [valid, frame_idx, x1, y1, x2, y2] (원본 256x256 기준)
    """
    B = face_window.shape[0]
    result = face_window.clone()

    for b in range(B):
        for f in range(N_FRAMES):
            valid, fidx, x1, y1, x2, y2 = lip_bbox[b, f].tolist()
            if valid != 1:
                continue
            # 원본 256×256 → 하단절반 128×256 좌표 변환
            y1h = max(0,   min(127, int(y1) - 128))
            y2h = max(1,   min(128, int(y2) - 128))
            x1i = max(0,   min(255, int(x1)))
            x2i = max(1,   min(256, int(x2)))
            if y2h <= y1h or x2i <= x1i:
                continue
            ph = y2h - y1h
            pw = x2i - x1i
            lip_resized = F.interpolate(
                pred_lips[b, f:f+1],
                size=(ph, pw),
                mode='bilinear',
                align_corners=False,
            )[0]
            ch_start = f * 3
            result[b, ch_start:ch_start+3, y1h:y2h, x1i:x2i] = lip_resized

    return result


def crop_mouth(x):
    if x.dim() == 5:
        return x[:, :, :, 48:96, 16:80]
    elif x.dim() == 4:
        return x[:, :, 48:96, 16:80]
    else:
        raise ValueError(f"Unexpected shape: {x.shape}")

def sobel_edge(x):
    """x: [B, 3, H, W] → edge map [B, 1, H, W]"""
    x = x.float()  # fp16 → fp32 강제
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=torch.float32, device=x.device).reshape(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                       dtype=torch.float32, device=x.device).reshape(1,1,3,3)
    ex = F.conv2d(gray, kx, padding=1)
    ey = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(ex**2 + ey**2 + 1e-6)

def sync_loss_contrastive(face_embed, audio_embed, margin=0.2):
    pos  = F.cosine_similarity(face_embed, audio_embed, dim=1)
    perm = torch.roll(torch.arange(face_embed.shape[0], device=face_embed.device), 1)
    neg  = F.cosine_similarity(face_embed, audio_embed[perm], dim=1)
    loss = F.relu(margin - pos + neg).mean()
    return loss, pos.mean().item(), neg.mean().item()


# ──────────────────────────────────────────
# 샘플 저장
# ──────────────────────────────────────────
def save_samples(model, vae, cfg, device, samples_dir, epoch, writer=None):
    """
    center frame 비교 이미지 + sequence 이미지 저장
    GT/Ref/Pred 모두 CENTER frame으로 통일
    """
    def load_lip_window(frame_dir, start, n_total):
        imgs = []
        for i in range(N_FRAMES):
            idx = min(start + i, n_total - 1)
            path = frame_dir / f"{idx:05d}.png"
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
            t = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device) / 127.5 - 1.0
            imgs.append(t)
        return torch.cat(imgs, dim=0).unsqueeze(0)  # [1, 16, 3, 96, 96]

    def get_audio_windows(af, start, n):
        windows = []
        for i in range(N_FRAMES):
            idx = min(start + i, n - 1)
            idxs = np.arange(idx - 2, idx + 3)
            idxs = np.clip(idxs, 0, n - 1)
            windows.append(af[idxs])
        return torch.tensor(np.stack(windows), dtype=torch.float32).unsqueeze(0).to(device)

    def to_np(t):
        return ((t.clamp(-1,1)+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

    def put_label(img, text):
        img = img.copy()
        cv2.putText(img, text, (3,12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        return img

    samples = [
        ("WDA_HillaryClinton_000", 68,  648, 826, "Clinton"),
        ("RD_Radio5_000",          660, 6,   608, "Radio5"),
        ("RD_Radio27_000",         1,   13,  131, "Radio27"),
    ]

    center_rows = []
    seq_rows    = []

    with torch.no_grad():
        for video_name, t, j_A, j_B, label in samples:
            base = Path(cfg.data.root) / video_name
            af   = np.load(str(base / "audio_feature.npy"))
            n    = len(af)
            fd   = base / "frames"

            target_lips = load_lip_window(fd, t,   n)  # [1,16,3,96,96]
            ref_lips_A  = load_lip_window(fd, j_A, n)
            ref_lips_B  = load_lip_window(fd, j_B, n)
            audios      = get_audio_windows(af, t, n)   # [1,16,5,384]
            zero_audio  = torch.zeros_like(audios)

            # neg audio: center frame 기준 100프레임 뒤
            neg_start = min(t + 100, n - N_FRAMES)
            audio_neg = get_audio_windows(af, neg_start, n)

            z_A  = vae.encode(ref_lips_A.reshape(-1, 3, 96, 96))
            _, _, H, W = z_A.shape
            z_A  = z_A.reshape(1, N_FRAMES, -1, H, W)
            z_B  = vae.encode(ref_lips_B.reshape(-1, 3, 96, 96)).reshape(1, N_FRAMES, -1, H, W)

            z_out_A,    _ = model(z_A, audios)
            z_out_B,    _ = model(z_B, audios)
            z_out_zero, _ = model(z_A, zero_audio)
            z_out_neg,  _ = model(z_A, audio_neg)

            def decode_seq(z):
                imgs = vae.decode(z.reshape(-1, z.shape[2], H, W))
                return imgs.reshape(1, N_FRAMES, 3, 96, 96)

            pred_A    = decode_seq(z_out_A)
            pred_B    = decode_seq(z_out_B)
            pred_zero = decode_seq(z_out_zero)
            pred_neg  = decode_seq(z_out_neg)

            # GT/Ref/Pred 모두 CENTER frame으로 통일
            c = CENTER
            gt_c  = to_np(target_lips[0, c])
            rA_c  = to_np(ref_lips_A[0, c])
            rB_c  = to_np(ref_lips_B[0, c])
            pA_c  = to_np(pred_A[0, c])
            pB_c  = to_np(pred_B[0, c])
            pZ_c  = to_np(pred_zero[0, c])
            pN_c  = to_np(pred_neg[0, c])

            center_row = np.concatenate([
                put_label(gt_c,  f"GT(f{t+c})"),
                put_label(rA_c,  f"RefA(f{j_A+c})"),
                put_label(rB_c,  f"RefB(f{j_B+c})"),
                put_label(pA_c,  "PredA"),
                put_label(pB_c,  "PredB"),
                put_label(pZ_c,  "PredZero"),
                put_label(pN_c,  "PredNeg"),
            ], axis=1)
            center_rows.append(center_row)

            # sequence: 각 샘플별 7행 block
            def label_first(imgs, text):
                imgs = [x.copy() for x in imgs]
                cv2.putText(imgs[0], text, (3,12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                return imgs

            gt_seq    = [to_np(target_lips[0, f]) for f in range(N_FRAMES)]
            refA_seq  = [to_np(ref_lips_A[0, f])  for f in range(N_FRAMES)]
            refB_seq  = [to_np(ref_lips_B[0, f])  for f in range(N_FRAMES)]
            predA_seq = [to_np(pred_A[0, f])      for f in range(N_FRAMES)]
            predB_seq = [to_np(pred_B[0, f])      for f in range(N_FRAMES)]
            zero_seq  = [to_np(pred_zero[0, f])   for f in range(N_FRAMES)]
            neg_seq   = [to_np(pred_neg[0, f])    for f in range(N_FRAMES)]

            block = np.concatenate([
                np.concatenate(label_first(gt_seq,    f"GT   f{t}~{t+15}"),    axis=1),
                np.concatenate(label_first(refA_seq,  f"RefA f{j_A}~{j_A+15}"), axis=1),
                np.concatenate(label_first(refB_seq,  f"RefB f{j_B}~{j_B+15}"), axis=1),
                np.concatenate(label_first(predA_seq, "PredA (RefA+GT audio)"),  axis=1),
                np.concatenate(label_first(predB_seq, "PredB (RefB+GT audio)"),  axis=1),
                np.concatenate(label_first(zero_seq,  "PredZero (RefA+zero)"),   axis=1),
                np.concatenate(label_first(neg_seq,   "PredNeg (RefA+neg)"),     axis=1),
            ], axis=0)
            seq_rows.append(block)

    center_combined = cv2.cvtColor(np.concatenate(center_rows, axis=0), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(samples_dir / f"center_epoch_{epoch:04d}.png"), center_combined)

    sample_names = ["Clinton", "Radio5", "Radio27"]
    for block, name in zip(seq_rows, sample_names):
        img_bgr = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(samples_dir / f"seq_{name}_epoch_{epoch:04d}.png"), img_bgr)

    print(f"  samples saved: epoch {epoch}", flush=True)

    if writer is not None:
        img_t = torch.tensor(cv2.cvtColor(center_combined, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255.0
        writer.add_image("sample/center", img_t, epoch)
        for block, name in zip(seq_rows, sample_names):
            img_t = torch.tensor(block).permute(2,0,1).float() / 255.0
            writer.add_image(f"sample/seq_{name}", img_t, epoch)


# ──────────────────────────────────────────
# 학습
# ──────────────────────────────────────────
def train(cfg_path, device_str, ckpt_path=None):
    cfg    = OmegaConf.load(cfg_path)
    device = torch.device(device_str)
    set_seed(cfg.train.seed)

    run_name    = cfg.get("run_name", "")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{run_name}_{timestamp}" if run_name else timestamp
    save_dir    = Path(cfg.results.root) / folder_name
    samples_dir = save_dir / "samples"
    tb_dir      = save_dir / "tb"
    save_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(exist_ok=True)
    tb_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=str(save_dir / "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger()

    # VAE
    vae = VAEWrapper(cfg.vae.model_name).to(device)
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    # VGG (perceptual loss)
    vgg = vgg16(pretrained=True).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    lambda_perceptual = cfg.loss.get("lambda_perceptual", 0.0)

    with torch.no_grad():
        z_dummy = vae.encode(torch.zeros(1,3,96,96).to(device))
        _, C, H, W = z_dummy.shape
        spatial_len = H * W
    print(f"latent: {z_dummy.shape}", flush=True)

    # Generator
    model = TemporalLipGenerator(
        latent_dim     = C,
        spatial_len    = spatial_len,
        audio_in       = 384,
        audio_len      = cfg.model.audio_len,
        n_frames       = N_FRAMES,
        dim            = cfg.model.dim,
        num_heads      = cfg.model.num_heads,
        spatial_layers = cfg.model.spatial_layers,
        temporal_layers= cfg.model.temporal_layers,
        ffn_dim_ratio  = cfg.model.ffn_dim_ratio,
        dropout        = cfg.model.dropout,
        alpha          = cfg.model.alpha,
    ).to(device)

    # Stage 2: load Stage 1 checkpoint
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint: {ckpt_path}", flush=True)
        log.info(f"Loaded checkpoint: {ckpt_path}")

    # loss params
    lambda_latent     = cfg.loss.lambda_latent
    lambda_same       = cfg.loss.get("lambda_same",       0.1)
    lambda_diff       = cfg.loss.get("lambda_diff",       0.05)
    diff_margin       = cfg.loss.get("diff_margin",       0.3)
    lambda_silence    = cfg.loss.get("lambda_silence",    0.05)
    lambda_dreg       = cfg.loss.get("lambda_delta_reg",  0.01)
    lambda_smooth     = cfg.loss.get("lambda_smooth",     0.05)
    lambda_sync       = cfg.loss.get("lambda_sync",       0.0)
    sync_margin       = cfg.loss.get("sync_margin",       0.2)
    lambda_mouth      = cfg.loss.get("lambda_mouth",      0.0)
    lambda_mouth_edge = cfg.loss.get("lambda_mouth_edge", 0.0)

    # our TTA
    lambda_our = cfg.loss.get("lambda_our", 0.0)
    our_margin = cfg.loss.get("our_margin", 1.0)
    our_lambda_pos = cfg.loss.get("our_lambda_pos", 1.0)
    our_lambda_neg = cfg.loss.get("our_lambda_neg", 1.0)

    # SyncNet (Stage 2)
    syncnet = None
    if lambda_sync > 0:
        syncnet = load_latentsync(cfg.syncnet.ckpt_path, device)

    our_syncnet = None
    our_criterion = None

    if lambda_our > 0:
        our_syncnet = load_our_syncnet(cfg, device)
        our_criterion = SyncNetContrastiveLoss(
            margin=our_margin,
            lambda_pos=our_lambda_pos,
            lambda_neg=our_lambda_neg,
        )

    if our_syncnet is not None:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": cfg.train.lr},
                {
                    "params": our_syncnet.parameters(),
                    "lr": cfg.train.lr * cfg.train.get("our_lr_mult", 5.0),
                },
            ],
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=1e-4
        )
    scaler = GradScaler()
    l1_loss   = nn.L1Loss()

    lambda_adv = cfg.loss.get("lambda_adv", 0.0)
    disc, opt_D, scaler_D = None, None, None
    if lambda_adv > 0:
        disc     = MouthDiscriminator(ndf=cfg.model.get("ndf", 32), in_ch=5).to(device)
        opt_D = torch.optim.Adam(disc.parameters(), lr=cfg.train.get("lr_d", 1e-5), betas=(0.5, 0.999))        
        scaler_D = GradScaler()

    def decode_seq_train(z_out, chunk_size=2):
        b, t, c, h, w = z_out.shape
        z_flat = z_out.reshape(b*t, c, h, w)

        imgs = []
        for i in range(0, z_flat.shape[0], chunk_size):
            imgs.append(vae.decode(z_flat[i:i+chunk_size]))

        imgs = torch.cat(imgs, dim=0)
        return imgs.reshape(b, t, *imgs.shape[1:])
    
    def encode_seq_train(lips):
        b, t, c, h, w = lips.shape
        z = vae.encode(lips.reshape(b*t, c, h, w))
        return z.reshape(b, t, *z.shape[1:])

    dataset = LipSyncDataset(
        cfg.data.root, split="train",
        samples_per_video=cfg.data.samples_per_video,
        splits_dir=cfg.data.get("splits_dir", None),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.train.num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
    )

    val_dataset = LipSyncDataset(
        cfg.data.root, split="val",
        samples_per_video=cfg.data.samples_per_video,
        splits_dir=cfg.data.get("splits_dir", None),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    writer      = SummaryWriter(log_dir=str(tb_dir))
    best_loss   = float("inf")
    global_step = 0
    our_scale = 1.0  # adaptive scaling factor
    our_scale_update_freq = 100  # 100 step마다 업데이트 # OOM나면 200으로 늘리기

    model.eval()
    save_samples(model, vae, cfg, device, samples_dir, epoch=0, writer=writer)
    model.train()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        ep = {k: [] for k in [
            "total","latent","same","diff","silence",
            "dreg","smooth","mouth","mouth_edge","perceptual","sync","pos_sim","neg_sim","dn",
            "our", "our_d_pos", "our_d_neg"
        ]}

        for batch in loader:
            target_lips = batch["target_lips"].to(device)  # [B,16,3,96,96]
            ref_lips_A  = batch["ref_lips_A"].to(device)
            ref_lips_B  = batch["ref_lips_B"].to(device)
            audios      = batch["audios"].to(device)        # [B,16,5,384]
            mel         = batch["mel"].to(device)
            face_window = batch["face_window"].to(device)
            lip_bbox    = batch["lip_bbox"].to(device)

            B_cur      = audios.shape[0]
            zero_audio = torch.zeros_like(audios)
            perm       = torch.roll(torch.arange(B_cur, device=device), 1)
            audio_neg  = audios[perm]

            # VAE encode (배치×16 묶어서)
            with torch.no_grad():
                z_A  = encode_seq_train(ref_lips_A)
                z_B  = encode_seq_train(ref_lips_B)
                z_gt = encode_seq_train(target_lips)

            # ref scaling: shortcut 방지
            z_A_in = z_A * 0.5
            z_B_in = z_B * 0.5

            with autocast():
                # forward
                z_out_A,    delta_A    = model(z_A_in, audios)
                z_out_B,    delta_B    = model(z_B_in, audios)
                z_out_zero, delta_zero = model(z_A_in, zero_audio)
                z_out_neg,  delta_neg  = model(z_A_in, audio_neg)

                # losses
                loss_latent = (l1_loss(z_out_A, z_gt) + l1_loss(z_out_B, z_gt)) / 2
                loss_same   = l1_loss(z_out_A, z_out_B)
                diff_dist   = l1_loss(delta_A, delta_neg.detach())
                loss_diff   = torch.relu(diff_margin - diff_dist)
                loss_silence= delta_zero.abs().mean()
                loss_dreg   = (delta_A.abs().mean() + delta_B.abs().mean()) / 2
                loss_smooth = (
                    l1_loss(delta_A[:, 1:], delta_A[:, :-1]) +
                    l1_loss(delta_B[:, 1:], delta_B[:, :-1])
                ) / 2

                need_pixel = (lambda_mouth > 0) or (lambda_sync > 0) or (lambda_perceptual > 0) or (lambda_mouth_edge > 0)
                pred_A_imgs = decode_seq_train(z_out_A) if need_pixel else None
                pred_B_imgs = decode_seq_train(z_out_B) if lambda_mouth > 0 else None

                # mouth loss
                loss_mouth_val = torch.tensor(0., device=device)
                if lambda_mouth > 0:
                    loss_mouth_val = (
                        l1_loss(crop_mouth(pred_A_imgs), crop_mouth(target_lips)) +
                        l1_loss(crop_mouth(pred_B_imgs), crop_mouth(target_lips))
                    ) / 2

                # mouth edge loss
                loss_mouth_edge_val = torch.tensor(0., device=device)
                if lambda_mouth_edge > 0 and pred_A_imgs is not None:
                    pred_mouth = crop_mouth(pred_A_imgs).reshape(-1, 3, 48, 64)
                    gt_mouth   = crop_mouth(target_lips).reshape(-1, 3, 48, 64)
                    pred_mouth = (pred_mouth + 1) / 2
                    gt_mouth   = (gt_mouth   + 1) / 2
                    loss_mouth_edge_val = F.l1_loss(
                        sobel_edge(pred_mouth),
                        sobel_edge(gt_mouth).detach()
                    )

                # perceptual loss
                loss_perceptual_val = torch.tensor(0., device=device)
                if lambda_perceptual > 0 and pred_A_imgs is not None:
                    pred_flat     = (pred_A_imgs.reshape(-1, 3, 96, 96) + 1) / 2
                    target_flat_p = (target_lips.reshape(-1, 3, 96, 96) + 1) / 2
                    loss_perceptual_val = F.l1_loss(vgg(pred_flat), vgg(target_flat_p))

                # sync loss
                loss_sync_val = torch.tensor(0., device=device)
                pos_sim = neg_sim = 0.0

                if lambda_sync > 0 and syncnet is not None:
                    face_with_pred = paste_lips_to_face(face_window, pred_A_imgs, lip_bbox)
                    face_embed  = syncnet.get_image_embed(face_with_pred)
                    audio_embed = syncnet.get_audio_embed(mel)
                    loss_sync_val, pos_sim, neg_sim = sync_loss_contrastive(
                        face_embed, audio_embed, margin=sync_margin
                    )

                # our loss
                loss_our_val = torch.tensor(0., device=device)
                our_d_pos = 0.0
                our_d_neg = 0.0

                if lambda_our > 0 and our_syncnet is not None:
                    # z_out_A std를 z_gt 분포에 맞게 rescaling
                    z_out_scaled = z_out_A * (z_gt.std() / z_out_A.std().clamp(min=1e-6))
                    
                    mel_pos = mel.squeeze(1)
                    mel_neg = mel[perm].squeeze(1)

                    if global_step == 0:
                        print(
                            "[DEBUG our input]",
                            "z_out_A=", z_out_A.shape,
                            "z_out_scaled std=", z_out_scaled.std().item(),
                            "z_gt std=", z_gt.std().item(),
                            "mel_pos=", mel_pos.shape,
                            flush=True,
                        )

                    v_embed = our_syncnet.forward_lip(z_out_scaled)
                    a_pos   = our_syncnet.forward_audio(mel_pos)
                    a_neg   = our_syncnet.forward_audio(mel_neg)

                    loss_our_val, our_info = our_criterion(v_embed, a_pos, a_neg)
                    our_d_pos = our_info["d_pos"]
                    our_d_neg = our_info["d_neg"]

                    # adaptive scaling: 100 step마다 측정
                    if global_step % our_scale_update_freq == 0 and lambda_sync > 0 and loss_sync_val.item() > 0:
                        sync_grads = torch.autograd.grad(
                            lambda_sync * loss_sync_val,
                            list(model.parameters()),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        sync_gnorm = sum(
                            g.norm()**2 for g in sync_grads if g is not None
                        ) ** 0.5

                        our_grads = torch.autograd.grad(
                            lambda_our * loss_our_val,
                            list(model.parameters()),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        our_gnorm = sum(
                            g.norm()**2 for g in our_grads if g is not None
                        ) ** 0.5

                        if our_gnorm > 1e-8:
                            new_scale = (sync_gnorm / our_gnorm).clamp(0.1, 5.0).item()
                            our_scale = 0.9 * our_scale + 0.1 * new_scale

                        writer.add_scalar("iter/our_scale", our_scale, global_step)
                        writer.add_scalar("iter/sync_gnorm", sync_gnorm, global_step)
                        writer.add_scalar("iter/our_gnorm", our_gnorm, global_step)
                        print(f"  [adaptive] sync_gnorm={sync_gnorm:.4f} our_gnorm={our_gnorm:.4f} scale={our_scale:.4f}", flush=True)

                loss = (lambda_latent   * loss_latent
                    + lambda_same       * loss_same
                    + lambda_diff       * loss_diff
                    + lambda_silence    * loss_silence
                    + lambda_dreg       * loss_dreg
                    + lambda_smooth     * loss_smooth
                    + lambda_mouth      * loss_mouth_val
                    + lambda_mouth_edge * loss_mouth_edge_val
                    + lambda_perceptual * loss_perceptual_val
                    + lambda_sync       * loss_sync_val
                    + lambda_our        * loss_our_val * our_scale)

            optimizer.zero_grad()

            # ── Discriminator update ──────────────────────────
            if lambda_adv > 0 and disc is not None:
                BT = B_cur * N_FRAMES

                # mel을 latent spatial size (H, W)=(12,12)에 맞게 resize
                mel_exp  = mel.unsqueeze(1).expand(-1, N_FRAMES, -1, -1, -1)
                mel_flat = mel_exp.reshape(BT, 1, mel.shape[2], mel.shape[3])
                mel_hw   = F.interpolate(mel_flat, size=(H, W), mode='bilinear', align_corners=False)

                # decode 없이 latent 직접 사용
                z_gt_flat   = z_gt.reshape(BT, C, H, W).detach()
                z_pred_flat = z_out_A.reshape(BT, C, H, W).detach()

                d_in_real = torch.cat([z_gt_flat,   mel_hw.detach()], dim=1)  # [BT, 5, 12, 12]
                d_in_fake = torch.cat([z_pred_flat, mel_hw.detach()], dim=1)

                opt_D.zero_grad()
                with autocast():
                    d_real_out, _ = disc.get_features(d_in_real)
                    d_fake_out, _ = disc.get_features(d_in_fake)
                    loss_d = ((d_real_out - 0.9)**2).mean() * 0.5 + (d_fake_out**2).mean() * 0.5
                scaler_D.scale(loss_d).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

                # G adv (decode 완전히 제거)
                z_pred_grad = z_out_A.reshape(BT, C, H, W)
                d_in_fake_g = torch.cat([z_pred_grad, mel_hw.detach()], dim=1)

                disc.requires_grad_(False)
                with autocast():
                    d_fake_out_g, feat_fake = disc.get_features(d_in_fake_g)
                    _, feat_real            = disc.get_features(d_in_real.detach())
                    loss_adv = ((d_fake_out_g - 1)**2).mean()
                    loss_fm  = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(feat_fake, feat_real))
                    loss = loss + lambda_adv * loss_adv + cfg.loss.get("lambda_fm", 0.0) * loss_fm
                disc.requires_grad_(True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            gen_grad_before = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5

            our_grad_before = 0.0
            if our_syncnet is not None:
                our_grad_before = sum(
                    p.grad.norm().item() ** 2
                    for p in our_syncnet.parameters()
                    if p.grad is not None
                ) ** 0.5

            params_to_clip = list(model.parameters())
            if our_syncnet is not None:
                params_to_clip += list(our_syncnet.parameters())

            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            scaler.step(optimizer)
            scaler.update()

            dn = delta_A.abs().mean().item()

            ep["total"].append(loss.item())
            ep["latent"].append(loss_latent.item())
            ep["same"].append(loss_same.item())
            ep["diff"].append(loss_diff.item())
            ep["silence"].append(loss_silence.item())
            ep["dreg"].append(loss_dreg.item())
            ep["smooth"].append(loss_smooth.item())
            ep["mouth"].append(loss_mouth_val.item())
            ep["mouth_edge"].append(loss_mouth_edge_val.item())
            ep["perceptual"].append(loss_perceptual_val.item())
            ep["sync"].append(loss_sync_val.item())
            ep["pos_sim"].append(pos_sim)
            ep["neg_sim"].append(neg_sim)
            ep["dn"].append(dn)
            ep["our"].append(loss_our_val.item())
            ep["our_d_pos"].append(our_d_pos)
            ep["our_d_neg"].append(our_d_neg)

            for k, v in [
                ("iter/loss_total",      loss.item()),
                ("iter/loss_latent",     loss_latent.item()),
                ("iter/loss_same",       loss_same.item()),
                ("iter/loss_diff",       loss_diff.item()),
                ("iter/loss_silence",    loss_silence.item()),
                ("iter/loss_smooth",     loss_smooth.item()),
                ("iter/loss_mouth",      loss_mouth_val.item()),
                ("iter/loss_mouth_edge", loss_mouth_edge_val.item()),
                ("iter/loss_perceptual", loss_perceptual_val.item()),
                ("iter/loss_sync",       loss_sync_val.item()),
                ("iter/diff_dist",       diff_dist.item()),
                ("iter/delta_A_norm",    delta_A.abs().mean().item()),
                ("iter/delta_B_norm",    delta_B.abs().mean().item()),
                ("iter/delta_neg_norm",  delta_neg.abs().mean().item()),
                ("iter/delta_zero_norm", delta_zero.abs().mean().item()),
                ("iter/pos_sim",         pos_sim),
                ("iter/neg_sim",         neg_sim),
                ("iter/delta_norm",      dn),
                ("iter/loss_our", loss_our_val.item()),
                ("iter/our_d_pos", our_d_pos),
                ("iter/our_d_neg", our_d_neg),
            ]:
                writer.add_scalar(k, v, global_step)
            global_step += 1

            if global_step % 100 == 0:
                msg = (f"[ep{epoch} iter{global_step}] "
                       f"loss={loss.item():.4f} latent={loss_latent.item():.4f} "
                       f"same={loss_same.item():.4f} "
                       f"diff={loss_diff.item():.4f} diff_dist={diff_dist.item():.4f} "
                       f"silence={loss_silence.item():.4f} smooth={loss_smooth.item():.4f} "
                       f"dA={delta_A.abs().mean().item():.4f} "
                       f"dN={delta_neg.abs().mean().item():.4f} "
                       f"dZ={delta_zero.abs().mean().item():.4f} "
                       f"mouth={loss_mouth_val.item():.4f} "
                       f"mouth_edge={loss_mouth_edge_val.item():.4f} "
                       f"sync={loss_sync_val.item():.4f} "
                       f"our={loss_our_val.item():.4f} "
                       f"d_pos={our_d_pos:.4f} d_neg={our_d_neg:.4f} "
                       f"pos={pos_sim:.4f} neg={neg_sim:.4f} "
                       f"dn={dn:.4f}")
                print(msg, flush=True)
                log.info(msg)

                # gradient norm 로깅 추가
                gen_grad = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5

                our_grad = 0.0
                if our_syncnet is not None:
                    our_grad = sum(
                        p.grad.norm().item() ** 2
                        for p in our_syncnet.parameters()
                        if p.grad is not None
                    ) ** 0.5

                grad_msg = (
                    f"  grad_norm_before_clip: generator={gen_grad_before:.4f} "
                    f"our_syncnet={our_grad_before:.4f}"
                )
                print(grad_msg, flush=True)
                log.info(grad_msg)
                writer.add_scalar("iter/grad_generator", gen_grad, global_step)
                writer.add_scalar("iter/grad_our_syncnet", our_grad, global_step)

        avg = lambda lst: float(np.mean(lst))
        msg = (f"[epoch {epoch}] "
               f"loss={avg(ep['total']):.4f} latent={avg(ep['latent']):.4f} "
               f"same={avg(ep['same']):.4f} diff={avg(ep['diff']):.4f} "
               f"silence={avg(ep['silence']):.4f} "
               f"dreg={avg(ep['dreg']):.4f} smooth={avg(ep['smooth']):.4f} "
               f"mouth={avg(ep['mouth']):.4f} "
               f"mouth_edge={avg(ep['mouth_edge']):.4f} "
               f"sync={avg(ep['sync']):.4f} "
               f"our={avg(ep['our']):.4f} "
               f"d_pos={avg(ep['our_d_pos']):.4f} d_neg={avg(ep['our_d_neg']):.4f} "
               f"pos={avg(ep['pos_sim']):.4f} neg={avg(ep['neg_sim']):.4f} "
               f"dn={avg(ep['dn']):.4f}")
        
        print(msg, flush=True)
        log.info(msg)

        for k, v in [
            ("train/loss_total",   avg(ep["total"])),
            ("train/loss_latent",  avg(ep["latent"])),
            ("train/loss_same",    avg(ep["same"])),
            ("train/loss_diff",    avg(ep["diff"])),
            ("train/loss_sync",    avg(ep["sync"])),
            ("train/loss_smooth",  avg(ep["smooth"])),
            ("train/loss_silence", avg(ep["silence"])),
            ("train/loss_dreg",    avg(ep["dreg"])),
            ("train/loss_mouth",   avg(ep["mouth"])),
            ("train/loss_mouth_edge", avg(ep["mouth_edge"])),
            ("train/loss_perceptual", avg(ep["perceptual"])),
            ("train/pos_sim",      avg(ep["pos_sim"])),
            ("train/neg_sim",      avg(ep["neg_sim"])),
            ("train/delta_norm",   avg(ep["dn"])),
            ("train/loss_our", avg(ep["our"])),
            ("train/our_d_pos", avg(ep["our_d_pos"])),
            ("train/our_d_neg", avg(ep["our_d_neg"])),
        ]:
            writer.add_scalar(k, v, epoch)

# val loop
        if epoch % cfg.sample.save_every == 0:
            model.eval()
            if our_syncnet is not None:
                our_syncnet.eval()
            save_samples(model, vae, cfg, device, samples_dir, epoch, writer)

            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    vt = val_batch["target_lips"].to(device)
                    vA = val_batch["ref_lips_A"].to(device)
                    va = val_batch["audios"].to(device)

                    B_v, T_v, c_v, h_v, w_v = vA.shape
                    z_vA = vae.encode(vA.reshape(B_v*T_v, c_v, h_v, w_v))
                    z_vA = z_vA.reshape(B_v, T_v, *z_vA.shape[1:])
                    z_vgt = vae.encode(vt.reshape(B_v*T_v, c_v, h_v, w_v))
                    z_vgt = z_vgt.reshape(B_v, T_v, *z_vgt.shape[1:])

                    z_vo, _ = model(z_vA, va)
                    val_losses.append(l1_loss(z_vo, z_vgt).item())

            val_loss = float(np.mean(val_losses))
            log.info(f"[val epoch {epoch}] loss={val_loss:.4f}")
            writer.add_scalar("val/loss_latent", val_loss, epoch)
            print(f"  [val] loss={val_loss:.4f}", flush=True)
            model.train()
            if our_syncnet is not None:
                our_syncnet.train()

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg(ep["total"]),
        }
        if disc is not None:
            ckpt["disc"]  = disc.state_dict()
            ckpt["opt_D"] = opt_D.state_dict()
        
        torch.save(ckpt, save_dir / "last.pt")
        if avg(ep["total"]) < best_loss:
            best_loss = avg(ep["total"])
            torch.save(ckpt, save_dir / "best.pt")
            log.info(f"  [best] epoch={epoch} loss={best_loss:.4f}")

    writer.close()
    print(f"done. results: {save_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="/home/jiweon/projects/ADLip2/configs/train_syncnet.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt",   default=None)
    args = parser.parse_args()
    train(args.cfg, args.device, args.ckpt)
