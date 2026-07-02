"""
train_ihj.py
친구 train.py 기반 + our syncnet loss 추가
Stage 1 : latent + same + diff + silence + delta_reg + smooth
Stage 2 : + sync loss (lambda_sync > 0)
Stage 3 : + GAN mouth discriminator (lambda_adv > 0)
Stage 3+: + our syncnet loss (lambda_our > 0)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import random
import logging
import argparse
from contextlib import nullcontext
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, "/home/ihjung/2026/ADLip2")
sys.path.insert(0, "/home/jiweon/projects/lip-sync-score/src")

from models.vae_wrapper    import VAEWrapper
from models.generator      import TemporalLipGenerator
from models.discriminator  import MouthDiscriminator
from data.dataset          import LipSyncDataset
from losses                import LossWeights, compute_losses, d_loss_lsgan, g_loss_lsgan, feature_matching_loss, syncnet_feature_matching_loss
from utils.lip_utils       import encode_seq, decode_seq, paste_lips_to_face
from utils.syncnet_loader  import load_latentsync
from utils.visualizer      import save_samples

from lipsyncscore.models.modified.syncnet_temporal import SyncNetTemporal
from lipsyncscore.loss.contrastive import SyncNetContrastiveLoss


N_FRAMES = 16


# ── 시드 ─────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


# ── our syncnet 로드 ──────────────────────────────────────────────
def load_our_syncnet(cfg, device):
    ckpt = torch.load(cfg.our.ckpt_path, map_location=device, weights_only=False)

    # 체크포인트에 저장된 config로 구조 복원
    saved_cfg    = ckpt.get("config", {})
    model_cfg    = saved_cfg.get("model", {})
    temporal_cfg = model_cfg.get("temporal", {}) or {}

    model = SyncNetTemporal(
        in_frames       = model_cfg.get("in_frames", cfg.our.get("in_frames", 16)),
        emb_dim         = model_cfg.get("emb_dim", cfg.our.get("emb_dim", 256)),
        pooling         = model_cfg.get("pooling", cfg.our.get("pooling", "mean")),
        lip_in_channels = model_cfg.get("lip_in_channels", cfg.our.get("lip_in_channels", 4)),
        temporal_cfg    = temporal_cfg if temporal_cfg else None,
    ).to(device)

    state = ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Our SyncNetTemporal loaded & trainable: {cfg.our.ckpt_path}", flush=True)
    print(f"  temporal: {temporal_cfg.get('type', 'none')}, num_layers: {temporal_cfg.get('num_layers', '-')}", flush=True)

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    return model


# ── TensorBoard 키 매핑 ───────────────────────────────────────────
ITER_KEYS = [
    ("iter/loss_total",      "total"),
    ("iter/loss_latent",     "latent"),
    ("iter/loss_same",       "same"),
    ("iter/loss_diff",       "diff"),
    ("iter/loss_delta_reg",  "delta_reg"),
    ("iter/loss_silence",    "silence"),
    ("iter/loss_smooth",     "smooth"),
    ("iter/loss_sync",       "sync"),
    ("iter/loss_gan",        "adv"),
    ("iter/loss_disc",       "d_loss"),
    ("iter/loss_fm",         "fm"),
    ("iter/loss_sync_fm",    "sync_fm"),
    ("iter/loss_our",        "our"),
    ("iter/our_d_pos",       "our_d_pos"),
    ("iter/our_d_neg",       "our_d_neg"),
    ("iter/diff_dist",       "diff_dist"),
    ("iter/pos_sim",         "pos_sim"),
    ("iter/neg_sim",         "neg_sim"),
    ("iter/delta_A_norm",    "delta_A_norm"),
    ("iter/delta_B_norm",    "delta_B_norm"),
    ("iter/delta_neg_norm",  "delta_neg_norm"),
    ("iter/delta_zero_norm", "delta_zero_norm"),
    ("iter/delta_norm",      "dn"),
    ("iter/d_real_mean",     "d_real_mean"),
    ("iter/d_fake_mean",     "d_fake_mean"),
]
EPOCH_KEYS = [
    ("train/loss_total",     "total"),
    ("train/loss_latent",    "latent"),
    ("train/loss_same",      "same"),
    ("train/loss_diff",      "diff"),
    ("train/loss_delta_reg", "delta_reg"),
    ("train/loss_silence",   "silence"),
    ("train/loss_smooth",    "smooth"),
    ("train/loss_sync",      "sync"),
    ("train/loss_gan",       "adv"),
    ("train/loss_disc",      "d_loss"),
    ("train/loss_fm",        "fm"),
    ("train/loss_sync_fm",   "sync_fm"),
    ("train/loss_our",       "our"),
    ("train/our_d_pos",      "our_d_pos"),
    ("train/our_d_neg",      "our_d_neg"),
    ("train/diff_dist",      "diff_dist"),
    ("train/pos_sim",        "pos_sim"),
    ("train/neg_sim",        "neg_sim"),
    ("train/delta_norm",     "dn"),
]

def log_iter(writer, m, step):
    for tb_k, m_k in ITER_KEYS:
        if m_k in m:
            writer.add_scalar(tb_k, m[m_k], step)

def log_epoch(writer, ep, epoch):
    avg = lambda k: float(np.mean(ep[k])) if ep[k] else 0.0
    for tb_k, m_k in EPOCH_KEYS:
        if m_k in ep and ep[m_k]:
            writer.add_scalar(tb_k, avg(m_k), epoch)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in ep.items()}


# ── 학습 ─────────────────────────────────────────────────────────
def train(cfg_path, device_str, ckpt_path=None, out_dir=None):

    # DDP 초기화
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_master = (local_rank == 0)

    cfg    = OmegaConf.load(cfg_path)
    set_seed(cfg.train.get("seed", 42) + local_rank)

    run_name    = cfg.get("run_name", "")

    # rank 0 timestamp를 브로드캐스트해서 전 rank 동일하게
    if is_master:
        ts_tensor = torch.tensor(
            [int(datetime.now().strftime("%Y%m%d%H%M%S"))], dtype=torch.long
        ).to(device)
    else:
        ts_tensor = torch.zeros(1, dtype=torch.long).to(device)
    dist.broadcast(ts_tensor, src=0)
    timestamp = datetime.strptime(str(ts_tensor.item()), "%Y%m%d%H%M%S").strftime("%Y%m%d_%H%M%S")

    folder_name = f"{run_name}_{timestamp}" if run_name else timestamp
    save_dir    = Path(out_dir) if out_dir else Path(cfg.results.root) / folder_name
    samples_dir = save_dir / "samples"
    tb_dir      = save_dir / "tb"
    if is_master:
        for d in [save_dir, samples_dir, tb_dir]:
            d.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_master:
        logging.basicConfig(
            filename=str(save_dir / "train.log"), level=logging.INFO,
            format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
            force=True,
        )
        OmegaConf.save(cfg, save_dir / "config.yaml")
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        writer = None
    log = logging.getLogger()

    # VAE (frozen)
    vae = VAEWrapper(cfg.vae.model_name).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    with torch.no_grad():
        z_dummy = vae.encode(torch.zeros(1, 3, 96, 96).to(device))
        _, C, H, W = z_dummy.shape
    print(f"latent: {z_dummy.shape}", flush=True)

    # Generator
    model = TemporalLipGenerator(
        latent_dim=C, spatial_len=H*W, audio_in=384,
        audio_len=cfg.model.audio_len, n_frames=N_FRAMES,
        dim=cfg.model.dim, num_heads=cfg.model.num_heads,
        spatial_layers=cfg.model.spatial_layers,
        temporal_layers=cfg.model.temporal_layers,
        ffn_dim_ratio=cfg.model.ffn_dim_ratio,
        dropout=cfg.model.dropout, alpha=cfg.model.alpha,
    ).to(device)

    if ckpt_path:
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd  = raw["model"] if "model" in raw else raw
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded: {ckpt_path}", flush=True)
        log.info(f"Loaded: {ckpt_path}")
        if missing:
            print(f"  missing ({len(missing)}): {missing[:3]}", flush=True)

    model = DDP(model, device_ids=[local_rank])

    # Loss / SyncNet
    weights    = LossWeights.from_cfg(cfg)
    syncnet    = load_latentsync(cfg.syncnet.ckpt_path, device) if weights.lambda_sync > 0 else None
    ref_scale  = cfg.model.get("ref_scale", 1.0)
    lambda_adv = cfg.loss.get("lambda_adv", 0.0)

    # our syncnet
    lambda_our     = cfg.loss.get("lambda_our", 0.0)
    our_margin     = cfg.loss.get("our_margin", 1.0)
    our_lambda_pos = cfg.loss.get("our_lambda_pos", 1.0)
    our_lambda_neg = cfg.loss.get("our_lambda_neg", 1.0)

    our_syncnet   = None
    our_criterion = None
    if lambda_our > 0:
        our_syncnet = load_our_syncnet(cfg, device)
        our_syncnet = DDP(our_syncnet, device_ids=[local_rank])
        our_criterion = SyncNetContrastiveLoss(
            margin=our_margin,
            lambda_pos=our_lambda_pos,
            lambda_neg=our_lambda_neg,
        )

    # Discriminator (Stage 3)
    disc  = None
    opt_D = None
    if lambda_adv > 0:
        ndf   = cfg.model.get("ndf", 64)
        disc  = MouthDiscriminator(ndf=ndf, in_ch=4).to(device)
        disc  = DDP(disc, device_ids=[local_rank])
        opt_D = torch.optim.Adam(
            disc.parameters(),
            lr=cfg.train.get("lr_d", 1e-5),
            betas=(0.5, 0.999),
        )
        print(f"Discriminator 초기화 (ndf={ndf})", flush=True)
        if ckpt_path:
            raw = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "disc" in raw:
                try:
                    disc.module.load_state_dict(raw["disc"])
                    opt_D.load_state_dict(raw["opt_D"])
                    print("  discriminator checkpoint 로드됨", flush=True)
                except RuntimeError as e:
                    print(f"  disc ckpt 불일치, 새로 초기화: {e}", flush=True)

    # optimizer: generator + our_syncnet 파라미터
    if our_syncnet is not None:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": cfg.train.lr},
                {"params": our_syncnet.parameters(), "lr": cfg.train.lr * cfg.train.get("our_lr_mult", 1.0)},
            ],
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=1e-4
        )

    if ckpt_path:
        _raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "optimizer" in _raw:
            optimizer.load_state_dict(_raw["optimizer"])
            print("  optimizer checkpoint 로드됨", flush=True)
        if our_syncnet is not None and "our_syncnet" in _raw:
            our_syncnet.module.load_state_dict(_raw["our_syncnet"])
            print("  our_syncnet checkpoint 로드됨", flush=True)

    # adaptive scaling
    our_scale            = 1.0
    our_scale_update_freq = 100

    # DataLoaders
    def make_loader(split, shuffle):
        ds = LipSyncDataset(
            cfg.data.root, split=split,
            samples_per_video=cfg.data.samples_per_video,
            splits_dir=cfg.data.get("splits_dir", None),
            ref_repeat=cfg.data.get("ref_repeat", False),
        )
        sampler = DistributedSampler(ds, shuffle=shuffle)
        return DataLoader(
            ds, batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers, pin_memory=True,
        ), sampler

    loader,     train_sampler = make_loader("train", shuffle=True)
    val_loader, _             = make_loader("val",   shuffle=False)

    best_loss   = float("inf")
    global_step = 0
    start_epoch = 1



    if ckpt_path:
        _raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "epoch" in _raw:
            loaded_epoch = _raw["epoch"]
            # cfg.train.epochs보다 크면 이어서 할 수 있도록 epochs 자동 조정
            if loaded_epoch >= cfg.train.epochs:
                extra = cfg.train.get("extra_epochs", 30)
                cfg.train.epochs = loaded_epoch + extra
                print(f"체크포인트 epoch={loaded_epoch}, epochs를 {cfg.train.epochs}로 자동 조정", flush=True)
            start_epoch = loaded_epoch + 1
            print(f"이어서 학습: epoch {start_epoch}부터", flush=True)

    if is_master:
        model.eval()
        save_samples(model, vae, cfg, device, samples_dir, epoch=0, writer=writer)
        model.train()

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        if disc is not None:
            disc.train()
        if our_syncnet is not None:
            our_syncnet.train()

        ep = {k: [] for k in [
            "total","latent","same","diff","diff_dist","silence",
            "delta_reg","smooth","mouth","sync","adv","d_loss",
            "fm","sync_fm","d_real_mean","d_fake_mean","pos_sim","neg_sim","dn",
            "delta_A_norm","delta_B_norm","delta_neg_norm","delta_zero_norm",
            "our","our_d_pos","our_d_neg",
        ]}

        accum_steps = cfg.train.get("grad_accum_steps", 4)
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            target_lips = batch["target_lips"].to(device)
            ref_lips_A  = batch["ref_lips_A"].to(device)
            ref_lips_B  = batch["ref_lips_B"].to(device)
            audios      = batch["audios"].to(device)
            mel         = batch["mel"].to(device)
            face_window = batch["face_window"].to(device)
            lip_bbox    = batch["lip_bbox"].to(device)
            B_cur       = audios.shape[0]

            zero_audio = torch.zeros_like(audios)
            perm       = torch.roll(torch.arange(B_cur, device=device), 1)
            audio_neg  = audios[perm]

            with torch.no_grad():
                z_A  = encode_seq(vae, ref_lips_A)
                z_B  = encode_seq(vae, ref_lips_B)
                z_gt = encode_seq(vae, target_lips)

            z_A_in = z_A * ref_scale
            z_B_in = z_B * ref_scale

            z_out_A,    delta_A    = model(z_A_in, audios)
            z_out_B,    delta_B    = model(z_B_in, audios)
            z_out_zero, delta_zero = model(z_A_in, zero_audio)
            z_out_neg,  delta_neg  = model(z_A_in, audio_neg)

            fwd = dict(
                z_out_A=z_out_A, z_out_B=z_out_B, z_gt=z_gt,
                delta_A=delta_A, delta_B=delta_B,
                delta_zero=delta_zero, delta_neg=delta_neg,
            )

            if weights.lambda_mouth > 0:
                fwd.update(
                    pred_A_imgs=decode_seq(vae, z_out_A),
                    pred_B_imgs=decode_seq(vae, z_out_B),
                    target_lips=target_lips,
                )

            if weights.lambda_sync > 0 and syncnet is not None:
                pred_A_imgs = fwd.get("pred_A_imgs", decode_seq(vae, z_out_A))
                face_w_pred = paste_lips_to_face(face_window, pred_A_imgs, lip_bbox)
                fwd["face_embed"]  = syncnet.get_image_embed(face_w_pred)
                fwd["audio_embed"] = syncnet.get_audio_embed(mel)
                if weights.lambda_sync_fm > 0:
                    face_w_real = paste_lips_to_face(face_window, target_lips, lip_bbox)
                    fwd["sync_real_feats"] = syncnet.get_image_features(face_w_real.detach())
                    fwd["sync_fake_feats"] = syncnet.get_image_features(face_w_pred)

            loss, m = compute_losses(fwd, weights)

            # ── our loss ─────────────────────────────────────────
            loss_our_val = torch.zeros([], device=device)
            our_d_pos = 0.0
            our_d_neg = 0.0

            if lambda_our > 0 and our_syncnet is not None:
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

                v_embed = our_syncnet.module.forward_lip(z_out_scaled)
                a_pos   = our_syncnet.module.forward_audio(mel_pos)
                a_neg   = our_syncnet.module.forward_audio(mel_neg)

                loss_our_val, our_info = our_criterion(v_embed, a_pos, a_neg)
                our_d_pos = our_info["d_pos"]
                our_d_neg = our_info["d_neg"]

                # adaptive scaling: 100 step마다 측정
                if global_step % our_scale_update_freq == 0 and weights.lambda_sync > 0:
                    sync_loss_for_grad = m.get("sync_loss_tensor", None)
                    if sync_loss_for_grad is not None and sync_loss_for_grad > 1e-8:
                        sync_grads = torch.autograd.grad(
                            weights.lambda_sync * sync_loss_for_grad,
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

                        if is_master:
                            writer.add_scalar("iter/our_scale", our_scale, global_step)

                        print(f"  [adaptive] sync_gnorm={sync_gnorm:.4f} our_gnorm={our_gnorm:.4f} scale={our_scale:.4f}", flush=True)

                loss = loss + lambda_our * loss_our_val * our_scale

            # ── Discriminator update ──────────────────────────────
            loss_adv_val = torch.zeros([], device=device)
            loss_d_val   = torch.zeros([], device=device)
            loss_fm_val  = torch.zeros([], device=device)

            if lambda_adv > 0 and disc is not None:
                pred_A_imgs = fwd.get("pred_A_imgs", decode_seq(vae, z_out_A))
                BT = B_cur * N_FRAMES

                mel_exp     = mel.unsqueeze(1).expand(-1, N_FRAMES, -1, -1, -1)
                mel_flat    = mel_exp.reshape(BT, 1, mel.shape[2], mel.shape[3])
                mel_96      = F.interpolate(mel_flat, size=(96,96), mode='bilinear', align_corners=False)

                target_flat = target_lips.reshape(BT, 3, 96, 96).detach()
                pred_detach = pred_A_imgs.reshape(BT, 3, 96, 96).detach()
                pred_grad   = pred_A_imgs.reshape(BT, 3, 96, 96)

                d_in_real   = torch.cat([target_flat, mel_96.detach()], dim=1)
                d_in_fake   = torch.cat([pred_detach, mel_96.detach()], dim=1)
                d_in_fake_g = torch.cat([pred_grad,   mel_96.detach()], dim=1)

                do_d_update = (i + 1) % accum_steps == 0

                if do_d_update:
                    opt_D.zero_grad()
                    d_real_out, _ = disc.module.get_features(d_in_real)
                    d_fake_out, _ = disc.module.get_features(d_in_fake)
                    loss_d        = d_loss_lsgan(d_real_out, d_fake_out)
                    loss_d.backward()
                    opt_D.step()
                    loss_d_val = loss_d.detach()

                disc.requires_grad_(False)
                d_real_out, feat_real = disc.module.get_features(d_in_real.detach())
                d_fake_out, feat_fake = disc.module.get_features(d_in_fake_g)
                loss_adv_val = g_loss_lsgan(d_fake_out)
                loss         = loss + lambda_adv * loss_adv_val
                if weights.lambda_fm > 0:
                    loss_fm_val = feature_matching_loss(feat_real, feat_fake)
                    loss        = loss + weights.lambda_fm * loss_fm_val
                m["d_real_mean"] = d_real_out.detach().mean().item()
                m["d_fake_mean"] = d_fake_out.detach().mean().item()
                disc.requires_grad_(True)

            if "d_real_mean" not in m: m["d_real_mean"] = 0.0
            if "d_fake_mean" not in m: m["d_fake_mean"] = 0.0
            m["sync_fm"]  = m.get("sync_fm", 0.0)
            m["adv"]      = loss_adv_val.item()
            m["d_loss"]   = loss_d_val.item()
            m["fm"]       = loss_fm_val.item() if lambda_adv > 0 and disc is not None else 0.0
            m["our"]      = loss_our_val.item()
            m["our_d_pos"] = our_d_pos
            m["our_d_neg"] = our_d_neg
            m["total"]    = loss.item()

            # accum 완료 시점에만 DDP gradient sync
            if (i + 1) % accum_steps == 0:
                (loss / accum_steps).backward()
            else:
                ctx_model = model.no_sync()
                ctx_our   = our_syncnet.no_sync() if our_syncnet is not None else nullcontext()
                with ctx_model, ctx_our:
                    (loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
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

                params_to_clip = list(model.parameters())
                if our_syncnet is not None:
                    params_to_clip += list(our_syncnet.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                for k, v in m.items():
                    if k in ep:
                        ep[k].append(v)
                if is_master:
                    log_iter(writer, m, global_step)
                global_step += 1

                if is_master and global_step % 100 == 0:
                    msg = (
                        f"[ep{epoch} iter{global_step}] "
                        f"loss={m['total']:.4f} latent={m['latent']:.4f} "
                        f"same={m['same']:.4f} diff={m['diff']:.4f} dist={m['diff_dist']:.4f} "
                        f"sil={m['silence']:.4f} smo={m['smooth']:.4f} "
                        f"sync={m['sync']:.4f} pos={m['pos_sim']:.4f} neg={m['neg_sim']:.4f} "
                        f"gan={m['adv']:.4f} disc={m['d_loss']:.4f} fm={m['fm']:.4f} "
                        f"our={m['our']:.4f} d_pos={m['our_d_pos']:.4f} d_neg={m['our_d_neg']:.4f} "
                        f"delta_A={m['delta_A_norm']:.4f} dn={m['dn']:.4f} "
                        f"grad_gen={gen_grad:.4f} grad_our={our_grad:.4f}"
                    )
                    print(msg, flush=True); log.info(msg)

        if is_master:
            avgs = log_epoch(writer, ep, epoch)
            msg  = (
                f"[epoch {epoch}] "
                + " ".join(f"{k}={avgs[k]:.4f}" for k in
                        ["total","latent","same","diff","silence","delta_reg",
                            "smooth","sync","adv","d_loss","our","dn"])
            )
            print(msg, flush=True); log.info(msg)

            if epoch % cfg.sample.save_every == 0:
                model.eval()
                if our_syncnet is not None:
                    our_syncnet.eval()
                save_samples(model, vae, cfg, device, samples_dir, epoch, writer)

                with torch.no_grad():
                    val_lat = []
                    for vb in val_loader:
                        vA   = vb["ref_lips_A"].to(device)
                        vt   = vb["target_lips"].to(device)
                        va   = vb["audios"].to(device)

                        z_vA  = encode_seq(vae, vA) * ref_scale
                        z_vgt = encode_seq(vae, vt)
                        z_vo, _ = model(z_vA, va)
                        val_lat.append(F.l1_loss(z_vo, z_vgt).item())

                val_loss = float(np.mean(val_lat))
                writer.add_scalar("val/loss_latent", val_loss, epoch)
                log.info(f"[val epoch {epoch}] loss={val_loss:.4f}")
                print(f"  [val] loss={val_loss:.4f}", flush=True)
                model.train()
                if our_syncnet is not None:
                    our_syncnet.train()
                if disc is not None:
                    disc.train()

            ckpt_dict = {
                "epoch": epoch, "loss": avgs["total"],
                "model": model.module.state_dict(), 
                "optimizer": optimizer.state_dict(),
            }
            if disc is not None:
                ckpt_dict["disc"]  = disc.module.state_dict()
                ckpt_dict["opt_D"] = opt_D.state_dict()
            
            if our_syncnet is not None:
                ckpt_dict["our_syncnet"] = our_syncnet.module.state_dict()

            torch.save(ckpt_dict, save_dir / "last.pt")
            if avgs["total"] < best_loss:
                best_loss = avgs["total"]
                torch.save(ckpt_dict, save_dir / "best.pt")
                log.info(f"  [best] epoch={epoch} loss={best_loss:.4f}")
        dist.barrier()

    if is_master:
        writer.close()
        print(f"done. results: {save_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",     default="/home/ihjung/2026/ADLip2/configs/train.yaml")
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--ckpt",    default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    train(args.cfg, args.device, args.ckpt, args.out_dir)