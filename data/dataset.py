"""
data/dataset.py

반환:
    target_lips  [16, 3, 96, 96]
    ref_lips_A   [16, 3, 96, 96]
    ref_lips_B   [16, 3, 96, 96]
    audios       [16, 5, 384]
    mel          [1, 80, 52]
    face_window  [48, 128, 256]

샘플링:
    t   = target 시작
    j_A = ref_A 시작, |j_A - t| >= 50
    j_B = ref_B 시작, |j_B - t| >= 50, |j_B - j_A| >= 50
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

MEL_STEP      = 52
FACE_SIZE     = 256
N_FRAMES      = 16
MIN_GAP       = 50
AUDIO_HALF    = 2  # window_size=5, half=2

# 고정 샘플 비디오는 split에서 제외됨


class LipSyncDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=96,
                 samples_per_video=100, splits_dir=None):
        self.data_root = Path(data_root)
        self.img_size  = img_size
        self.samples_per_video = samples_per_video

        if splits_dir is not None:
            split_file = Path(splits_dir) / f"{split}.txt"
            with open(str(split_file)) as f:
                video_names = [l.strip() for l in f if l.strip()]
            videos = [
                self.data_root / name for name in video_names
                if self._valid_video(self.data_root / name)
            ]
        else:
            all_videos = sorted([
                d for d in self.data_root.iterdir()
                if d.is_dir() and self._valid_video(d)
            ])
            n_val = max(1, int(len(all_videos) * 0.1))
            videos = all_videos[n_val:] if split == "train" else all_videos[:n_val]

        self.video_data = []
        self.samples    = []
        self._build(videos)
        print(f"{split} videos={len(self.video_data)} samples={len(self.samples)}")

    def _valid_video(self, d):
        return (
            d.is_dir()
            and (d / "audio_feature.npy").exists()
            and (d / "face_frames").exists()
            and (d / "lip_meta.npy").exists()
            and (d / "frame_indices.npy").exists()
            and (d / "mel_latentsync.npy").exists()
        )

    def _sample_ref_start(self, t, n, exclude_starts, max_try=200):
        """t와 MIN_GAP 이상 떨어진 ref 시작 index 샘플링"""
        for _ in range(max_try):
            j = np.random.randint(0, n - N_FRAMES)
            if abs(j - t) < MIN_GAP:
                continue
            if j + N_FRAMES > n:
                continue
            ok = True
            for ex in exclude_starts:
                if abs(j - ex) < MIN_GAP:
                    ok = False
                    break
            if ok:
                return j
        return None

    def _build(self, videos):
        for video_dir in videos:
            audio_npy      = video_dir / "audio_feature.npy"
            lip_meta_path  = video_dir / "lip_meta.npy"
            frame_idx_path = video_dir / "frame_indices.npy"
            mel_path       = video_dir / "mel_latentsync.npy"

            audio_feature = np.load(str(audio_npy), mmap_mode='r')
            lip_meta      = np.load(str(lip_meta_path), mmap_mode='r')
            frame_indices = np.load(str(frame_idx_path), mmap_mode='r')

            frame_paths = sorted((video_dir / "frames").glob("*.png"))
            face_paths  = sorted((video_dir / "face_frames").glob("*.png"))
            n = min(len(frame_paths), len(audio_feature),
                    len(lip_meta), len(frame_indices))

            # 최소 16*3 + 여유 = 150프레임 이상 필요
            if n < 150:
                continue

            # valid frame mask (lip_meta valid=1인 것만)
            valid_mask = lip_meta[:n, 0] == 1

            # target 후보: valid이고 t+16 범위 안에 전부 valid인 것
            target_starts = []
            for t in range(n - N_FRAMES):
                window = lip_meta[t:t+N_FRAMES, 0]
                if np.all(window == 1):
                    target_starts.append(t)

            if len(target_starts) < 10:
                continue

            # samples_per_video개 샘플링
            chosen = np.random.choice(
                target_starts,
                size=min(self.samples_per_video, len(target_starts)),
                replace=False
            )

            vid_idx = len(self.video_data)
            self.video_data.append({
                "video_dir":     str(video_dir),
                "frame_paths":   [str(p) for p in frame_paths],
                "face_paths":    [str(p) for p in face_paths],
                "audio_npy":     str(audio_npy),
                "mel_path":      str(mel_path),
                "frame_indices": np.array(frame_indices),
                "lip_meta":      np.array(lip_meta[:n]),
                "n":             n,
                "target_starts": target_starts,
            })

            for t in chosen:
                self.samples.append((vid_idx, int(t)))

    def _load_lip_window(self, frame_paths, start, n):
        """start:start+16 프레임 로드 → [16, 3, 96, 96]"""
        imgs = []
        for i in range(N_FRAMES):
            idx = min(start + i, n - 1)
            img = cv2.imread(frame_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            t = torch.tensor(img).permute(2,0,1).float() / 127.5 - 1.0
            imgs.append(t)
        return torch.stack(imgs, dim=0)  # [16, 3, 96, 96]

    def _get_audio_windows(self, audio_feature, start, n):
        """16시점 각각의 audio window → [16, 5, 384]"""
        windows = []
        for i in range(N_FRAMES):
            idx = min(start + i, n - 1)
            idxs = np.arange(idx - AUDIO_HALF, idx + AUDIO_HALF + 1)
            idxs = np.clip(idxs, 0, n - 1)
            windows.append(audio_feature[idxs])
        return torch.tensor(np.stack(windows), dtype=torch.float32)  # [16, 5, 384]

    def _get_face_window(self, face_paths, start, n):
        """16프레임 얼굴 하단 절반 concat → [48, 128, 256]"""
        frames = []
        for i in range(N_FRAMES):
            idx = min(start + i, n - 1)
            img = cv2.imread(face_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            t = torch.tensor(img).permute(2,0,1).float() / 127.5 - 1.0
            frames.append(t)
        face_stack  = torch.stack(frames, dim=0)          # [16, 3, 256, 256]
        h           = face_stack.shape[2]
        face_stack  = face_stack[:, :, h//2:, :]          # [16, 3, 128, 256]
        face_window = face_stack.permute(1,0,2,3).reshape(48, 128, FACE_SIZE)
        return face_window

    def _get_mel_window(self, mel_path, start, frame_indices):
        """target 시작 프레임 기준 mel window → [1, 80, 52]"""
        mel = np.load(mel_path, mmap_mode='r')
        orig_frame_idx = int(frame_indices[start])
        s = int(80. * (orig_frame_idx / 25.))
        chunk = mel[:, s:s + MEL_STEP]
        if chunk.shape[1] < MEL_STEP:
            chunk = np.pad(chunk, ((0,0),(0, MEL_STEP - chunk.shape[1])))
        return torch.tensor(np.array(chunk), dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, t = self.samples[idx]
        vd = self.video_data[vid_idx]
        n  = vd["n"]

        # ref_A, ref_B 샘플링
        j_A = self._sample_ref_start(t, n, exclude_starts=[t])
        if j_A is None:
            j_A = (t + 100) % max(1, n - N_FRAMES)
        j_B = self._sample_ref_start(t, n, exclude_starts=[t, j_A])
        if j_B is None:
            j_B = (t + 200) % max(1, n - N_FRAMES)

        audio_feature = np.load(vd["audio_npy"], mmap_mode='r')

        target_lips = self._load_lip_window(vd["frame_paths"], t,   n)
        ref_lips_A  = self._load_lip_window(vd["frame_paths"], j_A, n)
        ref_lips_B  = self._load_lip_window(vd["frame_paths"], j_B, n)
        audios      = self._get_audio_windows(audio_feature, t, n)
        face_window = self._get_face_window(vd["face_paths"], t, n)
        mel         = self._get_mel_window(vd["mel_path"], t, vd["frame_indices"])

        lip_bbox = torch.tensor(vd["lip_meta"][t:t+N_FRAMES], dtype=torch.long)  # [16, 6]

        return {
            "target_lips": target_lips,   # [16, 3, 96, 96]
            "ref_lips_A":  ref_lips_A,    # [16, 3, 96, 96]
            "ref_lips_B":  ref_lips_B,    # [16, 3, 96, 96]
            "audios":      audios,        # [16, 5, 384]
            "mel":         mel,           # [1, 80, 52]
            "face_window": face_window,   # [48, 128, 256]
            "lip_bbox":    lip_bbox,      # [16, 6]
        }
