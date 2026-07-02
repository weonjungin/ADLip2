# ADLip2

Audio-driven talking head generation. 입술 영역(Lip-ROI)에만 집중해 학습하는 생성 모델 **ADLip**과, 립 latent 공간에서 직접 동기화를 학습하는 손실 함수 **SyncLT**를 제안한 4인 팀 캡스톤 프로젝트입니다.

**[프로젝트 데모 페이지](https://weonjungin.github.io/ADLip2/)** | **[포스터 전체 보기](docs/poster.jpg)**

## 배경

MuseTalk 등 기존 lip-sync 모델은 얼굴 전체(하관) 단위로 동기화를 학습하기 때문에 입술 정렬 신호가 희석되어 LSE-C 점수가 낮게 나오는 한계가 있었습니다. AttnWav2Lip은 attention 메커니즘으로 입술에 집중하려 시도했지만, 여전히 얼굴 전체 생성 파이프라인에 묶여있어 정밀한 입술 정렬에는 한계가 있었습니다.

이 프로젝트는 **입술 영역(96×96)만을 명시적으로 분리해 학습하면 더 정밀한 동기화가 가능하다**는 가설을 검증하기 위해, 얼굴 전체 정보 없이 입술만으로 동작하는 생성 모델과 손실 함수를 처음부터 설계했습니다.

## 본인 역할

4인 팀 프로젝트에서 다음을 담당했습니다.

- **SyncLT(SyncNetTemporal) 설계 및 학습** — [lip-sync-score](https://github.com/weonjungin/lip-sync-score) 레포에서 별도 개발
- **ADLip2에 SyncLT 손실 결합** — LatentSync(face-level) + SyncLT(lip-level) dual-level supervision 구조 구현
- **Generator 성능 개선**
- **모델 비교 실험** — D/F/G-시리즈 ablation (손실 가중치, GAN 결합, DDP 대규모 학습)
- **추론 파이프라인 발전 및 후처리 구축** — 초안 추론 코드를 발전시키고, GFPGAN/Poisson blending 기반 후처리 파이프라인 구현

## 파이프라인

    Ref Lip (16프레임 × 96×96)          Audio (waveform)
            │                                │
      VAE Encoder(frozen)            Whisper Encoder(frozen)
            │                                │
            └──────────► ADLip Generator ◄───┘
                    Spatial Cross-Attention ×2
                    Temporal Self-Attention ×2
                    Delta Head (z_out = z_ref + α·delta, α=0.35)
                              │
                      VAE Decoder(frozen)
                              │
                  Generated Lip Frames (16)

Generator는 입 모양을 처음부터 생성하지 않고, reference latent 대비 **변화량(delta)**만 예측합니다. 학습에는 Reconstruction loss, GAN loss(Mouth Discriminator), LatentSync loss(face-level, frozen), SyncLT loss(lip-level, 직접 제안)를 사용했습니다.

### 3단계 학습 전략 (`train_base.py`, `train_ddp.py` 공통)

1. **Stage 1** — 기본 재구성 (`latent + same + diff + silence + delta_reg`)
2. **Stage 2** — LatentSync loss 추가 (face-level 동기화)
3. **Stage 3** — GAN loss + SyncLT loss 추가 (영상 품질 + 입술 동기화 동시 향상)

## 핵심 결과

**테스트 42개 비디오 기준**, 손실 구성 요소별 기여도:

| 조건 | LatentSync | SyncLT | LSE-C ↑ | LSE-D ↓ |
|---|---|---|---|---|
| L1 (no syncnet) | X | X | 0.083 | 14.983 |
| LatentSync only | O | X | 5.472 | 9.071 |
| SyncLT only | X | O | 1.898 | 12.146 |
| **LatentSync + SyncLT** | **O** | **O** | **5.966** | **8.583** |

LatentSync(얼굴 단위)와 SyncLT(입술 단위)는 서로 다른 층위의 동기화 정보를 제공하며, 두 손실을 함께 사용했을 때 가장 높은 LSE-C와 가장 낮은 LSE-D를 달성했습니다. 입술 정보만으로는(SyncLT only) 동기화 학습에 한계가 있다는 것도 함께 확인했습니다.

추가로 진행한 세부 실험(주요 6개):

| 실험 | 스크립트 | 핵심 설정 | 결과 |
|---|---|---|---|
| `expD3` | `train_base.py` | `lambda_our=0.15`, `our_lambda_neg=2.5` | SyncLT/LatentSync gradient 비율이 가장 이상적(~2배)이었던 최적 조합 |
| `expD5` | `train_base.py` | `our_lr_mult=5.0` (adaptive lr 시도) | gradient 비율이 오히려 11배로 악화되어 폐기. D3의 고정 lambda가 더 나은 선택이었음을 보여주는 대조군 |
| `expF1` | `train_base.py` | GAN(`lambda_adv=0.03`) + SyncLT | LSE 수치는 낮지만 정성적으로 가장 자연스러운 결과. 정량 지표만으로는 영상 품질을 온전히 판단할 수 없다는 것을 확인 |
| `expG1` | `train_ddp.py` | DDP, batch_size=2 | 멀티 GPU 학습 초기 검증 |
| `expG2` | `train_ddp.py` | DDP, batch_size=8, H100 서버 | 대규모 배치 학습 (협업 데이터 경로 사용) |
| `expG3` | `train_ddp.py` | DDP, `grad_accum_steps=4`, `lambda_adv=0.05` | gradient accumulation을 적용한 최종 대규모 학습 (epoch 60 완주) |

`train_base.py`는 단일 GPU 학습 파이프라인, `train_ddp.py`는 `DistributedDataParallel` 기반 멀티 GPU 학습 파이프라인입니다. 대규모 배치가 필요한 G-시리즈 실험을 위해 별도로 DDP 파이프라인을 구축했습니다.

## 후처리 실험

`inference_postprocessed.py`는 GFPGAN을 이용한 얼굴 보정과 Poisson blending을 추가로 적용합니다. GFPGAN을 적용하면 시각적 자연스러움은 올라가지만, 영상에 노이즈가 있는 경우 벌어진 입 모양 등 실제 움직임을 아티팩트로 오인해 잘못 "보정"하면서 오히려 동기화 점수가 낮아지는 문제가 있었습니다. 이 때문에 GFPGAN weight를 0.2로 낮게 설정해 과도한 보정을 방지했습니다.

두 스크립트를 모두 남겨 후처리 전(`inference_raw.py`)/후(`inference_postprocessed.py`) 결과를 비교할 수 있도록 했습니다.

## 추론 관련 개선 사항

- 슬라이딩 윈도우 추론 시 프레임 간 지터(jitter)가 발생하는 문제를 `z_ref_fixed`를 윈도우 루프 바깥에서 고정하는 방식으로 해결했습니다.
- reference 프레임 선택 기준을 입술 bounding box 면적이 가장 큰(정면을 향한) 프레임으로 변경해 화질을 개선했습니다.

## 레포 구조

    ADLip2/
    ├── docs/                      # 프로젝트 데모 페이지 (GitHub Pages)
    │   ├── index.html
    │   └── poster.jpg
    │
    ├── configs/
    │   ├── expD3.yaml, expD5.yaml, expF1.yaml   # train_base.py 실험
    │   └── expG1.yaml, expG2.yaml, expG3.yaml   # train_ddp.py 실험
    │
    ├── scripts/
    │   ├── train_base.py             # 단일 GPU 학습
    │   ├── train_ddp.py              # DistributedDataParallel 멀티 GPU 학습
    │   ├── inference_raw.py          # 추론 (후처리 없음)
    │   └── inference_postprocessed.py # 추론 + GFPGAN/Poisson blending 후처리
    │
    ├── models/
    │   ├── generator.py          # ADLip Generator
    │   ├── discriminator.py      # Mouth Discriminator (GAN)
    │   ├── latentsync.py         # LatentSync (frozen, face-level sync 기준 모델)
    │   └── vae_wrapper.py        # VAE latent 인코딩/디코딩
    │
    ├── data/
    │   └── dataset.py            # HDTF 기반 학습 데이터로더
    │
    ├── results/
    │   ├── D3_20260513_150330/   # config, 체크포인트, 학습 로그, 샘플 이미지
    │   ├── D5_20260513_162646/
    │   ├── F1_20260518_155648/
    │   ├── G1_20260520_180628/
    │   └── G3_20260529_104341/
    │   (G2는 체크포인트만 존재하며 별도 학습 로그가 남아있지 않음)
    │
    ├── inference/
    │   ├── D3/, D5/, F1/, G1/    # 조건별 추론 샘플 (A: ref 변경, B: audio 변경, C: zero audio)
    │   └── G3_test/              # 후처리 전/후 비교 샘플
    │
    └── eval_results/
        └── lse_results_ADLip2.csv   # 전체 실험(28개)의 LSE-C/LSE-D 정량 평가 기록

## 환경 설정

    conda env create -f environment.yml
    conda activate adlip

## 실행 방법

### 1. 학습

    # 단일 GPU
    python scripts/train_base.py --cfg configs/expD3.yaml

    # DDP 멀티 GPU
    torchrun --nproc_per_node=3 scripts/train_ddp.py --cfg configs/expG3.yaml

### 2. 추론

    python scripts/inference_postprocessed.py \
        --ref_dir <참조 비디오 폴더> \
        --ckpt results/G3_20260529_104341/best.pt \
        --cfg configs/expG3.yaml \
        --audio_dir1 <오디오 폴더> \
        --out <출력 폴더> \
        --device cuda:0

### 3. 정량 평가 (LSE-C / LSE-D)

[syncnet_python](https://github.com/joonson/syncnet_python)의 `run_pipeline.py` + `run_syncnet.py`로 평가합니다.

    python run_pipeline.py --videofile <생성된 mp4> --reference <이름> --data_dir <임시 폴더>
    python run_syncnet.py --videofile <생성된 mp4> --reference <이름> --data_dir <임시 폴더>

## 체크포인트

`.pt` 체크포인트는 용량 문제로 GitHub에 포함하지 않았습니다. [Google Drive](https://drive.google.com/drive/folders/1kRF8PGcIMjcGzjbLZh6Jc6HIfCz-Z6Ek?usp=sharing)에서 받으실 수 있습니다.

## 관련 프로젝트

- [lip-sync-score](https://github.com/weonjungin/lip-sync-score): 이 프로젝트에서 사용한 SyncNetTemporal(SyncLT) 모델 개발 레포
- [avsync_project](https://github.com/weonjungin/avsync_project): 초기 SyncNet 연구 (AVSpeech 기반)

## 관련 연구

- MuseTalk (Zhang et al., 2024) — 얼굴 전체 단위 동기화, 본 프로젝트의 동기가 된 한계점(LSE-C 저하)을 제시
- AttnWav2Lip (Wang et al., 2022) — attention 기반 입술 강조 시도, 여전히 얼굴 전체 생성 파이프라인에 결합됨
