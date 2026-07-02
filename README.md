# ADLip2

Audio-driven lip-sync 영상 생성 모델. MuseTalk을 기반으로, 입술 영역에 특화된 동기화 손실(SyncLT)을 결합해 기존 얼굴 전체 단위 동기화 방식의 한계를 개선했습니다.

## 배경

MuseTalk 등 기존 lip-sync 모델은 얼굴 전체 단위로 동기화를 학습(LatentSync loss)하기 때문에, 입술 자체의 미세한 움직임 정확도가 떨어지는 경우가 있었습니다. 이 프로젝트는 [lip-sync-score](https://github.com/weonjungin/lip-sync-score)에서 개발한 입술 특화 동기화 모델 **SyncNetTemporal(SyncLT)**을 MuseTalk 파이프라인에 결합해, 얼굴 전체 손실과 입술 손실을 함께 최적화합니다.

캡스톤 경진대회 평가는 4가지 조건(L1 only / LatentSync only / SyncLT only / LatentSync + SyncLT)으로 진행되었으며, 이 레포는 그중 SyncLT 단독 조건과 LatentSync + SyncLT 결합 조건을 다룹니다.

## 핵심 결과

LatentSync loss와 SyncLT를 결합했을 때 42개 테스트 비디오 기준 **LSE-C 5.696, LSE-D 8.853**을 달성했습니다 (수치가 높을수록/LSE-D는 낮을수록 동기화 품질이 좋음).

주요 실험 6개를 남겨두었습니다.

| 실험 | 스크립트 | 핵심 설정 | 결과 |
|---|---|---|---|
| `expD3` | `train_base.py` | `lambda_our=0.15`, `our_lambda_neg=2.5` | SyncLT/LatentSync gradient 비율이 가장 이상적(~2배)이었던 최적 조합 |
| `expD5` | `train_base.py` | `our_lr_mult=5.0` (adaptive lr 시도) | gradient 비율이 오히려 11배로 악화되어 폐기. D3의 고정 lambda가 더 나은 선택이었음을 보여주는 대조군 |
| `expF1` | `train_base.py` | GAN(`lambda_adv=0.03`) + SyncLT | LSE 수치는 낮지만 정성적으로 가장 자연스러운 결과. 정량 지표만으로는 영상 품질을 온전히 판단할 수 없다는 것을 확인 |
| `expG1` | `train_ddp.py` | DDP, batch_size=2 | 멀티 GPU 학습 초기 검증 |
| `expG2` | `train_ddp.py` | DDP, batch_size=8, H100 서버 | 대규모 배치 학습 (협업 데이터 경로 사용) |
| `expG3` | `train_ddp.py` | DDP, `grad_accum_steps=4`, `lambda_adv=0.05` | gradient accumulation을 적용한 최종 대규모 학습 (epoch 60 완주) |

`train_base.py`는 단일 GPU 학습 파이프라인이고, `train_ddp.py`는 `DistributedDataParallel` 기반 멀티 GPU 학습 파이프라인입니다. 둘 다 동일한 SyncLT 손실을 사용하며, 대규모 배치가 필요한 G-시리즈 실험을 위해 별도로 DDP 파이프라인을 구축했습니다.

## 후처리 실험

`inference_postprocessed.py`는 GFPGAN을 이용한 얼굴 보정과 Poisson blending을 추가로 적용합니다. GFPGAN을 적용하면 시각적 자연스러움은 올라가지만, 영상에 노이즈가 있는 경우 벌어진 입 모양 등 실제 움직임을 아티팩트로 오인해 잘못 "보정"하면서 오히려 동기화 점수가 낮아지는 문제가 있었습니다. 이 때문에 GFPGAN weight를 0.2로 낮게 설정해 과도한 보정을 방지했습니다.

두 스크립트를 모두 남겨 후처리 전(`inference_raw.py`)/후(`inference_postprocessed.py`) 결과를 비교할 수 있도록 했습니다.

## 추론 관련 개선 사항

- 슬라이딩 윈도우 추론 시 프레임 간 지터(jitter)가 발생하는 문제를 `z_ref_fixed`를 윈도우 루프 바깥에서 고정하는 방식으로 해결했습니다.
- reference 프레임 선택 기준을 입술 bounding box 면적이 가장 큰(정면을 향한) 프레임으로 변경해 화질을 개선했습니다.

## 레포 구조

    ADLip2/
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
    │   ├── generator.py          # ADLip2 Generator
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

`.pt` 체크포인트는 용량 문제로 GitHub에 포함하지 않았습니다. Google Drive에서 받으실 수 있습니다: [Google Drive](https://drive.google.com/drive/folders/1kRF8PGcIMjcGzjbLZh6Jc6HIfCz-Z6Ek?usp=sharing)

## 관련 프로젝트

- [lip-sync-score](https://github.com/weonjungin/lip-sync-score): 이 프로젝트에서 사용한 SyncNetTemporal(SyncLT) 모델 개발 레포
- [avsync_project](https://github.com/weonjungin/avsync_project): 초기 SyncNet 연구 (AVSpeech 기반)
