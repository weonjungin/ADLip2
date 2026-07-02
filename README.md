conda activate adlip
cd /home/jiweon/projects/ADLip2

nohup python train.py \
  --cfg configs/expB4.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/no_lambdamouthL/best.pt \
  > /home/jiweon/projects/ADLip2/log/B4.log 2>&1 &

tensorboard --logdir /home/jiweon/projects/ADLip2/results

### stage3 구조로 
nohup python train.py \
  --cfg configs/expD7.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/batch2_alpha0.35_sync0.15_20260502_214513/best.pt \
  > log/D7.log 2>&1 &

### inference
CUDA_VISIBLE_DEVICES=3 python scripts/inference_eval.py \
    --ref_dir /media/HDD/jiweon/processed/HDTF/HDTF_processed/RD_Radio1_000 \
    --ckpt results/D1_20260511_175012/best.pt \
    --cfg configs/expD1.yaml \
    --out inference/D1 \
    --device cuda:0

nohup python scripts/inference_eval.py \
  --ref_dir /media/HDD/jiweon/processed/HDTF/HDTF_processed/WDA_HillaryClinton_000 \
  --audio_dir1 /media/HDD/jiweon/processed/HDTF/HDTF_processed/RD_Radio5_000 \
  --audio_dir2 /media/HDD/jiweon/processed/HDTF/HDTF_processed/RD_Radio5_000 \
  --ckpt results/D1_20260511_175012/best.pt \
  --cfg configs/expD1.yaml \
  --out inference/D1_radio5_hillary \
  --device cuda:0 \
  > inference_D1_radio5_hillary.out 2>&1 &

## expE1
nohup python train.py \
  --cfg configs/expE1.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/no_lambdamouthL/best.pt \
  > /home/jiweon/projects/ADLip2/log/E1.log 2>&1 &
  
## expE2
nohup python train.py \
  --cfg configs/expE2.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/gan/best.pt \
  > /home/jiweon/projects/ADLip2/log/E2.log 2>&1 &

#############
tensorboard --logdir /home/jiweon/projects/ADLip2/results


## expF1,2,4,5
nohup python train.py \
  --cfg configs/expF5.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/D3_20260513_150330/best.pt \
  > /home/jiweon/projects/ADLip2/log/F5.log 2>&1 &

# F3
nohup python train.py \
  --cfg configs/expF3.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/F1_20260518_155648/best.pt \
  > /home/jiweon/projects/ADLip2/log/F3.log 2>&1 &

## expD8
nohup python train.py \
  --cfg configs/expD8.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/batch2_alpha0.35_sync0.15_20260502_214513/best.pt \
  > log/D8.log 2>&1 &


## expG1
nohup python train_ihj.py \
  --cfg configs/expG1.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/stage3_b8_gan_best.pt \
  > log/G1.log 2>&1 &

## D9, for 입술 튀는 현상 방지, same 올리기 
nohup python train.py \
  --cfg configs/expD9.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/D3_20260513_150330/best.pt \
  > /home/jiweon/projects/ADLip2/log/D9.log 2>&1 &


# 실행
python scripts/inference_eval.py \
    --ref_dir /media/HDD/jiweon/processed/HDTF/HDTF_processed/RD_Radio1_000 \
    --ckpt    /home/jiweon/projects/ADLip2/results/G1_20260520_180628/best.pt \
    --cfg     configs/expG1.yaml \
    --out     inference/G1 \
    --device  cuda:0

### G3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup torchrun --nproc_per_node=3 train_ihj.py \
  --cfg configs/expG3.yaml \
  --ckpt /home/jiweon/projects/ADLip2/results/G2_best.pt \
  > log/G3.log 2>&1 &

##########################
# LSE
cd /home/jiweon/projects/syncnet_python
conda activate syncnet

for EXP in F4 F5 G1; do
    for video in /home/jiweon/projects/ADLip2/inference/${EXP}/expB_*.mp4; do
        name=$(basename "$video" .mp4)
        ref="${EXP}_${name}"
        face_video="eval/tmp/${ref}_face.mp4"

        ffmpeg -y -loglevel error -i "$video" -vf "crop=256:256:256:0" "$face_video"

        TMP_DIR="eval/tmp/${ref}"
        mkdir -p "$TMP_DIR"

        python run_pipeline.py --videofile "$face_video" --reference "$ref" --data_dir "$TMP_DIR" > /dev/null 2>&1

        sync_out=$(python run_syncnet.py --videofile "$face_video" --reference "$ref" --data_dir "$TMP_DIR" 2>&1)
        lse_c=$(echo "$sync_out" | grep "Confidence:" | tail -n 1 | awk '{print $2}')
        lse_d=$(echo "$sync_out" | grep "Min dist:" | tail -n 1 | awk '{print $3}')
        echo "${EXP},${name},${lse_c},${lse_d}" | tee -a /home/jiweon/projects/ADLip2/eval_results/lse_results_ADLip2.csv
    done
done