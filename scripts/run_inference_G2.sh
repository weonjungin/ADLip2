#!/bin/bash

CKPT=/home/jiweon/projects/ADLip2/results/G2_best.pt
CFG=/home/jiweon/projects/ADLip2/configs/expG2.yaml
AUDIO=/home/ihjung/2026/ADLip2/eng_audio
DATA=/media/HDD/jiweon/processed/HDTF/HDTF_processed
OUT_ROOT=/media/HDD/jiweon/inference/G2
DEVICE=cuda:0

VIDEOS=(
RD_Radio52_000 RD_Radio49_000 WDA_LloydDoggett0_000 WDA_NancyPelosi0_000
WRA_TomCotton_000 WDA_MikeDoyle_000 WDA_KatieHill_000 RD_Radio59_000
RD_Radio47_000 WRA_LynnJenkins_000 WDA_XavierBecerra_002 WRA_BobGoodlatte0_003
RD_Radio30_000 WRA_JohnBoehner0_000 WDA_JohnSarbanes1_000 WRA_DebFischer1_000
WDA_NancyPelosi3_000 WRA_GregWalden1_002 WRA_LisaMurkowski0_000 WRA_CarlyFiorina0_000
WRA_JimRisch_000 WDA_JoeCrowley1_000 WDA_FrankPallone1_000 WDA_JoeCrowley1_002
WRA_MittRomney_000 WDA_ChrisCoons_000 WRA_RandPaul0_000 WDA_BobMenendez_000
WRA_KayBaileyHutchison_000 WDA_StenyHoyer_000 WDA_DonnaShalala1_000 WRA_VickyHartzler_000
RD_Radio2_000 WDA_MikeThompson0_000 WDA_AndyKim_000 WDA_ColinAllred_000
WRA_DaveCamp_000 WRA_MarkwayneMullin_001 WDA_ChrisVanHollen0_000
RD_Radio5_000 RD_Radio27_000 WDA_HillaryClinton_000
)

for VIDEO in "${VIDEOS[@]}"; do
    REF_DIR=$DATA/$VIDEO
    OUT_DIR=$OUT_ROOT/$VIDEO
    if [ ! -d "$REF_DIR" ]; then
        echo "SKIP (not found): $VIDEO"
        continue
    fi
    echo "Processing: $VIDEO"
    python scripts/inference_eval.py \
        --ref_dir $REF_DIR \
        --ckpt $CKPT \
        --cfg $CFG \
        --audio_dir1 $AUDIO \
        --out $OUT_DIR \
        --device $DEVICE
    echo "Done: $VIDEO"
done

echo "All done."
