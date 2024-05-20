#!/bin/bash

BASE_DIR=/path/to/your/datasets/MedicalImage/data
LIST_DIR=/path/to/your/FairDiff/SAMed/lists/Equal_Distribution/language/000

CENTER_CROP_SIZE=512
NUM_CLASS=2
EXP_NAME=repeat
EPOCH=( 59 69 79 89 99 )
ATTRIBUTE=ethnicity

LORA_CKPT=()
for epoch in 59 69 79 89 99; do
    LORA_CKPT+=( "/path/to/your/FairDiff/SAMed/out/repeat_sample_out/train_outputs/${ATTRIBUTE}_1/FairSeg_512_pretrain_vit_b_epo100_bs48_lr0.005/epoch_${epoch}.pth" )
done

for (( j=0; j<${#EPOCH[@]}; j++ )); do
    CUDA_VISIBLE_DEVICES=4 python test.py \
        --datadir ${BASE_DIR} \
        --output ./out/repeat_sample_out/test_outputs/${EXP_NAME}_epoch_${EPOCH[$j]}/${ATTRIBUTE} \
        --list_dir ${LIST_DIR} \
        --center_crop_size ${CENTER_CROP_SIZE} \
        --lora_ckpt ${LORA_CKPT[$j]} \
        --attribute ${ATTRIBUTE} \
        --exp_name ${EXP_NAME} \
        --epoch ${EPOCH[$j]}
done
