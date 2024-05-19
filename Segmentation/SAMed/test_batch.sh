#!/bin/bash

BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Distribution/language/000
# LIST_DIR="./lists/FairSeg_generate"

CENTER_CROP_SIZE=512
NUM_CLASS=2
EXP_NAME=repeat
EPOCH=( 59 69 79 89 99 )
ATTRIBUTE=ethnicity

LORA_CKPT=()
for epoch in 59 69 79 89 99; do
    LORA_CKPT+=( "/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/repeat_sample_out/train_outputs/${ATTRIBUTE}_1/FairSeg_512_pretrain_vit_b_epo100_bs48_lr0.005/epoch_${epoch}.pth" )
done

# /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/repeat_sample_out/train_outputs/gender/FairSeg_512_pretrain_vit_b_epo100_bs48_lr0.005/epoch_9.pth


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
