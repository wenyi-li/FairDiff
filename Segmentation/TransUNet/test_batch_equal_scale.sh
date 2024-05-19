#!/bin/bash

BASE_DIR="/DATA_EDS2/liwy/datasets/MedicalImage/data"


CENTER_CROP_SIZE=512
NUM_CLASS=2
BATCH_SIZE=48
EXP_NAME=412
EPOCH=( 209 219 229 239 249 259 269 279 289 299 )
ATTRIBUTE=race
NUM_CLASS=3
EXP_TYPE=ER_ratio
# EXP_TYPE=mix_ratio_out

LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/ER_ratio/race/42


LORA_CKPT=()
for epoch in 209 219 229 239 249 259 269 279 289 299; do
    LORA_CKPT+=( "/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/TransUNet/out/ER_ratio/train_outputs/${ATTRIBUTE}/${EXP_NAME}/TU_FairSeg224/TU_pretrain_ViT-B_16_skip0_epo300_bs48_lr0.02_224_${ATTRIBUTE}/epoch_${epoch}.pth" )
done


for (( j=0; j<${#EPOCH[@]}; j++ )); do
    CUDA_VISIBLE_DEVICES=5 python test_batch.py \
        --datadir ${BASE_DIR} \
        --output /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/TransUNet/out/${EXP_NAME}/test_outputs/${ATTRIBUTE} \
        --list_dir ${LIST_DIR} \
        --center_crop_size ${CENTER_CROP_SIZE} \
        --lora_ckpt ${LORA_CKPT[$j]} \
        --attribute ${ATTRIBUTE} \
		--num_classes ${NUM_CLASS} \
		--batch_size ${BATCH_SIZE} \
        --exp_name ${EXP_NAME} \
        --epoch ${EPOCH[$j]}
done