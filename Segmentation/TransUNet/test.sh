#!/bin/bash

BASE_DIR="/path/to/your/datasets/MedicalImage/data"
LIST_DIR=/path/to/your/FairSegDiff/SAMed/lists/Equal_Distribution/language/000

CENTER_CROP_SIZE=512
NUM_CLASS=2
BATCH_SIZE=24
EXP_NAME=syn_8000_additional
EPOCH=( 209 219 229 239 249 259 269 279 289 299 )
ATTRIBUTE=language
NUM_CLASS=3

LORA_CKPT=()
for epoch in 209 219 229 239 249 259 269 279 289 299; do
    LORA_CKPT+=( "/path/to/your/FairSegDiff/TransUNet/out/equal_scale_out/train_outputs/additional/${ATTRIBUTE}/TU_FairSeg224/TU_pretrain_ViT-B_16_skip0_epo300_bs48_lr0.02_224_${ATTRIBUTE}/epoch_${epoch}.pth" )
done


for (( j=0; j<${#EPOCH[@]}; j++ )); do
    CUDA_VISIBLE_DEVICES=6 python test_batch.py \
        --datadir ${BASE_DIR} \
        --output /path/to/your/FairSegDiff/TransUNet/out/equal_scale_out/test_outputs/${ATTRIBUTE} \
        --list_dir ${LIST_DIR} \
        --center_crop_size ${CENTER_CROP_SIZE} \
        --lora_ckpt ${LORA_CKPT[$j]} \
        --attribute ${ATTRIBUTE} \
		--num_classes ${NUM_CLASS} \
		--batch_size ${BATCH_SIZE} \
        --exp_name ${EXP_NAME} \
        --epoch ${EPOCH[$j]}
done

