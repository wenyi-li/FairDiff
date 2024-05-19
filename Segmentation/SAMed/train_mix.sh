#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Distribution/gender/025
BATCH_SIZE=48
CENTER_CROP_SIZE=512
NUM_CLASS=2
MAX_EPOCHS=100
STOP_EPOCH=100
NUM_GPU=2
MIX_RATIO=0.50
# ATTRIBUTE=( race gender language ethnicity )
ATTRIBUTE=( gender )
for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=6,7 python train.py \
	--root_path ${BASE_DIR} \
	--output ./out/mix_ratio_out/train_outputs/syn_025/${ATTRIBUTE[$j]} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--warmup \
	--AdamW \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--n_gpu ${NUM_GPU} \
	--mix_ratio ${MIX_RATIO} \
	--attribute ${ATTRIBUTE[$j]}
done
