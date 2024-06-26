#!/bin/bash
BASE_DIR=/path/to/your/datasets/MedicalImage/data
LIST_DIR=/path/to/your/FairSegDiff/SAMed/lists/Equal_Scale/language/total_15000
BATCH_SIZE=42
CENTER_CROP_SIZE=512
NUM_CLASS=2
MAX_EPOCHS=200
STOP_EPOCH=160
NUM_GPU=2
# ATTRIBUTE=( race gender language ethnicity )
ATTRIBUTE=( language )
for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=4,5 python train.py \
	--root_path ${BASE_DIR} \
	--output ./out/supp_out/train_outputs/${ATTRIBUTE[$j]} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--warmup \
	--AdamW \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--n_gpu ${NUM_GPU} \
	--attribute ${ATTRIBUTE[$j]}
done