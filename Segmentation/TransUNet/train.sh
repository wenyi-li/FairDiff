#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
# OUTPUT_DIR=./model/FairSeg_Output
# LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Distribution/abs/000
EXP_NAME=412
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/ER_ratio/race/${EXP_NAME}
BATCH_SIZE=48
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=300
# STOP_EPOCH=130
NUM_GPU=2
ATTRIBUTE=race
# for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
# do
# for (( i=0; i<10; i++ ));
# do
CUDA_VISIBLE_DEVICES=3,4 python train.py \
	--root_path ${BASE_DIR} \
	--output ./out/ER_ratio/train_outputs/${ATTRIBUTE}/${EXP_NAME} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--n_gpu ${NUM_GPU} \
	--attribute ${ATTRIBUTE} \
	
# done
# done