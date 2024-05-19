#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=lists/FairSeg_final
BATCH_SIZE=16
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=200
STOP_EPOCH=130
NUM_GPU=4
ATTRIBUTE=language
CKPT=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/model/TU_FairSeg224_FEBS_LOSS/TU_pretrain_ViT-B_16_skip0_epo5_bs16_224_attr_language/epoch_4.pth

CUDA_VISIBLE_DEVICES=0,2,4,5 python test.py \
	--datadir ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--attribute ${ATTRIBUTE} 
	
