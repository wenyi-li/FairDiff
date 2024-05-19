#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Scale/race/total_8000
BATCH_SIZE=48
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=200
STOP_EPOCH=130
NUM_GPU=2
ATTRIBUTE=race
CKPT=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/TransUNet/out/equal_scale_out/train_outputs/syn_8000/race/TU_FairSeg224/TU_pretrain_ViT-B_16_skip0_epo300_bs24_224_race/epoch_209.pth

CUDA_VISIBLE_DEVICES=3,4 python train_febs.py \
	--root_path ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--n_gpu ${NUM_GPU} \
	--attribute ${ATTRIBUTE} \
	--ckpt ${CKPT} 

