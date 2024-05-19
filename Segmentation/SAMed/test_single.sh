#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
OUTPUT_DIR=./test_outputs/
LIST_DIR=./lists/FairSeg_final
# LIST_DIR=./lists/FairSeg_generate

# LORA_CKPT=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/outputs/FairSeg_fair_loss_generate_110/race/FairSeg_512_pretrain_vit_b_epo10_bs20_lr0.0005/epoch_4.pth
# LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/finetune_outputs/real/race/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/finetune_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/finetune_outputs/real/language/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/finetune_outputs/real/ethnicity/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth )
LORA_CKPT=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/finetune_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth

CENTER_CROP_SIZE=512
NUM_CLASS=2
ATTRIBUTE=gender
# ATTRIBUTE=( race gender language ethnicity )


CUDA_VISIBLE_DEVICES=6,7 python test.py \
	--datadir ${BASE_DIR} \
	--output ./test_outputs/real/${ATTRIBUTE} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--lora_ckpt ${LORA_CKPT} \
	--attribute ${ATTRIBUTE}


