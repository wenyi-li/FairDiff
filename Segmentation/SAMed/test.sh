#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=./lists/Equal_Scale
# LIST_DIR=./lists/FairSeg_generate


# LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/race/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/language/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/ethnicity/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth )
LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/equal_scale_out/finetune_outputs/syn_075_epoch_99/race/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth ) 

CENTER_CROP_SIZE=512
NUM_CLASS=2
EXP_NAME=syn_075_epoch_99
EPOCH=( 59 69 79 89 99 )

# ATTRIBUTE=( race gender language ethnicity )
ATTRIBUTE=race

for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=5,6 python test.py \
	--datadir ${BASE_DIR} \
	--output ./equal_scale_out/test_outputs/syn_075_epoch_99/${ATTRIBUTE[$j]} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--lora_ckpt ${LORA_CKPT[$j]} \
	--attribute ${ATTRIBUTE[$j]} \
	--exp_name ${}\
	--epoch 
done