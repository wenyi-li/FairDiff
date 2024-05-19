#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Scale/race/total_15000
# LIST_DIR=./lists/FairSeg_generate


# LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/race/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/language/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/finetune_outputs/real/ethnicity/FairSeg_512_pretrain_vit_b_epo10_bs24_lr0.0005/epoch_4.pth )
LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/out/equal_scale_out/finetune_outputs/syn_15000/race/FairSeg_512_pretrain_vit_b_epo50_bs48_lr0.0005/epoch_39.pth ) 

CENTER_CROP_SIZE=512
NUM_CLASS=2

# ATTRIBUTE=( race gender language ethnicity )
ATTRIBUTE=race

for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=5 python test.py \
	--datadir ${BASE_DIR} \
	--output ./out/equal_scale_out/test_outputs/additional/${ATTRIBUTE[$j]}/39 \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--lora_ckpt ${LORA_CKPT[$j]} \
	--attribute ${ATTRIBUTE[$j]}
done