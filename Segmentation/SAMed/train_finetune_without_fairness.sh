#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Scale/race/total_15000
BATCH_SIZE=48
CENTER_CROP_SIZE=512
NUM_CLASS=2
MAX_EPOCHS=50
STOP_EPOCH=50
NUM_GPU=2
MIX_RATIO=0.1
# LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/race/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/language/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/ethnicity/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth)


LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/equal_scale_out/train_outputs/syn_15000/race/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_99.pth )
# ATTRIBUTE=( race language ethnicity gender)

ATTRIBUTE=( race )
for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=5,6 python train.py \
	--root_path ${BASE_DIR} \
	--output ./out/equal_scale_out/finetune_outputs/syn_15000/${ATTRIBUTE[$j]} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--base_lr 0.0005 \
	--AdamW \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE} \
	--n_gpu ${NUM_GPU} \
	--lora_ckpt ${LORA_CKPT[$j]} \
	--attribute ${ATTRIBUTE[$j]} \
	--mix_ratio ${MIX_RATIO}
done
