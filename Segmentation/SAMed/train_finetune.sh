#!/bin/bash
BASE_DIR=/DATA_EDS2/liwy/datasets/MedicalImage/data
LIST_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/lists/Equal_Distribution/abs/000
BATCH_SIZE=24
CENTER_CROP_SIZE=512
NUM_CLASS=2
MAX_EPOCHS=10
STOP_EPOCH=5
NUM_GPU=2
MIX_RATIO=0.25
# LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/race/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/language/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/ethnicity/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/train_outputs/real/gender/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_69.pth)


LORA_CKPT=( /DATA_EDS2/AIGC/2312/xuhr2312/workspace/FairSegDiff/SAMed/mix_ratio_out/transfered/train_outputs/syn_000/race/FairSeg_512_pretrain_vit_b_epo100_bs24_lr0.005/epoch_99.pth )
# ATTRIBUTE=( race language ethnicity gender)

ATTRIBUTE=( race )
for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=4,5 python train_finetune.py \
	--root_path ${BASE_DIR} \
	--output ./mix_ratio_out/transfered/finetune_outputs/syn_000_epoch_99/${ATTRIBUTE[$j]} \
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
