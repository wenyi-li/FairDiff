#!/bin/bash
ATTR_TYPE="race"
NAME="Black"

CHECKPOINT="/path_to_ckpt"
SAVE_PATH="/path_to_save_images"


CUDA_VISIBLE_DEVICES=0 python infer.py  \
    --attr_type ${ATTR_TYPE} --name ${NAME} \
    --ckpt ${CHECKPOINT} \
    --images 10000 \
    --save_path ${SAVE_PATH}
