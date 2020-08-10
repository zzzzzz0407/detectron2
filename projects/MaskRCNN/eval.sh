#!/usr/bin/env bash

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}
######################################################

# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/zhang/faster_rcnn_R_50_FPN_1x.yaml
GPU_NUM=4
OUTPUT_DIR=${CURDIR}/models/faster_rcnn_R_50_FPN_1x_train
WEIGHTS=/data00/home/zhangrufeng1/pretrained/detectron2/faster_rcnn/faster_rcnn_R_50_FPN_1x.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${CURDIR}/train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} --eval-only MODEL.WEIGHTS ${WEIGHTS} OUTPUT_DIR ${OUTPUT_DIR}
