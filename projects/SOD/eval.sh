#!/usr/bin/env bash

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/SOD_R_50_FPN_2x_base_on_pe_on_loc_prior_on.yaml
GPU_NUM=2
OUTPUT_DIR=/data00/home/zhangrufeng1/models/SOD_R_50_FPN_2x_base_on_pe_on_loc_prior_on/model_final.pth

CUDA_VISIBLE_DEVICES=2,3 python ${CURDIR}/train_net.py --num-gpus ${GPU_NUM} --config-file ${CONFIG_FILE} \
--eval-only MODEL.WEIGHTS ${OUTPUT_DIR}