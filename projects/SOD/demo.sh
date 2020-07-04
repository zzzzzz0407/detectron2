#!/usr/bin/env bash

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou.yaml
MODEL_WEIGHTS=/data00/home/zhangrufeng1/models/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou/model_final.pth
INPUT_DIR=/data00/home/zhangrufeng1/datasets/coco/val2017/000000000885.jpg
OUTPUT_DIR=/data00/home/zhangrufeng1/vis/sod/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou

python ${CURDIR}/demo/demo.py --config-file ${CONFIG_FILE} \
--input ${INPUT_DIR} \
--output ${OUTPUT_DIR} \
--opts MODEL.WEIGHTS ${MODEL_WEIGHTS}
