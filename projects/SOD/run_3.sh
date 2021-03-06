#!/usr/bin/env bash

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

# build if needed.
# hadoop fs -get ${HDFS_ROOT}/build/detectron2/build.tar.gz ${CURDIR}
# tar -xzvf ${CURDIR}/build.tar.gz
# echo 'Start building.'
# sudo python3 ${CURDIR}/setup.py build develop
# echo 'Finish building.'

# pip install if needed.
######################################################

# data.
######################################################
HDFS_ROOT=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/zhangrufeng/
mkdir -p ${CURDIR}/datasets/coco
mkdir -p ${CURDIR}/models
echo 'Start Downloading Data.'
hadoop fs -get ${HDFS_ROOT}/pretrained/detectron2/R-50.pkl ${CURDIR}/models
hadoop fs -get ${HDFS_ROOT}/datasets/coco/annotations ${CURDIR}/datasets/coco
hadoop fs -get ${HDFS_ROOT}/datasets/coco/val2017.zip ${CURDIR}/datasets/coco
hadoop fs -get ${HDFS_ROOT}/datasets/coco/train2017.zip ${CURDIR}/datasets/coco
echo 'Finish Downloading Data.'
######################################################

# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou_max_pool_rescale_last.yaml
GPU_NUM=8
OUTPUT_DIR=${CURDIR}/models/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou_max_pool_rescale_last
HDFS_DIR=${HDFS_ROOT}/models/sod/SOD_R_50_FPN_2x_no_base_pe_scale_variance_norm_loc_prior_ciou_max_pool_rescale_last

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}

python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}
