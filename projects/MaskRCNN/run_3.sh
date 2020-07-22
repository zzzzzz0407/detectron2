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
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
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
URL=tcp://127.0.0.1:50003
CONFIG_FILE=${CURDIR}/configs/zhang/mask_rcnn_R_50_FPN_1x_semi_0.05.yaml
GPU_NUM=4
OUTPUT_DIR=${CURDIR}/models/mask_rcnn_R_50_FPN_1x_semi_0.05
HDFS_DIR=${HDFS_ROOT}/models/detectron2/semi/mask_rcnn_R_50_FPN_1x_semi_0.05

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}

python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}
