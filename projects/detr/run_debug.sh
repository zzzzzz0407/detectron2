#!/usr/bin/env bash

# overall.
######################################################
CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}

# build if needed.
# echo 'Start building.'
# cd ${CURDIR}/../../
# sudo python3 ./setup.py build develop
# cd ${CURDIR}
# echo 'Finish building.'

# pip install if needed.
######################################################

# data.
######################################################
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
mkdir -p ${CURDIR}/datasets/crowdhuman
mkdir -p ${CURDIR}/checkpoints
echo 'Start Downloading Data.'
hadoop fs -get ${HDFS_ROOT}/pretrained/detr/torchvision-R-50_double_stems.pkl ${CURDIR}/checkpoints
hadoop fs -get ${HDFS_ROOT}/datasets/crowdhuman/annotations ${CURDIR}/datasets/crowdhuman
hadoop fs -get ${HDFS_ROOT}/datasets/crowdhuman/CrowdHuman_val.zip ${CURDIR}/datasets/crowdhuman
hadoop fs -get ${HDFS_ROOT}/datasets/crowdhuman/CrowdHuman_train.zip ${CURDIR}/datasets/crowdhuman
echo 'Finish Downloading Data.'
######################################################

# process.
######################################################
URL=tcp://127.0.0.1:50001
CONFIG_FILE=${CURDIR}/configs/zhang/debug_arnold.yaml
GPU_NUM=2
OUTPUT_DIR=${CURDIR}/checkpoints/debug_arnold
HDFS_DIR=${HDFS_ROOT}/models/detr/debug_arnold

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}

python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}
