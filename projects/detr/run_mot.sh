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
mkdir -p ${CURDIR}/datasets/mot/mot17
mkdir -p ${CURDIR}/checkpoints
echo 'Start Downloading Data.'
hadoop fs -get ${HDFS_ROOT}/pretrained/detr/crowdhuman_baseline.pth ${CURDIR}/checkpoints
hadoop fs -get ${HDFS_ROOT}/datasets/mot/mot17/annotations ${CURDIR}/datasets/mot/mot17
hadoop fs -get ${HDFS_ROOT}/datasets/mot/mot17/train.zip ${CURDIR}/datasets/mot/mot17
echo 'Finish Downloading Data.'
######################################################

# process.
######################################################
URL=tcp://127.0.0.1:50001
CONFIG_FILE=${CURDIR}/configs/zhang/mot17_half_track_baseline.yaml
GPU_NUM=4
OUTPUT_DIR=${CURDIR}/checkpoints/mot17_half_track_baseline
HDFS_DIR=${HDFS_ROOT}/models/detr/mot17_half_track_baseline
PRETRAINED=${CURDIR}/checkpoints/crowdhuman_baseline.pth

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR} \
MODEL.WEIGHTS ${PRETRAINED}

python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}
