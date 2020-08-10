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

# process.
######################################################
URL=tcp://127.0.0.1:50002
CONFIG_FILE=${CURDIR}/configs/zhang/crowdhuman_baseline.yaml
GPU_NUM=2
OUTPUT_DIR=${CURDIR}/checkpoints/debug
HDFS_DIR=${HDFS_ROOT}/models/detr/debug

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --dist-url ${URL} \
--config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR} SOLVER.IMS_PER_BATCH 4

# python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}
