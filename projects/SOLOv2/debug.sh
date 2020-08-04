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


# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/zhang/SOLOv2_R50_800_1x_edge.yaml
GPU_NUM=4
OUTPUT_DIR=${CURDIR}/models/SOLOv2_R50_800_1x_edge.yaml
HDFS_DIR=${HDFS_ROOT}/models/solov2/debug

hdfs dfs -mkdir -p ${HDFS_DIR}

python3 ${CURDIR}/train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}