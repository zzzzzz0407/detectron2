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
######################################################



# process.
######################################################
CONFIG_FILE=${CURDIR}/configs/SOD_R_50_FPN_1x.yaml
GPU_NUM=2
OUTPUT_DIR=${CURDIR}/models/test_for_fpn


CUDA_VISIBLE_DEVICES=1,3 python ${CURDIR}/train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR} SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005

# python3 ${CURDIR}/zip_uploading.py --local-dir ${OUTPUT_DIR} --dst-dir ${HDFS_DIR}