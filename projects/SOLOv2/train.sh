CONFIG_FILE=configs/SOLOv2_R50_800_1x.yaml
GPU_NUM=1
OUTPUT_DIR=models/solov2_r50_1x

/mnt/cephfs_new_wj/mlnlp/wangxinlong/anaconda2/envs/solo/bin/python \
train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}

