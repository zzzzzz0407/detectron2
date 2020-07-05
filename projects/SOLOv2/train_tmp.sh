CONFIG_FILE=configs/SOLOv2_R50_800_3x.yaml
GPU_NUM=8
OUTPUT_DIR=models/solov2_r50_3x_rerun

python3 \
train_net.py --config-file ${CONFIG_FILE} \
--num-gpus ${GPU_NUM} OUTPUT_DIR ${OUTPUT_DIR}

