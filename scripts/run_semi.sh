#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=3
python ./main.py --generator Generator_1 --discriminator Discriminator_1 --loss BCE_Loss \
--category person --outf models/person_bceloss_1000 \
--imageSize 64 --batchSize 256 --nz 256 --ngf 64 --ndf 64 \
--root datasets --dataset coco_2017_train \
--cuda --ngpu 2 --workers 8 --sample_interval 1000 \
--niter 1000 --lr 0.0002 --beta1 0.5
