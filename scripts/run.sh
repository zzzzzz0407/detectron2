#!/usr/bin/env bash

python ./dcgan.py --manualSeed 666 \
--dataset mnist --dataroot /data00/home/zhangrufeng1/datasets/debug \
--workers 4 --batchSize 128 --imageSize 64 --nz 100 --ngf 64 --ndf 64 --niter 100 \
--lr 0.0002 --beta1 0.5 --cuda --ngpu 1 --outf models/mnist_dcgan_epoch_100