# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_solov2_config(cfg):
    """
    Add config for SOLOv2.
    """
    cfg.MODEL.SOLOV2 = CN()

    # Instance parameters
    cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
    cfg.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
    cfg.MODEL.SOLOV2.SIGMA = 0.2
    # Channel size for the instance head.
    cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
    cfg.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
    # Convolutions to use in the instance head.
    cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
    cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
    cfg.MODEL.SOLOV2.TYPE_DCN = 'DCN'
    cfg.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
    # Number of foreground classes
    cfg.MODEL.SOLOV2.NUM_CLASSES = 80
    cfg.MODEL.SOLOV2.NUM_KERNELS = 256
    cfg.MODEL.SOLOV2.NORM = "GN"
    cfg.MODEL.SOLOV2.USE_COORD_CONV = True
    cfg.MODEL.SOLOV2.PRIOR_PROB = 0.01

    # Mask parameters
    # Channel size for the mask tower
    cfg.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
    cfg.MODEL.SOLOV2.MASK_CHANNELS = 128
    cfg.MODEL.SOLOV2.NUM_MASKS = 256

    # Test cfg.
    cfg.MODEL.SOLOV2.NMS_PRE = 500
    cfg.MODEL.SOLOV2.SCORE_THR = 0.1
    cfg.MODEL.SOLOV2.UPDATE_THR = 0.05
    cfg.MODEL.SOLOV2.MASK_THR = 0.5
    cfg.MODEL.SOLOV2.MAX_PER_IMG = 100
    # matrix / mask
    cfg.MODEL.SOLOV2.NMS_TYPE = "matrix"
    # gaussian / linear
    cfg.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
    cfg.MODEL.SOLOV2.NMS_SIGMA = 2

    cfg.MODEL.SOLOV2.LOSS = CN()
    cfg.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
    cfg.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
    cfg.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
    cfg.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
    cfg.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0

    # Optional Params.
    cfg.MODEL.SOLOV2.FLAG_SEMI = False
    cfg.MODEL.SOLOV2.RATIO_SEMI = 0.1
