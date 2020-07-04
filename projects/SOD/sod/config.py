# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_sod_config(cfg):
    """
    Add config for SOD.
    """
    cfg.MODEL.SOD = CN()

    # ---------------------------------------------------------------------------- #
    # General Configs
    # ---------------------------------------------------------------------------- #
    # Number of foreground classes
    cfg.MODEL.SOD.NUM_CLASSES = 80
    cfg.MODEL.SOD.NUM_KERNELS = 256
    cfg.MODEL.SOD.TOP_LEVELS = 2
    cfg.MODEL.SOD.NORM = "GN"
    cfg.MODEL.SOD.USE_BASE = False
    cfg.MODEL.SOD.PE_ON = False
    cfg.MODEL.SOD.WITH_COORD = False
    cfg.MODEL.SOD.CENTER_SYMMETRY = False
    cfg.MODEL.SOD.SCALE_VARIANCE = False
    cfg.MODEL.SOD.SCALE_NORMALIZE = False
    cfg.MODEL.SOD.LOC_PRIOR = False
    cfg.MODEL.SOD.CIOU_ON = False
    cfg.MODEL.SOD.RESCALE_FIRST = True
    cfg.MODEL.SOD.MAX_POOL = False
    cfg.MODEL.SOD.CE_LOSS_COEFF = 1.0
    cfg.MODEL.SOD.BBOX_LOSS_COEFF = 5.0
    cfg.MODEL.SOD.GIOU_LOSS_COEFF = 2.0
    cfg.MODEL.SOD.EOS_COEFF = 0.1
    cfg.MODEL.SOD.SCALE_COEFF = 0.1
    cfg.MODEL.SOD.PRIOR_PROB = 0.05

    # ---------------------------------------------------------------------------- #
    # Instance Configs
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.SOD.INSTANCE_IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.SOD.FPN_INSTANCE_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.SOD.NUM_GRIDS = [5, 5, 5, 5, 5]  # it should be same.
    cfg.MODEL.SOD.SIZES_OF_INTEREST = [64, 128, 256, 512]
    # Channel size for the instance head.
    cfg.MODEL.SOD.INSTANCE_IN_CHANNELS = 256
    cfg.MODEL.SOD.INSTANCE_CHANNELS = 256
    # Convolutions to use (before query-key) in the instance head.
    cfg.MODEL.SOD.NUM_INSTANCE_CONVS_BEFORE = 4
    # Convolutions to use (after query-key) in the instance head.
    cfg.MODEL.SOD.NUM_INSTANCE_CONVS_AFTER = 1
    # The number of dense2spare operations.
    cfg.MODEL.SOD.NUM_ATTENTION = 4
    # FC to use for feature extraction.
    cfg.MODEL.SOD.FC_DIM = [1024, 1024]
    # attention type. / now support softmax/sigmoid.
    cfg.MODEL.SOD.TYPE_ATTENTION = "softmax"

    # ---------------------------------------------------------------------------- #
    # Base Configs
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.SOD.NUM_BASE_CONVS = 3
    cfg.MODEL.SOD.BASE_CHANNELS = 256

    # ---------------------------------------------------------------------------- #
    # Test Configs
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.SOD.MAX_PER_IMG = 100
    cfg.MODEL.SOD.SCORE_THR = -1
