# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def add_extra_config(cfg):
    """
    Add config for extra operation.
    """
    _C = cfg

    # add for mobilenet backbone
    _C.MODEL.MOBILENETV3 = False
    # add for semi params.
    _C.MODEL.FLAG_SEMI = False
    _C.MODEL.FLAG_GAP = False
    _C.MODEL.FLAG_SEMI_ON_LOSS = False  # Mask rcnn 的loss * semi.
    _C.MODEL.COEFF_SEMI = 0.1
    # add for mask head.
    _C.MODEL.ROI_MASK_HEAD.WITH_MASK_LOSS = True  # mask loss = 0 --> faster rcnn.
    _C.MODEL.FREEZE_FASTER_RCNN = False  # freeze faster rcnn.
    _C.MODEL.WITH_GAN = False
    _C.MODEL.COEFF_GAN = 0.1
    _C.MODEL.WEIGHTS_GAN = ""
