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
    _C.MODEL.FLAG_SEMI_ON_LOSS = False
    _C.MODEL.COEFF_SEMI = 0.1
    # add for mask head.
    _C.MODEL.ROI_MASK_HEAD.WITH_MASK_LOSS = True
    _C.MODEL.FREEZE_FASTER_RCNN = False
