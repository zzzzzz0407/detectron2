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
