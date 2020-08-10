# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from .MaskLoader import MaskLoader
from .gan_1 import Generator_1, Discriminator_1
from .utils import HingeLoss

generator_cfg = {
    'Generator_1': Generator_1,
}

discriminator_cfg = {
    'Discriminator_1': Discriminator_1,
}

loss_cfg = {
    'Hinge_Loss': HingeLoss(),
    'BCE_Loss': nn.BCEWithLogitsLoss(),
}


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
