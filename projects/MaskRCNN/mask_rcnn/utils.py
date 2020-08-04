# coding:utf-8
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            # state size. 1 x 1 x 1.
            # nn.Sigmoid()
        )
        self._freeze_params()

    def _freeze_params(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1)
        return x