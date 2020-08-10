# coding: utf-8
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input, flag='real'):
        if flag == 'real':
            loss = self.relu(1.0 - input).mean()  # --> 1.
        elif flag == 'fake':
            loss = self.relu(1.0 + input).mean()  # --> -1.
        else:
            raise NotImplementedError
        return loss
