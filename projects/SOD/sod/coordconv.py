import torch
import math


def CoordConv(feat_in):
    # coord conv.
    x_range = torch.linspace(-1, 1, feat_in.shape[-1], device=feat_in.device)  # -1, 1编码.
    y_range = torch.linspace(-1, 1, feat_in.shape[-2], device=feat_in.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([feat_in.shape[0], 1, -1, -1])
    x = x.expand([feat_in.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    feat_out = torch.cat([feat_in, coord_feat], 1)
    return feat_out


@torch.no_grad()
def center_prior(num_loc, device):

    if isinstance(num_loc, int):
        num_x = int(math.sqrt(num_loc))
        num_y = num_x
        step_x = 0.5 * 1/num_x
        step_y = 0.5 * 1/num_y
    else:
        raise NotImplementedError

    x_range = torch.linspace(step_x, 1 - step_x, num_x, device=device)
    y_range = torch.linspace(step_y, 1 - step_y, num_y, device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y, x = y.contiguous().view(-1, 1), x.contiguous().view(-1, 1)
    center = torch.cat([x, y], 1)  # [x, y]
    return center
