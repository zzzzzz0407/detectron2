# coding:utf-8

import os
import argparse

from torchvision.utils import save_image
import random
import torch.backends.cudnn as cudnn

from semi_gan import *


Root_dir = './'


def parse_args():
    parser = argparse.ArgumentParser(description='as u wish.')
    # main.
    parser.add_argument("--generator", default='Generator_1', type=str)
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")

    parser.add_argument('--outf', default='./debug', help='folder to output images and model checkpoints')
    # dataset.
    parser.add_argument('--imageSize', default=64, type=int)
    parser.add_argument('--category', default='person')
    parser.add_argument('--niter', type=int, default=200)
    parser.add_argument('--batchSize', type=int, default=25, help='input batch size')
    parser.add_argument('--nc', type=int, default=1, help='size of the latent c vector')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--manualSeed', default=666, type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    store_path = os.path.join(Root_dir, args.outf, args.category, 'vis')
    if not os.path.isdir(store_path):
        os.makedirs(store_path)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # ---------------------------------------------------------------------------- #
    # Network.
    # ---------------------------------------------------------------------------- #
    # Generator.
    netG = generator_cfg[args.generator](args.nz, args.ngf, args.nc, args.ngpu).to(device)
    netG.apply(weights_init)
    netG.load_state_dict(torch.load(args.netG))
    netG.eval()
    print(netG)

    for i in range(args.niter):
        noise = torch.randn(args.batchSize, args.nz, 1, 1, device=device)
        fake_imgs = netG(noise)
        fake_imgs = (fake_imgs > args.thresh).float()
        save_image(fake_imgs.detach(), '%s/fake_samples_%s.png' % (store_path, i), nrow=5)
        print("Process [{}/{}]".format(i, args.niter))
