# coding:utf-8

import os
import argparse

import torch.nn.functional as F
from torchvision.utils import save_image
import collections
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from semi_gan import *


Root_dir = './'


def parse_args():
    parser = argparse.ArgumentParser(description='as u wish.')
    # main.
    parser.add_argument("--generator", default='Generator_1', type=str)
    parser.add_argument("--discriminator", default='Discriminator_1', type=str)
    parser.add_argument("--loss", default='Hinge_Loss', type=str)
    parser.add_argument('--outf', default='./debug', help='folder to output images and model checkpoints')
    # dataset.
    parser.add_argument('--root', default='datasets', type=str)
    parser.add_argument('--dataset', default='coco_2017_val', type=str)
    parser.add_argument('--imageSize', default=64, type=int)
    parser.add_argument('--category', default='person')
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    # others.
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nc', type=int, default=1, help='size of the latent c vector')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--manualSeed', default=666, type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    store_path = os.path.join(Root_dir, args.outf, args.category)
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
        os.makedirs(os.path.join(store_path, 'images'))
    logger_writer = os.path.join(store_path, 'log.txt')
    logger = open(logger_writer, 'w')

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # ---------------------------------------------------------------------------- #
    # Prepare data.
    # ---------------------------------------------------------------------------- #
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find("detectron2") + len("detectron2")]
    dataset_root = os.path.join(root_path, args.root)
    mask_data = MaskLoader(root=dataset_root, dataset=args.dataset,
                           size=args.imageSize, category=args.category)
    mask_loader = DataLoader(mask_data, batch_size=args.batchSize, shuffle=True, **kwargs)

    # ---------------------------------------------------------------------------- #
    # Network.
    # ---------------------------------------------------------------------------- #
    # Generator.
    netG = generator_cfg[args.generator](args.nz, args.ngf, args.nc, args.ngpu).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print(netG)

    # Discriminator.
    netD = discriminator_cfg[args.discriminator](args.ndf, args.nc, args.ngpu).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print(netD)

    # ---------------------------------------------------------------------------- #
    # Optimizer & Losses & Others.
    # ---------------------------------------------------------------------------- #
    criterion = loss_cfg[args.loss]

    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(args.batchSize, args.nz, 1, 1, device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    if args.dry_run:
        args.niter = 1

    batches_done = 0
    for epoch in range(args.niter):
        for i, data in enumerate(mask_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizerD.zero_grad()
            # train with real.
            real_imgs = data.to(device)
            batch_size = real_imgs.size(0)
            real_validity = netD(real_imgs)

            # train with fake.
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            fake_validity = netD(fake_imgs.detach())

            if args.loss == 'Hinge_Loss':  # loss is too small, ~0.
                errD_real = criterion(real_validity, flag='real')
                errD_fake = criterion(fake_validity, flag='fake')
                errD = errD_real + errD_fake
            elif args.loss == 'BCE_Loss':
                real_labels = torch.full((batch_size,), real_label, dtype=real_imgs.dtype, device=device)
                errD_real = criterion(real_validity, real_labels)
                fake_labels = torch.full((batch_size,), fake_label, dtype=fake_imgs.dtype, device=device)
                errD_fake = criterion(fake_validity, fake_labels)
                errD = errD_real + errD_fake

            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            optimizerG.zero_grad()
            if i % args.n_critic == 0:
                fake_imgs = netG(noise)
                fake_validity = netD(fake_imgs)
                if args.loss == 'Hinge_Loss':
                    errG = -fake_validity.mean()
                else:
                    labels = torch.full((batch_size,), real_label, dtype=fake_imgs.dtype, device=device)
                    errG = criterion(fake_validity, labels)

                errG.backward()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %
                      (epoch, args.niter, i, len(mask_loader), errD.item(), errG.item()))
                logger.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f\n' %
                             (epoch, args.niter, i, len(mask_loader), errD.item(), errG.item()))

                if i % args.sample_interval == 0:
                    real_imgs = (real_imgs > args.thresh).float()
                    save_image(real_imgs.detach(), '%s/images/real_samples.png' % store_path)
                    fake_imgs = netG(fixed_noise)
                    fake_imgs = (fake_imgs > args.thresh).float()
                    save_image(fake_imgs.detach(), '%s/images/fake_samples_%s.png' % (store_path, batches_done))

                batches_done += args.n_critic

            if args.dry_run:
                break

        # save models.
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (store_path, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (store_path, epoch))

    logger.close()

