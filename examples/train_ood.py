"""
The project is developed based on "CompressAI" Library
https://github.com/InterDigitalInc/CompressAI

"""

import argparse
import math
import random
import shutil
import sys
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import autograd

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import msssim, ssim, msssim_norm
from tone_mapping import tone_mapping, inverse_mapping
import logging
from hdr_dataset import ImageFolder_HDR, ImageFolder_OOD
from pu_tran import Lu_trans, Pu_trans
from tmo import TMO, ITMO


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, sdr_w=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.sdr_w = sdr_w
        self.hdr_w = 1 - sdr_w
        
    def forward(self, epoch, output, batch_size, env_sdr, env_hdr, penalty_weight, regu_weight):
        _, _, H, W = env_hdr.size()
        out = {}
        num_pixels = batch_size * 8 * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        re = output["x_hat"].clamp_(0,1)

        N_sdr = int(batch_size)*7
        N_hdr = int(batch_size)

        re_sdr = re[0:N_sdr,:,:,:]
        re_hdr_mapped = re[N_sdr:N_sdr+N_hdr,:,:,:]

        re_log = inverse_mapping(re_hdr_mapped*255)
        re_hdr = 10**re_log

        distortion1 = 1 - msssim(255*env_sdr, 255*re_sdr)
        penalty1 = penalty(255*re_sdr, 255*env_sdr, flag_hdr=False)

        distortion2 = self.l1(re_hdr, env_hdr)

        penalty2 = penalty(env_hdr, re_hdr, flag_hdr=True)

        train_nll = self.sdr_w*distortion1 + self.hdr_w*distortion2
        train_penalty = self.sdr_w*penalty1 + self.hdr_w*penalty2

        out["distortion_loss"] = self.lmbda * train_nll + penalty_weight * train_penalty + regu_weight
        out["loss"] = out["distortion_loss"] + out["bpp_loss"]

        out["sdr_msssim"] = 1 - distortion1
        out["hdr_mae"] = distortion2

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def penalty(re_img, img, flag_hdr):
    scale = torch.tensor(1.).cuda().requires_grad_()
    if flag_hdr:
        l1 = nn.L1Loss()
        loss = l1(img, re_img * scale)
    else:
        loss = 1 - msssim(img, re_img * scale)

    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger, l2_regularizer_weight, penalty_weight
    ):
    model.train()
    device = next(model.parameters()).device

    for i, (hdr, sdr, _) in enumerate(train_dataloader):
        hdr = hdr.to(device)
        sdr = sdr.to(device)

        N, B, C, H, W = hdr.size()

        hdr = hdr.view(int(N*B),C,H,W)
        sdr = sdr.view(int(N*B*7),C,H,W)/255

        hdr_log = torch.log10(hdr).clamp_(-5, 5)
        mapped_hdr = tone_mapping(hdr_log)/255

        img = torch.cat([sdr,mapped_hdr], dim=0)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.autograd.set_detect_anomaly(True):

            out_net = model(img)

            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            regu_weight = l2_regularizer_weight * weight_norm

            out_criterion = criterion(epoch, out_net, N*B, sdr, hdr, penalty_weight, regu_weight)
            out_criterion["loss"].backward()
            
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 50 == 0:
            logger.info(
                f"Train epoch {epoch}: "
                f" [{100. * i / len(train_dataloader):.0f}%]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tSDR MSSSIM: {out_criterion["sdr_msssim"].item():.3f} |'
                f'\tHDR MAE: {out_criterion["hdr_mae"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f}'
            )


def test_epoch(epoch, sdr_valid_dataloader, hdr_valid_dataloader, model, criterion, logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_sdr = AverageMeter()
    bpp_hdr = AverageMeter()
    sdr_msssim = AverageMeter()
    hdr_msssim = AverageMeter()

    with torch.no_grad():
        for d in sdr_valid_dataloader:
            d = d.to(device)
            out_net = model(d)

            N, _, H, W = d.size()
            num_pixels = N * H * W

            out_criterion = {}
            out_criterion["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )

            out_criterion["sdr_msssim"] = msssim(255*d, 255*out_net["x_hat"].clamp_(0,255))

            bpp_sdr.update(out_criterion["bpp_loss"])
            sdr_msssim.update(out_criterion["sdr_msssim"])
            loss.update(out_criterion["sdr_msssim"])


        for (img, _) in hdr_valid_dataloader:
            
            img = img.to(device)

            img_log = torch.log10(img).clamp_(-5, 5)
            pu = Pu_trans(img_log)
            d, bins, s, v = TMO(pu,20)
            d = d/255 # [0 1]

            out_net = model(d)
            re = out_net["x_hat"].clamp_(0,1)

            re_Pu = ITMO(255*re, bins, s, v)
            re_log = Lu_trans(re_Pu)
            re_img = 10**re_log

            N, _, H, W = d.size()
            num_pixels = N * H * W

            # Bit for Image Coding
            bpp_loss = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )

            # Bit for TMO/ITMO
            ex_bit = 8 * (bins.nbytes + s.nbytes + v.nbytes)
            
            BPP = bpp_loss + ex_bit / (H * W)
            bpp_hdr.update(BPP)

            MSSSIM = msssim(img, re_img)
            hdr_msssim.update(MSSSIM)
            loss.update(MSSSIM)


    logger.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tSDR MSSSIM: {sdr_msssim.avg:.3f}, Bpp: {bpp_sdr.avg:.3f} |"
        f"\tHDR MSSSIM: {hdr_msssim.avg:.3f}, Bpp: {bpp_hdr.avg:.3f} |"
    )

    return loss.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cheng2020-attn",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--gpu",
        default='all',
        help="select GPU for train",
    )
    parser.add_argument('--rw',default=0.0001,type=float, help="l2_regularizer_weight")
    parser.add_argument('--pw',default=1,type=float, help="penalty_weight")
    parser.add_argument('--sdr_w',default=0.5,type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--num_workers',default=8,type=int) 
    
    args = parser.parse_args(argv)
    
    return args


def main(argv):
    args = parse_args(argv)

    head = time.strftime("%m%d%H%M%S", time.localtime()) 

    log_dir = './log/train/' + 'ood_' + str(args.lmbda) + '_' + head +'/'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_dir+'log.txt',
                        )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    logger.info('---------------------------------train ood---------------------------------')
    logger.info('| L2 Regularizer Weight: {} | Penalty Weight: {} | SDR Weight: {} |'\
        .format(args.rw, args.pw, args.sdr_w))
    logger.info('---------------------------------------------------------------------------')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.gpu != 'all':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print('selected device:'+args.gpu)

    device = "cuda"

    sdr_train_root = '/sdr train sets'
    hdr_train_root = '/hdr train sets'

    sdr_valid_root = '/sdr valid sets'
    hdr_valid_root = '/sdr valid sets'

    train_dataset = ImageFolder_OOD(hdr_root=hdr_train_root, 
                            sdr_root=sdr_train_root,
                            patch_size=args.patch_size[0],
                            train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size/8),
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    sdr_valid_transforms = transforms.Compose(
        [transforms.ToTensor()])
    sdr_valid_dataset = ImageFolder(sdr_valid_root, transform=sdr_valid_transforms)

    sdr_valid_dataloader = DataLoader(
        sdr_valid_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"))


    hdr_valid_dataset = ImageFolder_HDR(
                                    hdr_root=hdr_valid_root,
                                    train=False)

    hdr_valid_dataloader = DataLoader(
        hdr_valid_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model](quality=3)
    net = net.to(device)
    net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, sdr_w=args.sdr_w)
    
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger,
            args.rw,
            args.pw
        )
        loss = test_epoch(epoch, sdr_valid_dataloader, hdr_valid_dataloader, net, criterion, logger)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename = log_dir + "model.pth.tar"
            )
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.module.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename = log_dir + "best_model.pth.tar"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
