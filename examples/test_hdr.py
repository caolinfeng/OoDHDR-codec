
import argparse
import math
import random
import shutil
import sys
import os
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import msssim, ssim
from collections import OrderedDict
from hdr_dataset import ImageFolder_HDR, ImageFolder_SDR
from pu_tran import Lu_trans, Pu_trans
from tmo import TMO, ITMO

import matplotlib
matplotlib.use('Agg')


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
        

def test_epoch(test_dataloader, model, logger, log_dir):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    msssim_loss = AverageMeter()

    save_index = 0
    Duration = []
    with torch.no_grad():
        for (img, _) in test_dataloader:
            
            img = img.to(device)

            time_start=time.time()
            img_log = torch.log10(img).clamp_(-5, 5)
            pu = Pu_trans(img_log)
            d, bins, s, v = TMO(pu,20)
            d = d/255 # [0 1]

            out_net = model(d)
            re = out_net["x_hat"].clamp_(0,1)

            re_Pu = ITMO(255*re, bins, s, v)
            re_log = Lu_trans(re_Pu)
            re_img = 10**re_log

            time_end=time.time()
            duration = time_end - time_start
            Duration.append(duration)

            N, _, H, W = d.size()
            num_pixels = N * H * W

            # Bit for Image Coding

            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )

            # Bit for TMO/ITMO
            ex_bit = 8 * (bins.nbytes + s.nbytes + v.nbytes)
            
            BPP = bpp + ex_bit / (H * W)
            bpp_loss.update(BPP)

            # Quality Evaluation on MS-SSIM
            MSSSIM = msssim(img, re_img)
            msssim_loss.update(MSSSIM)
            
            logger.info(
                f"\tMSSSIM: {MSSSIM:.3f} |"
                f"\tBpp: {BPP:.3f} |"
                f"\tDuration: {duration:.3f} "
            )

            # Image Output
            re_hdr_save = np.squeeze(re_img.cpu().numpy(), 0).transpose(1,2,0).astype(np.float32)
            cv2.imwrite(log_dir + str(save_index) + '.hdr', re_hdr_save)

            sdr_save = np.squeeze((re*255).cpu().numpy(), 0).transpose(1,2,0).astype(np.uint8)
            cv2.imwrite(log_dir + 'SDR_layer_'+str(save_index) + '.png', sdr_save)

            save_index += 1

    logger.info(
        f"\tMean MS-SSIM: {msssim_loss.avg:.3f} |"
        f"\tMean Bpp: {bpp_loss.avg:.3f} |"
        f"\tMean Duration: {np.mean(Duration):.3f} |"
    )



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="cheng2020-attn",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--pth", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    
    return args


def main(argv):
    args = parse_args(argv)

    head = time.strftime("%m%d%H%M%S", time.localtime()) 

    log_dir = './log/test_hdr/' + head +'/'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_dir+'log.txt',
                        )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )


    test_root = '/your hdr test sets'
    test_dataset = ImageFolder_HDR(
                                    hdr_root=test_root,
                                    train=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print('Using GPU:' + str(np.argmax(memory_gpu)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
    os.system('rm tmp') 
    device = "cuda"  

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model](quality=3)
    net = net.to(device)

    if args.pth:  # load from previous checkpoint
        logger.info("Loading"+args.pth)
        checkpoint = torch.load(args.pth)
        last_epoch = checkpoint["epoch"] + 1
        logger.info("trained {} epoch".format(last_epoch))

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if 'module' in k:
                k = k[7:]
            new_state_dict[k]=v

        net.load_state_dict(new_state_dict)

    test_epoch(test_dataloader, net, logger, log_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
