# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import msssim, ssim
from collections import OrderedDict
from typing import Tuple, Union
import PIL.Image as Image


class Distortion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        y = np.array(output["likelihoods"]['y'].cpu())

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        re = output["x_hat"].clamp_(0,1)
        re_255 = (re*255+0.5).clamp_(0,255).floor()
        tar_255 = (target*255+0.5).clamp_(0,255).floor()

        out["msssim"] = msssim(re_255, tar_255)

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



def test_epoch(test_dataloader, model, criterion, logger):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    msssim_loss = AverageMeter()

    Duration = []

    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)

            time_start=time.time()
            out_net = model(d)

            time_end=time.time()
            duration = time_end - time_start
            Duration.append(duration)

            out_criterion = criterion(out_net, d)
            BPP = out_criterion["bpp_loss"]
            bpp_loss.update(BPP)

            MSSSIM = out_criterion["msssim"]
            msssim_loss.update(MSSSIM)
            
            logger.info(
                f"\tMSSSIM: {MSSSIM:.3f} |"
                f"\tBpp: {BPP:.3f} |"
                f"\tDuration: {duration:.3f} "
            )

    logger.info(
        f"\tMean MS-SSIM: {msssim_loss.avg:.3f} |"
        f"\tMean Bpp: {bpp_loss.avg:.3f} |"
        f"\tMean Duration: {np.mean(Duration):.3f} |"
    )



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
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--pth", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    
    return args

def main(argv):
    args = parse_args(argv)

    head = time.strftime("%m%d%H%M%S", time.localtime()) 

    log_dir = './log/test_sdr/'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_dir+'log_' +head+ '.txt',
                        )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_root = '/your sdr test sets'

    logger.info(test_root)

    test_dataset = ImageFolder(test_root, transform=test_transforms)

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
        logger.info(args.pth)
        checkpoint = torch.load(args.pth)
        last_epoch = checkpoint["epoch"] + 1
        logger.info("trained {} epoch".format(last_epoch))

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if 'module' in k:
                k = k[7:]
            new_state_dict[k]=v

        net.load_state_dict(new_state_dict)

    criterion = Distortion()

    test_epoch(test_dataloader, net, criterion, logger)



if __name__ == "__main__":
    main(sys.argv[1:])
