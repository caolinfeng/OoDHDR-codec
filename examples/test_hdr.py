
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

# def tone_mapping(input, N_segment):
#     hdr_log=np.array(input.detach().cpu()).transpose(0,2,3,1)
#     lu_log = (hdr_log[:, :, :,0] + hdr_log[:, :, :, 1] + hdr_log[:, :, :, 2]) / 3

#     batch_num,h,w,c=hdr_log.shape

#     v_max = 255

#     bins=[]
#     s=[]
#     v=[]
#     img_mapped=np.zeros((batch_num,h,w,c))
#     for i in range(batch_num):
#         n_i, bins_i, patches = plt.hist(lu_log[i,:,:].flatten(), N_segment)
#         # plt.show()

#         prob_i = n_i / (h * w)
#         P_i = np.sum(prob_i ** (1 / 3))
#         # print('prob sum:',np.sum(prob_i))
#         V = 0
#         s_i = []
#         v_i = []
#         bins_i[N_segment]=max(hdr_log.flatten())
#         bins_i[0]=min(hdr_log.flatten())
#         for j in range(N_segment):
#             s_item = (v_max * (prob_i[j] ** (1 / 3))) / ((bins_i[j + 1] - bins_i[j]) * P_i)
#             v_item = V
#             V = s_item * (bins_i[j + 1] - bins_i[j]) + V
#             s_i.append(s_item)
#             v_i.append(v_item)

#         v_i.append(v_max)

#         bins.append(bins_i)
#         s.append(s_i)
#         v.append(v_i)

#         f1_range = []
#         f1_value = []

#         for k in range(N_segment):
#             f1_item = ((hdr_log[i,:,:,:] - bins_i[k]) * s_i[k] + v_i[k])
#             f1_value.append(f1_item)
#             f1_range.append(hdr_log[i,:,:,:] <= bins_i[k + 1])

#         img_mapped[i,:,:,:] = np.select(f1_range, f1_value)

#     bins=np.array(bins)
#     s=np.array(s)
#     v=np.array(v)

#     img_mapped = torch.Tensor(img_mapped.transpose(0, 3, 1, 2)).cuda()
    
#     return img_mapped, bins, s, v


# def inverse_mapping(input, bins, s, v):

#     N_segment = s.shape[1]

#     re_sdr=(input.cpu()).detach().numpy().transpose(0,2,3,1)
#     batch_num,h,w,c=re_sdr.shape

#     re_hdr = np.zeros((batch_num, h, w, c))
#     for m in range(batch_num):
#         f2_range = []
#         f2_value = []
#         for n in range(N_segment):
#             if s[m, n]>0:
#                 f2_item = ((re_sdr[m,:,:,:] - v[m, n]) / s[m, n] + bins[m, n])
#             elif s[m, n]==0:
#                 f2_item = (0.5*bins[m, n] + 0.5*bins[m, n+1])
#             f2_value.append(f2_item)
#             f2_range.append(re_sdr[m,:,:,:] <= v[m, n + 1])

#         re_hdr[m,:,:,:] = np.select(f2_range, f2_value)

#     hdr_inversed = torch.Tensor(re_hdr.transpose(0, 3, 1, 2)).cuda()
    
#     return hdr_inversed


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


    test_root = '/home/wheneverwhy/project/CLF_GMCVQN/image/HDRIHAVEN_test'
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
