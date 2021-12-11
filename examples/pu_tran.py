import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from math import sqrt
from torch.autograd import gradcheck

class Lu_Pu(torch.autograd.Function):
    """
    Transform the original Luminance Space ("Log10[Luminance]" in our project) 
    to Perceptual Unit Space
    """

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)

        hdr_log=np.array(input.detach().cpu())

        batch_num,c,h,w=hdr_log.shape

        # PU curve fitting
        luminance = [-5,-1,0,0.698970004336019,1,2,4,5]
        pu = [-112.6,-42.1,7.1,80.4,121.5,269.4,570,721]
        s = [17.625,49.2,104.868591,136.53124469,147.9,150.3,151]
        N_segment = len(s)

        img_mapped=np.zeros((batch_num,c,h,w))

        f1_range = []
        f1_value = []
        for k in range(N_segment):
            f1_item = ((hdr_log - luminance[k]) * s[k] + pu[k])
            f1_value.append(f1_item)
            f1_range.append(hdr_log <= luminance[k + 1])

        img_mapped = np.select(f1_range, f1_value)

        img_mapped = torch.Tensor(img_mapped).cuda()
        
        return img_mapped

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        input = input.cuda()

        hdr_log=np.array(input.detach().cpu())
        batch_num,c,h,w=hdr_log.shape


        luminance = [-5,-1,0,0.698970004336019,1,2,4,5]
        pu = [-112.6,-42.1,7.1,80.4,121.5,269.4,570,721]
        s = [17.625,49.2,104.868591,136.53124469,147.9,150.3,151]
        N_segment = len(s)

        img_grad=np.zeros((batch_num,c,h,w))

        f1_range = []
        f1_value = []
        for k in range(N_segment):
            f1_item = s[k]
            f1_value.append(f1_item)
            f1_range.append(hdr_log <= luminance[k + 1])

        img_grad = np.select(f1_range, f1_value)

        Img_grad = torch.Tensor(img_grad).cuda()

        grad_input = torch.mul(grad_output, Img_grad)

        return grad_input





class Pu_Lu(torch.autograd.Function):

    """
    Transform the Perceptual Unit Space to original Luminance Space
    """

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)

        luminance = [-5,-1,0,0.698970004336019,1,2,4,5]
        pu = [-112.6,-42.1,7.1,80.4,121.5,269.4,570,721]
        s = [17.625,49.2,104.868591,136.53124469,147.9,150.3,151]
        N_segment = len(s)

        re_sdr=(input.cpu()).detach().numpy()
        np.clip(re_sdr,pu[0], pu[N_segment])

        batch_num,c,h,w=re_sdr.shape

        save_index = 0
        re_lu = np.zeros((batch_num,c,h,w))

        f2_range = []
        f2_value = []
        for n in range(N_segment):

            f2_item = ((re_sdr - pu[n]) / s[n] + luminance[n])
            f2_value.append(f2_item)
            f2_range.append(re_sdr <= pu[n + 1])

        re_lu = np.select(f2_range, f2_value) 

        lu_inversed = torch.Tensor(re_lu).cuda()
        
        return lu_inversed

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        input = input.cuda()

        luminance = [-5,-1,0,0.698970004336019,1,2,4,5]
        pu = [-112.6,-42.1,7.1,80.4,121.5,269.4,570,721]
        s = [17.625,49.2,104.868591,136.53124469,147.9,150.3,151]
        N_segment = len(s)

        re_sdr=(input.cpu()).detach().numpy()
        np.clip(re_sdr,pu[0], pu[N_segment])

        batch_num,c,h,w=re_sdr.shape

        lu_grad = np.zeros((batch_num,c,h,w))

        f2_range = []
        f2_value = []
        for n in range(N_segment):

            f2_item = (1 / s[n])
            f2_value.append(f2_item)
            f2_range.append(re_sdr <= pu[n + 1])

        lu_grad = np.select(f2_range, f2_value) 

        Lu_grad = torch.Tensor(lu_grad).cuda()

        grad_input = torch.mul(grad_output, Lu_grad)

        return grad_input





def Pu_trans(input):
    return Lu_Pu.apply(input)

def Lu_trans(input):
    return Pu_Lu.apply(input)
