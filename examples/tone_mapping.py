import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from math import sqrt
from torch.autograd import gradcheck
matplotlib.use('Agg')

class Mapping(torch.autograd.Function):
    """
    TMO Module Implementation
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)

        hdr_log=np.array(input.detach().cpu()).transpose(0,2,3,1)
        lu_log = (hdr_log[:, :, :,0] + hdr_log[:, :, :, 1] + hdr_log[:, :, :, 2]) / 3

        batch_num,h,w,c=hdr_log.shape

        N_segment = 10
        v_max = 255

        bins=[]
        s=[]
        v=[]
        img_mapped=np.zeros((batch_num,h,w,c))
        for i in range(batch_num):
            n_i, bins_i, patches = plt.hist(lu_log[i,:,:].flatten(), N_segment)

            prob_i = n_i / (h * w)
            P_i = np.sum(prob_i ** (1 / 3))
            V = 0
            s_i = []
            v_i = []
            bins_i[N_segment]=max(hdr_log.flatten())
            bins_i[0]=min(hdr_log.flatten())
            for j in range(N_segment):
                s_item = (v_max * (prob_i[j] ** (1 / 3))) / ((bins_i[j + 1] - bins_i[j]) * P_i)
                v_item = V
                V = s_item * (bins_i[j + 1] - bins_i[j]) + V
                s_i.append(s_item)
                v_i.append(v_item)

            v_i.append(v_max)

            bins.append(bins_i)
            s.append(s_i)
            v.append(v_i)

            f1_range = []
            f1_value = []

            for k in range(N_segment):
                f1_item = ((hdr_log[i,:,:,:] - bins_i[k]) * s_i[k] + v_i[k])
                f1_value.append(f1_item)
                f1_range.append(hdr_log[i,:,:,:] <= bins_i[k + 1])

            img_mapped[i,:,:,:] = np.select(f1_range, f1_value)

        bins=np.array(bins)
        s=np.array(s)
        v=np.array(v)

        index = 5
        np.save("tmo/bins"+str(index)+".npy",bins)
        np.save("tmo/s"+str(index)+".npy",s)
        np.save("tmo/v"+str(index)+".npy",v)

        img_mapped = torch.Tensor(img_mapped.transpose(0, 3, 1, 2)).cuda()
        
        return img_mapped

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors

        hdr_log=np.array(input.detach().cpu()).transpose(0,2,3,1)
        batch_num,h,w,c=hdr_log.shape

        index = 5
        bins = np.load("tmo/bins"+str(index)+".npy")
        v = np.load("tmo/v"+str(index)+".npy")
        s = np.load("tmo/s"+str(index)+".npy")

        N_segment = s.shape[1]

        img_grad=np.zeros((batch_num,h,w,c))
        for i in range(batch_num):

            f1_range = []
            f1_value = []

            for k in range(N_segment):
                f1_item = s[i, k]
                f1_value.append(f1_item)
                f1_range.append(hdr_log[i,:,:,:] <= bins[i, k + 1])

            img_grad[i,:,:,:] = np.select(f1_range, f1_value)
            
        Img_grad = torch.FloatTensor(img_grad).permute(0, 3, 1, 2).contiguous().cuda()

        grad_input = torch.mul(grad_output, Img_grad)

        return grad_input




class Inverse_Mapping(torch.autograd.Function):

    """
    ITMO Module Implementation
    """

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)

        index = 5
        bins = np.load("tmo/bins"+str(index)+".npy")
        v = np.load("tmo/v"+str(index)+".npy")
        s = np.load("tmo/s"+str(index)+".npy")

        N_segment = s.shape[1]

        re_sdr=(input.cpu()).detach().numpy().transpose(0,2,3,1)
        batch_num,h,w,c=re_sdr.shape

        save_index = 2
        re_hdr = np.zeros((batch_num, h, w, c))
        for m in range(batch_num):
            f2_range = []
            f2_value = []
            for n in range(N_segment):
                if s[m, n]>0:
                    f2_item = ((re_sdr[m,:,:,:] - v[m, n]) / s[m, n] + bins[m, n])
                elif s[m, n]==0:
                    f2_item = (0.5*bins[m, n] + 0.5*bins[m, n+1])
                f2_value.append(f2_item)
                f2_range.append(re_sdr[m,:,:,:] <= v[m, n + 1])

            re_hdr[m,:,:,:] = np.select(f2_range, f2_value)

            save_index = save_index + 1

        hdr_inversed = torch.Tensor(re_hdr.transpose(0, 3, 1, 2)).cuda()
        
        return hdr_inversed

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        index = 5
        bins = np.load("tmo/bins"+str(index)+".npy")
        v = np.load("tmo/v"+str(index)+".npy")
        s = np.load("tmo/s"+str(index)+".npy")

        N_segment = s.shape[1]

        re_sdr=(input.cpu()).detach().numpy().transpose(0,2,3,1)
        batch_num,h,w,c=re_sdr.shape

        hdr0_grad = np.zeros((batch_num, h, w, c))
        for m in range(batch_num):
            f2_range = []
            f2_value = []
            for n in range(N_segment):
                if s[m, n]>0:
                    f2_item = (1 / s[m, n])
                elif s[m, n]==0:
                    f2_item = (1 / (s[m, n]+0.001))
                f2_value.append(f2_item)
                f2_range.append(re_sdr[m,:,:,:] <= v[m, n + 1])

            hdr0_grad[m,:,:,:] = np.select(f2_range, f2_value)

        Hdr_grad = torch.FloatTensor(hdr0_grad).permute(0, 3, 1, 2).contiguous().cuda()

        grad_input = torch.mul(grad_output, Hdr_grad)

        return grad_input



def tone_mapping(input):
    return Mapping.apply(input)

def inverse_mapping(input):
    return Inverse_Mapping.apply(input)
