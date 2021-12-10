import numpy as np
import torch
import matplotlib.pyplot as plt


def TMO(input, N_segment):
    hdr_log=np.array(input.detach().cpu()).transpose(0,2,3,1)
    lu_log = (hdr_log[:, :, :,0] + hdr_log[:, :, :, 1] + hdr_log[:, :, :, 2]) / 3

    batch_num,h,w,c=hdr_log.shape

    # N_segment = 10
    v_max = 255

    bins=[]
    s=[]
    v=[]
    img_mapped=np.zeros((batch_num,h,w,c))
    for i in range(batch_num):
        n_i, bins_i, patches = plt.hist(lu_log[i,:,:].flatten(), N_segment)
        # plt.show()

        prob_i = n_i / (h * w)
        P_i = np.sum(prob_i ** (1 / 3))
        # print('prob sum:',np.sum(prob_i))
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

    img_mapped = torch.Tensor(img_mapped.transpose(0, 3, 1, 2)).cuda()
    
    return img_mapped, bins, s, v


def ITMO(input, bins, s, v):

    N_segment = s.shape[1]

    re_sdr=(input.cpu()).detach().numpy().transpose(0,2,3,1)
    batch_num,h,w,c=re_sdr.shape

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

    hdr_inversed = torch.Tensor(re_hdr.transpose(0, 3, 1, 2)).cuda()
    
    return hdr_inversed