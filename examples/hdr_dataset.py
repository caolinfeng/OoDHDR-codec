# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pdb
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt



HDR_EXTENSIONS = [
    '.hdr',
]

SDR_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_hdr_file(filename):
    return any(filename.endswith(extension) for extension in HDR_EXTENSIONS)

def is_sdr_file(filename):
    return any(filename.endswith(extension) for extension in SDR_EXTENSIONS)


def default_hdr_loader_loader(path):
    return cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH)   # np.araay

def default_sdr_loader_loader(path):
    return cv2.imread(path)   # np.araay


class ImageFolder_OOD(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, 
                hdr_root,
                sdr_root,  
                patch_size = 256, 
                train = False, 
                hdr_loader=default_hdr_loader_loader,
                sdr_loader=default_sdr_loader_loader):

        sdr_images = []
        hdr_images = []
        for filename in os.listdir(hdr_root):
            if is_hdr_file(filename):
                hdr_images.append('{}'.format(filename))

        for filename in os.listdir(sdr_root):
            if is_sdr_file(filename):
                sdr_images.append('{}'.format(filename))

        self.sdr_root = sdr_root
        self.hdr_root = hdr_root
        self.sdr_imgs = sdr_images
        self.hdr_imgs = hdr_images
        self.hdr_loader = hdr_loader
        self.sdr_loader = sdr_loader
        self.patch_size = patch_size
        self.train = train

        """ pre-set the number of images in one iteration (1/7 ratio in our project) """ 
        self.hdr_img_num = 1
        self.sdr_img_num = 7
        self.hdr_total_num = len(hdr_images)
        self.sdr_total_num = len(sdr_images)

    @staticmethod
    def _random_flip(img, mode):
        return cv2.flip(img, mode)

    @staticmethod
    def _random_rot(img, rot_coin):
        if rot_coin==0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rot_coin==1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if rot_coin==2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img

    def __getitem__(self, index):

        if self.train:

            flip_coin = np.random.random()
            flip_flag = 0
            if flip_coin > 0.5:
                flip_flag = 1
                mode = 0 if np.random.random() > 0.5 else 1
            rotate_coin = np.random.randint(0, 2)

            # HDR Image Folder
            final_hdr = None

            for i in range(self.hdr_img_num):

                if i == 0:
                    random_index = 0
                else:
                    random_index = np.random.randint(0,self.hdr_total_num)
                
                filename1 = self.hdr_imgs[(index+random_index)%self.hdr_total_num]
                hdr = self.hdr_loader(os.path.join(self.hdr_root, filename1))
                h, w, c = hdr.shape

                L = self.patch_size

                h1 = np.random.randint(hdr.shape[0] - L)
                w1 = np.random.randint(hdr.shape[1] - L)

                h2 = h1 + L
                w2 = w1 + L
                
                hdr=hdr[h1:h2,w1:w2,:]

                if flip_flag == 1:
                    hdr = self._random_flip(hdr, mode)
                
                hdr = self._random_rot(hdr, rotate_coin)

                hdr=torch.Tensor(hdr.transpose((2, 0, 1)))

                hdr = hdr.unsqueeze(dim=0)

                if final_hdr is None:
                    final_hdr = hdr
                else:
                    final_hdr = torch.cat([final_hdr, hdr],dim=0)


            # SDR Image Folder
            final_sdr = None

            filename2_list = []
            for i in range(self.sdr_img_num):

                if i == 0:
                    random_index = 0
                else:
                    random_index = np.random.randint(0,self.sdr_total_num)

                filename2 = self.sdr_imgs[(index+random_index)%self.sdr_total_num]
                sdr = self.sdr_loader(os.path.join(self.sdr_root, filename2))
                filename2_list.append(filename2)

                L = self.patch_size

                h1 = np.random.randint(sdr.shape[0] - L)
                w1 = np.random.randint(sdr.shape[1] - L)

                h2 = h1 + L
                w2 = w1 + L
                
                sdr=sdr[h1:h2,w1:w2,:]

                if flip_flag == 1:
                    sdr = self._random_flip(sdr, mode)
                
                sdr = self._random_rot(sdr, rotate_coin)

                sdr=torch.Tensor(sdr.transpose((2, 0, 1)))

                sdr = sdr.unsqueeze(dim=0)

                if final_sdr is None:
                    final_sdr = sdr
                else:
                    final_sdr = torch.cat([final_sdr, sdr],dim=0)            


        else:
            
            filename1 = self.hdr_imgs[index]
            filename2 = self.sdr_imgs[index]
            hdr = self.hdr_loader(os.path.join(self.hdr_root, filename1))
            sdr = self.sdr_loader(os.path.join(self.sdr_root, filename2))
            h=hdr.shape[0]
            w=hdr.shape[1]
            q=16
            h0=q*math.floor(h/q)
            w0=q*math.floor(w/q)
            hdr=hdr[0:h0,0:w0,:]

            h=sdr.shape[0]
            w=sdr.shape[1]
            q=16
            h0=q*math.floor(h/q)
            w0=q*math.floor(w/q)
            sdr=sdr[0:h0,0:w0,:]

            final_hdr=torch.Tensor(hdr.transpose((2, 0, 1)))
            final_sdr=torch.Tensor(sdr.transpose((2, 0, 1)))

        return final_hdr, final_sdr, filename2_list

    def __len__(self):
        return len(self.hdr_imgs)




class ImageFolder_SDR(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, 
                sdr_root,  
                patch_size = 256, 
                train = False, 
                sdr_loader=default_sdr_loader_loader):

        sdr_images = []

        for filename in os.listdir(sdr_root):
            if is_sdr_file(filename):
                sdr_images.append('{}'.format(filename))

        self.sdr_root = sdr_root
        self.sdr_imgs = sdr_images
        self.sdr_loader = sdr_loader
        self.patch_size = patch_size
        self.train = train

    @staticmethod
    def _random_flip(img, mode):
        return cv2.flip(img, mode)

    @staticmethod
    def _random_rot(img, rot_coin):
        if rot_coin==0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rot_coin==1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if rot_coin==2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img

    def __getitem__(self, index):
        filename2 = self.sdr_imgs[index]

        try:
            sdr = self.sdr_loader(os.path.join(self.sdr_root, filename2))
        except:
            print('img read fail')


        if self.train:

            L = self.patch_size
            h1 = np.random.randint(sdr.shape[0] - L)
            w1 = np.random.randint(sdr.shape[1] - L)
            h2 = h1 + L
            w2 = w1 + L
            
            sdr=sdr[h1:h2,w1:w2,:]

            flip_coin = np.random.random()
            if flip_coin > 0.5:
                mode = 0 if np.random.random() > 0.5 else 1
                sdr = self._random_flip(sdr, mode)

            rotate_coin = np.random.randint(0, 2)
            sdr = self._random_rot(sdr, rotate_coin)

        else:
            h=sdr.shape[0]
            w=sdr.shape[1]
            q=16
            h0=q*math.floor(h/q)
            w0=q*math.floor(w/q)
            sdr=sdr[0:h0,0:w0,:]

        sdr=torch.Tensor(sdr.transpose((2, 0, 1)))
        return sdr, filename2

    def __len__(self):
        return len(self.sdr_imgs)




class ImageFolder_HDR(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, 
                hdr_root,  
                patch_size = 256, 
                train = False, 
                hdr_loader=default_hdr_loader_loader):

        hdr_images = []

        for filename in os.listdir(hdr_root):
            if is_hdr_file(filename):
                hdr_images.append('{}'.format(filename))

        self.hdr_root = hdr_root
        self.hdr_imgs = hdr_images
        self.hdr_loader = hdr_loader
        self.patch_size = patch_size
        self.train = train

    @staticmethod
    def _random_flip(img, mode):
        return cv2.flip(img, mode)

    @staticmethod
    def _random_rot(img, rot_coin):
        if rot_coin==0:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rot_coin==1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if rot_coin==2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img

    def __getitem__(self, index):
        filename2 = self.hdr_imgs[index]

        try:
            hdr = self.hdr_loader(os.path.join(self.hdr_root, filename2))
        except:
            print('img read fail')


        if self.train:
            L = self.patch_size
            h1 = np.random.randint(hdr.shape[0] - L)
            w1 = np.random.randint(hdr.shape[1] - L)
            h2 = h1 + L
            w2 = w1 + L
            
            hdr=hdr[h1:h2,w1:w2,:]

            flip_coin = np.random.random()
            if flip_coin > 0.5:
                mode = 0 if np.random.random() > 0.5 else 1
                hdr = self._random_flip(hdr, mode)

            rotate_coin = np.random.randint(0, 2)
            hdr = self._random_rot(hdr, rotate_coin)

        else:
            h=hdr.shape[0]
            w=hdr.shape[1]
            q=16
            h0=q*math.floor(h/q)
            w0=q*math.floor(w/q)
            hdr=hdr[0:h0,0:w0,:]

        hdr=torch.Tensor(hdr.transpose((2, 0, 1)))
        return hdr, filename2

    def __len__(self):
        return len(self.hdr_imgs)
