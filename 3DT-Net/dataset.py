# -*- coding: UTF-8 -*-
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
import scipy.io as sio
import random
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from kornia.filters import get_gaussian_kernel2d
import kornia

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def load_img(filepath):
    x = sio.loadmat(filepath)
    x = x['msi']
    x = torch.tensor(x).float()
    return x

def load_img1(filepath):
    x = sio.loadmat(filepath)
    x = x['RGB']
    x = torch.tensor(x).float()
    return x

def load_img2(filepath):
    x = sio.loadmat(filepath)
    x = x['blur']
    x = torch.tensor(x).float()
    return x

def load_img3(filepath):
    x = sio.loadmat(filepath)
    x = x['LR']
    x = torch.tensor(x).float()
    return x


# def my_gaussian_blur2d(input, kernel_size, sigma, border_type = 'reflect'):
# 
#     kernel = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma, force_even=True), dim=0)
#     # print(kernel)
# 
#     return kornia.filters.filter2d(input, kernel, border_type)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3,upscale_factor, patch_size,input_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.patch_size = patch_size
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)

        self.lens = 20000

        self.xs = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.x_blurs = []
        for img in self.image_filenames3:
            self.x_blurs.append(load_img2(img))

        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

    def __getitem__(self, index):
        ind = index % 20
        img = self.xs[ind]
        img2 = self.ys[ind]
        img3 = self.x_blurs[ind]
        upscale_factor = self.upscale_factor
        w = np.random.randint(0, img.shape[0]-self.patch_size)
        h = np.random.randint(0, img.shape[1]-self.patch_size)
        X = img[w:w+self.patch_size, h:h+self.patch_size, :]
        Y = img2[w:w+self.patch_size, h:h+self.patch_size, :]

        # Z = my_gaussian_blur2d(X.unsqueeze(0), (8, 8), (2, 2)).squeeze(0)
        Z = img3[int(w+upscale_factor/2):w+self.patch_size:upscale_factor, int(h+upscale_factor/2):h+self.patch_size:upscale_factor, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        X = torch.rot90(X, rotTimes, [0,1])
        Y = torch.rot90(Y, rotTimes, [0,1])
        Z = torch.rot90(Z, rotTimes, [0,1])

        # Random vertical Flip
        for j in range(vFlip):
            X = X.flip(1)
            Y = Y.flip(1)
            Z = Z.flip(1)

        # Random Horizontal Flip
        for j in range(hFlip):
            X = X.flip(0)
            Y = Y.flip(0)
            Z = Z.flip(0)

        X = X.permute(2,0,1)
        Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)

        return Z, Y, X

    def __len__(self):
        return self.lens


class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, input_transform=None):
        super(DatasetFromFolder2, self).__init__()
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)
        # self.upscale_factor = upscale_factor
        self.input_transform = input_transform

        self.xs = []
        self.xs_name = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))
            self.xs_name.append(img)

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.zs = []
        for img in self.image_filenames3:
            self.zs.append(load_img3(img))

    def __getitem__(self, index):
        X = self.xs[index]
        Y = self.ys[index]
        Z = self.zs[index]

        # upscale_factor = self.upscale_factor

        # Z = F.interpolate(X.permute(2, 0, 1).unsqueeze(0), scale_factor=1.0 / upscale_factor, mode='bicubic',
        #                     align_corners=False, recompute_scale_factor=False).squeeze(0).permute(1, 2, 0)

        # 
        # Z = my_gaussian_blur2d(X.unsqueeze(0), (8, 8), (2, 2)).squeeze(0)
        # Z = Z[int(upscale_factor/2)::upscale_factor, int(upscale_factor/2)::upscale_factor, :]




        X = X.permute(2, 0, 1)
        # Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        return Z, Y, X, self.xs_name[index]


    def __len__(self):
        return len(self.image_filenames1)