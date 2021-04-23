from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import torch
import PIL
from PIL import Image


def compute_psnr(im1, im2):
    np1, np2 = np.array(im1), np.array(im2)
    p = psnr(np1, np2)
    return p


def compute_ssim(im1, im2):

    np1, np2 = np.array(im1), np.array(im2)
    isRGB = len(np1.shape) == 3 and np1.shape[-1] != 1
    s = ssim(np1, np2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

if __name__ == '__main__':
    comic = Image.open('comic_gt.png')
    zssr = Image.open('low_res.png')
    np_comic = np.array(comic)
    #print(np_comic.shape)
    print(compute_ssim(comic,zssr))
