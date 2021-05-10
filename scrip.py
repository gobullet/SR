import os
import PIL
from config import get_config
import torch
from dataset import Datasets
from dataset2 import Datasets2
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from model.conv8 import ZSSRNet
from model.resnet import ResNet
import matplotlib.pyplot as plt
import time
from utils import compute_ssim, compute_psnr


def GTmod12():
    route = r'../BSDS300/images/test'
    target = r'../BSDS300/GTmod12'
    if not os.path.exists(target):
        os.makedirs(target)
    for root, dirs, files in os.walk(route):
        for file in files:
            img = Image.open(os.path.join(root, file))
            size = img.size
            img = TF.crop(img, 0, 0, size[1] // 12 * 12, size[0] // 12 * 12)
            img.save(os.path.join(target, file))


def LRbicx2():
    route = r'../BSDS300/GTmod12'
    target = r'../BSDS300/LRbicx2'
    if not os.path.exists(target):
        os.makedirs(target)
    for root, dirs, files in os.walk(route):
        for file in files:
            img = Image.open(os.path.join(root, file))
            size = img.size
            img = img.resize([size[0] // 2, size[1] // 2], resample=PIL.Image.BICUBIC)
            img.save(os.path.join(target, file))


if __name__ == '__main__':
   LRbicx2()
