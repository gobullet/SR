import os
from config import get_config
import torch
from dataset import Datasets
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

from dataset2 import Datasets2
from model.conv8 import ZSSRNet
import matplotlib.pyplot as plt
import pytest

"""
if __name__ == '__main__':
    config = get_config()
    img = Image.open(config.img)
    train_dataset = Datasets2(img, config.scale_factor, config.noise_std, config.crop_size)
    data_sampler = WeightedRandomSampler(train_dataset.probability, num_samples=config.batch_size,
                                         replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               sampler=data_sampler)

    for step, image in enumerate(train_loader):
        for i in range(config.batch_size):
            input()
            low_resolution = image['lr']
            high_resolution = image['hr']
            h0 = high_resolution[i]
            h0 = transforms.ToPILImage()(h0)
            h0.show()
            l0 = low_resolution[i]
            l0 = transforms.ToPILImage()(l0)
            l0.show()
    a = transforms
"""

if __name__ == '__main__':
    config = get_config()

    crop_size = config.crop_size

    img = Image.open(config.img)
    t_img = transforms.ToTensor()(img)
    size = t_img.size()

    p_size = size[1:3]
    while crop_size > min(p_size[0:2]) or crop_size ** 2 > p_size[0] * p_size[1] // 4:
        crop_size = crop_size // 2





