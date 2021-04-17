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
from model.conv7 import ZSSRNet
import matplotlib.pyplot as plt


"""
if __name__ == '__main__':
    config = get_config()

    img = Image.open(config.img)
    train_dataset = Datasets(img, config.scale_factor, config.noise_std, config.sub_image_size)
    data_sampler = WeightedRandomSampler(train_dataset.probability, num_samples=4, replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4,
                                               sampler=data_sampler)
    
    for image in train_dataset:
        input()
        hr = image["hr"]
        lr = image["lr"]
        hr.show()



    
    for epoch in range(10):
        input()
        for step, batch in enumerate(train_loader):
            low_resolution = batch['lr']
            high_resolution = batch['hr']
            h0 = high_resolution[0]
            h0 = transforms.ToPILImage()(h0)


            h0.show()
"""
if __name__ == '__main__':
    print(format(1.23456, '.2f'))
