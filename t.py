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


'''
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn(2, 3)
    y = torch.randn(1, 3)
    out=torch.cat((x,y),-2)
    print(out)
    '''
