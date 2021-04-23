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
from model.conv8 import ZSSRNet
import matplotlib.pyplot as plt



if __name__ == '__main__':
    config = get_config()
    print("epoch: {epoch} Loss: {loss:.5f}, Learning Rate: {lr:.3f}".format(
        epoch=1, loss=2.2, lr=(3)))




    '''
    for epoch in range(10):
        input()
        for step, batch in enumerate(train_loader):
            low_resolution = batch['lr']
            high_resolution = batch['hr']
            h0 = high_resolution[0]
            h0 = transforms.ToPILImage()(h0)


            h0.show()
            '''

'''
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn(2, 3)
    y = torch.randn(1, 3)
    out=torch.cat((x,y),-2)
    print(out)
    '''
