import os

import PIL

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
import time

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, sr_factor, learnig_rate, num_batch, train_loader):
    model = model.to(device)

    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learnig_rate)
    l_loss = []

    start = time.perf_counter()
    progress = tqdm(range(num_batch))
    for epoch in progress:
        for step, image in enumerate(train_loader):
            low_resolution = Variable(image['lr'].to(device))
            high_resolution = Variable(image['hr'].to(device))
            optimizer.zero_grad()
            out = model(low_resolution)
            loss = loss_function(out, high_resolution)
            loss.backward()
            optimizer.step()

            cpu_loss = loss.data.cpu().numpy()
            progress.set_description("epoch: {epoch} Loss: {loss:.5f}, Learning Rate: {lr}".format(
                epoch=epoch, loss=cpu_loss, lr=learnig_rate))
            l_loss.append(cpu_loss)

    end = time.perf_counter()
    print('process time:' + str(end - start))

    plt.title('loss')
    plt.plot(l_loss)
    plt.savefig('loss.png')
    plt.show()


def test(model, img, sr_factor):
    model.eval()

    img = img.resize((int(img.size[0] * sr_factor),
                      int(img.size[1] * sr_factor)), resample=PIL.Image.BICUBIC)
    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)

    input = Variable(img.to(device))
    out=model(input)
    out=out.data.cpu()
    out=out.clamp(min=0, max=1)
    out=torch.squeeze(out,0)
    out = transforms.ToPILImage()(out)
    out.save('zssr.png')


if __name__ == "__main__":
    config = get_config()

    img = Image.open(config.img)
    t_img = transforms.ToTensor()(img)

    size = t_img.size()
    chanel = size[0]
    # t_img=torch.unsqueeze(t_img,0)

    model = ZSSRNet(input_channels=chanel)

    train_dataset = Datasets(img, config.scale_factor, config.noise_std, config.sub_image_size)
    data_sampler = WeightedRandomSampler(train_dataset.probability, num_samples=config.batch_size,
                                         replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               sampler=data_sampler)
    train(model, config.scale_factor, config.learning_rate, config.num_batches, train_loader)

    test(model, img, config.scale_factor)
