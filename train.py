import os
from config import get_config
import torch
from dataset import Datasets
from torch.utils.data import DataLoader
from model.architecture import IMDN
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.autograd import Variable

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def train(sr_factor, learnig_rate, num_epoch, train_loader):
    model = IMDN()
    model = model.to(device)

    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learnig_rate)

    progress = tqdm(range(num_epoch))
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
            progress.set_description("epoch: {epoch} Loss: {loss}, Learning Rate: {lr}".format( \
                epoch=epoch, loss=cpu_loss, lr=learnig_rate))


if __name__ == "__main__":
    config = get_config()

    # make directory not existed
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    train_dataset = Datasets(config.image_size, config.scale_factor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    train(config.scale_factor, config.learning_rate, config.num_epoch, train_loader)
