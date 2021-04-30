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
from torchvision import transforms
from model.conv8 import ZSSRNet
from model.resnet import ResNet
from model.test import TestNet
import matplotlib.pyplot as plt
import time
from utils import compute_ssim, compute_psnr

device = ('cuda' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'


def train(name_img, img, model, sr_factor, learnig_rate, num_epoch, noise_std, sub_image_size, batch_size):
    train_dataset = Datasets2(img, sr_factor, noise_std, sub_image_size)
    data_sampler = WeightedRandomSampler(train_dataset.probability, num_samples=batch_size,
                                         replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               sampler=data_sampler)

    model = model.to(device)
    loss_function = nn.L1Loss()
    l_loss = []
    optimizer = optim.Adam(model.parameters(), lr=learnig_rate, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [600, 1200, 1700, 2200, 2600, 3000], gamma=0.1,
    #                                                 last_epoch=-1)

    start = time.perf_counter()
    progress = tqdm(range(num_epoch))
    for epoch in progress:
        for iter, image in enumerate(train_loader):
            model.zero_grad()

            low_resolution = image['lr'].to(device)
            high_resolution = image['hr'].to(device)

            out = model(low_resolution)
            loss = loss_function(out, high_resolution)
            loss.backward()
            optimizer.step()
            scheduler.step()

            cpu_loss = loss.data.cpu()
            progress.set_description("epoch: {epoch} Loss: {loss:.4f}, Learning Rate: {lr:.1e}".format(
                epoch=epoch, loss=cpu_loss, lr=float(scheduler.get_last_lr()[-1])))

            l_loss.append(cpu_loss)

    end = time.perf_counter()
    print('process time:{:.1f}'.format(end - start))

    plt.title('loss')
    plt.ylim(0, 0.1)
    plt.plot(l_loss)
    plt.savefig('../output/' + name_img + '_loss.png')


def test(name_img, model, img, sr_factor, gt=False, img_gt=None):
    model.eval()

    img_bicubic = img.resize((int(img.size[0] * sr_factor),
                              int(img.size[1] * sr_factor)), resample=PIL.Image.BICUBIC)
    img_bicubic.save('../output/' + name_img + '_bicubic.png')

    input = transforms.ToTensor()(img_bicubic)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    with torch.no_grad():
        out = model(input)
    out = out.data.cpu()
    out = out.clamp(min=0, max=1)
    out = torch.squeeze(out, 0)
    out = transforms.ToPILImage()(out)
    out.save('../output/' + name_img + '_zssr.png')

    if gt:
        ssim_bicubic = compute_ssim(img_gt, img_bicubic)
        psnr_bicubic = compute_psnr(img_gt, img_bicubic)
        ssim_zssr = compute_ssim(img_gt, out)
        psnr_zssr = compute_psnr(img_gt, out)
        print("ssim_bicubic:\t{:.3f}".format(ssim_bicubic))
        print("ssim_zssr:\t{:.3f}".format(ssim_zssr))
        print("psnr_bicubic:\t{:.2f}".format(psnr_bicubic))
        print("psnr_zssr:\t{:.2f}".format(psnr_zssr))


def test2(name_img, model, img, sr_factor, gt=False, img_gt=None):
    model.eval()

    img_bicubic = img.resize((int(img.size[0] * sr_factor),
                              int(img.size[1] * sr_factor)), resample=PIL.Image.BICUBIC)
    img_bicubic.save('../output/' + name_img + '_bicubic.png')

    input = transforms.ToTensor()(img)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    with torch.no_grad():
        out = model(input)
    out = out.data.cpu()
    out = out.clamp(min=0, max=1)
    out = torch.squeeze(out, 0)
    out = transforms.ToPILImage()(out)
    out.save('../output/' + name_img + '_zssr.png')

    if gt:
        ssim_bicubic = compute_ssim(img_gt, img_bicubic)
        psnr_bicubic = compute_psnr(img_gt, img_bicubic)
        ssim_zssr = compute_ssim(img_gt, out)
        psnr_zssr = compute_psnr(img_gt, out)
        print("ssim_bicubic:\t{:.3f}".format(ssim_bicubic))
        print("ssim_zssr:\t{:.3f}".format(ssim_zssr))
        print("psnr_bicubic:\t{:.2f}".format(psnr_bicubic))
        print("psnr_zssr:\t{:.2f}".format(psnr_zssr))


if __name__ == "__main__":
    config = get_config()

    name_img = os.path.basename(config.img)
    name_img, ex = os.path.splitext(name_img)
    gt = False
    img_gt = None
    if os.path.exists(name_img + r"_gt.png"):
        gt = True
        gt_root = name_img + r"_gt.png"
        img_gt = Image.open(gt_root)

    img = Image.open(config.img)
    t_img = transforms.ToTensor()(img)
    size = t_img.size()
    channel = size[0]

    crop_size = config.crop_size
    p_size = size[1:3]
    while crop_size > min(p_size[0:2]) or crop_size ** 2 > p_size[0] * p_size[1] // 4:
        crop_size = crop_size // 2
    print("crop_size:" + str(crop_size))

    model = ResNet(input_channels=channel, sf=config.scale_factor)

    train(name_img, img, model, config.scale_factor, config.learning_rate, config.num_epoch, config.noise_std,
          crop_size, config.batch_size)

    test2(name_img, model, img, config.scale_factor, gt, img_gt)
