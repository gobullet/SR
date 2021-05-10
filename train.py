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
import matplotlib.pyplot as plt
import time
from utils import compute_ssim, compute_psnr
import math

device = ('cuda' if torch.cuda.is_available() else 'cpu')


def train(name_img, img, model, sr_factor, learnig_rate, num_epoch, noise_std, sub_image_size, batch_size):
    train_dataset = Datasets(img, sr_factor, noise_std, sub_image_size)
    data_sampler = WeightedRandomSampler(train_dataset.probability, num_samples=10,
                                         replacement=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               sampler=data_sampler)

    model = model.to(device)
    loss_function = nn.L1Loss()
    l_loss = []
    optimizer = optim.Adam(model.parameters(), lr=learnig_rate, betas=[0.9, 0.999])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [700, 1300, 1800, 2300, 2800, 3300], gamma=0.1,
                                                     last_epoch=-1)

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
    train_time = end - start
    print('process time:{:.1f}'.format(train_time))

    plt.figure()
    plt.title('loss')
    plt.ylim(0, 0.1)
    plt.plot(l_loss)
    out_file = r'../output'
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    plt.savefig(os.path.join(out_file, name_img + '_loss.png'))
    return train_time


def test(name_img, model, img, sr_factor, gt=False, img_gt=None):
    out_file = r'../output'
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    model.eval()

    img_bicubic = img.resize((int(img.size[0] * sr_factor),
                              int(img.size[1] * sr_factor)), resample=PIL.Image.BICUBIC)
    img_bicubic.save(os.path.join(out_file, name_img + '_bicubic.png'))

    input = transforms.ToTensor()(img_bicubic)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    with torch.no_grad():
        out = model(input)
    out = out.data.cpu()
    out = out.clamp(min=0, max=1)
    out = torch.squeeze(out, 0)
    out = transforms.ToPILImage()(out)
    out.save(os.path.join(out_file, name_img + '_zssr.png'))

    if gt:
        ssim_bicubic = compute_ssim(img_gt, img_bicubic)
        psnr_bicubic = compute_psnr(img_gt, img_bicubic)
        ssim_zssr = compute_ssim(img_gt, out)
        psnr_zssr = compute_psnr(img_gt, out)
        print("psnr_bicubic:\t{:.2f}".format(psnr_bicubic))
        print("ssim_bicubic:\t{:.4f}".format(ssim_bicubic))
        print("psnr_zssr:\t{:.2f}".format(psnr_zssr))
        print("ssim_zssr:\t{:.4f}".format(ssim_zssr))
        fo = open(os.path.join(out_file, 'PSNR_and_SSIM.txt'), mode='a')
        fo.write(str(name_img) + ':\n')
        fo.write('\tbicubic: psnr:{:.2f}\tssim:{:.4f}\tzssr: psnr:{:.2f}\tssim:{:.4f}\n'
                 .format(psnr_bicubic, ssim_bicubic, psnr_zssr, ssim_zssr))
        return ssim_bicubic, psnr_bicubic, ssim_zssr, psnr_zssr


def test2(name_img, model, img, sr_factor, gt=False, img_gt=None):
    out_file = r'../output'
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    model.eval()

    img_bicubic = img.resize((int(img.size[0] * sr_factor),
                              int(img.size[1] * sr_factor)), resample=PIL.Image.BICUBIC)
    img_bicubic.save(os.path.join(out_file, name_img + '_bicubic.png'))

    input = transforms.ToTensor()(img)
    input = torch.unsqueeze(input, 0)
    input = input.to(device)
    with torch.no_grad():
        out = model(input)
    out = out.data.cpu()
    out = out.clamp(min=0, max=1)
    out = torch.squeeze(out, 0)
    out = transforms.ToPILImage()(out)
    out.save(os.path.join(out_file, name_img + '_zssr.png'))

    if gt:
        ssim_bicubic = compute_ssim(img_gt, img_bicubic)
        psnr_bicubic = compute_psnr(img_gt, img_bicubic)
        ssim_zssr = compute_ssim(img_gt, out)
        psnr_zssr = compute_psnr(img_gt, out)
        print("psnr_bicubic:\t{:.2f}".format(psnr_bicubic))
        print("ssim_bicubic:\t{:.4f}".format(ssim_bicubic))
        print("psnr_zssr:\t{:.2f}".format(psnr_zssr))
        print("ssim_zssr:\t{:.4f}".format(ssim_zssr))
        fo = open(os.path.join(out_file, 'PSNR_and_SSIM.txt'), mode='a')
        fo.write(str(name_img) + ':\n')
        fo.write('\tbicubic: psnr:{:.2f}\tssim:{:.4f}\tzssr: psnr:{:.2f}\tssim:{:.4f}\n'
                 .format(psnr_bicubic, ssim_bicubic, psnr_zssr, ssim_zssr))
        return ssim_bicubic, psnr_bicubic, ssim_zssr, psnr_zssr


def com(name_img, img, gt, img_gt, config):
    print(gt)
    t_img = transforms.ToTensor()(img)
    size = t_img.size()
    channel = size[0]

    crop_size = config.crop_size
    p_size = size[1:3]
    if crop_size > min(p_size[0:2]) or crop_size ** 2 > p_size[0] * p_size[1] // 4:
        crop_size = min(round(math.pow(p_size[0] * p_size[1] // 4, 0.5)), min(p_size[0:2]))
        crop_size = crop_size // 12 * 12
    print("crop_size:" + str(crop_size))

    model = ZSSRNet(input_channels=channel, sf=config.scale_factor)

    train_time = train(name_img, img, model, config.scale_factor, config.learning_rate, config.num_epoch,
                       config.noise_std,
                       crop_size, config.batch_size)

    ssim_bicubic, psnr_bicubic, ssim_zssr, psnr_zssr = \
        test(name_img, model, img, config.scale_factor, gt, img_gt)
    return ssim_bicubic, psnr_bicubic, ssim_zssr, psnr_zssr, train_time


if __name__ == "__main__":
    config = get_config()
    # img单个图片
    if os.path.isfile(config.img):
        root, name_img = os.path.split(config.img)
        name_img, ex = os.path.splitext(name_img)
        gt = False
        img_gt = None
        if os.path.exists(os.path.join(root, name_img + r"_gt" +ex)):
            gt = True
            gt_route = os.path.join(root, name_img + r"_gt" +ex)
            img_gt = Image.open(gt_route)
        img = Image.open(config.img)
        com(name_img, img, gt, img_gt, config)

    # img为文件夹
    elif os.path.isdir(config.img):
        sum_ssim_bicubic = 0
        sum_psnr_bicubic = 0
        sum_ssim_zssr = 0
        sum_psnr_zssr = 0
        sum_time = 0
        num = 0
        for root, dirs, files in os.walk(config.img):
            progress = tqdm(files)
            for img in progress:
                name_img, ex = os.path.splitext(img)
                gt = True
                gt_route = os.path.join(config.gt, img)
                img_gt = Image.open(gt_route)
                img = Image.open(os.path.join(root, img))
                ssim_bicubic, psnr_bicubic, ssim_zssr, psnr_zssr, train_time = \
                    com(name_img, img, gt, img_gt, config)
                sum_psnr_bicubic = sum_psnr_bicubic + psnr_bicubic
                sum_ssim_bicubic = sum_ssim_bicubic + ssim_bicubic
                sum_psnr_zssr = sum_psnr_zssr + psnr_zssr
                sum_ssim_zssr = sum_ssim_zssr + ssim_zssr
                sum_time = sum_time + train_time
                num = num + 1
        print("ave_ssim_bicubic:\t{:.4f}".format(sum_ssim_bicubic / num))
        print("ave_ssim_zssr:\t\t{:.4f}".format(sum_ssim_zssr / num))
        print("ave_psnr_bicubic:\t{:.2f}".format(sum_psnr_bicubic / num))
        print("ave_psnr_zssr:\t\t{:.2f}".format(sum_psnr_zssr / num))
        print('train time:{:.1f}'.format(sum_time))
        print("total number:{}".format(num))
