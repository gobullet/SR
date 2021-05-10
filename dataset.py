from torch.utils.data import Dataset
from PIL import Image
import os
import PIL
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms.functional as TF
import random


class Datasets(Dataset):
    def __init__(self, image, sf, noise_std, crop_size):
        self.image = image
        self.sf = sf
        self.noise_std = noise_std
        self.crop_size = crop_size

        self.hr_fathers = []
        self.lr_sons = []
        # 面积越大，被选中的可能性越大
        self.probability = []

        # downscale factor
        dsf = self._find_factor()

        for i in dsf:
            hr = self.image.resize((int(self.image.size[0] * i),
                                    int(self.image.size[1] * i)),
                                   resample=PIL.Image.BICUBIC)
            self.hr_fathers.append(hr)

            lr = self._father_to_son(hr)
            self.lr_sons.append(lr)

            self.probability.append(hr.size[0] * hr.size[1])

    def __getitem__(self, item):
        lr = self.lr_sons[item]
        hr = self.hr_fathers[item]

        hr, lr = self._trans(hr, lr)

        # fill to subsieze*subsize
        size = lr.size()
        if size[1] < self.crop_size or size[2] < self.crop_size:
            hr = self._fill(hr, self.crop_size)
            lr = self._fill(lr, self.crop_size)

        images = {'lr': lr, 'hr': hr}
        return images

    def __len__(self):
        return len(self.hr_fathers)

    def _find_factor(self):
        smaller_side = min(self.image.size[0: 2])
        larger_side = max(self.image.size[0: 2])

        factors = []
        for i in range(smaller_side // 5, smaller_side + 1):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side) / smaller_side
            downsampled_larger_side = round(larger_side * zoom)
            if downsampled_smaller_side % self.sf == 0 and \
                    downsampled_larger_side % self.sf == 0:
                factors.append(zoom)
        return factors

    def _father_to_son(self, hr):
        # sf倍缩小
        lr = hr.resize(((hr.size[0] // self.sf),
                        (hr.size[1] // self.sf)))
        # 加噪
        t_lr = transforms.ToTensor()(lr)
        t_lr = t_lr + (self.noise_std * torch.randn(t_lr.size())).clamp(min=0, max=1)
        lr = transforms.ToPILImage()(t_lr)
        # 放大sf倍，作为输入
        lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)
        return lr

    def _trans(self, high_resolution, low_resolution):
        # mirror reflections
        if random.random() > 0.5:
            high_resolution = TF.vflip(high_resolution)
            low_resolution = TF.vflip(low_resolution)

        if random.random() > 0.5:
            high_resolution = TF.hflip(high_resolution)
            low_resolution = TF.hflip(low_resolution)

        # rotation
        angle = random.choice([0, 90, 180, 270])
        high_resolution = TF.rotate(high_resolution, angle, expand=True)
        low_resolution = TF.rotate(low_resolution, angle, expand=True)

        # random crop
        w, h = low_resolution.size
        tw, th = self.crop_size, self.crop_size
        if w < tw:
            tw = w // 2
        if h < th:
            th = h// 2

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        high_resolution = TF.crop(high_resolution, i, j, th, tw)
        low_resolution = TF.crop(low_resolution, i, j, th, tw)

        high_resolution = TF.to_tensor(high_resolution)
        low_resolution = TF.to_tensor(low_resolution)
        return high_resolution, low_resolution

    def _fill(self, tensor, size):
        t = tensor
        tensor_size = t.size()
        dh = size - tensor_size[1]
        dw = size - tensor_size[2]
        zero1 = torch.zeros((tensor_size[0], dh, tensor_size[2]), dtype=torch.float32)
        zero2 = torch.zeros((tensor_size[0], size, dw), dtype=torch.float32)
        t = torch.cat((t, zero1), 1)
        t = torch.cat((t, zero2), 2)
        return t
