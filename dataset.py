from torch.utils.data import Dataset
from PIL import Image
import os
from trans import trans
import PIL
import torchvision.transforms as transforms
import torch


class Datasets(Dataset):
    def __init__(self, image, sf, noise_std, sub_image_size):
        self.image = image
        self.sf = sf
        self.noise_std = noise_std
        self.sub_image_size = sub_image_size

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

        hr, lr = trans(hr, lr, self.sub_image_size)

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

        lr = hr.resize(((hr.size[0] // self.sf),
                        (hr.size[1] // self.sf)))

        # 加噪
        t_lr = transforms.ToTensor()(lr)
        t_lr = t_lr + (self.noise_std * torch.randn(t_lr.size())).clamp(min=0, max=1)
        lr = transforms.ToPILImage()(t_lr)

        lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)

        return lr
