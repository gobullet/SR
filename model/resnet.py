import torch
import torch.nn as nn
from model.res_block import ResBlock
import math
from torch.nn.functional import interpolate


class ResNet(nn.Module):
    def __init__(self, input_channels=3, sf=2, kernel_size=3, channels=64):
        super(ResNet, self).__init__()
        self.sf = sf
        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=True)
        self.res0 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res1 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res2 = ResBlock(channels, channels, kernel_size=kernel_size)

        self.conv1 = nn.Conv2d(channels, input_channels * sf ** 2, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=True)
        self.sub_pixel = nn.PixelShuffle(sf)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        with torch.no_grad():
            bibubic = interpolate(x, scale_factor=self.sf)
        fea = self.relu(self.conv0(x))
        res = self.res0(fea)
        res = self.res1(res)
        res = self.res2(res)
        res = 0.1 * res + fea
        res = self.sub_pixel(self.conv1(res))
        out = 0.1 * res + bibubic
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

