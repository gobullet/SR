import torch
import torch.nn as nn
from model.res_block import ResBlock
import math


class ResNet(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=128):
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=True)
        self.conv01 = nn.Conv2d(channels * 3 // 2, channels, kernel_size=kernel_size,padding=kernel_size //2,bias=True)
        self.zeroconv = nn.Conv2d(channels * 3 // 2, channels, kernel_size=1)
        self.res0 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res1 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res2 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res3 = ResBlock(channels, channels, kernel_size=kernel_size)

        self.conv1 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        fea = self.conv0(x)
        #fea = self.zeroconv(fea)
        #fea = self.conv01(fea)
        res = self.res0(fea)
        res = self.res1(res)
        res = self.res2(res)
        #res = self.res3(res)
        res = res + fea
        res = self.conv1(res)
        out = 0.1 * res + x
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    """
