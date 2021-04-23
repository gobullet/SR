import torch
import torch.nn as nn
from model.res_block import ResBlock


class ResNet(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.res0 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res1 = ResBlock(channels, channels, kernel_size=kernel_size)
        self.res2 = ResBlock(channels, channels, kernel_size=kernel_size)

        self.conv1 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        res = self.conv0(x)
        res = self.res0(res)
        res = self.res1(res)
        res = self.res2(res)
        res = self.conv1(res)
        out = res + x
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
