import torch
import torch.nn as nn
from model.res_block import ResBlock


class Res(nn.Module):
    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(Res, self).__init__()


        self.conv0 = nn.Conv2d(input_channels,channels,kernel_size=kernel_size,padding=kernel_size // 2, bias=True)
        self.res0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.res1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.res2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.oneconv = nn.Conv2d(channels, input_channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv0(x)
        res =self.res0(x)
        res = self.res1(res)
        res = self.res2(res)
        res = self.oneconv(res)
        out = self.relu(res + x)
        return out
