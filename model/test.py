import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class TestNet(nn.Module):
    def __init__(self, input_channels=3, sf=2, kernel_size=3, channels=64):
        super(TestNet, self).__init__()
        self.sf = sf

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv7 = nn.Conv2d(channels, input_channels * sf ** 2, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=True)
        self.sub_pixel = nn.PixelShuffle(sf)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        with torch.no_grad():
            bibubic = interpolate(x, scale_factor=self.sf)

        fea = self.relu(self.conv0(x))
        res = self.relu(self.conv1(fea))
        res = self.relu(self.conv2(res))
        res = self.relu(self.conv3(res))
        res = self.relu(self.conv4(res))
        res = self.relu(self.conv5(res))
        res = self.relu(self.conv6(res))
        res = self.sub_pixel(self.conv7(0.1 * res + fea))
        out = 0.1 * res + bibubic

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
