import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, input_channels=64, channels=64, kernel_size=3):
        super(ResBlock, self).__init__()

        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        res = self.relu(self.conv0(x))
        res = self.conv1(res)
        out = res + x
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')


