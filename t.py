import torch.nn.functional as F
import skimage
import skimage.io as io
from skimage import color, data
import matplotlib.pyplot as plt
from torchvision import transforms

if __name__ == '__main__':
    chelsea = data.logo()
    io.imshow(chelsea)
    t_chelsea = transforms.ToTensor()(chelsea)
    print(t_chelsea.size())