import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import torch


def trans(high_resolution, low_resolution, sub_image_size):
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
    w, h = high_resolution.size
    sw, sh = sub_image_size, sub_image_size

    if w < sw or h < sh:
        sh, sw = h // 2, w // 2

    i = random.randint(0, h - sh)
    j = random.randint(0, w - sw)

    high_resolution = TF.crop(high_resolution, i, j, sh, sw)
    low_resolution = TF.crop(low_resolution, i, j, sh, sw)


    high_resolution = TF.to_tensor(high_resolution)
    #high_resolution = torch.unsqueeze(high_resolution, 0)
    low_resolution = TF.to_tensor(low_resolution)
    #low_resolution = torch.unsqueeze(low_resolution, 0)


    return high_resolution, low_resolution
