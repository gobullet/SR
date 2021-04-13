import torchvision
import torchvision.transforms.functional as TF
from random import random


def transform(high_resolution, low_resolution):
    if random() > 0.5:
        high_resolution = TF.vflip(high_resolution)
        low_resolution = TF.vflip(low_resolution)

    if random() > 0.5:
        high_resolution = TF.hflip(high_resolution)
        low_resolution = TF.hflip(low_resolution)

    angle = random.choice([0., 90., 180., 270.])
    high_resolution = TF.rotate(high_resolution, angle)
    low_resolution = TF.rotate(low_resolution, angle)

    high_resolution = TF.to_tensor(high_resolution)
    low_resolution = TF.to_tensor(low_resolution)

    t = torchvision.transforms.RandomRotation
