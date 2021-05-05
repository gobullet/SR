import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--crop_size', type=int, default=120, help='the height / width of the hr image to network')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--num_epoch', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for super resolution')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Base learning rate for Adam')
parser.add_argument('--img', type=str, help='Path to input img', default=r'comic.png')
parser.add_argument('--gt', type=str, help='Path to ground truth')
parser.add_argument('--noise_std', type=float, default=0, help='add Gaussian noise to lrsons')


def get_config():
    return parser.parse_args()
