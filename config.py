import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--sub_image_size', type=int, default=128, help='the height / width of the hr image to network')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_batches', type=int, default=15000, help='number of epochs to train for')
parser.add_argument('--scale_factor', type=int, default=4, help='scale factor for super resolution')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Base learning rate for Adam')
parser.add_argument('--img', type=str, help='Path to input img', default=r'monarch.png')
parser.add_argument('--noise_std', type=float, default=0, help='add Gaussian noise to lrsons')


def get_config():
    return parser.parse_args()
