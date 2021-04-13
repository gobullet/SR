
from torch.utils.data import Dataset
from PIL import Image
import os
from tansform import transform


class Datasets(Dataset):
    def  __init__(self, image_size, scale):
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists(r'../data/'):
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(os.listdir(os.path.join(r'../data', 'DIV2K_train_HR')))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join(r'../data', 'hr', file_name)).convert('RGB')
        low_resolution = Image.open(os.path.join(r'../data', 'lr', file_name)).convert('RGB')

        high_resolution, low_resolution = transform(high_resolution, low_resolution)

        """if random() > 0.5:
            high_resolution = TF.vflip(high_resolution)
            low_resolution = TF.vflip(low_resolution)

        if random() > 0.5:
            high_resolution = TF.hflip(high_resolution)
            low_resolution = TF.hflip(low_resolution)

        high_resolution = TF.to_tensor(high_resolution)
        low_resolution = TF.to_tensor(low_resolution)
        """

        images = {'lr': low_resolution, 'hr': high_resolution}

        return images


    def __len__(self):
        return len(self.image_file_name)


