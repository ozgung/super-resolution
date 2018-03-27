import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img_gray(filepath):
    '''Load as grayscale'''
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def load_img_rgb(filepath):
    '''Load as grayscale'''
    img = Image.open(filepath)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, convert_gray=False):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.convert_gray = convert_gray

    def __getitem__(self, index):
        if self.convert_gray:
            input = load_img_gray(self.image_filenames[index])
        else:
            input = load_img_rgb(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
