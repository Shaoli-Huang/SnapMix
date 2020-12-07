import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
from datasets.tfs import get_nabirds_transform
import numpy as np

import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, tta=None):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'cat_num'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        cat_nums = data.cat_num.unique().tolist()
        cat_nums.sort()
        data['label'] = data['cat_num'].apply(lambda x: cat_nums.index(x))
        data = data[data['train_flag'] == train]

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.tta = tta

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))

        if self.tta is None:
            img = self.transform(img)

        elif self.tta == 'flip':
            img_1 = self.transform(img)
            img_2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_2 = self.transform(img_2)
            img = torch.stack((img_1, img_2), dim=0)
        else:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_dataset(conf):

    datadir = 'data/nabirds'

    if conf and 'datadir' in conf:
        datadir = conf.datadir

    conf['num_class'] = 555

    transform_train,transform_test = get_nabirds_transform(conf)

    ds_train = ImageLoader(datadir, train=True, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)


    return ds_train,ds_test
