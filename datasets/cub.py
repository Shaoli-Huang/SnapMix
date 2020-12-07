import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from datasets.tfs import get_cub_transform

import pdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        bounding_box = pd.read_csv(os.path.join(root, "bounding_boxes.txt"), sep=" ", header=None,  names=['idx', 'x', 'y', 'w', 'h'])
        data = pd.concat([img_paths, img_labels, train_test_split, bounding_box], axis=1)
        data['label'] = data['label'] - 1
        alldata = data.copy()

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
        print('num of data:{}'.format(len(imgs)))

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
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_dataset(conf):

    datadir = 'data/cub'

    if conf and 'datadir' in conf:
        datadir = conf.datadir

    conf['num_class'] = 200

    transform_train,transform_test = get_cub_transform(conf)

    ds_train = ImageLoader(datadir, train=True, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test)


    return ds_train,ds_test
