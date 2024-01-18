import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets

from .datasets import register

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, image_size=224, box_size=256, **kwargs):
        if box_size is None:
            box_size = image_size

        if 'image_size' in kwargs:
            image_size = kwargs.get('image_size')
        if 'box_size' in kwargs:
            box_size = kwargs.get('box_size')
        print('image_size:', image_size)
        print('box_size:', box_size)
        self.filepaths = []
        self.label = []
        self.classes = sorted(os.listdir(root_path))
        self.id2classes = {}
        self.rootpath = root_path

        for i, c in enumerate(self.classes):
            self.id2classes[i] = c
            if not os.path.isdir(os.path.join(root_path, c)):
                continue
            for filename in sorted(os.listdir(os.path.join(root_path, c))):
                self.filepaths.append(os.path.join(root_path, c, filename))
                self.label.append(i)
        self.n_classes = max(self.label) + 1
        
        mean, std = [0.5723625, 0.34657937, 0.2374997], [0.21822436, 0.19240488, 0.17723322]
        norm_params = {'mean': mean,
                       'std': std }
        normalize = transforms.Normalize(**norm_params)
        if kwargs.get('augment'):
            print('use data augment.')
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(box_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')
        return self.transform(img), self.label[i]

