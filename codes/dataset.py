import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

from .config import Config

class Dataset(Config):
    def __init__(self, phase="train", **kwargs):
        # intialize config
        super().__init__()

        # current phase
        self._phase = phase

    _data_transforms = None
    @property
    def data_transforms(self):
        if self._data_transforms is None:
            self._data_transforms = self._set_data_transforms()
        return self._data_transforms

    @data_transforms.setter
    def data_transforms(self, data_transforms):
        self._data_transforms = data_transforms

    def _set_data_transforms(self):
        """Function to set up data augmentation
        
        Data augmentation and normalization for training; just normalization for validation
        """
        if self._phase == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif self._phase == "val":
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

    _image_dataset = None
    @property
    def image_dataset(self):
        if self._image_dataset is None:
            self._image_dataset = self._get_image_dataset()
        return self._image_dataset
    
    @image_dataset.setter
    def image_dataset(self, image_dataset):
        self._image_dataset = image_dataset

    def _get_image_dataset(self):
        return datasets.ImageFolder(
            os.path.join(self._datapath, self._phase),
            self.data_transforms
        )