import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

from .config import Config

class FTDataset(Config, Dataset):
    def __init__(self, phase="train", **kwargs):
        # intialize config
        super().__init__(**kwargs)

        # current phase
        self._phase = phase
        
        # load raw data
        self._prepare_data()

    _image_transforms = None
    @property
    def image_transforms(self):
        if self._image_transforms is None:
            self._image_transforms = self._set_image_transforms()
        return self._image_transforms

    @image_transforms.setter
    def image_transforms(self, image_transforms):
        self._image_transforms = image_transforms

    def _set_image_transforms(self):
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
            ])

    @property
    def data_location(self):
        return os.path.join(self._datapath, self._phase)
    
    def _prepare_data(self):
        """Function to load data from raw images with targets
        
        Attributes:
            classes: list of classes
            class_to_idx: dict of class indices
            image_labels: [
                                (image_path_1, target_1), 
                                (image_path_2, target_2),
                                ...
                          ]
        """
        self.image_labels = [
            (f, os.path.basename(os.path.dirname(f)))
            for f in glob.glob(os.path.join(self.data_location, "*", "*"))
        ]
        
        self.classes = list(set([v[1] for v in self.image_labels]))
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(sorted(self.classes))
        }
    
    def __getitem__(self, idx):
        filename, target = self.image_labels[idx]
        
        img = Image.open(filename).convert('RGB')
        img_ = self.image_transforms(img)
        
        label_ = self.class_to_idx[target]

        return img_, label_
    
    def __len__(self):
        return len(self.image_labels)

class FTDataLoader(Config):
    def __init__(self, phase="train", **kwargs):
        # intialize config
        super().__init__(**kwargs)

        # current phase
        self._phase = phase
        self.image_dataset = FTDataset(phase=phase, **kwargs)
        
        # set global batch size (for multi-device training)
        self._global_batch_size = kwargs.get("global_batch_size", self._batch_size)

    _dataloader = None
    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = self._get_data_loader()
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader
        
    def _get_data_loader(self):
        return DataLoader(
            self.image_dataset, 
            batch_size=self._global_batch_size,
            shuffle=self._shuffle, 
            num_workers=self._num_worker
        )

    @property
    def _size(self):
        return len(self.image_dataset)

    @property
    def _classes(self):
        return self.image_dataset.classes