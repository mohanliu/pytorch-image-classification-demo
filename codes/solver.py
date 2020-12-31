from __future__ import print_function, division

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
from .dataset import FTDataset
from .model import FineTuneModel

class Solver(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # get device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare data
        self._get_dataset()

    def _get_dataset(self):
        for p in ["train", "val"]:
            temp_d = FTDataset(phase=p)
            setattr(self, "{}_dataloader".format(p), temp_d.dataloader)
            setattr(self, "{}_datasize".format(p), temp_d._size)

            if p == "train":
                self.classes = temp_d._classes

    def _get_or_load_model(self):
        self.model = FineTuneModel().get_model(len(self.classes))

    def _set_optimizer(self, parameters, **kwargs):
        pass

    def _set_lossfunc(self):
        pass

    def _set_learningrate(self):
        pass

    def train(self):
        pass



