from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import gc
import shutil
import logging

from .config import Config
from .dataset import FTDataLoader
from .model import FineTuneModel

logger = logging.getLogger("image_classification.inference")

class ImageClassification(Config):
    def __init__(self, gpu_number=0, **kwargs):
        super().__init__(**kwargs)
        
        # get device
        self._set_device(gpu_number)

        # prepare data
        self._get_dataset()
        
        # prepare model
        self._get_or_load_model()
        
    def _set_device(self, gpu_number):
        if isinstance(gpu_number, int):
            self._device = torch.device("cuda:{}".format(gpu_number) if torch.cuda.is_available() else "cpu")
            self._multi_gpu_mode = False
            self._num_gpu = 1
            logger.info("Using single device: {}".format(self._device))

        elif isinstance(gpu_number, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_number))
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._multi_gpu_mode = True
            self._num_gpu = len(gpu_number)
            logger.info("Using {} GPUs: {}".format(torch.cuda.device_count(), str(gpu_number)))
            
    def _preprocess_data(self, image_dict):
        for p in ["train", "val"]:
            temp_d = FTDataLoader(phase=p, global_batch_size=self._num_gpu * self._batch_size)
            setattr(self, "{}_dataloader".format(p), temp_d.dataloader)
            setattr(self, "{}_datasize".format(p), temp_d._size)

            if p == "train":
                self.classes = temp_d._classes

    def _get_or_load_model(self):
        self.model = FineTuneModel().get_model(len(self.classes))
        if self._multi_gpu_mode:
            self.model = nn.DataParallel(self.model)
            self.model.to(self._device)
        else:
            self.model.to(self._device)
            
    def predict(self, image_dict):
        return