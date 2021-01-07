from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import gc
import shutil
import logging
from PIL import Image

from .config import Config
from .model import FineTuneModel

logger = logging.getLogger("image_classification.inference")

class ImageClassification(Config):
    def __init__(self, weight_path, gpu_number=0, **kwargs):
        super().__init__(**kwargs)
        
        # get device
        self._set_device(gpu_number)
        
        # prepare model
        self._load_model_weights(weight_path)
        
    def _set_device(self, gpu_number):
        if isinstance(gpu_number, int):
            gpu_number_list = [gpu_number]
        else:
            gpu_number_list = gpu_number

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_number_list))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_gpu = len(gpu_number_list)
        logger.info("Using {} GPUs: {}".format(torch.cuda.device_count(), str(gpu_number_list)))
            
    def _preprocess_data(self, image_path):
        inference_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_ = Image.open(image_path).convert('RGB')
        
        image_tensor = inference_transforms(image_).float().unsqueeze_(0)
    
        return image_tensor
    
    def _process_output(self, image_tensor):
        input_ = image_tensor.to(self._device)
        output_ = self.model(input_)
        
        raw_output = [
            np.round(v, 4) 
            for v in output_.data.cpu().numpy().tolist()[0]
        ]
        
        _, preds = torch.max(output_, 1)
        
        pred_index = preds.data.cpu().numpy()[0]
        
        pred_class = [
            k 
            for k, v in self._target_class_to_idx.items()
            if v == pred_index 
        ][0]
        
        return {
            "predicted_class": pred_class,
            "raw_output": raw_output,
            "predicted_label": pred_index
        }

    def _load_model_weights(self, weight_path):
        logger.info("Preparing model: {} ...".format(self._model_backbone))
        self.model = FineTuneModel().get_model(len(self._target_classes))
        
        logger.info("Preparing model: mapping to devices...")
        self.model = nn.DataParallel(self.model)
        self.model.to(self._device)
        
        logger.info("Loading weights: {} ...".format(weight_path))  
        checkpoint = torch.load(weight_path, map_location=self._device)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self._device)
        
        logger.info("Model is ready!")
        
        self.model.eval()
        
            
    def predict(self, image_path):
        image_tensor_ = self._preprocess_data(image_path)
        output_ = self._process_output(image_tensor_)
        
        output_.update({"image_path": image_path})
        
        return output_