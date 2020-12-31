import torch.nn as nn
from torchvision import models

from .config import Config

class FineTuneModel(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self, num_labels):
        if self._model_backbone == "resnet18":
            model_ft = models.resnet18(pretrained=self._pretrain)
            num_ftrs = model_ft.fc.in_features

            model_ft.fc = nn.Linear(num_ftrs, num_labels)

            return model_ft

