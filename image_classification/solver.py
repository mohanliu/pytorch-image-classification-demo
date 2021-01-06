from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import gc
import shutil

from .config import Config
from .dataset import FTDataLoader
from .model import FineTuneModel

class Solver(Config):
    def __init__(self, gpu_number=0, **kwargs):
        super().__init__(**kwargs)

        # get device
        self._device = torch.device("cuda:{}".format(gpu_number) if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self._device))

        # prepare data
        self._get_dataset()
        
        # prepare model
        self._get_or_load_model()
        self._set_optimizer(self.model.parameters())
        self._set_criterion()
        self._set_learningrate_scheduler()

    def _get_dataset(self):
        for p in ["train", "val"]:
            temp_d = FTDataLoader(phase=p)
            setattr(self, "{}_dataloader".format(p), temp_d.dataloader)
            setattr(self, "{}_datasize".format(p), temp_d._size)

            if p == "train":
                self.classes = temp_d._classes

    def _get_or_load_model(self):
        self.model = FineTuneModel().get_model(len(self.classes))
        self.model.to(self._device)

    def _set_optimizer(self, parameters):
        self.optimizer = optim.SGD(
            parameters,
            lr=self._learning_rate,
            momentum=self._momentum,
        )

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _set_learningrate_scheduler(self):
        if self._lr_scheduler_dict["__name__"] == "step_lr":
            self.lr_scheduler = lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self._lr_scheduler_dict.get("step_size", 7), 
                gamma=self._lr_scheduler_dict.get("gamma", 0.1)
            )
            
    def save_checkpoint(self, state, epoch, filename='checkpoint.pth.tar'):
        if not os.path.exists(self._snapshot_folder):
            os.makedirs(self._snapshot_folder)
        
        absolute_path = os.path.join(self._snapshot_folder, "epoch_{}_{}".format(epoch, filename))
        torch.save(state, absolute_path)
        
    def update_best_model(self, epoch, filename='checkpoint.pth.tar'):
        current_absolute_path = os.path.join(self._snapshot_folder, "epoch_{}_{}".format(epoch, filename))
        best_absolute_path = os.path.join(self._snapshot_folder, "best_{}".format(filename))
        shutil.copyfile(current_absolute_path, best_absolute_path)
        
    def restore_model(self, epoch=-1, filename='checkpoint.pth.tar'):
        if epoch == -1:
            model_path = "best_{}".format(filename)
        else:
            model_path = "epoch_{}_{}".format(epoch, filename)
        
        model_fullpath = os.path.join(self._snapshot_folder, model_path)
        
        checkpoint = torch.load(model_fullpath, map_location=self._device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self._device)

    def train(self):
        print('Start training...')
        since_ = time.time()

#         best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self._num_epochs):
            print('Epoch {}/{}'.format(epoch, self._num_epochs - 1))
            print('-' * 10)

            
            loss_dict = {}
            acc_dict = {}
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in getattr(self, "{}_dataloader".format(phase)):
                    # map data to device
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                    # zero the parameter gradients
                    # clears old gradients from the last step 
                    # (otherwise you’d just accumulate the gradients from all loss.backward() calls).
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    # set_grad_enabled() can be used to conditionally enable gradients.
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # computes the derivative of the loss w.r.t. the parameters
                            # (or anything requiring gradients) using backpropagation.
                            loss.backward()
                            
                            # Performs a single optimization step (parameter update).
                            self.optimizer.step() 

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # learning rate update for training only
                if phase == 'train':
                    # update learning rate with learning rate scheduler
                    self.lr_scheduler.step()

                # save stats and display
                epoch_loss = running_loss / getattr(
                    self, "{}_datasize".format(phase)
                )
                loss_dict[phase] = epoch_loss
                
                epoch_acc = running_corrects.double() / getattr(
                    self, "{}_datasize".format(phase)
                )
                acc_dict[phase] = epoch_acc

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc
                ))
                
                # save intermediate model state, params and etc.
                if phase == 'val':
                    self.save_checkpoint(
                        {
                            'lr': self.optimizer.param_groups[0]["lr"],
                            'state_dict': self.model.state_dict(),
                            'loss_stats': loss_dict,
                            'acc_stats': acc_dict
                        },
                        epoch
                    )

                    # deep copy the model
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.update_best_model(epoch)

            print()
            gc.collect()
            torch.cuda.empty_cache()

        time_elapsed = time.time() - since_
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        
    def evaluate(self, epoch):
        # load model
        self.restore_model(epoch)
        
        # Set model to evaluate mode
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
            
        for inputs, labels in self.val_dataloader:
            # map data to device
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        total_loss = running_loss / self.val_datasize
        total_acc = running_corrects / self.val_datasize
            
        print("Total Loss: {:.4f}, Acc: {:.4f}".format(total_loss, total_acc))
        
    def inference(self, epoch):
        # TO DO
        pass