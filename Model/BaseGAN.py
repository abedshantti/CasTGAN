import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn


class BaseGAN:

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
    
    def save(self, path):
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model

    # TODO: define dataloading

    # TODO: define all supporting functions

    # TODO: define sample
