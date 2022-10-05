import shutil
from typing import Optional

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from src.models.unet import UNet  # type: ignore
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from datetime import datetime


class TestingConfig:
    def __init__(self, custom_layer: Optional[nn.Module], batch_size: int):
        # Batch size for training
        self.batch_size: int = batch_size

        # Input shape
        self.input_h = 512
        self.input_w = 512
        self.channels = 1
        self.classes = 1
        # Mode layers definition
        self.net = UNet(
            n_channels=self.channels,
            n_classes=self.classes,
            window_layer=custom_layer,
        ).float()

        self.tb: Optional[SummaryWriter] = None
