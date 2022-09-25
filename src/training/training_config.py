import shutil

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


class TrainingConfig:
    def __init__(
        self,
        custom_layer: nn.Module,
        overwrite: bool,
        batch_size: int,
        epochs: int,
        learning_rate: float,
    ):
        # Batch size for training
        self.batch_size: int = batch_size

        # Number of folds for cross validation
        self.k_folds: int = 2

        # Number of epochs to train for
        self.epochs: int = epochs

        # Learning rate
        self.learning_rate: float = learning_rate
        self.learning_rate_patience: int = 3

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

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=self.learning_rate_patience,
            threshold=0.0000001,
            threshold_mode="abs",
        )
        self.loss = nn.CrossEntropyLoss()

        self.overwrite_previous = overwrite
        self.tb = None
