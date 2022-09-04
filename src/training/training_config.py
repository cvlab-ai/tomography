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


class TrainingConfig:
    def __init__(self, custom_layer: nn.Module = None):
        # Batch size for training
        self.batch_size: int = 4

        # Number of folds for cross validation
        self.k_folds: int = 2

        # Number of epochs to train for
        self.epochs: int = 100

        # Learning rate
        self.learning_rate: float = 0.01
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
        ).float()

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.learning_rate,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        self.loss = nn.CrossEntropyLoss()

    def create_tensorboard(self, log_dir: str) -> None:
        """
        Create the tensorboard object to use for logging.
        :param log_dir: Directory to store logs in
        """
        # Check if dir runs/name exists, if yes, print error and exit
        if os.path.exists(os.path.join("runs", log_dir)):
            print("Training already exists")
            exit()
        self.tb = SummaryWriter(f"runs/{log_dir}")

    def close_tensorboard(self) -> None:
        """
        Close the tensorboard writer.
        """
        self.tb.close()

    def dice_loss(self, pred, target):
        return 1 - self.dice_coeff(pred, target)

    def dice_coeff(self, pred, target):
        smooth = 1.0
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    def jsc(self, pred, target):
        """
        Calculate jacard similarity coefficient
        """
        smooth = 1.0
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (intersection + smooth) / (
            iflat.sum() + tflat.sum() - intersection + smooth
        )

    def calc_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: dict,
    ) -> None:
        dice = self.dice_coeff(pred, target)
        metrics["dice"] += dice * target.size(0)
        metrics["jsc"] += self.jsc(pred, target) * target.size(0)

    def print_metrics(
        self, metrics: dict, epoch_samples: int, phase: str, epoch: int
    ) -> None:
        """
        Print out the metrics to stdout.
        :param metrics: dictionary containing the metrics
        :param epoch_samples: number of samples in this epoch
        :param phase: train/val
        """
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))
        # Dump the metrics to tensorboard
        for k, v in metrics.items():
            self.tb.add_scalar(f"{phase}-{k}", v / epoch_samples, epoch)
