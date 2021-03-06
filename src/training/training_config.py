from src.models.unet import UNet
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class TrainingConfig:
    def __init__(self):
        # Batch size for training
        self.batch_size: int = 2

        # Number of folds for cross validation
        self.k_folds: int = 5

        # Number of epochs to train for
        self.epochs: int = 50

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3

        # Input shape
        self.input_h = 512
        self.input_w = 512
        self.channels = 1
        self.classes = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Mode layers definition
        self.net = UNet(
            n_channels=self.channels, n_classes=self.classes, bilinear=False
        ).float()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=1e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

    @staticmethod
    def dice_loss(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = 1 - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        )

        return loss.mean()

    def calc_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: dict,
        bce_weight: float = 0.5,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
        metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
        metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

        return loss

    @staticmethod
    def print_metrics(metrics: dict, epoch_samples: int, phase: str) -> None:
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
