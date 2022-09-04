import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from src.models.unet import UNet
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os


class TrainingConfig:
    def __init__(self, custom_layer: nn.Module = None):
        # Batch size for training
        self.batch_size: int = 4

        # Number of folds for cross validation
        self.k_folds: int = 5

        # Number of epochs to train for
        self.epochs: int = 100

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3

        # Input shape
        self.input_h = 512
        self.input_w = 512
        self.channels = 1
        self.classes = 2
        # Mode layers definition
        self.net = UNet(
            n_channels=self.channels,
            n_classes=self.classes,
            bilinear=False,
            custom_window_layer=custom_layer,
        ).float()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=1e-4
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

    def dice_coeff(
        self,
        input: Tensor,
        target: Tensor,
        reduce_batch_first: bool = False,
        epsilon: float = 1e-6,
    ) -> float:
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(
                f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
            )

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0.0
            for i in range(input.shape[0]):
                dice += self.dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]

    def multiclass_dice_coeff(
        self,
        input: Tensor,
        target: Tensor,
        reduce_batch_first: bool = False,
        epsilon: float = 1e-6,
    ) -> float:
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0.0
        for channel in range(input.shape[1]):
            dice += self.dice_coeff(
                input[:, channel, ...],
                target[:, channel, ...],
                reduce_batch_first,
                epsilon,
            )

        return dice / input.shape[1]

    def dice_loss(
        self, input: Tensor, target: Tensor, multiclass: bool = False
    ) -> float:
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

    def calc_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: dict,
    ) -> None:
        bce = F.binary_cross_entropy_with_logits(pred, target)

        dice = self.multiclass_dice_coeff(pred, target, reduce_batch_first=True)

        metrics["bce"] += bce.item() * target.size(0)
        metrics["dice"] += dice * target.size(0)

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
