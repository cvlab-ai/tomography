import torch
import os
import torch.nn as nn
from torchmetrics import Dice
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard(log_dir: str) -> SummaryWriter:
    """
    Create the tensorboard object to use for logging.
    :param log_dir: Directory to store logs in
    """
    # Check if dir runs/name exists, if yes, print error and exit
    path = os.path.join("runs", log_dir)
    if os.path.exists(path):
        print("Training already exists")
        exit()
    return SummaryWriter(f"runs/{log_dir}")


def close_tensorboard(tb: SummaryWriter) -> None:
    """
    Close the tensorboard writer.
    """
    tb.close()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - dice_coeff(pred, target)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate dice coefficient
    :param pred: predicted mask
    :param target: target mask
    :return:
    """
    smooth = 1.0
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def jsc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate jacard similarity coefficient
    :param pred: predicted mask
    :param target: target mask
    """
    smooth = 1.0
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (intersection + smooth) / (iflat.sum() + tflat.sum() - intersection + smooth)


def calc_metrics(
    pred: torch.Tensor, target: torch.Tensor, metrics: dict, device: str
) -> None:
    dice_coeff_metric = Dice(average="micro", multiclass=True).to(device)
    dice = dice_coeff_metric(pred, target)
    metrics["dice"] += dice.item() * target.size(0)
    metrics["jsc"] += jsc(pred, target).item() * target.size(0)


def print_metrics(
    tb: SummaryWriter, metrics: dict, epoch_samples: int, phase: str, epoch: int
) -> None:
    """
    Print out the metrics to stdout.
    :param tb: SummaryWriter object to write to
    :param metrics: dictionary containing the metrics
    :param epoch_samples: number of samples in this epoch
    :param phase: train/val
    :param epoch: current epoch
    """
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))
    # Dump the metrics to tensorboard
    for k, v in metrics.items():
        tb.add_scalar(f"{phase}-{k}", v / epoch_samples, epoch)


def save_model(net: nn.Module, name: str) -> None:
    """
    Save the model to disk.
    """
    torch.save(net.state_dict(), name)
