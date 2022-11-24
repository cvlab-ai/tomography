import torch
import os
import torch.nn as nn
from torchmetrics import Dice
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard(log_dir: str, log_name: str) -> SummaryWriter:
    """
    Create the tensorboard object to use for logging.
    :param log_dir: Directory to store logs in
    :param log_name: Name of log
    """
    # Check if dir runs/name exists, if yes, print error and exit

    path = os.path.join("runs", log_dir, log_name)
    if os.path.exists(path):
        print("Training already exists")
        exit()
    return SummaryWriter(path)


def close_tensorboard(tb: SummaryWriter) -> None:
    """
    Close the tensorboard writer.
    """
    tb.close()


def multi_label_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    num_of_classes: int,
) -> float:
    """
    Calculate the dice score for a batch of images.
    :param preds: Batch of predicted masks
    :param targets: Batch of target masks
    :param device: device str
    :param num_of_classes: number of classes
    :return: Dice score, mean of all classes
    """
    dice = Dice(multiclass=True, num_classes=num_of_classes, ignore_index=0).to(device)
    return dice(preds, targets).item()


def calc_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: dict,
    device: torch.device,
    num_of_classes: int,
) -> None:
    dice_score = torch.Tensor([0 for _ in range(num_of_classes)]).to(device)
    dice_coeff_metric = Dice().to(device)
    for i in range(num_of_classes):
        # Ignore background
        if num_of_classes > 1 and i == 0:
            continue
        elif num_of_classes == 1:
            dice_score = dice_coeff_metric(pred, target)
        else:
            dice_score[i] = dice_coeff_metric(pred[:, i], target[:, i])
    if num_of_classes > 1:
        metrics["dice_liver"] += dice_score[1].item()
        metrics["dice_tumor"] += dice_score[2].item()
        metrics["dice"] += dice_score[1:].mean().item()
    else:
        metrics["dice"] += dice_score.item()


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


def norm_point(point):
    return (point + 1024) / 2560 - 1


def denorm_point(point):
    return (point + 1) * 2560 - 1024


def torch_renorm(x, width, center):
    width_back = torch.sub(torch.mul(torch.add(width, 1), 2560), 1024)
    return torch.div(torch.sub(x, center), torch.div(width_back, 5120))


class BinaryDiceLoss(nn.Module):
    r"""Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception(f"Unexpected reduction {self.reduction}")


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape do not match"
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert (
                        self.weight.shape[0] == target.shape[1]
                    ), "Expect weight shape [{}], get[{}]".format(
                        target.shape[1], self.weight.shape[0]
                    )
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / (target.shape[1] - (0 if self.ignore_index is None else 1))
