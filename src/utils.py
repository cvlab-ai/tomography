from typing import Union

import torch
import numpy as np
from torch.autograd import Variable
from torch.autograd.function import Function


def dice_loss(outputs, labels):

    outputs, labels = outputs.float(), labels.float()
    intersect = torch.dot(outputs, labels)
    union = torch.add(torch.sum(outputs), torch.sum(labels))
    dice = 1 - (2 * intersect + 1e-5) / (union + 1e-5)
    return dice
