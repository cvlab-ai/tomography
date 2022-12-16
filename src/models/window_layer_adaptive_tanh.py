from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from src.utils import norm_point, torch_renorm


class WindowLayerAdaptiveTanh(nn.Module):
    """
    Implementation of HU window activation. It applies tanh((x-center)/width) to the input elementwise.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - center - trainable parameter
        - width - trainable parameter
    """

    def __init__(self, n_windows: int = 1, centers: List[float] = None, widths: List[float] = None):
        """
        Initialization.
        INPUT:
            - center: center point of the window
            - width: trainable parameter
        """
        super().__init__()

        if centers is None:
            centers = [150.0] * n_windows

        if widths is None:
            widths = [2000.0] * n_windows

        # Apply norm point to each center and width
        centers = [norm_point(center) for center in centers]
        widths = [norm_point(width) for width in widths]

        # Initialize centers and widths tensors
        self.centers = Parameter(torch.Tensor(centers), requires_grad=True)
        self.widths = Parameter(torch.Tensor(widths), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        n = tuple(torch.tanh(torch_renorm(x, width, center)) for width, center in torch.stack((self.widths, self.centers), dim=0, out=None))
        x = torch.concat(n, dim=1)
        return x
