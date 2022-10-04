import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class WindowLayerAdaptiveSigmoid(nn.Module):
    """
    Implementation of HU window activation. It applies sigmoid((x-center)/width) to the input elementwise.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - center - trainable parameter
        - width - trainable parameter
    """

    def __init__(self, center=150.0, width=2000.0):
        """
        Initialization.
        INPUT:
            - center: center point of the window
            - width: trainable parameter
        """
        super().__init__()

        # initialize center and width with the given values
        if center is None:
            self.center = Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.center = Parameter(torch.tensor(center), requires_grad=True)
        if width is None:
            self.width = Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.width = Parameter(torch.tensor(width / 5), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return nn.functional.sigmoid((x - self.center.item()) / self.width.item())
