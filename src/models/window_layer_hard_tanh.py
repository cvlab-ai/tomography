import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class WindowLayerHardTanH(nn.Module):
    """
    Implementation of HU window activation. It applies hardtanh((x, center-width, center + width) to the input
    elementwise.
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
        self.center = Parameter(torch.Tensor([center]), requires_grad=True)
        self.width = Parameter(torch.Tensor([width]), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return torch.clamp_(
            x,
            torch.sub(self.center, torch.div(self.width, 2)),
            torch.add(self.center, torch.div(self.width, 2)),
        )
