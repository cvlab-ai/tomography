import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class window_layer(nn.Module):
    """
    Implementation of HU window activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - center - trainable parameter
        - width - trainable parameter
    """

    def __init__(self, in_features, center=300.0, width=400.0):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - width: trainable parameter
            aplha is initialized with zero value by default
        """
        super().__init__()
        self.in_features = in_features

        # initialize center and width with the given values
        if center is None:
            self.center = Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.center = Parameter(torch.tensor(center), requires_grad=True)
        if width is None:
            self.width = Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.width = Parameter(torch.tensor(width), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return nn.functional.hardtanh(
            x,
            self.center.item() - self.width.item(),
            self.center.item() + self.width.item(),
            True,
        )
