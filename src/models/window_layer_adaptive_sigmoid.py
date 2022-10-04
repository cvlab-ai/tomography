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

        center_grad = center is not None
        width_grad = width is not None
        center_init = 0 if center is None else center
        width_init = 1 if width is None else width

        # initialize center and width with the given values
        self.center = Parameter(torch.Tensor(1), requires_grad=center_grad)
        torch.nn.init.constant(self.center, center_init)

        self.width = Parameter(torch.Tensor(1), requires_grad=width_grad)
        torch.nn.init.constant(self.center, width_init)

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        lower_level = self.center - (self.width / 2)
        upper_level = self.center + (self.width / 2)
        y = torch.clamp_(x, lower_level, upper_level)
        return nn.functional.sigmoid(y)
