import torch
import torch.nn as nn


class CustomLeaklyRelu(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        """

        Custom LeaklyRelu layer for modules. Default negative slope is 0.2.

        :param negative_slope : negative slope value for leaky relu.

        """
        super().__init__()

        self.leakly_relu = nn.LeaklyRelu(negative_slope=negative_slope)

    def forward(self, x) -> torch.Tensor:
        """

        :param x: torch.Tensor
            :shape: (b, c, h, w)

        :return : torch.Tensor
            :shape: (b, c, h, w)

        """

        return self.leakly_relu(x)
