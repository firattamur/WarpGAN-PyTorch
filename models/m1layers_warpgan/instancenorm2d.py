import torch
import torch.nn as nn


class CustomInstanceNorm2d(nn.Module):

    def __init__(self, num_features: int):
        """

        Custom InstanceNorm layer for modules to multiple norm with gamma and sum with beta.

        :param num_features: C from an expected input of size (N, C, H, W).

        """
        super(CustomInstanceNorm2d, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features)


    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """

        :param x: torch.Tensor
            :shape: (b, c, h, w)

        :param gamma: style controller encodings
            :shape: (batch_size, c, 1, 1)

        :param beta : style controller encodings
            :shape: (batch_size, c, 1, 1)

        """

        return gamma * self.instance_norm(x) + beta