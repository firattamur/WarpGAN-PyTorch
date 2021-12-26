import torch
import torch.nn as nn


class CustomUpScale2d(nn.Module):
    """

    Upscales an input tensor by specified factor.
    Upscaling done by torch.tile: https://pytorch.org/docs/stable/generated/torch.tile.html

    """

    def __init__(self, factor: int = 2):
        """

        :param factor: scaling factor.

        """
        super(CustomUpScale2d, self).__init__()

        assert factor >= 1, "Scaling factor must be greater than 0!"

        self.factor = factor

    def forward(self, x) -> torch.Tensor:
        """

        :param x: torch.Tensor
            :shape: (b, c, h, w)

        :return : torch.Tensor
            :shape: (b, c, h * self.factor, w * self.factor)

        """

        shape = x.shape

        # input : (b, c, h, w)
        # output: (b, c, h, 1, w, 1)
        x = torch.reshape(x, (-1, shape[1], shape[2], 1, shape[3], 1))

        # input : (b, c, h, 1, w, 1)
        # output: (b, c, h, self.factor, w, self.factor)
        x = torch.tile(x, (1, 1, 1, self.factor, 1, self.factor))

        # input : (b, c, h, self.factor, w, self.factor)
        # output: (b, c, h * self.factor, w * self.factor)
        x = torch.reshape(
            x, (-1, shape[1], shape[2] * self.factor, shape[3] * self.factor)
        )

        return x
