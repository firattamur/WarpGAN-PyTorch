import torch
import torch.nn as nn
from typing import Union


class CustomSequential(nn.Sequential):

    def forward(self, x: Union[torch.Tensor, tuple(torch.Tensor, torch.Tensor, torch.Tensor)]) -> torch.Tensor:
        """

        :param x: torch.Tensor or tuple of torch.Tensors if it is a tuple it contains gamma and beta.
            :shape: (b, c, h, w)

            if type(x) is tuple:

                :param gamma: style controller encodings
                    :shape: (batch_size, c, 1, 1)

                :param beta : style controller encodings
                    :shape: (batch_size, c, 1, 1)

        """

        for module in self._modules.values():

            # this case is applied for our custom instance norm layer
            # custom instance norm layer accepts x, gamma and beta
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)

        return inputs