import torch
import torch.nn as nn


from models.m1layers_warpgan.instancenorm2d import CustomInstanceNorm2d


class CustomSequential(nn.Sequential):
    """

    In sequential module we have convolution layers and after convolution we have our CustomInstanceNorm2d layer. 
    Because CustomInstanceNorm2d accepts 3 inputs, x: torch.Tensor, gamma: torch.Tensor and beta: torch.Tensor. 
    We need to pass 3 inputs to Sequential layer. Normal pytorch nn.Sequential only support single input.
    This class is created to pass multiple inputs to CustomInstanceNorm2d in Sequential layer.
    

    """


    def forward(self, x: tuple) -> torch.Tensor:
        """
        
        :param x: tuple of torch.Tensors
            :shape: (b, c, h, w)

            if type(x) is tuple:

                :param gamma: style controller encodings
                    :shape: (batch_size, c, 1, 1)

                :param beta : style controller encodings
                    :shape: (batch_size, c, 1, 1)

        """

        for module in self._modules.values():

            if isinstance(module, CustomInstanceNorm2d):
                x = (module(*x),   x[1], x[2])
            else:
                x = (module(x[0]), x[1], x[2])

        return x 