import torch
import torch.nn as nn
import torch.functional as tf


class CustomConv2d(nn.Module):


    def __init__(self, activation: nn.Module, pad: int = 1, **kwargs):
        """

        Custom convolution module that applies padding before convolution and activation after convolution.

        :param in_channels : number of input  channels
        :param out_channels: number of output channels
        :param kernel_size : size of kernel
        :param stride      : stride of convolution

        """
        super(CustomConv2d, self).__init__()

        self.pad = pad

        self.conv_block = nn.Sequential(
            nn.Conv2d(**kwargs, padding="valid"), activation(inplace=True)
        )

        self.initialize_weights()

    def forward(self, x) -> torch.Tensor:
        """

        Calculate convolution following by activation.

        :param x: torch.Tensor
            :shape: (b, c, h, w)

        :return : torch.Tensor
            :shape: (b, c, h_new, w_new)

        """

        padded = self.padding(x, pad=self.pad)
        return self.conv_block(padded)

    def padding(self, x: torch.Tensor, pad: int, pad_mode="reflect") -> torch.Tensor:
        """

        Custom padding to apply before convolution layer.

        :param x: input image
        :param pad: padding size
        :param pad_mode: padding mode
            :options:
                - reflect
                - zero

        """

        if pad_mode == "reflect":
            return tf.pad(input=x, pad=(0, 0, pad, pad, pad, pad, 0, 0), mode="reflect")

        if pad_mode == "zero":
            return tf.pad(input=x, pad=(0, 0, pad, pad, pad, pad, 0, 0), mode="constant")

        raise ValueError(f"{pad_mode} must be one of ['reflect', 'zero']!")


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules.
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias:
                    nn.init.zeros_(module.bias)