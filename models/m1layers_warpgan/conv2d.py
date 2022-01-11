import torch
import torch.nn as nn
import torch.nn.functional as tf


class CustomConv2d(nn.Module):


    def __init__(self, activation: nn.Module, in_channels: int, out_channels: int, kernel_size: int, stride: int,  pad: int = 1):
        """

        Custom convolution module that applies padding before convolution and activation after convolution.

        :param in_channels : number of input  channels
        :param out_channels: number of output channels
        :param kernel_size : size of kernel
        :param stride      : stride of convolution

        """
        super().__init__()

        self.pad = pad

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding="valid"),
            activation(inplace=True)
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

        if pad_mode == "zero":
            return tf.pad(input=x, pad=(pad, pad, pad, pad), mode="constant")

        if pad_mode == "reflect":
            return tf.pad(input=x, pad=(pad, pad, pad, pad), mode="reflect")

        raise ValueError(f"{pad_mode} must be one of ['reflect', 'zero']!")


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules.
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)