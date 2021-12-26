import torch
import torch.nn as nn
import torch.functional as tf


class CustomDeConv2d(nn.Module):
    def __init__(self, activation: nn.Module, pad: int = 1, **kwargs):
        """

        Custom deconvolution module that applies padding before deconvolution and activation after deconvolution.

        :param in_channels : number of input  channels
        :param out_channels: number of output channels
        :param kernel_size : size of kernel
        :param stride : stride of convolution

        """
        super(CustomDeConv2d, self).__init__()

        self.pad = pad

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(**kwargs, padding="valid"), activation(inplace=True)
        )

    def forward(self, x) -> torch.Tensor:
        """

        Calculate deconvolution following by activation.

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
            return tf.pad(
                input=x, pad=(0, 0, pad, pad, pad, pad, 0, 0), mode="constant"
            )

        raise ValueError(f"{pad_mode} must be one of ['reflect', 'zero']!")
