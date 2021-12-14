import torch
import torch.nn as nn
import torch.functional as tf


class Conv2dLeaklyRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2):
        """
        
        Custom convolution following by activation and batch normalization.

        :param in_channels : number of input  channels
        :param out_channels: number of output channels
        :param kernel_size : size of kernel 
        :param stride : stride of convolution
        
        """

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=1, padding_mode='reflect'),

    
    def forward(self, x):
        """
        
        Calculate convolution following by activation.

        :param x: input image
        
        """

        out = self.conv(x)
        out = tf.leakyReLu(out)

        return out


