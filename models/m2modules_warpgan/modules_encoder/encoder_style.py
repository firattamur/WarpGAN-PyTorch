import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import LeakyReLU, ReLU


from ..helpers.conv2d import Conv2d


class StyleEncoder(nn.Module):
    """
    
    Style Encoder network.
    
    """

    def __init__(self, in_channels: int, n_classes: int, in_batch: int, in_height: int, in_width: int, style_size: int = 8, initial=64):
        """
        
        Style Encoder network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param style_size       : full connected layer size
        :param initial          : initial channel number for convolution
        
        """


        """
        
        Output dimension calculation from pytorch docs
        
        Convolution:
            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            
            Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        Pooling    :
            source: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

            Dout = ( Din + 2 * pad - kernel_size) / ( stride ) + 1

        """

        # inp: (in_batch, in_height, in_width,   in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        self.convs = nn.Sequential(

            # inp: (in_batch, in_height, in_width, in_channels)
            # out: (in_batch, in_height, in_width, initial)
            Conv2d(activation=nn.ReLU, in_channels=in_channels, out_channels=initial,     kernel_size=7, stride=1, pad=3),

            # inp: (in_batch, in_height,   in_width,   initial)
            # out: (in_batch, in_height/2, in_width/2, initial * 2)
            Conv2d(activation=nn.ReLU, in_channels=initial,     out_channels=initial * 2, kernel_size=4, stride=2, pad=3),

            # inp: (in_batch, in_height/2, in_width/2, initial * 2)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            Conv2d(activation=nn.ReLU, in_channels=initial * 2, out_channels=initial * 4, kernel_size=4, stride=2, pad=3),

        )

        # calculate height and width after convolution
        # convs out: (in_batch, in_height/4, in_width/4, initial * 4)
        out_height = in_height / 4
        out_width  = in_width  / 4

        # inp: (in_batch, in_height/2, in_width/2, initial * 4)
        # out: (in_batch, 1,           1,          initial * 4)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(out_height, out_width), stride=2, padding='valid')

        # inp: (in_batch, 1,           1,          initial * 4)
        # out: (in_batch, 1 * 1 * initial * 4)
        self.flatten = nn.Flatten()

        in_features = 1 * 1 * initial * 4

        # inp: (in_batch, 1 * 1 * initial * 4)
        # out: (in_batch, style_size)
        self.linear = nn.Linear(in_features=in_features, out_features=style_size)

        
    def forward(self, x) -> torch.Tensor:
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: (in_batch, in_height, in_width, in_channels)

        :return : style vector
            :shape: (in_batch, style_size)
        
        """
        
        # inp: (in_batch, in_height, in_width,     in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        out = self.convs(x)

        # inp: (in_batch, in_height/2, in_width/2, initial * 4)
        # out: (in_batch, 1,           1,          initial * 4)
        pooled = self.avg_pool2d(out)

        # inp: (in_batch, 1,           1,          initial * 4)
        # out: (in_batch, 1 * 1 * initial * 4)
        flatted = self.flatten(pooled)
        
        # inp: (in_batch, 1 * 1 * initial * 4)
        # out: (in_batch, style_size)
        style_vector = self.linear(flatted)

        # in the tensorflow implementation they have:
        #    -> style_vec = tf.identity(style_vec, name='style_vec')
        # but for pytorch no need to that...
        # here an answer: https://github.com/pytorch/pytorch/issues/9160#issuecomment-402494129
        return style_vector
        


