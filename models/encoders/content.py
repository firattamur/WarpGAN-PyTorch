import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU


from .helpers.conv2d import Conv2d


class ContentEncoder(nn.Module):
    """
    
    Content Encoder network.
    
    """

    def __init__(self, in_channels: int, n_classes: int, in_batch: int, in_height: int, in_width: int, initial=64):
        """
        
        Content Encoder network.
        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
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
            Conv2d(activation=nn.ReLU, in_channels=in_channels, out_channels=initial, kernel_size=7, stride=1, pad=3),

            # inp: (in_batch, in_height,   in_width,   initial)
            # out: (in_batch, in_height/2, in_width/2, initial * 2)
            Conv2d(activation=nn.ReLU, in_channels=initial, out_channels=initial * 2, kernel_size=4, stride=2),

            # inp: (in_batch, in_height/2, in_width/2, initial * 2)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            Conv2d(activation=nn.ReLU, in_channels=initial * 2, out_channels=initial * 4, kernel_size=4, stride=2),

        )
        # convs out: (in_batch, in_height/4, in_width/4, initial * 4)

        '''
        for i in range(3):
            net_ = conv(net, 4*k, 3, scope='res{}_0'.format(i))
            net += conv(net_, 4*k, 3, activation_fn=None, biases_initializer=None, scope='res{}_1'.format(i))
            print('module res{} shape:'.format(i), [dim.value for dim in net.shape])

        '''
        self.res1 = nn.Sequential(

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            Conv2d(activation=nn.ReLU, in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1),

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            nn.Conv2d(in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1, padding=1),
        )
        self.res2 = nn.Sequential(

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            Conv2d(activation=nn.ReLU, in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1),

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            nn.Conv2d(in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1, padding=1),
        )
        self.res3 = nn.Sequential(

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            Conv2d(activation=nn.ReLU, in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1),

            # inp: (in_batch, in_height/4, in_width/4, initial * 4)
            # out: (in_batch, in_height/4, in_width/4, initial * 4)
            nn.Conv2d(in_channels=initial * 4, out_channels=initial * 4, kernel_size=3, stride=1, padding=1),
        )

        
    def forward(self, x) -> torch.Tensor:
        """
        
        Forward function for Discriminator.
        :param x: input image
            :shape: (in_batch, in_height, in_width, in_channels)
        :return : out
            :shape: (in_batch, in_height/4, in_width/4, initial * 4)
        
        """
        
        # inp: (in_batch, in_height, in_width,     in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        out = self.convs(x)
        # inp: (in_batch, in_height, in_width,     in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        out += self.res1(out)
        # inp: (in_batch, in_height, in_width,     in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        out += self.res2(out)
        # inp: (in_batch, in_height, in_width,     in_channels)
        # out: (in_batch, in_height/4, in_width/4, initial * 4)
        out += self.res3(out)

        return out
        
        
        
