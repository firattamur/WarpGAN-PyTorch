import torch
import torch.nn as nn
import torch.functional as tf

from .helpers.conv2d import Conv2dLeaklyRelu


class Discriminator(nn.Module):
    """
    
    Discriminator network.
    
    """

    def __init__(self, in_channels: int, n_classes: int, bottleneck_size: int = 512):

        # collection of convs with kernel=4, stride=2
        # leakyReLu applied after each conv
        # padding=1 and mode is reflect
        self.convs = nn.Sequential(

            """
            
            Output dimension calculation from pytorch docs
            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

            Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

            Dout = (Dint 2 * 1 - 1 * (4 - 1) - 1 / ( 2 ) + 1) -> (Din - 2) / 2 + 1 => Din / 2

            """

            Conv2dLeaklyRelu(in_channels=in_channels, out_channels=32),

            Conv2dLeaklyRelu(in_channels=32, out_channels=64),

            Conv2dLeaklyRelu(in_channels=64, out_channels=128),
            
            Conv2dLeaklyRelu(in_channels=128, out_channels=256),

            Conv2dLeaklyRelu(in_channels=256, out_channels=512),

        )

        # patch discriminator
        self.conv2d = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding='same')
        
        # global discriminator
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=100, out_features=100)


        self.linear2 = nn.Linear(in_features=100, out_features=100)



    
