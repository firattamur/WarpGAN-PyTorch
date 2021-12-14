import torch
import torch.nn as nn
import torch.functional as tf

from .helpers.conv2d import Conv2d
from .helpers.deconv2d import Deconv2d


class Discriminator(nn.Module):
    """
    
    Discriminator network.
    
    """

    def __init__(self, in_channels: int, n_classes: int, in_batch: int, in_height: int, in_width: int, bottleneck_size: int = 512):
        """
        
        Discriminator network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
        :param bottleneck_size  : size of bottleneck

        """

        """
        
        Output dimension calculation from pytorch docs
        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

        Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        """

        self.convs = nn.Sequential(

            # inp: (in_batch, in_height,   in_width,   in_channels)
            # out: (in_batch, in_height/2, in_width/2, 32)
            Conv2d(activation=nn.LeakyReLU, in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),

            # inp: (in_batch, in_height/2, in_width/2, 32)
            # out: (in_batch, in_height/4, in_width/4, 64)
            Conv2d(activation=nn.LeakyReLU, in_channels=32,         out_channels=64, kernel_size=4, stride=2),

            # inp: (in_batch, in_height/4, in_width/4, 64)
            # out: (in_batch, in_height/8, in_width/8, 128)
            Conv2d(activation=nn.LeakyReLU, in_channels=64,         out_channels=128, kernel_size=4, stride=2),
            
            # inp: (in_batch, in_height/8,  in_width/8, 128)
            # out: (in_batch, in_height/16, in_width/16, 256)
            Conv2d(activation=nn.LeakyReLU, in_channels=128,        out_channels=256, kernel_size=4, stride=2),

            # inp: (in_batch, in_height/16, in_width/16, 256)
            # out: (in_batch, in_height/32, in_width/32, 512)
            Conv2d(activation=nn.LeakyReLU, in_channels=256,        out_channels=512, kernel_size=4, stride=2),

        )

        # patch discriminator
        # inp: (in_batch, in_height/32, in_width/32, 512)
        # out: (in_batch, in_height/32, in_width/32, 3)
        self.conv2d = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding='same')
        
        # global discriminator
        # inp: (in_batch, in_height/32, in_width/32, 512)
        # out: (in_batch, in_height/32 * in_width/32 * 512)
        self.flatten = nn.Flatten()

        # size of flatten tensor
        # dimension reduces to half after each conv layer that's why:

        out_height = in_height / 32
        out_width  = in_width  / 32

        in_features = 512 * out_height * out_width * in_batch

        # inp: (in_batch, in_height/32 * in_width/32 * 512)
        # out: (in_batch, 512)
        self.linear1 = nn.Linear(in_features=in_features, out_features=bottleneck_size)

        # inp: (in_batch, 512)
        # out: (in_batch, n_classes)
        self.linear2 = nn.Linear(in_features=bottleneck_size, out_features=n_classes)

        # initalize all network weights
        self.initialize_weights()


    def forward(self, x: torch.Tensor) -> tuple:
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: (in_batch, in_height,    in_width, in_channels)

        :return : patch5_logits and logits
            :shape patch5_logits: (in_batch, in_height/32, in_width/32, 3)
            :shape        logits: (in_batch, n_classes)
        
        """

        # inp: (in_batch, in_height,    in_width,    in_channels)
        # out: (in_batch, in_height/32, in_width/32, 512)
        out = self.convs(x)

        # Patch Discrminator
        # inp: (in_batch, in_height/32, in_width/32, 512)
        # out: (in_batch, in_height/32, in_width/32, 3)
        patch5_logits = self.conv2d(out)

        # inp: (in_batch, in_height/32, in_width/32, 3)
        # out: (in_batch * in_height/32 * in_width/32, 3)
        patch5_logits = patch5_logits.reshape(-1, 3)

        # Global Discriminator
        # inp: (in_batch, in_height/32, in_width/32, 512)
        # out: (in_batch, in_height/32 * in_width/32 * 512)
        flatten = self.flatten(out)

        # inp: (in_batch, in_height/32 * in_width/32 * 512)
        # out: (in_batch, 512)
        prelogits = self.linear1(flatten)

        # inp: (in_batch, in_height/32 * in_width/32 * 512)
        # out: (in_batch, 512)
        prelogits = tf.normalize(prelogits, axis=1)

        # inp: (in_batch, 512)
        # out: (in_batch, n_classes)
        logits = self.linear2(prelogits)

        return patch5_logits, logits


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules in Discriminator.

        :param modules: list of modules to initialize weights
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    

