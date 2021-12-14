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

        # collection of convs with kernel=4, stride=2
        # leakyReLu applied after each conv
        # padding=1 and mode is reflect

        """
        
        Output dimension calculation from pytorch docs
        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

        Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        Dout = (Dint 2 * 1 - 1 * (4 - 1) - 1 / ( 2 ) + 1) -> (Din - 2) / 2 + 1 => Din / 2

        """

        self.convs = nn.Sequential(

            Conv2d(activation=nn.LeakyReLU, in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),

            Conv2d(activation=nn.LeakyReLU, in_channels=32, out_channels=64, kernel_size=4, stride=2),

            Conv2d(activation=nn.LeakyReLU, in_channels=64, out_channels=128, kernel_size=4, stride=2),
            
            Conv2d(activation=nn.LeakyReLU, in_channels=128, out_channels=256, kernel_size=4, stride=2),

            Conv2d(activation=nn.LeakyReLU, in_channels=256, out_channels=512, kernel_size=4, stride=2),

        )

        # patch discriminator
        self.conv2d = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding='same')
        
        # global discriminator
        self.flatten = nn.Flatten()

        # size of flatten tensor
        # dimension reduces to half after each conv layer that's why:

        final_height = in_height / (2 ** 5)
        final_width  = in_width  / (2 ** 5)

        in_features = 512 * final_height * final_width * in_batch

        self.linear1 = nn.Linear(in_features=in_features, out_features=bottleneck_size)
        self.linear2 = nn.Linear(in_features=bottleneck_size, out_features=n_classes)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: ?

        :return : patch5_logits and logits
            :shape patch5_logits: ?
            :shape        logits: ?
        
        """

        out = self.convs(x)

        # Patch Discrminator
        patch5_logits = self.conv2d(out)
        patch5_logits = patch5_logits.reshape(-1, 3)

        # Global Discriminator
        flatten = self.flatten(out)

        prelogits = self.linear1(flatten)
        prelogits = tf.normalize(prelogits, axis=1)

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
    

