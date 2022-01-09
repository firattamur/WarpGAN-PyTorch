import torch
import torch.nn as nn


from modules_encoder.encoder_content import ContentEncoder
from modules_encoder.encoder_style   import StyleEncoder


class Encoder(nn.Module):
    """
    
    The Encoder Network. 
    
    """

    def __init__(self, args):
        """
        
        Style Encoder network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param style_size       : full connected layer size
        :param initial          : initial channel number for convolution
        
        """
        super(Encoder, self).__init__()

        # content encoder
        self.content_encoder = ContentEncoder(args)

        # style encoder
        self.style_encoder   = StyleEncoder(args)


    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: (in_batch, in_height, in_width, in_channels)

        :return : 
            content vector:
                :shape: (in_batch, initial * 4, in_height/4, in_width/4)
            
            style vector
                :shape: (in_batch, style_size)
            
        """

        # inp: (in_batch, in_channels, in_height,   in_width)
        # out: (in_batch, initial * 4, in_height/4, in_width/4)
        content = self.content_encoder(x)

        # inp: (in_batch, in_channels, in_height,   in_width)
        # out: (in_batch, style_size)
        style   = self.style_encoder(x)

        return content, style

