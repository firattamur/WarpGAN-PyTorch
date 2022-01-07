import torch
import torch.nn as nn


from modules_decoder.controller_warp import WarpController
from modules_decoder.controller_style import StyleController
from modules_decoder.controller_decoder import DecoderController


class Decoder(nn.Module):
    """
    
    The Decoder network.
    
    """

    def __init__(self, args):
        """
        
        Decoder Network


        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
        :param initial          : initial channel number for convolution


        """
        super(Decoder).__init__()

        self.controller_warp    = WarpController(args)
        self.controller_style   = StyleController(args)
        self.controller_decoder = DecoderController(args)
        

    def forward(self, x: torch.Tensor, scales: torch.Tensor, styles: torch.Tensor, texture_only: bool = False) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        
        Forward function for Decoder.
        
        :param x        : Encoded Image features
            :shape: (in_batch, initial(default=64) * 4, in_height/4, in_width/4)

        :param styles   : Encoded style features
            :shape: (in_batch, style_size(default=8))

        :param scales   : Scale values for images
            :shape: (in_batch, 1)

        :return : 
            images_transformed:
                :shape: (in_batch, initial(default=64), in_height, in_width)
            images_rendered   :
                :shape: (in_batch, initial(default=64), in_height, in_width)
            landmarks_pred    :
                :shape: (in_batch, n_ldmark * 2)
            landmarks_norm    :
                :shape: (1)

        """

        # inp: (in_batch, style_size(default=8))
        # out_beta : (in_batch, 4 * k(default=8), 1, 1)
        # out_gamma: (in_batch, 4 * k(default=8), 1, 1)
        beta, gamma = self.controller_style(styles)

        # inp_x    : (in_batch, initial(default=64) * 4, in_height/4, in_width/4)
        # inp_beta : (in_batch, 4 * k(default=8), 1, 1)
        # inp_gamma: (in_batch, 4 * k(default=8), 1, 1)
        # out_encoded          : (in_batch, initial(default=64), in_height, in_width)
        # out_images_rendered  : (in_batch, initial(default=64), in_height, in_width)
        encoded, images_rendered = self.controller_decoder(x, beta, gamma)

        if texture_only:
            return images_rendered

        # inp_encoded        : (in_batch, initial(default=64), in_height, in_width)
        # inp_images_rendered: (in_batch, initial(default=64), in_height, in_width)
        # inp_scales         : (in_batch, 1)

        # out_images_transformed : (in_batch, initial(default=64), in_height, in_width)
        # out_landmarks_pred     : (in_batch, n_ldmark * 2)
        # out_landmarks_norm     : (1)
        images_transformed, landmarks_pred, landmarks_norm = self.controller_warp(encoded, images_rendered, scales)


        return images_transformed, images_rendered, landmarks_pred, landmarks_norm






        
        
        
        
