import torch
import torch.nn as nn


from .m2modules_warpgan.module_encoder import Encoder
from .m2modules_warpgan.module_decoder import Decoder
from .m2modules_warpgan.module_discriminator import Discriminator


class WarpGAN(nn.Module):
    """
    
    The WarpGAN Model.
    
    """

    def __init__(self, args):
        """
        
        The WarpGAN Model.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param style_size       : full connected layer size
        :param initial          : initial channel number for convolution
        
        """
        super(WarpGAN, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.discriminator = Discriminator(args)


    def forward(self, input: dict) -> tuple(torch.Tensor, torch.Tensor):
        """
        
        Forward function for Discriminator.

        :param input: input dict contains input images, labels and scales
            :shape: {

                images_A: (in_batch, in_channels, in_height, in_width)
                images_B: (in_batch, in_channels, in_height, in_width)

                labels_A: (in_batch, 1)
                labels_B: (in_batch, 1)

                scales_A: (in_batch, 1)
                scales_B: (in_batch, 1)

            }

        :return : output dict contains required inputs for loss modules
            :shape: {

                patch_logits_A: (in_batch * in_height/32 * in_width/32, 3)
                patch_logits_B: (in_batch * in_height/32 * in_width/32, 3)

                logits_A      : (in_batch, n_classes)
                logits_B      : (in_batch, n_classes)

                rendered_AA   : (in_batch, in_channels, in_height, in_width)
                rendered_BB   : (in_batch, in_channels, in_height, in_width)

            }
            
        """

        images_A = input["images_A"]
        images_B = input["images_B"]

        labels_A = input["labels_A"]
        labels_B = input["labels_B"]

        scales_A = input["scales_A"]
        scales_B = input["scales_B"] 

        # --------------------------------------------------------
        # Module Encoder
        # --------------------------------------------------------

        encoded_A, styles_A = self.encoder(images_A)
        encoded_B, styles_B = self.encoder(images_B)

        # --------------------------------------------------------
        # Module Decoder
        # --------------------------------------------------------

        deformed_BA, rendered_BA, landmark_pred, landmark_norm = self.decoder(encoded_B, scales_B, None)
        
        rendered_AA = self.decoder(encoded_A, scales_A, styles_A, texture_only=True)
        rendered_BB = self.decoder(encoded_B, scales_B, styles_B, texture_only=True)

        # --------------------------------------------------------
        # Module Discriminator
        # --------------------------------------------------------

        patch_logits_A, logits_A = self.discriminator(images_A)
        patch_logits_B, logits_B = self.discriminator(images_B)

        patch_logits_BA, logits_BA = self.discriminator(deformed_BA)

        return {

            # to calculate Patch Advesarial loss for deform_BA

            "patch_logits_A": patch_logits_A,
            "patch_logits_B": patch_logits_B,

            "logits_A": logits_A,
            "logits_B": logits_B,

            # to calculate identity loss

            "rendered_AA" : rendered_AA,
            "rendered_BB" : rendered_BB,

        }



