import torch
import typing
import torch.nn as nn


from models.m2modules_warpgan.module_encoder import Encoder
from models.m2modules_warpgan.module_decoder import Decoder
from models.m2modules_warpgan.module_discriminator import Discriminator


class WarpGANGenerator(nn.Module):
    """
    
    The WarpGAN Generator Model.
    
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
        super().__init__()

        self.is_train = args.is_train
        self.encoder  = Encoder(args)
        self.decoder  = Decoder(args)


    def forward(self, input: typing.Dict[str, torch.Tensor]) -> tuple:
        """
        
        Forward function for Discriminator.

        :param input: input dict contains input images, labels and scales
            :shape: {

                images_caric: (in_batch, in_channels, in_height, in_width)
                images_photo: (in_batch, in_channels, in_height, in_width)

                labels_caric: (in_batch, 1)
                labels_photo: (in_batch, 1)

                scales_caric: (in_batch, 1)
                scales_photo: (in_batch, 1)

            }

        :return : output dict contains required inputs for loss modules
            :shape: {

                generated_caric   : (in_batch, in_channels, in_height, in_width)
                rendered_generated_caric   : (in_batch, in_channels, in_height, in_width)

                landmark_pred : (in_batch, n_ldmark * 2)
                landmark_norm : (1)

                rendered_caric   : (in_batch, in_channels, in_height, in_width)
                rendered_photo   : (in_batch, in_channels, in_height, in_width)

            }
            
        """

        # model is in evaluation mode just return caricature
        if not self.is_train:

            images_photo = input["images_photo"]
            scales_photo = input["scales_photo"] 

            encoded_photo, styles_B = self.encoder(images_photo)
            generated_caric, _, _, _    = self.decoder(encoded_photo, scales_photo, None)

            return generated_caric

        images_caric = input["images_caric"]
        images_photo = input["images_photo"]

        labels_caric = input["labels_caric"]
        labels_photo = input["labels_photo"]

        scales_caric = input["scales_caric"]
        scales_photo = input["scales_photo"] 

        # --------------------------------------------------------
        # Module Encoder
        # --------------------------------------------------------

        encoded_caric, styles_A = self.encoder(images_caric)
        encoded_photo, styles_B = self.encoder(images_photo)

        # --------------------------------------------------------
        # Module Decoder
        # --------------------------------------------------------

        generated_caric, rendered_generated_caric, landmark_pred, landmark_norm = self.decoder(encoded_photo, scales_photo, None)
        
        rendered_caric = self.decoder(encoded_caric, scales_caric, styles_A, texture_only=True)
        rendered_photo = self.decoder(encoded_photo, scales_photo, styles_B, texture_only=True)

        return {

            "rendered_caric" : rendered_caric,
            "rendered_photo" : rendered_photo,

            "generated_caric" : generated_caric,
            "rendered_generated_caric" : rendered_generated_caric,

            "landmark_pred" : landmark_pred,
            "landmark_norm" : landmark_norm,

        }


class WarpGANDiscriminator(nn.Module):
    """
    
    The WarpGAN Discriminator Model.
    
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
        super().__init__()

        self.is_train = args.is_train
        self.discriminator = Discriminator(args)


    def forward(self, input: typing.Dict[str, torch.Tensor]) -> tuple:
        """
        
        Forward function for Discriminator.

        :param input: input dict contains input images, labels and scales
            :shape: {

                images_caric: (in_batch, in_channels, in_height, in_width)
                images_photo: (in_batch, in_channels, in_height, in_width)

                labels_caric: (in_batch, 1)
                labels_photo: (in_batch, 1)

                scales_caric: (in_batch, 1)
                scales_photo: (in_batch, 1)

            }

        :return : output dict contains required inputs for loss modules
            :shape: {

                logits_caric      : (in_batch, n_classes)
                logits_photo      : (in_batch, n_classes)
                logits_generated_caric     : (in_batch, n_classes)

                patch_logits_caric : (in_batch * in_height/32 * in_width/32, 3)
                patch_logits_photo : (in_batch * in_height/32 * in_width/32, 3)
                patch_logits_generated_caric: (in_batch * in_height/32 * in_width/32, 3)

            }
            
        """

        images_caric = input["images_caric"]
        images_photo = input["images_photo"]

        labels_caric = input["labels_caric"]
        labels_photo = input["labels_photo"]

        scales_caric = input["scales_caric"]
        scales_photo = input["scales_photo"] 

        # --------------------------------------------------------
        # Module Discriminator
        # --------------------------------------------------------

        patch_logits_caric, logits_caric   = self.discriminator(images_caric)
        patch_logits_photo, logits_photo   = self.discriminator(images_photo)

        patch_logits_generated_caric, logits_generated_caric = self.discriminator(generated_caric)

        return {

            "logits_caric" : logits_caric,
            "logits_photo" : logits_photo,
            "logits_generated_caric": logits_generated_caric,

            "patch_logits_caric" : patch_logits_caric,
            "patch_logits_photo" : patch_logits_photo,
            "patch_logits_generated_caric": patch_logits_generated_caric,

        }
