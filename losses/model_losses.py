
import torch
import typing
import torch.nn as nn
from torch.nn import functional as F


class PatchAdversarialLoss(nn.Module):
    """
    
    Patch adversarial loss
    
    """

    def __init__(self, args):
        super().__init__() 

        self.device = args.device

    def forward(self, logits_caric, logits_photo, logits_generated_caric) -> tuple:
        """       
        Forward function for patch adversarial loss.

        :param logits_caric: 
            :shape: (in_height/32 * in_width/32, in_channels)
        :param logits_photo: 
            :shape: (in_height/32 * in_width/32, in_channels)
        :param logits_generated_caric: 
            :shape: (in_height/32 * in_width/32, in_channels)

        :return : loss_D and loss_G
        
        """

        labels_D_A  = torch.zeros(logits_caric.shape[0:1],  dtype = torch.int64).to(self.device)
        labels_D_B  = torch.ones(logits_photo.shape[0:1],   dtype = torch.int64).to(self.device)
        
        labels_D_BA = torch.ones(logits_generated_caric.shape[0:1],  dtype = torch.int64).to(self.device) * 2
        labels_G_BA = torch.zeros(logits_generated_caric.shape[0:1], dtype = torch.int64).to(self.device)

        loss_D_A  = F.cross_entropy(input=logits_caric,  target=labels_D_A)
        loss_D_B  = F.cross_entropy(input=logits_photo,  target=labels_D_B)
        loss_D_BA = F.cross_entropy(input=logits_generated_caric, target=labels_D_BA)
        
        loss_G    = F.cross_entropy(input=logits_generated_caric, target=labels_G_BA)

        loss_D    = loss_D_A + loss_D_B + loss_D_BA

        return loss_D, loss_G


class AdversarialLoss(nn.Module):
    """
    
    Adversarial Loss.
        
    """

    def __init__(self, args):
        super().__init__()

        self.n_classes = args.n_classes
        self.device    = args.device

    def forward(self, output: typing.Dict[str, torch.Tensor]) -> tuple:
        """       
        Forward function for patch adversarial loss.

        :param output_dict: dictionary containing the outputs of discriminator and labels

            :shape: {

                logits_caric : (in_batch, n_classes)
                logits_photo : (in_batch, n_classes)
                logits_generated_caric: (in_batch, n_classes)

                labels_caric : (in_batch, 1)
                labels_photo : (in_batch, 1)
                labels_generated_caric : (in_batch, 1)

            }

        :return : loss_D and loss_G : double
        
        """

        # unpack output dict
        logits_caric  = output["logits_caric"]
        logits_photo  = output["logits_photo"]
        logits_generated_caric = output["logits_generated_caric"]

        labels_caric  = output["labels_caric"]
        labels_photo  = output["labels_photo"]
        labels_generated_caric = output["labels_generated_caric"]

        # Cross entropy function expects label input as long: int64 ??

        loss_D_A  = F.cross_entropy(input=logits_caric,  target=labels_caric)
        loss_D_B  = F.cross_entropy(input=logits_photo,  target=labels_photo)
        loss_D_BA = F.cross_entropy(input=logits_generated_caric, target=labels_generated_caric)

        loss_D    = loss_D_A + loss_D_B + loss_D_BA

        loss_G    = F.cross_entropy(input=logits_generated_caric, target=labels_generated_caric)

        return loss_D, loss_G