
import torch
import typing
import torch.nn as nn
from torch.nn import functional as F


class PatchAdversarialLoss(nn.Module):
    """
    
    Patch adversarial loss
    
    """

    def _init_(self):
        super().__init__() 


    def forward(self, logits_A, logits_B, logits_BA) -> tuple:
        """       
        Forward function for patch adversarial loss.

        :param logits_A: 
            :shape: (in_height/32 * in_width/32, in_channels)
        :param logits_B: 
            :shape: (in_height/32 * in_width/32, in_channels)
        :param logits_BA: 
            :shape: (in_height/32 * in_width/32, in_channels)

        :return : loss_D and loss_G
        
        """

        labels_D_A  = torch.zeros(logits_A.shape[0:1],  dtype = torch.int64)
        labels_D_B  = torch.ones(logits_B.shape[0:1],   dtype = torch.int64)
        
        labels_D_BA = torch.ones(logits_BA.shape[0:1],  dtype = torch.int64) * 2
        labels_G_BA = torch.zeros(logits_BA.shape[0:1], dtype = torch.int64)

        loss_D_A  = F.cross_entropy(input=logits_A,  target=labels_D_A)
        loss_D_B  = F.cross_entropy(input=logits_B,  target=labels_D_B)
        loss_D_BA = F.cross_entropy(input=logits_BA, target=labels_D_BA)
        loss_G    = F.cross_entropy(input=logits_BA, target=labels_G_BA)

        loss_D = loss_D_A + loss_D_B + loss_D_BA

        return loss_D, loss_G


class AdversarialLoss(nn.Module):
    """
    
    Adversarial Loss.
        
    """

    def _init_(self):
        super().__init__()


    def forward(self, output: typing.Dict[str, torch.Tensor], num_classes: int) -> tuple:
        """       
        Forward function for patch adversarial loss.

        :param output_dict: dictionary containing the outputs of discriminator and labels
            :shape: {

                logits_A : (in_batch, n_classes)
                logits_B : (in_batch, n_classes)
                logits_BA: (in_batch, n_classes)

                labels_A : (in_batch, 1)
                labels_B : (in_batch, 1)
                label_BA : (in_batch, 1)

            }

        :return : loss_D and loss_G : double
        
        """

        # unpack output dict
        logits_A  = output["logits_A"]
        logits_B  = output["logits_B"]
        logits_BA = output["logits_BA"]

        labels_A  = output["labels_A"]
        labels_B  = output["labels_B"]
        labels_BA = output["label_BA"]

        # Cross entropy function expects label input as long: int64 ??

        loss_D_A  = F.cross_entropy(input=logits_A,  target=labels_A)
        loss_D_B  = F.cross_entropy(input=logits_B,  target=labels_B  + num_classes)
        loss_D_BA = F.cross_entropy(input=logits_BA, target=labels_BA + 2 * num_classes)

        loss_D = loss_D_A + loss_D_B + loss_D_BA

        loss_G = F.cross_entropy(input=logits_BA, target=labels_BA)

        return loss_D, loss_G