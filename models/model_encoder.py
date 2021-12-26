import torch
import torch.nn as nn
import torch.functional as tf


class Encoder(nn.Module):
    """
    
    The proposed deformable generator in WarpGAN is composed of three sub-networks: 

    - Content Encoder Ec,
    - Decoder R 
    - Warp controller.
    
    """


    def __init__(self):
        pass


    def forward(self, x):
        pass


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of Endoder modules.

        :param modules: list of modules to initialize weights
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    