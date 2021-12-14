import torch
import torch.nn as nn


def initialize_weights(modules: list) -> None:
    """
    
    Initialize weights of given module list.

    :param modules: list of modules to initialize weights
    
    """

    for module in modules:

        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)

        if isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)

        if isinstance(module, nn.LeakyReLU):
            pass

        if isinstance(module, nn.Sequential):
            pass

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)


        
