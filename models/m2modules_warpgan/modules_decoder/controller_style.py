import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU


from models.m1layers_warpgan.conv2d import CustomConv2d


class StyleController(nn.Module):
    """
    
    Style Controller network.
    
    """

    def __init__(self, args):
        """
        
        Style Controller Network

        :param batch_size      : number of examples in a batch
        :param input_size      : dimension of the style vectors
        
        """
        super().__init__()

        # unpack input parameters from args        
        self.batch_size = args.in_batch
        self.k          = args.k
        self.style_size = args.style_size
        self.device     = args.device

        # inp: (in_batch, input_size)
        # out: (in_batch, 128)
        self.linears = nn.Sequential(
            
            # inp: (in_batch, input_size)
            # out: (in_batch, 128)
            nn.Linear(self.style_size, 128),

            # inp: (in_batch, 128)
            # out: (in_batch, 128)
            nn.LayerNorm(128),

            # inp: (in_batch, 128)
            # out: (in_batch, 128)
            nn.ReLU(),

            # inp: (in_batch, 128)
            # out: (in_batch, 128)
            nn.Linear(128, 128),

            # inp: (in_batch, 128)
            # out: (in_batch, 128)
            nn.LayerNorm(128),

            # inp: (in_batch, 128)
            # out: (in_batch, 128)
            nn.ReLU(),

        )

        # inp: (in_batch, 128)
        # out: (in_batch, 4 * k)
        self.linear_gamma = nn.Linear(128, 4 * self.k, bias = True)
        
        # inp: (in_batch, 128)
        # out: (in_batch, 4 * k)
        self.linear_beta  = nn.Linear(128, 4 * self.k, bias = True)

        # initialize all weights for module
        self.initialize_weights()

        
    def forward(self, x) -> tuple:
        """
        
        Forward function for Style Controller.

        Returns two (batch_size, 1, 1, 4 * k) shaped tensors, gamma and beta coefficients
        
        :param x: style encodings
            :shape: (batch_size, style_size)
        :return : out
            :shape: (batch_size, 2, 1, 4 * k)
        
        """
        
        if x is None:
            x = torch.randn((self.batch_size, self.style_size)).to(self.device)
        
        # inp: (batch_size, style_size)
        # out: (batch_size, 128)
        out = self.linears(x)

        # inp: (batch_size, 128)
        # out: (batch_size, 4 * k)
        gamma = self.linear_gamma(out)

        # inp: (batch_size, 4 * k)
        # out: (batch_size, 4 * k, 1, 1)
        gamma = gamma.view([-1, 4 * self.k, 1, 1])
        
        # inp: (batch_size, 128)
        # out: (batch_size, 4 * k, 1, 1)
        beta = self.linear_beta(out)

        # inp: (batch_size, 4 * k)
        # out: (batch_size, 4 * k, 1, 1)
        beta = beta.view([-1, 4 * self.k, 1, 1])
        
        return beta, gamma
        

    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules.
        
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
