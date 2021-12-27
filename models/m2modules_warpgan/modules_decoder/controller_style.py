import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU


from ...m1layers_warpgan.conv2d import CustomConv2d


class StyleController(nn.Module):
    """
    
    Style Controller network.
    
    """

    def __init__(self, batch_size: int, input_size = 8):
        """
        
        Style Controller Network
        :param batch_size      : number of examples in a batch
        :param input_size      : dimension of the style vectors
        
        """
        super(StyleController, self).__init__()
        
        self.input_size = input_size
        self.batch_size = batch_size

        # Used in output channel calculations
        # Authors of the paper set it to 64 
        self.k = 64

        """

        # inp: (in_batch, input_size)
        # out: (in_batch, 128)

        # TODO: @nmutlu18
        # TODO: should we use bias? https://github.com/google-research/tf-slim/blob/e00575ad39d19112a4b1342930825258316cf233/tf_slim/layers/layers.py#L1881

        self.fc1 = nn.Linear(self.input_size, 128, bias = True)     
        self.initialize_weights_with_he_biases_with_zero(self.fc1)
        
        # inp: (in_batch, 128)
        # out: (in_batch, 128)
        self.ln1 = nn.LayerNorm(128)
        
        # inp: (in_batch, 128)
        # out: (in_batch, 128)
        self.relu1 = nn.ReLU()
        
        # inp: (in_batch, 128)
        # out: (in_batch, 128)
        self.fc2 = nn.Linear(128, 128, bias = True)
        self.initialize_weights_with_he_biases_with_zero(self.fc2)
        
        # inp: (in_batch, 128)
        # out: (in_batch, 128)
        self.ln2 = nn.LayerNorm(128)
        
        # inp: (in_batch, 128)
        # out: (in_batch, 128)
        self.relu2 = nn.ReLU()

        """


        # inp: (in_batch, input_size)
        # out: (in_batch, 128)
        self.linears = nn.Sequential(
            
            # inp: (in_batch, input_size)
            # out: (in_batch, 128)
            nn.Linear(self.input_size, 128),

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

        # TODO: remove this
        # self.initialize_weights_with_he_biases_with_zero(self.fc3) 
        
        # inp: (in_batch, 128)
        # out: (in_batch, 4 * k)
        self.linear_beta  = nn.Linear(128, 4 * self.k, bias = True)

        # TODO: remove this
        # self.initialize_weights_with_he_biases_with_zero(self.fc4)

        # initialize all weights for module
        self.initialize_weights()

        
    def forward(self, x) -> tuple(torch.Tensor, torch.Tensor):
        """
        
        Forward function for Style Controller.
        Returns two concatenated (batch_size, 1, 1, 4 * k) shaped tensors, gamma and beta coefficients
        
        :param x: style encodings
            :shape: (batch_size, input_size)
        :return : out
            :shape: (batch_size, 2, 1, 4 * k)
        
        """
        
        if x is None:
            x = torch.randn((self.batch_size, self.input_size))
        
        # TODO: nmutlu
        # TODO: remove this
        """
        # inp: (batch_size, input_size)
        # out: (batch_size, 128)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        
        # inp: (batch_size, 128)
        # out: (batch_size, 128)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        """

        # inp: (batch_size, input_size)
        # out: (batch_size, 128)
        out = self.linears(x)

        # inp: (batch_size, 128)
        # out: (batch_size, 4 * k)
        gamma = self.linear_gamma(out)

        # inp: (batch_size, 4 * k)
        # out: (batch_size, 4 * k, 1, 1)
        gamma = torch.reshape(gamma, [-1, 4 * self.k, 1, 1])
        
        # inp: (batch_size, 128)
        # out: (batch_size, 4 * k, 1, 1)
        beta = self.linear_beta(out)

        # inp: (batch_size, 4 * k)
        # out: (batch_size, 4 * k, 1, 1)
        beta = torch.reshape(beta, [-1, 4 * self.k, 1, 1])
        

        # TODO: nmutlu
        # why do we have this?
        # return torch.cat((beta, gamma), 0)

        return beta, gamma
        
