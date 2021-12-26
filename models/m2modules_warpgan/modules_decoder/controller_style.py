import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU


from .helpers.conv2d import Conv2d


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
        super().__init__()
        
        self.input_size = input_size
        self.batch_size = batch_size
        # Used in output channel calculations
        # Authors of the paper set it to 64 
        self.k = 64

        # inp: (in_batch, input_size)
        # out: (in_batch, 128)
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
        
        # inp: (in_batch, 128)
        # out: (in_batch, 4 * k)
        self.fc3 = nn.Linear(128, 4 * self.k, bias = True)
        self.initialize_weights_with_he_biases_with_zero(self.fc3) 
        
        # inp: (in_batch, 128)
        # out: (in_batch, 4 * k)
        self.fc4 = nn.Linear(128, 4 * self.k, bias = True)
        self.initialize_weights_with_he_biases_with_zero(self.fc4)

        
    def forward(self, x) -> torch.Tensor:
        """
        
        Forward function for Style Controller.
        Returns two concatenated (batch_size, 1, 1, 4 * k) shaped tensors, gamma and beta coefficients
        
        :param x: style encodings
            :shape: (batch_size, input_size)
        :return : out
            :shape: (batch_size, 2, 1, 4 * k)
        
        """
        
        def forward(self, x):
        
        if x is None:
            x = torch.randn((self.batch_size, self.input_size))
        
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
        
        # inp: (batch_size, 128)
        # out: (batch_size, 1, 1, 4 * k)
        gamma = self.fc3(x)
        gamma = torch.reshape(gamma, [-1, 1, 1, 4 * self.k])
        
        # inp: (batch_size, 128)
        # out: (batch_size, 1, 1, 4 * k)
        beta = self.fc4(x)
        beta = torch.reshape(beta, [-1, 1, 1, 4 * self.k])
        
        return torch.cat((beta, gamma), 0)
        
        
        
