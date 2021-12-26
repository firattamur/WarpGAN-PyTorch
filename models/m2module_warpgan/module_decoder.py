import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU



class Decoder(nn.Module):
    """
    
    Decoder network.
    
    """

    def __init__(self, in_channels: int, k: int, gamma: torch.tensor, beta: torch.tensor):
        """
        
        Decoder Network
        :param in_channels      : number of channels in the content encoder output
        :k                      : k parameter of style controller
        :gamma                  : tensor of size batch_size x 1 x 1 x (4 * style_encoder.k)
        :beta                   : tensor of batch_size x 1 x 1 x (4 * style_encoder.k)
        """
        super().__init__()
        self.gamma = gamma
        self.beta = beta

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 4 * self.k, kernel_size=3)
        self.initialize_weights_and_biases(self.conv1)
        
        self.instance_norm_layer1 = nn.InstanceNorm2d(4 * self.k)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 3, bias = False)
        nn.init.kaiming_normal_(self.conv2.weight)
        
        self.instance_norm_layer2 = nn.InstanceNorm2d(4 * self.k)
        
        self.conv3 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 3)
        self.initialize_weights_and_biases(self.conv3)
        
        self.instance_norm_layer3 = nn.InstanceNorm2d(4 * self.k)
        self.relu2 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 3, bias = False)
        nn.init.kaiming_normal_(self.conv4.weight) 
        
        self.instance_norm_layer4 = nn.InstanceNorm2d(4 * self.k)

        self.conv5 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 3)
        self.initialize_weights_and_biases(self.conv5)
        
        self.instance_norm_layer5 = nn.InstanceNorm2d(4 * self.k)
        self.relu3 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 3, bias = False)
        nn.init.kaiming_normal_(self.conv6.weight)
        
        self.instance_norm_layer6 = nn.InstanceNorm2d(4 * self.k)
        
        self.conv7 = nn.Conv2d(in_channels = 4 * self.k, out_channels = 4 * self.k, kernel_size = 5)
        self.initialize_weights_and_biases(self.conv7)
        
        self.instance_norm_layer7 = nn.InstanceNorm2d(2 * self.k)
        self.relu4 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(2 * self.k, self.k, 5)
        self.initialize_weights_and_biases(self.conv8)
        
        self.instance_norm_layer8 = nn.InstanceNorm2d(self.k)
        self.relu5 = nn.ReLU()
        
        self.conv9 = nn.Conv2d(self.k, 3, 7)
        self.initialize_weights_and_biases(self.conv9, True)
        
        self.tanh = nn.Tanh()
    
    def upscale2dtorch(x, factor=2):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
            s = x.shape
            x = torch.reshape(x, (-1, s[1], 1, s[2], 1, s[3]))
            x = torch.tile(x, (1, 1, factor, 1, factor, 1))
            x = torch.reshape(x, (-1, s[1] * factor, s[2] * factor, s[3]))
        return x

    def padding2(x: torch.Tensor, pad: int, pad_mode='reflect') -> torch.Tensor:
        """
        
        Custom padding to apply before convolution layer.
        :param x: input image
        :param pad: padding size
        :param pad_mode: padding mode
            :options:
                - reflect
                - zero
        
        """

        if pad_mode == 'reflect':
            return F.pad(input=x, pad=(pad, pad, pad, pad, 0, 0, 0, 0), mode='reflect')

        if pad_mode == 'zero':
            return F.pad(input=x, pad=(pad, pad, pad, pad, 0, 0, 0, 0), mode='constant')

        raise ValueError(f"{pad_mode} must be one of ['reflect', 'zero']!")
        
    def initialize_weights_and_biases(self, layer : nn.Module, bothWeightAndBias = False):
        if not bothWeightAndBias:
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.fill_(0.0)
        else:
            layer.weight.data.fill_(0.0)
            layer.bias.data.fill_(0.0)
            
    def forward(self, x) -> torch.Tensor:
        """
        
        Forward function for Decoder.
        Returns  (...) shaped tensor
        
        :param x: content encodings
            :shape: ()
        :return : out
            :shape: ()
        
        """
        # inp: ()
        # out: ()
        
        x_ = self.conv1(padding2(x, 1, pad_mode="zero"))
        x_ = self.instance_norm_layer1(x_)
        x_ = gamma * x_ + beta
        x_ = self.relu1(x_)
        
        x += self.instance_norm_layer2(self.conv2(padding2(x_, 1, pad_mode="zero")))
        
        x_ = self.conv3(padding2(x, 1, pad_mode="zero"))
        x_ = self.instance_norm_layer3(x_)
        x_ = gamma * x_ + beta
        x_ = self.relu2(x_)
        
        x += self.instance_norm_layer4(self.conv4(padding2(x_, 1, pad_mode="zero")))
        
        x_ = self.conv5(padding2(x, 1, pad_mode="zero"))
        x_ = self.instance_norm_layer5(x_)
        x_ = gamma * x_ + beta
        x_ = self.relu3(x_)
        
        x += self.instance_norm_layer6(self.conv6(padding2(x_, 1, pad_mode="zero")))
        
        x = upscale2d(x, 2, pad_mode="zero")
        x = self.conv7(padding2(x, 2))
        x = self.instance_norm_layer7(x)
        x = self.relu4(x)
        
        x = upscale2d(x, 2)
        x = self.conv8(padding2(x, 2, pad_mode="zero"))
        x = self.instance_norm_layer8(x)
        x = self.relu5(x)
        
        x = self.conv9(padding2(x, 3, pad_mode="zero"))
        return self.tanh(x)
        
        
        
