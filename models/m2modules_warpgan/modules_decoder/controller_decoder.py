import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import ReLU


from models.m1layers_warpgan.conv2d import CustomConv2d
from models.m1layers_warpgan.deconv2d import CustomDeConv2d
from models.m1layers_warpgan.upscale2d import CustomUpScale2d
from models.m1layers_warpgan.sequential import CustomSequential
from models.m1layers_warpgan.instancenorm2d import CustomInstanceNorm2d


class DecoderController(nn.Module):
    """
    
    Decoder Controller network.
    
    """

    def __init__(self, args):
        """
        
        Content Encoder network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
        :param initial          : initial channel number for convolution
        
        """

        """
        
        Output dimension calculation from pytorch docs
        
        Convolution:
            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            
            Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        Convolution Transpose:
            source: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#convtranspose2d

            Dout = (Din - 1) x stride - 2 * pad + dilation * ( kernel_size - 1 ) + out_pad + 1

        Pooling    :
            source: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
            Dout = ( Din + 2 * pad - kernel_size) / ( stride ) + 1

        """
        super().__init__()

        # unpack input parameters from args
        self.in_channels  = args.initial   * 4
        self.in_width     = args.in_width  // 4
        self.in_height    = args.in_height // 4

        self.res1 = CustomSequential(

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1),
            
            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

        )

        self.res2 = CustomSequential(

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
        
            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

        )

        self.res3 = CustomSequential(

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
        
            # inp: (in_batch, initial*4, in_height, in_width)
            # out: (in_batch, initial*4, in_height, in_width)
            CustomInstanceNorm2d(self.in_channels),

        )

        # inp: (in_batch, initial*4, in_height,   in_width)
        # out: (in_batch, initial,   in_height*4, in_width*4)
        self.deconvs = nn.Sequential(

            # inp: (in_batch, initial*4, in_height,   in_width)
            # out: (in_batch, initial*4, in_height*2, in_width*2)
            CustomUpScale2d(factor=2),

            # inp: (in_batch, initial*4, in_height*2, in_width*2)
            # out: (in_batch, initial*2, in_height*2, in_width*2)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels,      out_channels=self.in_channels // 2, kernel_size=5, stride=1, pad=2),
            
            # inp: (in_batch, initial*4, in_height*2, in_width*2)
            # out: (in_batch, initial*2, in_height*2, in_width*2)
            nn.InstanceNorm2d(self.in_channels // 2),

            # inp: (in_batch, initial*2, in_height*2, in_width*2)
            # out: (in_batch, initial*2, in_height*4, in_width*4)
            CustomUpScale2d(factor=2),

            # inp: (in_batch, initial*2, in_height*4, in_width*4)
            # out: (in_batch, initial,   in_height*4, in_width*4)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels // 2, out_channels=self.in_channels // 4, kernel_size=5, stride=1, pad=2),
            
            # inp: (in_batch, initial, in_height*4, in_width*4)
            # out: (in_batch, initial, in_height*4, in_width*4)
            nn.InstanceNorm2d(self.in_channels // 4)

        )

        # inp: (in_batch, initial,   in_height*4, in_width*4)
        # out: (in_batch, 3,         in_height*4, in_width*4)
        self.conv = nn.Conv2d(in_channels=self.in_channels//4, out_channels=3, kernel_size=7, stride=1, padding=3)
        # nn.init.zeros_(self.conv.weight)

        # inp: (in_batch, initial, in_height*4, in_width*4)
        # out: (in_batch, initial, in_height*4, in_width*4)
        self.tanh = nn.Tanh()

        # initalize layer weights
        self.initialize_weights()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> tuple:
        """
        
        Forward function for Style Controller.
        Returns two concatenated (batch_size, 1, 1, 4 * k) shaped tensors, gamma and beta coefficients
        
        :param x: content encodings
            :shape: (in_batch, initial * 4, in_height/4, in_width/4)

        :param gamma: style controller encodings
            :shape: (batch_size, 4 * k, 1, 1)

        :param beta : style controller encodings
            :shape: (batch_size, 4 * k, 1, 1)

        :return : out
            :shape: (in_batch, initial, in_height, in_width)
        
        """

        # inp: (in_batch, initial*4, in_height, in_width)
        # out: (in_batch, initial*4, in_height, in_width)
        out1 = self.res1((x,   gamma, beta))

        # inp: (in_batch, initial*4, in_height, in_width)
        # out: (in_batch, initial*4, in_height, in_width)
        out2 = out1[0] + self.res2(out1)[0]
        
        # inp: (in_batch, initial*4, in_height, in_width)
        # out: (in_batch, initial*4, in_height, in_width)
        out3 = out2 + self.res3((out2, gamma, beta))[0]

        # inp: (in_batch, initial*4, in_height,   in_width)
        # out: (in_batch, initial,   in_height*4, in_width*4)
        out4 = self.deconvs(out3)

        # inp: (in_batch, initial,   in_height*4, in_width*4)
        # out: (in_batch, 3,         in_height*4, in_width*4)
        out5 = self.conv(out4)

        # inp: (in_batch, 3,         in_height*4, in_width*4)
        # out: (in_batch, 3,         in_height*4, in_width*4)
        images_rendered  = self.tanh(out5)

        return out5, images_rendered


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules.
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)