import torch
import torch.nn as nn
import torch.functional as tf
from torch.nn.modules.activation import LeakyReLU, ReLU


from models.m1layers_warpgan.conv2d import CustomConv2d


class StyleEncoder(nn.Module):
    """
    
    Style Encoder network.
    
    """

    def __init__(self, args):
        """
        
        Style Encoder network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param style_size       : full connected layer size
        :param initial          : initial channel number for convolution
        
        """


        """
        
        Output dimension calculation from pytorch docs
        
        Convolution:
            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
            
            Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        Pooling    :
            source: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

            Dout = ( Din + 2 * pad - kernel_size) / ( stride ) + 1

        """
        super().__init__()


        # unpack input parameters from args
        self.initial      = args.initial
        self.in_channels  = args.in_channels
        self.out_channels = args.n_classes
        self.in_width     = args.in_width
        self.in_height    = args.in_height
        self.style_size   = args.style_size
        

        # inp: (in_batch, in_channels, in_height,   in_width)
        # out: (in_batch, initial * 4, in_height/4, in_width/4)
        self.convs = nn.Sequential(

            # inp: (in_batch, in_channels, in_height, in_width)
            # out: (in_batch, initial,     in_height, in_width)
            CustomConv2d(activation=nn.ReLU, in_channels=self.in_channels, out_channels=self.initial, kernel_size=7, stride=1, pad=3),

            # inp: (in_batch, initial,   in_height,   in_width)
            # out: (in_batch, initial*2, in_height/2, in_width/2)
            CustomConv2d(activation=nn.ReLU, in_channels=self.initial, out_channels=self.initial * 2, kernel_size=4, stride=2),

            # inp: (in_batch, initial*2, in_height/2, in_width/4)
            # out: (in_batch, initial*4, in_height/4, in_width/4)
            CustomConv2d(activation=nn.ReLU, in_channels=self.initial * 2, out_channels=self.initial * 4, kernel_size=4, stride=2),

        )

        # calculate height and width after convolution

        # convs out: (in_batch, initial*4, in_height/4, in_width/4)
        out_height = self.in_height // 4
        out_width  = self.in_width  // 4

        # inp: (in_batch, initial*4, in_height/4, in_width/4)
        # out: (in_batch, initial*4, 1,           1)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(out_height, out_width), stride=2)

        # inp: (in_batch, initial*4, 1,           1)
        # out: (in_batch, initial*4 * 1 * 1)
        self.flatten = nn.Flatten()

        in_features = 1 * 1 * self.initial * 4

        # inp: (in_batch, initial*4 * 1 * 1)
        # out: (in_batch, style_size)
        self.linear = nn.Linear(in_features=in_features, out_features=self.style_size)


        self.initialize_weights()    


    def forward(self, x) -> torch.Tensor:
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: (in_batch, in_channels, in_height, in_width)

        :return : style vector
            :shape: (in_batch, style_size)
        
        """
        
        # inp: (in_batch, in_channels, in_height,   in_width)
        # out: (in_batch, initial * 4, in_height/4, in_width/4)
        out = self.convs(x)

        # inp: (in_batch, initial*4, in_height/4, in_width/4)
        # out: (in_batch, initial*4, 1,           1)
        pooled = self.avg_pool2d(out)

        # inp: (in_batch, initial*4, 1,           1)
        # out: (in_batch, initial*4 * 1 * 1)
        flatted = self.flatten(pooled)
        
        # inp: (in_batch, initial*4 * 1 * 1)
        # out: (in_batch, style_size)
        style_vector = self.linear(flatted)

        # in the tensorflow implementation they have:
        #    -> style_vec = tf.identity(style_vec, name='style_vec')
        # but for pytorch no need to that...
        # here an answer: https://github.com/pytorch/pytorch/issues/9160#issuecomment-402494129
        
        return style_vector


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules.
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

