import torch
import torch.nn as nn
import torch.nn.functional as tf

from models.m1layers_warpgan.conv2d import CustomConv2d


class Discriminator(nn.Module):
    """
    
    Discriminator network.
    
    """

    def __init__(self, args):
        """
        
        Discriminator network.

        :param in_channels      : number of channels
        :param n_classes        : number of classes
        :param in_batch         : batch size
        :param in_height        : height of input image
        :param in_width         : width of input image
        :param bottleneck_size  : size of bottleneck

        """

        """
        
        Output dimension calculation from pytorch docs
        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

        Dout = ( Din + 2 * pad - dilation * ( kernel_size - 1 ) - 1 ) / ( stride ) + 1

        """
        super().__init__()

        # unpack input parameters from args
        self.in_channels     = args.in_channels
        self.out_channels    = args.n_classes
        self.in_batch        = args.in_batch
        self.in_width        = args.in_width
        self.in_height       = args.in_height
        self.bottleneck_size = args.bottleneck_size

        self.convs = nn.Sequential(

            # inp: (in_batch, in_channels, in_height,    in_width)
            # out: (in_batch, 32,          in_height/2,  in_width/2)
            CustomConv2d(activation=nn.LeakyReLU, in_channels=self.in_channels, out_channels=32, kernel_size=4, stride=2),

            # inp: (in_batch, 32,          in_height/2,  in_width/2)
            # out: (in_batch, 64,          in_height/4,  in_width/4)
            CustomConv2d(activation=nn.LeakyReLU, in_channels=32,          out_channels=64, kernel_size=4, stride=2),

            # inp: (in_batch, 64,          in_height/4,  in_width/4)
            # out: (in_batch, 128,         in_height/8,  in_width/8)
            CustomConv2d(activation=nn.LeakyReLU, in_channels=64,          out_channels=128, kernel_size=4, stride=2),
            
            # inp: (in_batch, 128,         in_height/8,   in_width/8)
            # out: (in_batch, 256,         in_height/16,  in_width/16)
            CustomConv2d(activation=nn.LeakyReLU, in_channels=128,         out_channels=256, kernel_size=4, stride=2),

            # inp: (in_batch, 256,         in_height/16,  in_width/16)
            # out: (in_batch, 512,         in_height/32,  in_width/32)
            CustomConv2d(activation=nn.LeakyReLU, in_channels=256,         out_channels=512, kernel_size=4, stride=2),

        )

        # patch discriminator
        # inp: (in_batch, 512,         in_height/32,  in_width/32)
        # out: (in_batch, 3,           in_height/32,  in_width/32)
        self.conv2d = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding='same')
        
        # global discriminator
        # inp: (in_batch, 512,  in_height/32,  in_width/32)
        # out: (in_batch, 512 * in_height/32 * in_width/32)
        self.flatten = nn.Flatten()

        # size of flatten tensor
        # dimension reduces to half after each conv layer that's why:

        out_height = self.in_height // 32
        out_width  = self.in_width  // 32

        in_features = 512 * out_height * out_width

        # inp: (in_batch, 512 * in_height/32 * in_width/32)
        # out: (in_batch, 512)
        self.linear1 = nn.Linear(in_features=in_features, out_features=self.bottleneck_size)

        # inp: (in_batch, 512)
        # out: (in_batch, n_classes)
        self.linear2 = nn.Linear(in_features=self.bottleneck_size, out_features=self.out_channels)

        # initalize all network weights
        self.initialize_weights()


    def forward(self, x: torch.Tensor) -> tuple:
        """
        
        Forward function for Discriminator.

        :param x: input image
            :shape: (in_batch, in_height, in_width, in_channels)

        :return : patch5_logits and logits
            :shape patch5_logits: (in_batch, in_height/32, in_width/32, 3)
            :shape        logits: (in_batch, n_classes)
        
        """

        # inp: (in_batch, in_channels, in_height,     in_width)
        # out: (in_batch, 512,         in_height/32,  in_width/32)
        out = self.convs(x)

        # Patch Discriminator
        # inp: (in_batch, 512,         in_height/32,  in_width/32)
        # out: (in_batch, 3,           in_height/32,  in_width/32)
        patch5_logits = self.conv2d(out)

        # inp: (in_batch, 3,  in_height/32,  in_width/32)
        # out: (in_batch * in_height/32 * in_width/32, 3)
        patch5_logits = patch5_logits.reshape(-1, 3)

        # Global Discriminator
        # inp: (in_batch, 512,  in_height/32,  in_width/32)
        # out: (in_batch, 512 * in_height/32 * in_width/32)
        flatten = self.flatten(out)

        # inp: (in_batch, 512 * in_height/32 * in_width/32)
        # out: (in_batch, 512)
        prelogits = self.linear1(flatten)

        # inp: (in_batch, 512)
        # out: (in_batch, 512)
        prelogits = tf.normalize(prelogits, dim=1)

        # inp: (in_batch, 512)
        # out: (in_batch, n_classes)
        logits = self.linear2(prelogits)

        return patch5_logits, logits


    def initialize_weights(self) -> None:
        """
        
        Initialize weights of modules in Discriminator.

        :param modules: list of modules to initialize weights
        
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    

