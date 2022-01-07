"""
Training file for WarpGAN

"""

import argparse
import sys


def main(args):
    pass

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    
    parser.add_argument("--name", help="Rename the log dir",
                        type=str, default=None)
    
    parser.add_argument("--in_channels", help="Initial channels in the input images", type = int, default = 3)
    
    parser.add_argument("--n_classes", help = "Number of identities", type = int, default = None)
    
    parser.add_argument("--in_batch", help = "Number of samples in a batch", type = int, default = None)
    
    parser.add_argument("--in_height", help = "Height of the input images", type = int, default = 256)
    
    parser.add_argument("--in_width", help = "Width of the input images", type = int, default = 256)
    
    parser.add_argument("--style_size", help = "Sizes of style vector", type = int, default=8)
    
    parser.add_argument("--initial", help = "Value for convolution layer sizes", type = int, default = 64)
    
    parser.add_argument("--k", help = "k parameter in style controller", type = int, default = 64)
    
    parser.add_argument("--bottleneck_size", help = "Size of bottleneck in discriminator", type = int, default = 512)
    
    parser.add_argument("--n_ldmark", help = "", type = int, default = 16)
    
    return parser.parse_args(argv)

if __name__ == "__main__":

    main(parse_arguments(sys[1:]))