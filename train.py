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
    
    return parser.parse_args(argv)

if __name__ == "__main__":

    main(parse_arguments(sys[1:]))