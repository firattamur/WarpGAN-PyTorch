"""

Test WarpGAN Model.

"""


import sys
from utils.commandline import load_config


if __name__ == "__main__":

    # load configuration file from specified configuration file path
    config = load_config(sys.argv)
    
    print(config.image_size)

