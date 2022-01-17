import argparse
from importlib.machinery import SourceFileLoader


def load_config(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",  
                        help="The path to the train config file",
                        type=str,
                        default="./config/config.py")

    args = parser.parse_args()

    config_path = args.config
    config = SourceFileLoader('config', config_path).load_module()
    
    return config